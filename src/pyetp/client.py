import asyncio
import contextlib
import datetime
import logging
import typing as T
import uuid
import warnings
from collections import defaultdict
from collections.abc import AsyncGenerator, Generator, Sequence
from types import TracebackType

import numpy as np
import numpy.typing as npt
import websockets
import websockets.client
from pydantic import SecretStr

import resqml_objects.v201 as ro
from energistics.avro_handler import (
    encode_message,
    decode_message,
    CompressionAlgorithm,
    GzipCompression,
)
from energistics.base import ETPBaseProtocolModel, Protocol, Role
from energistics.etp.v12.datatypes import (
    AnyArrayType,
    AnyLogicalArrayType,
    ArrayOfString,
    DataValue,
    EndpointCapabilityKind,
    ErrorInfo,
    MessageHeader,
    SupportedDataObject,
    SupportedProtocol,
    Uuid,
    Version,
)
from energistics.etp.v12.datatypes.data_array_types import (
    DataArrayIdentifier,
    DataArrayMetadata,
    GetDataSubarraysType,
    PutDataArraysType,
    PutDataSubarraysType,
    PutUninitializedDataArrayType,
)
from energistics.etp.v12.datatypes.message_header import MessageHeaderFlags
from energistics.etp.v12.datatypes.object import (
    ContextInfo,
    ContextScopeKind,
    DataObject,
    Dataspace,
    RelationshipKind,
    Resource,
)
from energistics.etp.v12.protocol.core import (
    Authorize,
    AuthorizeResponse,
    CloseSession,
    OpenSession,
    ProtocolException,
    RequestSession,
)
from energistics.etp.v12.protocol.data_array import (
    GetDataArrayMetadata,
    GetDataArrayMetadataResponse,
    GetDataArrays,
    GetDataArraysResponse,
    GetDataSubarrays,
    GetDataSubarraysResponse,
    PutDataArrays,
    PutDataArraysResponse,
    PutDataSubarrays,
    PutDataSubarraysResponse,
    PutUninitializedDataArrays,
    PutUninitializedDataArraysResponse,
)
from energistics.etp.v12.protocol.dataspace import (
    DeleteDataspaces,
    DeleteDataspacesResponse,
    GetDataspaces,
    GetDataspacesResponse,
    PutDataspaces,
    PutDataspacesResponse,
)
from energistics.etp.v12.protocol.discovery import (
    GetResources,
    GetResourcesResponse,
)
from energistics.etp.v12.protocol.store import (
    DeleteDataObjects,
    DeleteDataObjectsResponse,
    GetDataObjects,
    GetDataObjectsResponse,
    PutDataObjects,
    PutDataObjectsResponse,
)
from energistics.etp.v12.protocol.transaction import (
    CommitTransaction,
    RollbackTransaction,
    StartTransaction,
    StartTransactionResponse,
)
from energistics.uris import DataObjectURI, DataspaceURI
from pyetp import utils_arrays
from pyetp._version import version
from pyetp.errors import ETPTransactionFailure
from resqml_objects import parse_resqml_v201_object, serialize_resqml_v201_object

logger = logging.getLogger(__name__)


class ETPError(Exception):
    def __init__(self, message: str, code: int):
        self.message = message
        self.code = code
        super().__init__(f"{message} ({code=:})")

    @classmethod
    def from_proto(cls, error: ErrorInfo):
        assert error is not None, "passed no error info"
        return cls(error.message, error.code)

    @classmethod
    def from_protos(cls, errors: Sequence[ErrorInfo]):
        return list(map(cls.from_proto, errors))


class ReceiveWorkerExited(Exception):
    pass


class ETPMessageTooLarge(Exception):
    def __init__(self, message: str, message_size: int):
        self.message_size = message_size
        super().__init__(message)


class ETPClient:
    def __init__(
        self,
        ws: websockets.ClientConnection,
        etp_timeout: float | None = 10.0,
    ) -> None:
        self.ws = ws
        self.max_size = self.ws.protocol.max_message_size
        # We need to add some slack to the array messages to handle the rest of
        # the message body. This size is a guess! The only way to be absolutely
        # sure is to encode the message, and then check if it is too large.
        self.max_array_size_margin = 3000

        self.message_id = 2

        self.application_name = "pyetp"
        self.application_version = version

        self.client_instance_id = uuid.uuid4()

        # We request all protocols that we support.
        self.requested_protocols = [
            self.get_default_server_supported_protocols(rp)
            for rp in [
                Protocol.DISCOVERY,
                Protocol.STORE,
                Protocol.DATA_ARRAY,
                Protocol.TRANSACTION,
                Protocol.DATASPACE,
            ]
        ]
        # This is done for the initial `RequestSession`-message. When we get a
        # corresponding `OpenSession` we populate `negotiated_protocols`
        # appropriately.
        self.negotiated_protocols = self.requested_protocols

        self.supported_data_objects = [
            SupportedDataObject(qualified_type="eml20.*"),
            SupportedDataObject(qualified_type="resqml20."),
        ]

        self.supported_compression = [GzipCompression]
        self.negotiated_compression = None
        self.supported_formats = ["xml"]

        self.endpoint_capabilities = {
            EndpointCapabilityKind.MAX_WEB_SOCKET_MESSAGE_PAYLOAD_SIZE: DataValue(
                item=self.max_size
            ),
        }

        self._recv_events: dict[int, asyncio.Event] = {}
        self._recv_buffer: dict[int, list[ETPBaseProtocolModel]] = defaultdict(
            lambda: list()
        )

        # Ensure a minimum timeout of 10 seconds.
        self.etp_timeout = (
            etp_timeout if etp_timeout is None or etp_timeout > 10.0 else 10.0
        )
        self.__recvtask = asyncio.create_task(self.__receiver_loop())

    @staticmethod
    def get_default_server_supported_protocols(
        protocol: int | Protocol,
    ) -> SupportedProtocol:
        match Protocol(protocol):
            case Protocol.CORE:
                return SupportedProtocol(protocol=protocol, role=Role.SERVER)
            case Protocol.CHANNEL_STREAMING:
                return SupportedProtocol(protocol=protocol, role=Role.PRODUCER)
            case _:
                return SupportedProtocol(protocol=protocol, role=Role.STORE)

    @staticmethod
    def get_server_capabilities_url(url: str) -> str:
        url = urllib.parse.urlparse(url)
        if url.scheme == "ws":
            url = url._replace(scheme="http")
        elif url.scheme == "wss":
            url = url._replace(scheme="https")

        url = urllib.parse.urljoin(
            url.geturl(),
            (
                ".well-known/etp-server-capabilities"
                "?GetVersion=etp12.energistics.org&$format=binary"
            ),
        )

        return url

    def get_message_id(self) -> int:
        ret_id = self.message_id
        self.message_id += 2

        return ret_id

    async def send_and_recv(
        self,
        body: ETPBaseProtocolModel,
        multi_part_bodies: list[ETPBaseProtocolModel] = [],
    ) -> list[ETPBaseProtocolModel]:
        correlation_id = await self.send(body=body, multi_part_bodies=multi_part_bodies)
        return await self.recv(correlation_id)

    async def send(
        self,
        body: ETPBaseProtocolModel,
        multi_part_bodies: list[ETPBaseProtocolModel] = [],
    ) -> int:
        # The core protocol is _always_ supported.
        if (
            body._protocol != Protocol.CORE
            # Consider checking the role in the body as well. The
            # negotiated_protocols-list contains a list of the protocols and
            # roles that the _server uses_.
            and body._protocol not in [np.protocol for np in self.negotiated_protocols]
        ):
            raise ValueError(
                f"Message '{body.__class__.__name}' belongs to protocol "
                f"{body._protocol} which is not included in the negotiated protocols "
                f"{self.negotiated_protocols}."
            )

        message_id = self.get_message_id()

        compression_flag = (
            MessageHeaderFlags.COMPRESSED
            if self.negotiated_compression is not None
            and body._protocol != Protocol.CORE
            else 0x0
        )
        fin_flag = MessageHeaderFlags.FIN if len(multi_part_bodies) == 0 else 0x0

        header = MessageHeader.from_etp_protocol_body(
            body=body,
            message_id=message_id,
            message_flags=fin_flag | compression_flag,
        )

        message = encode_message(
            header=header,
            body=body,
            compression_func=self.negotiated_compression.compress
            if self.negotiated_compression is not None
            else None,
        )

        if len(message) > self.max_size:
            raise ETPMessageTooLarge(
                message=(
                    f"Message with header {header} is too large: {len(message) = } "
                    f"> max_size = {self.max_size}"
                ),
                message_size=len(message),
            )

        self._recv_events[header.message_id] = asyncio.Event()

        tasks = [self.ws.send(message)]
        for i, mpb in enumerate(multi_part_bodies):
            fin_flag = 0x0
            if i == len(multi_part_bodies) - 1:
                fin_flag = MessageHeaderFlags.FIN
            mpb_header = MessageHeader.from_etp_protocol_body(
                body=mpb,
                message_id=self.get_message_id(),
                message_flags=fin_flag | compression_flag,
                correlation_id=header.message_id,
            )
            message = encode_message(
                header=mpb_header,
                body=mpb,
                compression_func=self.negotiated_compression.compress,
            )

            if len(message) > self.max_size:
                raise ETPMessageTooLarge(
                    message=(
                        f"Message with header {mpb_header} is too large: "
                        f"{len(message) = } > max_size = {self.max_size}"
                    ),
                    message_size=len(message),
                )
            tasks.append(self.ws_send(message))

        await asyncio.gather(*tasks)

        return header.message_id

    async def recv(self, correlation_id: int) -> list[ETPBaseProtocolModel]:
        assert correlation_id in self._recv_events, (
            "Trying to receive a response on non-existing message"
        )

        for ti in timeout_intervals(self.etp_timeout):
            try:
                # Wait for an event for `ti` seconds.
                async with asyncio.timeout(ti):
                    await self._recv_events[correlation_id].wait()
            except TimeoutError:
                # Check if the receiver task is still running.
                if self.__recvtask.done():
                    # Raise any errors by waiting for the task to finish.
                    await self.__recvtask

                    # Check that the receiver task stopped due to a
                    # (successfully) closed websockets connection.
                    try:
                        await self.ws.recv()
                    except websockets.ConnectionClosedOK:
                        pass

                    # Terminate client with an error.
                    raise ReceiveWorkerExited(
                        "Receiver task terminated prematurely due to a closed "
                        "websockets connection"
                    )
            else:
                # Break out of for-loop, and start processing message.
                break
        else:
            # The for-loop finished without breaking. In other words, we have
            # timed out.
            assert self.etp_timeout is not None
            raise TimeoutError(
                f"Receiver task did not set event within {self.etp_timeout} seconds"
            )

        # Remove event from list of events.
        del self._recv_events[correlation_id]
        # Read message bodies from buffer.
        bodies = self._recv_buffer.pop(correlation_id)

        # Check if there are errors in the received messages.
        errors = self.parse_error_info(bodies)

        # Raise errors in case there are any.
        if len(errors) == 1:
            raise ETPError.from_proto(errors.pop())
        elif len(errors) > 1:
            raise ExceptionGroup(
                "Server responded with ETPErrors:", ETPError.from_protos(errors)
            )

        return bodies

    async def __receiver_loop(self):
        logger.debug("Starting receiver loop")

        # Using `async for` makes the receiver task exit without errors on a
        # `websockets.exceptions.ConnectionClosedOK`-exception. This ensures a
        # smoother clean-up in case the main-task errors resulting in a closed
        # websockets connection down the line.
        async for message in self.ws:
            header, body = decode_message(
                message,
                decompression_func=self.negotiated_compression.decompress
                if self.negotiated_compression is not None
                else None,
            )

            logger.debug(
                f"Receiver task got message type {body.__class__.__name__} with "
                f"header {header}"
            )
            self._recv_buffer[header.correlation_id].append(body)

            if header.is_final_message():
                self._recv_events[header.correlation_id].set()

        logger.info("Websockets connection closed and receiver task stopped")

    async def __aenter__(self) -> T.Self:
        rs = RequestSession(
            application_name=self.application_name,
            application_version=self.application_version,
            client_instance_id=self.client_instance_id,
            requested_protocols=self.requested_protocols,
            supported_data_objects=self.supported_data_objects,
            supported_compression=[sc.name for sc in self.supported_compression],
            supported_formats=self.supported_formats,
            endpoint_capabilities=self.endpoint_capabilities,
        )

        responses = await self.send_and_recv(rs)
        assert len(responses) == 1
        os = responses[0]
        assert_response(os, OpenSession)
        logger.info(f"Session opened:\n{os}")

        self.server_application_name = os.application_name
        self.server_application_version = os.application_version
        self.server_instance_id = os.server_instance_id
        self.negotiated_protocols = os.supported_protocols
        # We currently do not use this information, but should in the future.
        # The way to use it is to limit which type of objects can be passed in
        # the client.
        self.negotiated_data_objects = os.supported_data_objects

        # There should only be a single compression algorithm in the
        # `OpenSession.supported_compression`-field corresponding to the first
        # hit in the requested compression list. We therefore locate the
        # algorithm that first matches.
        if self.supported_compression and os.supported_compression:
            for sc in self.supported_compression:
                if sc.name in os.supported_compression:
                    self.negotiated_compression = sc
                    break

        elif self.supported_compression and not os.supported_compression:
            logger.info(
                "The server does not support any of the compression algorithms "
                "requested. Continuing without compression."
            )

        # We don't use the negotiated_formats for anything yet, as we only
        # support XML for now.
        assert "xml" in os.negotiated_formats
        self.negotiated_formats = os.negotiated_formats

        self.session_id = os.session_id

        server_max_size = os.endpoint_capabilities[
            EndpointCapabilityKind.MAX_WEB_SOCKET_MESSAGE_PAYLOAD_SIZE
        ].item

        if server_max_size < self.max_size:
            self.max_size = server_max_size

        # We currently do not use this capability as it is quite large for the
        # open-etp-server. Most likely the limit will be the max size of the
        # web socket message. However, both should ideally be checked.
        self.max_uncompressed_size = os.endpoint_capabilities[
            EndpointCapabilityKind.MAX_MESSAGE_PAYLOAD_UNCOMPRESSED_SIZE
        ].item

    async def send(self, body: ETPBaseProtocolModel) -> list[ETPBaseProtocolModel]:
        correlation_id = await self._send(body)
        return await self._recv(correlation_id)

    async def _send(self, body: ETPBaseProtocolModel) -> int:
        msg = Message.get_object_message(body, message_flags=MessageFlags.FINALPART)
        if msg is None:
            raise TypeError(f"{type(body)} not valid etp protocol")

        msg.header.message_id = self.consume_msg_id()
        logger.debug(f"sending {msg.body.__class__.__name__} {repr(msg.header)}")

        # create future recv event
        self._recv_events[msg.header.message_id] = asyncio.Event()

        tasks = []
        for msg_part in msg.encode_message_generator(self.max_size, self):
            tasks.append(self.ws.send(msg_part))

        await asyncio.gather(*tasks)

        return msg.header.message_id

    @staticmethod
    def parse_error_info(bodies: list[ETPBaseProtocolModel]) -> list[ErrorInfo]:
        # returns all error infos from bodies
        errors = []
        for body in bodies:
            if isinstance(body, ProtocolException):
                if body.error is not None:
                    errors.append(body.error)
                errors.extend(body.errors.values())
        return errors

    async def __aexit__(
        self,
        exc_type: T.Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        close_session_sent = False
        try:
            await self._send(CloseSession(reason="Client exiting"))
            close_session_sent = True
        except websockets.ConnectionClosed:
            logger.error(
                "Websockets connection is closed, unable to send a CloseSession-message"
                " to the server"
            )
        finally:
            # Check if the receive task is done, and if not, stop it.
            if not self.__recvtask.done():
                self.__recvtask.cancel("stopped")

            self.is_connected = False

        try:
            # Raise any potential exceptions that might have occured in the
            # receive task
            await self.__recvtask
        except asyncio.CancelledError:
            # No errors except for a cancellation, which is to be expected.
            pass
        except websockets.ConnectionClosed as e:
            # The receive task errored on a closed websockets connection.
            logger.error(
                "The receiver task errored on a closed websockets connection. The "
                f"message was: {e.__class__.__name__}: {e}"
            )

        if len(self._recv_buffer) > 0:
            logger.error(
                f"Connection is closed, but there are {len(self._recv_buffer)} "
                "messages left in the buffer"
            )

        # Check if there were any messages left in the websockets connection.
        # Reading them will speed up the closing of the connection.
        counter = 0
        try:
            # In some cases the server does not drop the connection after we
            # have sent the `CloseSession`-message. We therefore add a timeout
            # to the reading of possibly lost messages.
            async with asyncio.timeout(self.etp_timeout or 10):
                async for msg in self.ws:
                    counter += 1
        except websockets.ConnectionClosed:
            # The websockets connection has already closed. Either successfully
            # or with an error, but we ignore both cases.
            pass
        except TimeoutError:
            if close_session_sent:
                logger.warning(
                    "Websockets connection was not closed within "
                    f"{self.etp_timeout or 10} seconds after the "
                    "`CloseSession`-message was sent"
                )

        if counter > 0:
            logger.error(
                f"There were {counter} unread messages in the websockets connection "
                "after the session was closed"
            )

        logger.debug("Client closed")

    async def close(self) -> None:
        """Closing method that tears down the ETP-connection via the
        `ETPClient.__aexit__`-method, and closes the websockets connection.
        This method should _only_ be used if the user has set up a connection
        via `etp_client = await connect(...)` or `etp_client = await
        etp_connect(...)` and will handle the closing of the connection
        manually.
        """

        await self.__aexit__(None, None, None)
        # The websockets connection should be closed from the ETP-server once
        # it has received a `CloseSession`-message. However, calling close on
        # the websockets connection does not do anything if it is already
        # closed.
        await self.ws.close()

    async def __aenter__(self) -> T.Self:
        return await self.request_session()

    async def request_session(self):
        # Handshake protocol
        etp_version = Version(major=1, minor=2, revision=0, patch=0)

        def get_protocol_server_role(protocol: CommunicationProtocol) -> str:
            match protocol:
                case CommunicationProtocol.CORE:
                    return "server"
                case CommunicationProtocol.CHANNEL_STREAMING:
                    return "producer"

            return "store"

        msgs = await self.send(
            RequestSession(
                applicationName=self.application_name,
                applicationVersion=self.application_version,
                clientInstanceId=uuid.uuid4(),  # type: ignore
                requestedProtocols=[
                    SupportedProtocol(
                        protocol=p.value,
                        protocolVersion=etp_version,
                        role=get_protocol_server_role(p),
                    )
                    for p in CommunicationProtocol
                ],
                supportedDataObjects=[
                    SupportedDataObject(qualifiedType="resqml20.*"),
                    SupportedDataObject(qualifiedType="eml20.*"),
                ],
                currentDateTime=self.timestamp,
                earliestRetainedChangeTime=0,
                endpointCapabilities=dict(
                    MaxWebSocketMessagePayloadSize=DataValue(item=self.max_size)
                ),
            )
        )

        assert len(msgs) == 1
        msg = msgs[0]

        assert msg and isinstance(msg, OpenSession)

        self.is_connected = True

        # ignore this endpoint
        _ = msg.endpoint_capabilities.pop("MessageQueueDepth", None)
        self.client_info.negotiate(msg)

        return self

    async def authorize(
        self, authorization: str, supplemental_authorization: T.Mapping[str, str] = {}
    ):
        msgs = await self.send(
            Authorize(
                authorization=authorization,
                supplementalAuthorization=supplemental_authorization,
            )
        )
        assert len(msgs) == 1
        msg = msgs[0]

        assert msg and isinstance(msg, AuthorizeResponse)

        return msg

    @staticmethod
    def assert_response(
        response: ETPBaseProtocolModel, expected_type: T.Type[ETPBaseProtocolModel]
    ) -> None:
        assert isinstance(response, expected_type), (
            f"Expected {expected_type}, got {type(response)} with content {response}"
        )

    @property
    def max_array_size(self):
        if self.max_size <= self.max_array_size_margin:
            raise AttributeError(
                "The maximum size of a websocket message must be greater than "
                f"{self.max_array_size_margin}"
            )
        return self.max_size - self.max_array_size_margin

    @property
    def timestamp(self):
        return int(datetime.datetime.now(datetime.timezone.utc).timestamp())

    def dataspace_uri(self, ds: str) -> DataspaceURI:
        if ds.count("/") > 1:
            raise Exception("Max one / in dataspace name")
        return DataspaceURI.from_name(ds)

    async def list_objects(
        self, dataspace_uri: DataspaceURI | str, depth: int = 1
    ) -> GetResourcesResponse:
        responses = await self.send(
            GetResources(
                scope=ContextScopeKind.TARGETS_OR_SELF,
                context=ContextInfo(
                    uri=str(dataspace_uri),
                    depth=depth,
                    dataObjectTypes=[],
                    navigableEdges=RelationshipKind.PRIMARY,
                ),
            )
        )
        assert len(responses) == 1
        return responses[0]

    #
    # dataspace
    #

    async def get_dataspaces(
        self, store_last_write_filter: int = None
    ) -> GetDataspacesResponse:
        responses = await self.send(
            GetDataspaces(store_last_write_filter=store_last_write_filter)
        )

        assert all(
            [isinstance(response, GetDataspacesResponse) for response in responses]
        ), "Expected GetDataspacesResponse"
        assert len(responses) == 1

        return responses[0]

    async def put_dataspaces(
        self,
        legaltags: list[str],
        otherRelevantDataCountries: list[str],
        owners: list[str],
        viewers: list[str],
        *dataspace_uris: DataspaceURI,
    ) -> dict[str, str]:
        _uris = list(map(DataspaceURI.from_any, dataspace_uris))
        for i in _uris:
            if i.raw_uri.count("/") > 4:  # includes the 3 eml
                raise Exception("Max one / in dataspace name")
        time = self.timestamp
        responses = await self.send(
            PutDataspaces(
                dataspaces={
                    d.raw_uri: Dataspace(
                        uri=d.raw_uri,
                        storeCreated=time,
                        storeLastWrite=time,
                        path=d.dataspace,
                        custom_data={
                            "legaltags": DataValue(
                                item=ArrayOfString(values=legaltags)
                            ),
                            "otherRelevantDataCountries": DataValue(
                                item=ArrayOfString(values=otherRelevantDataCountries)
                            ),
                            "owners": DataValue(item=ArrayOfString(values=owners)),
                            "viewers": DataValue(item=ArrayOfString(values=viewers)),
                        },
                    )
                    for d in _uris
                }
            )
        )
        assert all(
            [isinstance(response, PutDataspacesResponse) for response in responses]
        ), "Expected PutDataspacesResponse"

        successes = {}
        for response in responses:
            successes = {**successes, **response.success}

        assert len(successes) == len(dataspace_uris), (
            f"expected {len(dataspace_uris)} successes"
        )

        return successes

    async def put_dataspaces_no_raise(
        self,
        legaltags: list[str],
        otherRelevantDataCountries: list[str],
        owners: list[str],
        viewers: list[str],
        *dataspace_uris: DataspaceURI,
    ) -> dict[str, str]:
        try:
            return await self.put_dataspaces(
                legaltags, otherRelevantDataCountries, owners, viewers, *dataspace_uris
            )
        except ETPError:
            return {}

    async def delete_dataspaces(self, *dataspace_uris: DataspaceURI) -> dict[str, str]:
        _uris = list(map(str, dataspace_uris))

        responses = await self.send(DeleteDataspaces(uris=dict(zip(_uris, _uris))))
        assert all(
            [isinstance(response, DeleteDataspacesResponse) for response in responses]
        ), "Expected DeleteDataspacesResponse"

        successes = {}
        for response in responses:
            successes = {**successes, **response.success}

        assert len(successes) == len(dataspace_uris), (
            f"expected {len(dataspace_uris)} successes"
        )
        return successes

    async def get_data_objects(self, *uris: T.Union[DataObjectURI, str]):
        tasks = []
        for uri in uris:
            task = self.send(GetDataObjects(uris={str(uri): str(uri)}))
            tasks.append(task)

        task_responses = await asyncio.gather(*tasks)
        responses = [r for tr in task_responses for r in tr]
        assert len(responses) == len(uris)

        data_objects = []
        errors = []
        for uri, response in zip(uris, responses):
            if not isinstance(response, GetDataObjectsResponse):
                errors.append(
                    TypeError(
                        "Expected GetDataObjectsResponse, got "
                        f"{response.__class__.__name} with content: {response}",
                    )
                )
            data_objects.append(response.data_objects[str(uri)])

        if len(errors) > 0:
            raise ExceptionGroup(
                f"There were {len(errors)} errors in ETPClient.get_data_objects",
                errors,
            )

        return data_objects

    async def put_data_objects(self, *objs: DataObject):
        tasks = []
        for obj in objs:
            task = self.send(
                PutDataObjects(
                    data_objects={f"{obj.resource.name} - {obj.resource.uri}": obj},
                ),
            )
            tasks.append(task)

        task_responses = await asyncio.gather(*tasks)
        responses = [r for tr in task_responses for r in tr]

        errors = []
        for response in responses:
            if not isinstance(response, PutDataObjectsResponse):
                errors.append(
                    TypeError(
                        "Expected PutDataObjectsResponse, got "
                        f"{response.__class__.__name} with content: {response}",
                    )
                )
        if len(errors) > 0:
            raise ExceptionGroup(
                f"There were {len(errors)} errors in ETPClient.put_data_objects",
                errors,
            )

        sucesses = {}
        for response in responses:
            sucesses = {**sucesses, **response.success}

        return sucesses

    async def get_resqml_objects(
        self, *uris: T.Union[DataObjectURI, str]
    ) -> T.List[ro.AbstractObject]:
        data_objects = await self.get_data_objects(*uris)
        return [
            parse_resqml_v201_object(data_object.data) for data_object in data_objects
        ]

    async def put_resqml_objects(
        self, *objs: ro.AbstractObject, dataspace_uri: DataspaceURI
    ):
        time = self.timestamp
        uris = [DataObjectURI.from_obj(dataspace_uri, obj) for obj in objs]
        dobjs = [
            DataObject(
                format="xml",
                data=serialize_resqml_v201_object(obj),
                resource=Resource(
                    uri=uri.raw_uri,
                    name=obj.citation.title if obj.citation else obj.__class__.__name__,
                    lastChanged=time,
                    storeCreated=time,
                    storeLastWrite=time,
                    activeStatus="Inactive",  # type: ignore
                    sourceCount=None,
                    targetCount=None,
                ),
            )
            for uri, obj in zip(uris, objs)
        ]

        _ = await self.put_data_objects(*dobjs)
        return uris

    async def delete_data_objects(
        self, *uris: T.Union[DataObjectURI, str], prune_contained_objects=False
    ):
        _uris = list(map(str, uris))

        responses = await self.send(
            DeleteDataObjects(
                uris=dict(zip(_uris, _uris)),
                prune_contained_objects=prune_contained_objects,
            )
        )

        assert len(responses) == 1
        response = responses[0]

        assert isinstance(response, DeleteDataObjectsResponse), (
            "Expected DeleteDataObjectsResponse"
        )

        return response.deleted_uris

    async def start_transaction(
        self, dataspace_uri: DataspaceURI | str, read_only: bool = True
    ) -> Uuid:
        dataspace_uri = str(DataspaceURI.from_any(dataspace_uri))
        responses = await self.send(
            StartTransaction(read_only=read_only, dataspace_uris=[dataspace_uri])
        )
        assert all(
            [isinstance(response, StartTransactionResponse) for response in responses]
        ), "Expected StartTransactionResponse"

        assert len(responses) == 1
        response = responses[0]

        if not response.successful:
            raise ETPTransactionFailure(f"Failed starting transaction {dataspace_uri}")

        return response.transaction_uuid

    async def commit_transaction(self, transaction_uuid: Uuid):
        responses = await self.send(
            CommitTransaction(transaction_uuid=transaction_uuid)
        )
        assert len(responses) == 1
        response = responses[0]

        if response.successful is False:
            raise ETPTransactionFailure(response.failure_reason)
        return response

    async def rollback_transaction(self, transaction_id: Uuid):
        return await self.send(RollbackTransaction(transactionUuid=transaction_id))

    async def get_array_metadata(self, *uids: DataArrayIdentifier):
        responses = await self.send(
            GetDataArrayMetadata(dataArrays={i.path_in_resource: i for i in uids})
        )
        assert len(responses) == 1
        response = responses[0]

        assert isinstance(response, GetDataArrayMetadataResponse)

        if len(response.array_metadata) != len(uids):
            raise ETPError(f"Not all uids found ({uids})", 11)

        # return in same order as arguments
        return [response.array_metadata[i.path_in_resource] for i in uids]

    async def get_array(self, uid: DataArrayIdentifier):
        # Check if we can download the full array in one go.
        (meta,) = await self.get_array_metadata(uid)
        if (
            utils_arrays.get_transport_array_size(
                meta.transport_array_type, meta.dimensions
            )
            > self.max_array_size
        ):
            return await self._get_array_chunked(uid)

        responses = await self.send(
            GetDataArrays(dataArrays={uid.path_in_resource: uid})
        )

        assert len(responses) == 1
        response = responses[0]

        assert isinstance(response, GetDataArraysResponse), (
            "Expected GetDataArraysResponse"
        )

        arrays = list(response.data_arrays.values())
        return utils_arrays.get_numpy_array_from_etp_data_array(arrays[0])

    async def download_array(
        self,
        epc_uri: str | DataObjectURI,
        path_in_resource: str,
    ) -> npt.NDArray[utils_arrays.LogicalArrayDTypes]:
        # Create identifier for the data.
        dai = DataArrayIdentifier(
            uri=str(epc_uri),
            path_in_resource=path_in_resource,
        )

        responses = await self.send(
            GetDataArrayMetadata(data_arrays={dai.path_in_resource: dai}),
        )

        assert len(responses) == 1
        response = responses[0]

        self.assert_response(response, GetDataArrayMetadataResponse)
        assert (
            len(response.array_metadata) == 1
            and dai.path_in_resource in response.array_metadata
        )

        metadata = response.array_metadata[dai.path_in_resource]

        # Check if we can download the full array in a single message.
        if (
            utils_arrays.get_transport_array_size(
                metadata.transport_array_type, metadata.dimensions
            )
            >= self.max_array_size
        ):
            transport_dtype = utils_arrays.get_dtype_from_any_array_type(
                metadata.transport_array_type,
            )
            # NOTE: The logical array type is not yet supported by the
            # open-etp-server. As such the transport array type will be actual
            # array type used. We only add this call to prepare for when it
            # will be used.
            logical_dtype = utils_arrays.get_dtype_from_any_logical_array_type(
                metadata.logical_array_type,
            )
            if logical_dtype != np.dtype(np.bool_):
                # If this debug message is triggered we should test the
                # mapping.
                logger.debug(
                    "Logical array type has changed: "
                    f"{metadata.logical_array_type = }, with {logical_dtype = }"
                )

            # Create a buffer for the data.
            data = np.zeros(metadata.dimensions, dtype=transport_dtype)

            # Get list with starting indices in each block, and a list with the
            # number of elements along each axis for each block.
            block_starts, block_counts = utils_arrays.get_array_block_sizes(
                data.shape, data.dtype, self.max_array_size
            )

            def data_subarrays_key(pir: str, i: int) -> str:
                return pir + f" ({i})"

            tasks = []
            for i, (starts, counts) in enumerate(zip(block_starts, block_counts)):
                task = self.send(
                    GetDataSubarrays(
                        data_subarrays={
                            data_subarrays_key(
                                dai.path_in_resource, i
                            ): GetDataSubarraysType(
                                uid=dai,
                                starts=starts,
                                counts=counts,
                            ),
                        },
                    ),
                )
                tasks.append(task)

            task_responses = await asyncio.gather(*tasks)
            responses = [
                response
                for task_response in task_responses
                for response in task_response
            ]

            data_blocks = []
            for i, response in enumerate(responses):
                self.assert_response(response, GetDataSubarraysResponse)
                assert (
                    len(response.data_subarrays) == 1
                    and data_subarrays_key(dai.path_in_resource, i)
                    in response.data_subarrays
                )

                data_block = utils_arrays.get_numpy_array_from_etp_data_array(
                    response.data_subarrays[
                        data_subarrays_key(dai.path_in_resource, i)
                    ],
                )
                data_blocks.append(data_block)

            for data_block, starts, counts in zip(
                data_blocks, block_starts, block_counts
            ):
                # Create slice-objects for each block.
                slices = tuple(
                    map(
                        lambda s, c: slice(s, s + c),
                        np.array(starts).astype(int),
                        np.array(counts).astype(int),
                    )
                )
                data[slices] = data_block

            # Return after fetching all sub arrays.
            return data

        # Download the full array in one go.
        responses = await self.send(
            GetDataArrays(data_arrays={dai.path_in_resource: dai}),
        )

        assert len(responses) == 1
        response = responses[0]

        self.assert_response(response, GetDataArraysResponse)
        assert (
            len(response.data_arrays) == 1
            and dai.path_in_resource in response.data_arrays
        )

        return utils_arrays.get_numpy_array_from_etp_data_array(
            response.data_arrays[dai.path_in_resource]
        )

    async def upload_array(
        self,
        epc_uri: str | DataObjectURI,
        path_in_resource: str,
        data: npt.NDArray[utils_arrays.LogicalArrayDTypes],
    ) -> None:
        # Fetch ETP logical and transport array types
        logical_array_type, transport_array_type = (
            utils_arrays.get_logical_and_transport_array_types(data.dtype)
        )

        # Create identifier for the data.
        dai = DataArrayIdentifier(
            uri=str(epc_uri),
            path_in_resource=path_in_resource,
        )

        # Get current time as a UTC-timestamp.
        now = self.timestamp

        # Allocate space on server for the array.
        responses = await self.send(
            PutUninitializedDataArrays(
                data_arrays={
                    dai.path_in_resource: PutUninitializedDataArrayType(
                        uid=dai,
                        metadata=DataArrayMetadata(
                            dimensions=list(data.shape),
                            transport_array_type=transport_array_type,
                            logical_array_type=logical_array_type,
                            store_last_write=now,
                            store_created=now,
                        ),
                    ),
                },
            ),
        )

        assert len(responses) == 1
        response = responses[0]

        self.assert_response(response, PutUninitializedDataArraysResponse)
        assert len(response.success) == 1 and dai.path_in_resource in response.success

        # Check if we can upload the entire array in go, or if we need to
        # upload it in smaller blocks.
        if data.nbytes > self.max_array_size:
            tasks = []

            # Get list with starting indices in each block, and a list with the
            # number of elements along each axis for each block.
            block_starts, block_counts = utils_arrays.get_array_block_sizes(
                data.shape, data.dtype, self.max_array_size
            )

            for starts, counts in zip(block_starts, block_counts):
                # Create slice-objects for each block.
                slices = tuple(
                    map(
                        lambda s, c: slice(s, s + c),
                        np.array(starts).astype(int),
                        np.array(counts).astype(int),
                    )
                )

                # Slice the array, and convert to the relevant ETP-array type.
                # Note in the particular the extra `.data`-after the call. The
                # data should not be of type `DataArray`, but `AnyArray`, so we
                # need to fetch it from the `DataArray`.
                etp_subarray_data = utils_arrays.get_etp_data_array_from_numpy(
                    data[slices]
                ).data

                # Create an asynchronous task to upload a block to the
                # ETP-server.
                task = self.send(
                    PutDataSubarrays(
                        data_subarrays={
                            dai.path_in_resource: PutDataSubarraysType(
                                uid=dai,
                                data=etp_subarray_data,
                                starts=starts,
                                counts=counts,
                            ),
                        },
                    ),
                )
                tasks.append(task)

            # Upload all blocks.
            task_responses = await asyncio.gather(*tasks)

            # Flatten list of responses.
            responses = [
                response
                for task_response in task_responses
                for response in task_response
            ]

            # Check for successful responses.
            for response in responses:
                self.assert_response(response, PutDataSubarraysResponse)
                assert (
                    len(response.success) == 1
                    and dai.path_in_resource in response.success
                )

            # Return after uploading all sub arrays.
            return

        # Convert NumPy data-array to an ETP-transport array.
        etp_array_data = utils_arrays.get_etp_data_array_from_numpy(data)

        # Pass entire array in one message.
        responses = await self.send(
            PutDataArrays(
                data_arrays={
                    dai.path_in_resource: PutDataArraysType(
                        uid=dai,
                        array=etp_array_data,
                    ),
                }
            )
        )

        assert len(responses) == 1
        response = responses[0]

        self.assert_response(response, PutDataArraysResponse)
        assert len(response.success) == 1 and dai.path_in_resource in response.success

    async def put_array(
        self,
        uid: DataArrayIdentifier,
        data: np.ndarray,
    ):
        logical_array_type, transport_array_type = (
            utils_arrays.get_logical_and_transport_array_types(data.dtype)
        )
        await self._put_uninitialized_data_array(
            uid,
            data.shape,
            transport_array_type=transport_array_type,
            logical_array_type=logical_array_type,
        )
        # Check if we can upload the full array in one go.
        if data.nbytes > self.max_array_size:
            return await self._put_array_chunked(uid, data)

        responses = await self.send(
            PutDataArrays(
                data_arrays={
                    uid.path_in_resource: PutDataArraysType(
                        uid=uid,
                        array=utils_arrays.get_etp_data_array_from_numpy(data),
                    )
                }
            )
        )

        assert len(responses) == 1
        response = responses[0]

        assert isinstance(response, PutDataArraysResponse), (
            "Expected PutDataArraysResponse"
        )
        assert len(response.success) == 1, "expected one success from put_array"

        return response.success

    async def get_subarray(
        self,
        uid: DataArrayIdentifier,
        starts: T.Union[np.ndarray, T.List[int]],
        counts: T.Union[np.ndarray, T.List[int]],
    ):
        starts = np.array(starts).astype(np.int64)
        counts = np.array(counts).astype(np.int64)

        logger.debug(f"get_subarray {starts=:} {counts=:}")

        payload = GetDataSubarraysType(
            uid=uid,
            starts=starts.tolist(),
            counts=counts.tolist(),
        )
        responses = await self.send(
            GetDataSubarrays(dataSubarrays={uid.path_in_resource: payload})
        )

        assert len(responses) == 1
        response = responses[0]

        assert isinstance(response, GetDataSubarraysResponse), (
            "Expected GetDataSubarraysResponse"
        )

        arrays = list(response.data_subarrays.values())
        return utils_arrays.get_numpy_array_from_etp_data_array(arrays[0])

    async def put_subarray(
        self,
        uid: DataArrayIdentifier,
        data: np.ndarray,
        starts: T.Union[np.ndarray, T.List[int]],
        counts: T.Union[np.ndarray, T.List[int]],
    ):
        # NOTE: This function assumes that the user (or previous methods) have
        # called _put_uninitialized_data_array.

        # starts [start_X, starts_Y]
        # counts [count_X, count_Y]
        # len = 2 [x_start_index, y_start_index]
        starts = np.array(starts).astype(np.int64)
        counts = np.array(counts).astype(np.int64)  # len = 2
        ends = starts + counts  # len = 2

        slices = tuple(map(lambda s, e: slice(s, e), starts, ends))
        dataarray = utils_arrays.get_etp_data_array_from_numpy(data[slices])
        payload = PutDataSubarraysType(
            uid=uid,
            data=dataarray.data,
            starts=starts.tolist(),
            counts=counts.tolist(),
        )

        logger.debug(
            f"put_subarray {data.shape=:} {starts=:} {counts=:} "
            f"{dataarray.data.item.__class__.__name__}"
        )

        responses = await self.send(
            PutDataSubarrays(dataSubarrays={uid.path_in_resource: payload})
        )

        assert len(responses) == 1
        response = responses[0]

        assert isinstance(response, PutDataSubarraysResponse), (
            "Expected PutDataSubarraysResponse"
        )
        assert len(response.success) == 1, "expected one success"
        return response.success

    def _get_chunk_sizes(
        self, shape, dtype: np.dtype[T.Any] = np.dtype(np.float32), offset=0
    ):
        warnings.warn(
            "This function is deprecated and will be removed in a later version of "
            "pyetp. The replacement is located via the import "
            "`from pyetp.utils_arrays import get_array_block_sizes`.",
            DeprecationWarning,
            stacklevel=2,
        )
        shape = np.array(shape)

        # capsize blocksize
        max_items = self.max_array_size / dtype.itemsize
        block_size = np.power(max_items, 1.0 / len(shape))
        block_size = min(2048, int(block_size // 2) * 2)

        assert block_size > 8, "computed blocksize unreasonable small"

        all_ranges = [range(s // block_size + 1) for s in shape]
        indexes = np.array(np.meshgrid(*all_ranges)).T.reshape(-1, len(shape))
        for ijk in indexes:
            starts = ijk * block_size
            if offset != 0:
                starts = starts + offset
            ends = np.fmin(shape, starts + block_size)
            if offset != 0:
                ends = ends + offset
            counts = ends - starts
            if any(counts == 0):
                continue
            yield starts, counts

    async def _get_array_chuncked(self, *args, **kwargs):
        warnings.warn(
            "This function is deprecated and will be removed in a later version of "
            "pyetp. Please use the updated function 'pyetp._get_array_chunked'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._get_array_chunked(*args, **kwargs)

    async def _get_array_chunked(
        self,
        uid: DataArrayIdentifier,
        offset: int = 0,
        total_count: T.Union[int, None] = None,
    ):
        metadata = (await self.get_array_metadata(uid))[0]
        if len(metadata.dimensions) != 1 and offset != 0:
            raise Exception("Offset is only implemented for 1D array")

        if isinstance(total_count, (int, float)):
            buffer_shape = np.array([total_count], dtype=np.int64)
        else:
            buffer_shape = np.array(metadata.dimensions, dtype=np.int64)
        dtype = utils_arrays.get_dtype_from_any_array_type(
            metadata.transport_array_type
        )
        buffer = np.zeros(buffer_shape, dtype=dtype)
        params = []

        async def populate(starts, counts):
            params.append([starts, counts])
            array = await self.get_subarray(uid, starts, counts)
            ends = starts + counts
            slices = tuple(
                map(lambda se: slice(se[0], se[1]), zip(starts - offset, ends - offset))
            )
            buffer[slices] = array
            return

        _ = await asyncio.gather(
            *[
                populate(starts, counts)
                for starts, counts in self._get_chunk_sizes(buffer_shape, dtype, offset)
            ]
        )

        return buffer

    async def _put_array_chuncked(self, *args, **kwargs):
        warnings.warn(
            "This function is deprecated and will be removed in a later version of "
            "pyetp. Please use the updated function 'pyetp._put_array_chunked'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._put_array_chunked(*args, **kwargs)

    async def _put_array_chunked(self, uid: DataArrayIdentifier, data: np.ndarray):
        for starts, counts in self._get_chunk_sizes(data.shape, data.dtype):
            await self.put_subarray(uid, data, starts, counts)

        return {uid.uri: ""}

    async def _put_uninitialized_data_array(
        self,
        uid: DataArrayIdentifier,
        shape: T.Tuple[int, ...],
        transport_array_type: AnyArrayType,
        logical_array_type: AnyLogicalArrayType,
    ):
        payload = PutUninitializedDataArrayType(
            uid=uid,
            metadata=(
                DataArrayMetadata(
                    dimensions=list(shape),  # type: ignore
                    transportArrayType=transport_array_type,
                    logicalArrayType=logical_array_type,
                    storeLastWrite=self.timestamp,
                    storeCreated=self.timestamp,
                )
            ),
        )
        responses = await self.send(
            PutUninitializedDataArrays(dataArrays={uid.path_in_resource: payload})
        )

        assert len(responses) == 1
        response = responses[0]

        assert isinstance(response, PutUninitializedDataArraysResponse), (
            "Expected PutUninitializedDataArraysResponse"
        )
        assert len(response.success) == 1, "expected one success"
        return response.success


class etp_connect:
    """
    Connect to an ETP server via websockets.

    This class can act as:

    1. A context manager handling setup and tear-down of the connection.
    2. An asynchronous iterator which can be used to persistently retry to
    connect if the websockets connection drops.
    3. An awaitable connection that must be manually closed by the user.

    See below for examples of all three cases.

    Parameters
    ----------
    uri: str
        The uri to the ETP server. This should be the uri to a websockets
        endpoint.
    data_partition_id: str | None
        The data partition id used when connecting to the OSDU open-etp-server
        in multi-partition mode. Default is `None`.
    authorization: str | SecretStr | None
        Bearer token used for authenticating to the ETP server. This token
        should be on the form `"Bearer 1234..."`. Default is `None`.
    etp_timeout: float | None
        The timeout in seconds for when to stop waiting for a message from the
        ETP server. Setting it to `None` will persist the connection
        indefinetly. Default is `None`.
    max_message_size: float
        The maximum number of bytes for a single websockets message. Default is
        `2**20` corresponding to `1` MiB.


    Examples
    --------
    An example of connecting to the ETP server using :func:`etp_connect` as a
    context manager is:

        async with etp_connect(...) as etp_client:
            ...

    In this case the closing message and the websockets connection is closed
    once the program exits the context manager.


    To persist a connection if the websockets connection is dropped (for any
    reason), use :func:`etp_connect` as an asynchronous generator, viz.:

        import websockets

        async for etp_client in etp_connect(...):
            try:
                ...
            except websockets.ConnectionClosed:
                continue

            # Include `break` to avoid re-running the whole block if the
            # iteration runs without any errors.
            break

    Note that in this case the whole program under the `try`-block is re-run
    from the start if the iteration completes normally, or if the websockets
    connection is dropped. Therefore, make sure to include a `break` at the end
    of the `try`-block (as in the example above).


    The third option is to set up a connection via `await` and then manually
    close the connection once done:

        etp_client = await etp_connect(...)
        ...
        await etp_client.close()
    """

    def __init__(
        self,
        uri: str,
        data_partition_id: str | None = None,
        authorization: str | SecretStr | None = None,
        etp_timeout: float | None = None,
        max_message_size: float = 2**20,
    ) -> None:
        self.uri = uri
        self.data_partition_id = data_partition_id

        if isinstance(authorization, SecretStr):
            self.authorization = authorization
        else:
            self.authorization = SecretStr(authorization)

        self.etp_timeout = etp_timeout
        self.max_message_size = max_message_size
        self.subprotocols = ["etp12.energistics.org"]

    def __await__(self) -> ETPClient:
        # The caller is responsible for calling `close()` on the client.
        return self.__aenter__().__await__()

    def get_additional_headers(self) -> dict[str, str]:
        additional_headers = {}

        if self.authorization.get_secret_value() is not None:
            additional_headers["Authorization"] = self.authorization.get_secret_value()

        if self.data_partition_id is not None:
            additional_headers["data-partition-id"] = self.data_partition_id

        return additional_headers

    async def __aenter__(self) -> ETPClient:
        self.stack = contextlib.AsyncExitStack()
        try:
            ws = await self.stack.enter_async_context(
                websockets.connect(
                    uri=self.uri,
                    subprotocols=self.subprotocols,
                    max_size=self.max_message_size,
                    additional_headers=self.get_additional_headers(),
                )
            )
            etp_client = await self.stack.enter_async_context(
                ETPClient(
                    ws=ws,
                    etp_timeout=self.etp_timeout,
                )
            )
        except BaseException:
            await self.stack.aclose()
            raise

        return etp_client

    async def __aexit__(
        self,
        exc_type: T.Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return await self.stack.aclose()

    async def __aiter__(self) -> AsyncGenerator[ETPClient]:
        async for ws in websockets.connect(
            uri=self.uri,
            subprotocols=self.subprotocols,
            max_size=self.max_message_size,
            additional_headers=self.get_additional_headers(),
        ):
            async with ETPClient(
                ws=ws,
                etp_timeout=self.etp_timeout,
            ) as etp_client:
                yield etp_client


def timeout_intervals(total_timeout: float) -> Generator[float]:
    # Local function generating progressively longer timeout intervals.

    # Use the timeout-interval generator from the Python websockets
    # library.
    backoff_generator = websockets.client.backoff(
        initial_delay=5.0, min_delay=5.0, max_delay=20.0
    )

    # Check if we should never time out.
    if total_timeout is None:
        # This is an infinite generator, so it should never exit.
        yield from backoff_generator
        return

    # Generate timeout intervals until we have reached the
    # `total_timeout`-threshold.
    csum = 0.0
    for d in backoff_generator:
        yield d

        csum += d

        if csum >= total_timeout:
            break
