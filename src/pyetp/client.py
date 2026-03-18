import asyncio
import contextlib
import logging
import typing as T
import urllib
import uuid
from collections import defaultdict
from collections.abc import AsyncGenerator, Generator, Sequence
from types import TracebackType

import websockets
import websockets.client
from pydantic import SecretStr

from energistics.avro_handler import (
    GzipCompression,
    decode_message,
    encode_message,
)
from energistics.base import ETPBaseProtocolModel, Protocol, Role
from energistics.etp.v12.datatypes import (
    DataValue,
    EndpointCapabilityKind,
    ErrorInfo,
    MessageHeader,
    SupportedDataObject,
    SupportedProtocol,
)
from energistics.etp.v12.datatypes.message_header import MessageHeaderFlags
from energistics.etp.v12.protocol.core import (
    CloseSession,
    OpenSession,
    ProtocolException,
    RequestSession,
)
from pyetp._version import version

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

        if etp_timeout is not None and etp_timeout < 10:
            logger.warning(
                "A timeout shorter than 10 seconds can make the client close slow "
                "connections too soon. Consider increasing `etp_timeout` if it "
                "becomes unstable."
            )
        self.etp_timeout = etp_timeout
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
        self.assert_response(os, OpenSession)
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
        assert "xml" in os.supported_formats and len(os.supported_formats) == 1
        self.negotiated_formats = os.supported_formats

        self.session_id = os.session_id

        server_max_size = self.max_size
        dv = os.endpoint_capabilities.get(
            EndpointCapabilityKind.MAX_WEB_SOCKET_MESSAGE_PAYLOAD_SIZE
        )
        if dv is not None:
            del os.endpoint_capabilities[
                EndpointCapabilityKind.MAX_WEB_SOCKET_MESSAGE_PAYLOAD_SIZE
            ]
            server_max_size = dv.item

        if server_max_size < self.max_size:
            self.max_size = server_max_size

        if len(os.endpoint_capabilities) > 0:
            logger.info(
                "Remaining unprocessed endpoint capabilities "
                f"{os.endpoint_capabilities}"
            )

        return self

    async def __aexit__(
        self,
        exc_type: T.Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # We catch and log a lot of errors instead of letting them be raised.
        # The reason is that we are trying to close down the connection as fast
        # as possible, and by raising an error it can take a while for the
        # websockets connection to drop making the program hang.
        close_session_sent = False
        try:
            await self.send(CloseSession(reason="Client exiting"))
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
        via `etp_client = await etp_connect(...)` and will handle the closing
        of the connection manually.
        """

        await self.__aexit__(None, None, None)
        # The websockets connection should be closed from the ETP-server once
        # it has received a `CloseSession`-message. However, calling close on
        # the websockets connection does not do anything if it is already
        # closed.
        await self.ws.close()

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
