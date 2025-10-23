import asyncio
import datetime
import logging
import sys
import typing as T
import uuid
import warnings
from collections import defaultdict
from types import TracebackType

import numpy as np
import websockets
from etpproto.connection import CommunicationProtocol, ConnectionType, ETPConnection
from etpproto.messages import Message, MessageFlags
from etptypes import ETPModel
from etptypes.energistics.etp.v12.datatypes.any_array_type import AnyArrayType
from etptypes.energistics.etp.v12.datatypes.any_logical_array_type import (
    AnyLogicalArrayType,
)
from etptypes.energistics.etp.v12.datatypes.array_of_string import ArrayOfString
from etptypes.energistics.etp.v12.datatypes.data_array_types.data_array_identifier import (
    DataArrayIdentifier,
)
from etptypes.energistics.etp.v12.datatypes.data_array_types.data_array_metadata import (
    DataArrayMetadata,
)
from etptypes.energistics.etp.v12.datatypes.data_array_types.get_data_subarrays_type import (
    GetDataSubarraysType,
)
from etptypes.energistics.etp.v12.datatypes.data_array_types.put_data_arrays_type import (
    PutDataArraysType,
)
from etptypes.energistics.etp.v12.datatypes.data_array_types.put_data_subarrays_type import (
    PutDataSubarraysType,
)
from etptypes.energistics.etp.v12.datatypes.data_array_types.put_uninitialized_data_array_type import (
    PutUninitializedDataArrayType,
)
from etptypes.energistics.etp.v12.datatypes.data_value import DataValue
from etptypes.energistics.etp.v12.datatypes.error_info import ErrorInfo
from etptypes.energistics.etp.v12.datatypes.object.context_info import ContextInfo
from etptypes.energistics.etp.v12.datatypes.object.context_scope_kind import (
    ContextScopeKind,
)
from etptypes.energistics.etp.v12.datatypes.object.data_object import DataObject
from etptypes.energistics.etp.v12.datatypes.object.dataspace import Dataspace
from etptypes.energistics.etp.v12.datatypes.object.relationship_kind import (
    RelationshipKind,
)
from etptypes.energistics.etp.v12.datatypes.object.resource import Resource
from etptypes.energistics.etp.v12.datatypes.supported_data_object import (
    SupportedDataObject,
)
from etptypes.energistics.etp.v12.datatypes.supported_protocol import SupportedProtocol
from etptypes.energistics.etp.v12.datatypes.uuid import Uuid
from etptypes.energistics.etp.v12.datatypes.version import Version
from etptypes.energistics.etp.v12.protocol.core.authorize import Authorize
from etptypes.energistics.etp.v12.protocol.core.authorize_response import (
    AuthorizeResponse,
)
from etptypes.energistics.etp.v12.protocol.core.close_session import CloseSession
from etptypes.energistics.etp.v12.protocol.core.open_session import OpenSession
from etptypes.energistics.etp.v12.protocol.core.protocol_exception import (
    ProtocolException,
)
from etptypes.energistics.etp.v12.protocol.core.request_session import RequestSession
from etptypes.energistics.etp.v12.protocol.data_array.get_data_array_metadata import (
    GetDataArrayMetadata,
)
from etptypes.energistics.etp.v12.protocol.data_array.get_data_array_metadata_response import (
    GetDataArrayMetadataResponse,
)
from etptypes.energistics.etp.v12.protocol.data_array.get_data_arrays import (
    GetDataArrays,
)
from etptypes.energistics.etp.v12.protocol.data_array.get_data_arrays_response import (
    GetDataArraysResponse,
)
from etptypes.energistics.etp.v12.protocol.data_array.get_data_subarrays import (
    GetDataSubarrays,
)
from etptypes.energistics.etp.v12.protocol.data_array.get_data_subarrays_response import (
    GetDataSubarraysResponse,
)
from etptypes.energistics.etp.v12.protocol.data_array.put_data_arrays import (
    PutDataArrays,
)
from etptypes.energistics.etp.v12.protocol.data_array.put_data_arrays_response import (
    PutDataArraysResponse,
)
from etptypes.energistics.etp.v12.protocol.data_array.put_data_subarrays import (
    PutDataSubarrays,
)
from etptypes.energistics.etp.v12.protocol.data_array.put_data_subarrays_response import (
    PutDataSubarraysResponse,
)
from etptypes.energistics.etp.v12.protocol.data_array.put_uninitialized_data_arrays import (
    PutUninitializedDataArrays,
)
from etptypes.energistics.etp.v12.protocol.data_array.put_uninitialized_data_arrays_response import (
    PutUninitializedDataArraysResponse,
)
from etptypes.energistics.etp.v12.protocol.dataspace.delete_dataspaces import (
    DeleteDataspaces,
)
from etptypes.energistics.etp.v12.protocol.dataspace.delete_dataspaces_response import (
    DeleteDataspacesResponse,
)
from etptypes.energistics.etp.v12.protocol.dataspace.get_dataspaces import GetDataspaces
from etptypes.energistics.etp.v12.protocol.dataspace.get_dataspaces_response import (
    GetDataspacesResponse,
)
from etptypes.energistics.etp.v12.protocol.dataspace.put_dataspaces import PutDataspaces
from etptypes.energistics.etp.v12.protocol.dataspace.put_dataspaces_response import (
    PutDataspacesResponse,
)
from etptypes.energistics.etp.v12.protocol.discovery.get_resources import GetResources
from etptypes.energistics.etp.v12.protocol.store.delete_data_objects import (
    DeleteDataObjects,
)
from etptypes.energistics.etp.v12.protocol.store.delete_data_objects_response import (
    DeleteDataObjectsResponse,
)
from etptypes.energistics.etp.v12.protocol.store.get_data_objects import GetDataObjects
from etptypes.energistics.etp.v12.protocol.store.get_data_objects_response import (
    GetDataObjectsResponse,
)
from etptypes.energistics.etp.v12.protocol.store.put_data_objects import PutDataObjects
from etptypes.energistics.etp.v12.protocol.store.put_data_objects_response import (
    PutDataObjectsResponse,
)
from etptypes.energistics.etp.v12.protocol.transaction.commit_transaction import (
    CommitTransaction,
)
from etptypes.energistics.etp.v12.protocol.transaction.rollback_transaction import (
    RollbackTransaction,
)
from etptypes.energistics.etp.v12.protocol.transaction.start_transaction import (
    StartTransaction,
)
from pydantic import SecretStr
from xtgeo import RegularSurface

import resqml_objects.v201 as ro
from pyetp import utils_arrays, utils_xml
from pyetp.config import SETTINGS
from pyetp.uri import DataObjectURI, DataspaceURI
from resqml_objects import parse_resqml_v201_object, serialize_resqml_v201_object

# from asyncio import timeout

try:
    # for py >3.11, we can raise grouped exceptions
    from builtins import ExceptionGroup  # type: ignore
except ImportError:

    def ExceptionGroup(msg, errors):
        return errors[0]


try:
    from asyncio import timeout
except ImportError:
    from contextlib import asynccontextmanager

    import async_timeout

    @asynccontextmanager
    async def timeout(delay: T.Optional[float]) -> T.Any:
        try:
            async with async_timeout.timeout(delay):
                yield None
        except asyncio.CancelledError as e:
            raise asyncio.TimeoutError(f"Timeout ({delay}s)") from e


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
    def from_protos(cls, errors: T.Iterable[ErrorInfo]):
        return list(map(cls.from_proto, errors))


def get_all_etp_protocol_classes():
    """Update protocol - all exception protocols are now per message"""

    pddict = ETPConnection.generic_transition_table
    # protocol exception
    pexec = ETPConnection.generic_transition_table["0"]["1000"]

    for v in pddict.values():
        v["1000"] = pexec

    return pddict


class ETPClient(ETPConnection):
    generic_transition_table = get_all_etp_protocol_classes()

    _recv_events: T.Dict[int, asyncio.Event]
    _recv_buffer: T.Dict[int, T.List[ETPModel]]

    def __init__(self, ws: websockets.ClientConnection, timeout=10.0):
        super().__init__(connection_type=ConnectionType.CLIENT)
        self._recv_events = {}
        self._recv_buffer = defaultdict(lambda: list())  # type: ignore
        self.ws = ws

        self.timeout = timeout
        self.client_info.endpoint_capabilities["MaxWebSocketMessagePayloadSize"] = (
            SETTINGS.MaxWebSocketMessagePayloadSize
        )
        self.__recvtask = asyncio.create_task(self.__recv__())

    #
    # client
    #

    async def send(self, body: ETPModel):
        correlation_id = await self._send(body)
        return await self._recv(correlation_id)

    async def _send(self, body: ETPModel):
        msg = Message.get_object_message(body, message_flags=MessageFlags.FINALPART)
        if msg is None:
            raise TypeError(f"{type(body)} not valid etp protocol")

        msg.header.message_id = self.consume_msg_id()
        logging.debug(f"sending {msg.body.__class__.__name__} {repr(msg.header)}")

        # create future recv event
        self._recv_events[msg.header.message_id] = asyncio.Event()

        for msg_part in msg.encode_message_generator(self.max_size, self):
            await self.ws.send(msg_part)

        return msg.header.message_id

    async def _recv(self, correlation_id: int) -> ETPModel:
        assert correlation_id in self._recv_events, (
            "trying to recv response on non-existing message"
        )

        async with timeout(self.timeout):
            await self._recv_events[correlation_id].wait()

        # cleanup
        bodies = self._clear_msg_on_buffer(correlation_id)

        # error handling
        errors = self._parse_error_info(bodies)

        if len(errors) == 1:
            raise ETPError.from_proto(errors.pop())
        elif len(errors) > 1:
            raise ExceptionGroup(
                "Server responded with ETPErrors:", ETPError.from_protos(errors)
            )

        if len(bodies) > 1:
            logging.warning(f"Recived {len(bodies)} messages, but only expected one")

        # ok
        return bodies[0]

    @staticmethod
    def _parse_error_info(bodies: list[ETPModel]) -> list[ErrorInfo]:
        # returns all error infos from bodies
        errors = []
        for body in bodies:
            if isinstance(body, ProtocolException):
                if body.error is not None:
                    errors.append(body.error)
                errors.extend(body.errors.values())
        return errors

    async def close(self, reason=""):
        try:
            await self._send(CloseSession(reason=reason))
        except websockets.ConnectionClosed:
            logging.error(
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
            logging.error(
                "The receiver task errored on a closed websockets connection. The "
                f"message was: {e.__class__.__name__}: {e}"
            )

        if len(self._recv_buffer) > 0:
            logging.error(
                f"Connection is closed, but there are {len(self._recv_buffer)} "
                "messages left in the buffer"
            )

        # Check if there were any messages left in the websockets connection.
        # Reading them will speed up the closing of the connection.
        counter = 0
        try:
            async for msg in self.ws:
                counter += 1
        except websockets.ConnectionClosed:
            # The websockets connection had already closed. Either successfully
            # or with an error, but we ignore both cases.
            pass

        if counter > 0:
            logging.error(
                f"There were {counter} unread messages in the websockets connection "
                "after the session was closed"
            )

        logging.debug("Client closed")

    #
    #
    #

    def _clear_msg_on_buffer(self, correlation_id: int):
        del self._recv_events[correlation_id]
        return self._recv_buffer.pop(correlation_id)

    def _add_msg_to_buffer(self, msg: Message):
        self._recv_buffer[msg.header.correlation_id].append(msg.body)

        # NOTE: should we add task to autoclear buffer message if never waited on ?
        if msg.is_final_msg():
            # set response on send event
            self._recv_events[msg.header.correlation_id].set()

    async def __recv__(self):
        logging.debug("starting recv loop")

        while True:
            msg_data = await self.ws.recv()
            msg = Message.decode_binary_message(
                T.cast(bytes, msg_data), ETPClient.generic_transition_table
            )

            if msg is None:
                logging.error(f"Could not parse {msg_data}")
                continue

            logging.debug(f"recv {msg.body.__class__.__name__} {repr(msg.header)}")
            self._add_msg_to_buffer(msg)

    #
    # session related
    #

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

        msg = await self.send(
            RequestSession(
                applicationName=SETTINGS.application_name,
                applicationVersion=SETTINGS.application_version,
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
        assert msg and isinstance(msg, OpenSession)

        self.is_connected = True

        # ignore this endpoint
        _ = msg.endpoint_capabilities.pop("MessageQueueDepth", None)
        self.client_info.negotiate(msg)

        return self

    async def authorize(
        self, authorization: str, supplemental_authorization: T.Mapping[str, str] = {}
    ):
        msg = await self.send(
            Authorize(
                authorization=authorization,
                supplementalAuthorization=supplemental_authorization,
            )
        )
        assert msg and isinstance(msg, AuthorizeResponse)

        return msg

    #

    @property
    def max_size(self):
        return SETTINGS.MaxWebSocketMessagePayloadSize
        # return self.client_info.getCapability("MaxWebSocketMessagePayloadSize")

    @property
    def max_array_size(self):
        return self.max_size - 512  # maxsize - 512 bytes for header and body

    @property
    def timestamp(self):
        return int(datetime.datetime.now(datetime.timezone.utc).timestamp())

    def dataspace_uri(self, ds: str) -> DataspaceURI:
        if ds.count("/") > 1:
            raise Exception("Max one / in dataspace name")
        return DataspaceURI.from_name(ds)

    def list_objects(self, dataspace_uri: DataspaceURI, depth: int = 1) -> list:
        return self.send(
            GetResources(
                scope=ContextScopeKind.TARGETS_OR_SELF,
                context=ContextInfo(
                    uri=dataspace_uri.raw_uri,
                    depth=depth,
                    dataObjectTypes=[],
                    navigableEdges=RelationshipKind.PRIMARY,
                ),
            )
        )

    #
    # dataspace
    #

    async def get_dataspaces(
        self, store_last_write_filter: int = None
    ) -> GetDataspacesResponse:
        return await self.send(
            GetDataspaces(store_last_write_filter=store_last_write_filter)
        )

    async def put_dataspaces(
        self,
        legaltags: list[str],
        otherRelevantDataCountries: list[str],
        owners: list[str],
        viewers: list[str],
        *dataspace_uris: DataspaceURI,
    ):
        _uris = list(map(DataspaceURI.from_any, dataspace_uris))
        for i in _uris:
            if i.raw_uri.count("/") > 4:  # includes the 3 eml
                raise Exception("Max one / in dataspace name")
        time = self.timestamp
        response = await self.send(
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
        assert isinstance(response, PutDataspacesResponse), (
            "Expected PutDataspacesResponse"
        )

        assert len(response.success) == len(dataspace_uris), (
            f"expected {len(dataspace_uris)} success's"
        )

        return response.success

    async def put_dataspaces_no_raise(
        self,
        legaltags: list[str],
        otherRelevantDataCountries: list[str],
        owners: list[str],
        viewers: list[str],
        *dataspace_uris: DataspaceURI,
    ):
        try:
            return await self.put_dataspaces(
                legaltags, otherRelevantDataCountries, owners, viewers, *dataspace_uris
            )
        except ETPError:
            pass

    async def delete_dataspaces(self, *dataspace_uris: DataspaceURI):
        _uris = list(map(str, dataspace_uris))

        response = await self.send(DeleteDataspaces(uris=dict(zip(_uris, _uris))))
        assert isinstance(response, DeleteDataspacesResponse), (
            "Expected DeleteDataspacesResponse"
        )
        return response.success

    #
    # data objects
    #

    async def get_data_objects(self, *uris: T.Union[DataObjectURI, str]):
        _uris = list(map(str, uris))

        msg = await self.send(GetDataObjects(uris=dict(zip(_uris, _uris))))
        assert isinstance(msg, GetDataObjectsResponse), "Expected dataobjectsresponse"
        assert len(msg.data_objects) == len(_uris), (
            "Here we assume that all three objects fit in a single record"
        )

        return [msg.data_objects[u] for u in _uris]

    async def put_data_objects(self, *objs: DataObject):
        response = await self.send(
            PutDataObjects(
                data_objects={f"{p.resource.name} - {p.resource.uri}": p for p in objs},
            )
        )

        assert isinstance(response, PutDataObjectsResponse), (
            "Expected PutDataObjectsResponse"
        )

        return response.success

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

        response = await self.send(
            DeleteDataObjects(
                uris=dict(zip(_uris, _uris)),
                prune_contained_objects=prune_contained_objects,
            )
        )
        assert isinstance(response, DeleteDataObjectsResponse), (
            "Expected DeleteDataObjectsResponse"
        )

        return response.deleted_uris

    async def start_transaction(
        self, dataspace_uri: DataspaceURI, read_only: bool = True
    ) -> Uuid:
        trans_id = await self.send(
            StartTransaction(
                read_only=read_only, dataspace_uris=[dataspace_uri.raw_uri]
            )
        )
        if trans_id.successful is False:
            raise Exception(f"Failed starting transaction {dataspace_uri.raw_uri}")
        # uuid.UUID(bytes=trans_id.transaction_uuid)
        return Uuid(trans_id.transaction_uuid)

    async def commit_transaction(self, transaction_uuid: Uuid):
        r = await self.send(CommitTransaction(transaction_uuid=transaction_uuid))
        if r.successful is False:
            raise Exception(r.failure_reason)
        return r

    async def rollback_transaction(self, transaction_id: Uuid):
        return await self.send(RollbackTransaction(transactionUuid=transaction_id))

    #
    # xtgeo
    #
    async def get_xtgeo_surface(
        self,
        epc_uri: T.Union[DataObjectURI, str],
        gri_uri: T.Union[DataObjectURI, str],
        crs_uri: T.Union[DataObjectURI, str],
    ):
        (
            gri,
            crs,
        ) = await self.get_resqml_objects(gri_uri, crs_uri)
        rotation = crs.areal_rotation.value
        # some checks

        assert isinstance(gri, ro.Grid2dRepresentation), (
            "obj must be Grid2DRepresentation"
        )
        sgeo = gri.grid2d_patch.geometry.points.supporting_geometry  # type: ignore
        if sys.version_info[1] != 10:
            assert isinstance(
                gri.grid2d_patch.geometry.points, ro.Point3dZValueArray
            ), "Points must be Point3dZValueArray"
            assert isinstance(
                gri.grid2d_patch.geometry.points.zvalues, ro.DoubleHdf5Array
            ), "Values must be DoubleHdf5Array"
            assert isinstance(
                gri.grid2d_patch.geometry.points.supporting_geometry,
                ro.Point3dLatticeArray,
            ), "Points support_geo must be Point3dLatticeArray"
            assert isinstance(sgeo, ro.Point3dLatticeArray), (
                "supporting_geometry must be Point3dLatticeArray"
            )
        assert isinstance(
            gri.grid2d_patch.geometry.points.zvalues.values, ro.Hdf5Dataset
        ), "Values must be Hdf5Dataset"

        # get array
        array = await self.get_array(
            DataArrayIdentifier(
                uri=str(epc_uri),
                pathInResource=gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file,
            )
        )

        return RegularSurface(
            ncol=array.shape[0],
            nrow=array.shape[1],
            xinc=sgeo.offset[0].spacing.value,
            yinc=sgeo.offset[1].spacing.value,
            xori=sgeo.origin.coordinate1,
            yori=sgeo.origin.coordinate2,
            values=array,
            rotation=rotation,
            masked=True,
        )

    async def put_xtgeo_surface(
        self,
        surface: RegularSurface,
        epsg_code: int,
        dataspace_uri: DataspaceURI,
        handle_transaction: bool = True,
    ):
        """Returns (epc_uri, crs_uri, gri_uri).

        If `handle_transaction == True` we start a transaction, and commit it
        after the data has been uploaded. Otherwise, we do not handle the
        transactions at all and assume that the user will start and commit the
        transaction themselves.
        """
        assert surface.values is not None, "cannot upload empty surface"

        if handle_transaction:
            transaction_uuid = await self.start_transaction(
                dataspace_uri, read_only=False
            )

        epc, crs, gri = utils_xml.parse_xtgeo_surface_to_resqml_grid(surface, epsg_code)
        epc_uri, crs_uri, gri_uri = await self.put_resqml_objects(
            epc, crs, gri, dataspace_uri=dataspace_uri
        )

        await self.put_array(
            DataArrayIdentifier(
                uri=epc_uri.raw_uri if isinstance(epc_uri, DataObjectURI) else epc_uri,
                pathInResource=gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file,  # type: ignore
            ),
            surface.values.filled(np.nan).astype(np.float32),
        )

        if handle_transaction:
            await self.commit_transaction(transaction_uuid=transaction_uuid)

        return epc_uri, gri_uri, crs_uri

    #
    # array
    #

    async def get_array_metadata(self, *uids: DataArrayIdentifier):
        response = await self.send(
            GetDataArrayMetadata(dataArrays={i.path_in_resource: i for i in uids})
        )
        assert isinstance(response, GetDataArrayMetadataResponse)

        if len(response.array_metadata) != len(uids):
            raise ETPError(f"Not all uids found ({uids})", 11)

        # return in same order as arguments
        return [response.array_metadata[i.path_in_resource] for i in uids]

    async def get_array(self, uid: DataArrayIdentifier):
        # Check if we can download the full array in one go.
        (meta,) = await self.get_array_metadata(uid)
        if utils_arrays.get_transport_array_size(meta) > self.max_array_size:
            return await self._get_array_chunked(uid)

        response = await self.send(
            GetDataArrays(dataArrays={uid.path_in_resource: uid})
        )
        assert isinstance(response, GetDataArraysResponse), (
            "Expected GetDataArraysResponse"
        )

        arrays = list(response.data_arrays.values())
        return utils_arrays.get_numpy_array_from_etp_data_array(arrays[0])

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

        response = await self.send(
            PutDataArrays(
                data_arrays={
                    uid.path_in_resource: PutDataArraysType(
                        uid=uid,
                        array=utils_arrays.get_etp_data_array_from_numpy(data),
                    )
                }
            )
        )

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

        logging.debug(f"get_subarray {starts=:} {counts=:}")

        payload = GetDataSubarraysType(
            uid=uid,
            starts=starts.tolist(),
            counts=counts.tolist(),
        )
        response = await self.send(
            GetDataSubarrays(dataSubarrays={uid.path_in_resource: payload})
        )
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

        slices = tuple(map(lambda se: slice(se[0], se[1]), zip(starts, ends)))
        dataarray = utils_arrays.get_etp_data_array_from_numpy(data[slices])
        payload = PutDataSubarraysType(
            uid=uid,
            data=dataarray.data,
            starts=starts.tolist(),
            counts=counts.tolist(),
        )

        logging.debug(
            f"put_subarray {data.shape=:} {starts=:} {counts=:} "
            f"{dataarray.data.item.__class__.__name__}"
        )

        response = await self.send(
            PutDataSubarrays(dataSubarrays={uid.path_in_resource: payload})
        )
        assert isinstance(response, PutDataSubarraysResponse), (
            "Expected PutDataSubarraysResponse"
        )
        assert len(response.success) == 1, "expected one success"
        return response.success

    #
    # chunked get array - ETP will not chunk response - so we need to do it manually
    #

    def _get_chunk_sizes(
        self, shape, dtype: np.dtype[T.Any] = np.dtype(np.float32), offset=0
    ):
        shape = np.array(shape)

        # capsize blocksize
        # remove 512 bytes for headers and body
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
        response = await self.send(
            PutUninitializedDataArrays(dataArrays={uid.path_in_resource: payload})
        )
        assert isinstance(response, PutUninitializedDataArraysResponse), (
            "Expected PutUninitializedDataArraysResponse"
        )
        assert len(response.success) == 1, "expected one success"
        return response.success


# define an asynchronous context manager
class connect:
    def __init__(self, authorization: T.Optional[SecretStr] = None):
        self.server_url = SETTINGS.etp_url
        self.authorization = authorization
        self.data_partition = SETTINGS.data_partition
        self.timeout = SETTINGS.etp_timeout

    # ... = await connect(...)

    def __await__(self):
        return self.__aenter__().__await__()

    # async with connect(...) as ...:

    async def __aenter__(self):
        headers = {}
        if isinstance(self.authorization, str):
            headers["Authorization"] = self.authorization
        elif isinstance(self.authorization, SecretStr):
            headers["Authorization"] = self.authorization.get_secret_value()
        if self.data_partition is not None:
            headers["data-partition-id"] = self.data_partition

        self.ws = await websockets.connect(
            self.server_url,
            subprotocols=[ETPClient.SUB_PROTOCOL],  # type: ignore
            additional_headers=headers,
            max_size=SETTINGS.MaxWebSocketMessagePayloadSize,
            ping_timeout=self.timeout,
            open_timeout=None,
        )

        self.client = ETPClient(self.ws, timeout=self.timeout)

        try:
            await self.client.request_session()
        except Exception as e:
            # aexit not called if raised in aenter - so manual cleanup here needed
            await self.client.close("Failed to request session")
            raise e

        return self.client

    # exit the async context manager
    async def __aexit__(self, exc_type, exc: Exception, tb: TracebackType):
        await self.client.close()
        await self.ws.close()
