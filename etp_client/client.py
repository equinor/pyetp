import asyncio
import datetime
import logging
import typing as T
import uuid
from collections import defaultdict
from types import TracebackType

import numpy as np
import websockets
from etpproto.connection import (CommunicationProtocol, ConnectionType,
                                 ETPConnection)
from etpproto.messages import Message, MessageFlags
from xtgeo import RegularSurface

import map_api.resqml_objects as ro
from map_api.config import SETTINGS
from map_api.etp_client.utils import *

from .types import *
from .uri import DataObjectURI, DataspaceURI

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


MAXPAYLOADSIZE = 10_000_000  # 10MB


class ETPError(Exception):
    def __init__(self, message: str, code: int):
        self.message = message
        self.code = code
        super().__init__(f"{message} ({code=:})")

    @classmethod
    def from_proto(cls, msg: ProtocolException):
        assert msg.error is not None or msg.errors is not None, "passed no error info"
        error = msg.error or list(msg.errors.values())[0]
        return cls(error.message, error.code)


class ETPClient(ETPConnection):

    _recv_events: T.Dict[int, asyncio.Event]
    _recv_buffer: T.Dict[int, T.List[ETPModel]]

    def __init__(self, ws: websockets.WebSocketClientProtocol, default_dataspace_uri: DataspaceURI | None, timeout=10.):
        super().__init__(connection_type=ConnectionType.CLIENT)
        self._recv_events = {}
        self._recv_buffer = defaultdict(lambda: list())  # type: ignore
        self._default_duri = default_dataspace_uri
        self.ws = ws

        self.timeout = timeout
        self.client_info.endpoint_capabilities['MaxWebSocketMessagePayloadSize'] = MAXPAYLOADSIZE
        self.__recvtask = asyncio.create_task(self.__recv__())

    #
    # client
    #

    async def send(self, body: ETPModel):
        correlation_id = await self._send(body)
        return await self._recv(correlation_id)

    async def _send(self, body: ETPModel):

        msg = Message.get_object_message(
            body, message_flags=MessageFlags.FINALPART
        )
        if msg == None:
            raise TypeError(f"{type(body)} not valid etp protocol")

        msg.header.message_id = self.consume_msg_id()
        logger.debug(f"sending {msg.body.__class__.__name__} {repr(msg.header)}")

        # create future recv event
        self._recv_events[msg.header.message_id] = asyncio.Event()

        async for msg_part in msg.encode_message_generator(self.max_size, self):
            await self.ws.send(msg_part)

        return msg.header.message_id

    async def _recv(self, correlation_id: int) -> ETPModel:
        assert correlation_id in self._recv_events, "trying to recv response on non-existing message"

        async with asyncio.timeout(self.timeout):
            await self._recv_events[correlation_id].wait()

        # cleanup
        bodies = self._clear_msg_on_buffer(correlation_id)

        for body in bodies:
            if isinstance(body, ProtocolException):
                raise ETPError.from_proto(body)

        if len(bodies) > 1:
            logger.warning(f"Recived {len(bodies)} messages, but only expected one")

        return bodies[0]

    async def close(self, reason=''):
        try:
            await self._send(CloseSession(reason=reason))
        finally:
            await self.ws.close(reason=reason)
            self.is_connected = False
            self.__recvtask.cancel("stopped")

            if len(self._recv_buffer):
                logger.error(f"Closed connection - but had stuff left in buffers")
                logger.warning(self._recv_buffer)

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
            self._recv_events[msg.header.correlation_id].set()  # set response on send event

    async def __recv__(self):

        logger.debug(f"starting recv loop")

        while (True):
            msg_data = await self.ws.recv()
            msg = Message.decode_binary_message(
                T.cast(bytes, msg_data), ETPConnection.generic_transition_table
            )

            if msg is None:
                logger.error(f"Could not parse {msg_data}")
                continue

            logger.debug(f"recv {msg.body.__class__.__name__} {repr(msg.header)}")
            self._add_msg_to_buffer(msg)

    async def connect(self):
        msg = await self.send(
            RequestSession(
                applicationName=SETTINGS.application_name,
                applicationVersion=SETTINGS.application_version,
                clientInstanceId=uuid.uuid4(),  # type: ignore
                requestedProtocols=[
                    SupportedProtocol(protocol=p.value, protocolVersion=Version(major=1, minor=2), role='store')
                    for p in [CommunicationProtocol.DISCOVERY, CommunicationProtocol.STORE, CommunicationProtocol.DATA_ARRAY, CommunicationProtocol.DATASPACE]
                ],
                supportedDataObjects=[SupportedDataObject(qualifiedType="resqml20.*"), SupportedDataObject(qualifiedType="eml20.*")],
                currentDateTime=self.timestamp,
                earliestRetainedChangeTime=0,
                endpointCapabilities=dict(
                    MaxWebSocketMessagePayloadSize=DataValue(item=self.max_size)
                )
            )
        )
        assert msg and isinstance(msg, OpenSession)

        self.is_connected = True

        # ignore this endpoint
        _ = msg.endpoint_capabilities.pop('MessageQueueDepth', None)
        self.client_info.negotiate(msg)

        return self

    #

    @property
    def max_size(self):
        return self.client_info.getCapability("MaxWebSocketMessagePayloadSize")

    @property
    def max_array_size(self):
        return self.max_size - 512  # maxsize - 512 bytes for header and body

    @property
    def timestamp(self):
        return int(datetime.datetime.now(datetime.timezone.utc).timestamp())

    @property
    def default_dataspace_uri(self):
        return self._default_duri

    @default_dataspace_uri.setter
    def default_dataspace_uri(self, v: DataspaceURI | str | None):
        self._default_duri = None if v is None else DataspaceURI.from_any(v)

    def get_dataspace_or_default_uri(self, ds: DataspaceURI | str | None) -> DataspaceURI:
        """Returns default dataspace or user spefied one"""

        if ds is None and self._default_duri is None:
            raise ValueError("Could not get dataspace from userinput or default")

        if isinstance(ds, str):
            return DataspaceURI(ds)

        return ds or self._default_duri  # type: ignore

    #
    # dataspace
    #

    async def put_dataspaces(self, *uris: DataspaceURI | str):
        from etptypes.energistics.etp.v12.protocol.dataspace.put_dataspaces import \
            PutDataspaces
        from etptypes.energistics.etp.v12.protocol.dataspace.put_dataspaces_response import \
            PutDataspacesResponse

        _uris = [DataspaceURI(u) if isinstance(u, str) else u for u in uris]  # type: ignore

        time = self.timestamp
        response = await self.send(
            PutDataspaces(dataspaces={
                d.raw_uri: Dataspace(uri=d.raw_uri, storeCreated=time, storeLastWrite=time, path=d.dataspace)
                for d in _uris
            })
        )
        assert isinstance(response, PutDataspacesResponse), "Expected PutDataspacesResponse"

        assert len(response.success) == len(uris), f"expected {len(uris)} success's"

        return response.success

    async def put_dataspaces_no_raise(self, *uris: DataspaceURI | str):
        try:
            return await self.put_dataspaces(*uris)
        except ETPError:
            pass

    async def delete_dataspaces(self, *uris: DataspaceURI | str):
        from etptypes.energistics.etp.v12.protocol.dataspace.delete_dataspaces import \
            DeleteDataspaces
        from etptypes.energistics.etp.v12.protocol.dataspace.delete_dataspaces_response import \
            DeleteDataspacesResponse

        _uris = list(map(str, uris))

        response = await self.send(DeleteDataspaces(uris=dict(zip(_uris, _uris))))
        assert isinstance(response, DeleteDataspacesResponse), "Expected DeleteDataspacesResponse"
        return response.success

    #
    # data objects
    #

    async def get_data_objects(self, *uris: DataObjectURI | str):

        from etptypes.energistics.etp.v12.protocol.store.get_data_objects import \
            GetDataObjects
        from etptypes.energistics.etp.v12.protocol.store.get_data_objects_response import \
            GetDataObjectsResponse

        _uris = list(map(str, uris))

        msg = await self.send(
            GetDataObjects(uris=dict(zip(_uris, _uris)))
        )
        assert isinstance(msg, GetDataObjectsResponse), "Expected dataobjectsresponse"
        assert len(msg.data_objects) == len(_uris), "Here we assume that all three objects fit in a single record"

        return [msg.data_objects[u] for u in _uris]

    async def put_data_objects(self, *objs: DataObject):

        from etptypes.energistics.etp.v12.protocol.store.put_data_objects import \
            PutDataObjects
        from etptypes.energistics.etp.v12.protocol.store.put_data_objects_response import \
            PutDataObjectsResponse

        response = await self.send(
            PutDataObjects(dataObjects={p.resource.name: p for p in objs})
        )
        # logger.info(f"objects {response=:}")
        assert isinstance(response, PutDataObjectsResponse), "Expected PutDataObjectsResponse"
        # assert len(response.success) == len(objs)  # might be 0 if objects exists

        return response.success

    async def get_resqml_objects(self, *uris: DataObjectURI | str) -> T.List[ro.AbstractObject]:
        data_objects = await self.get_data_objects(*uris)
        return parse_resqml_objects(data_objects)

    async def put_resqml_objects(self, *objs: ro.AbstractObject, dataspace: DataspaceURI | str | None = None):
        from etptypes.energistics.etp.v12.datatypes.object.resource import \
            Resource

        time = self.timestamp
        duri = self.get_dataspace_or_default_uri(dataspace)

        uris = [DataObjectURI.from_obj(duri, obj) for obj in objs]

        dobjs = [DataObject(
            format="xml",
            data=resqml_to_xml(obj),
            resource=Resource(
                uri=uri.raw_uri,
                name=obj.citation.title if obj.citation else obj.__class__.__name__,
                lastChanged=time,
                storeCreated=time,
                storeLastWrite=time,
                activeStatus="Inactive",  # type: ignore
                sourceCount=None,
                targetCount=None
            )
        ) for uri, obj in zip(uris, objs)]

        response = await self.put_data_objects(*dobjs)
        return uris

    async def delete_data_objects(self, *uris: DataObjectURI | str, pruneContainedObjects=False):
        from etptypes.energistics.etp.v12.protocol.store.delete_data_objects import \
            DeleteDataObjects
        from etptypes.energistics.etp.v12.protocol.store.delete_data_objects_response import \
            DeleteDataObjectsResponse

        _uris = list(map(str, uris))

        response = await self.send(
            DeleteDataObjects(
                uris=dict(zip(_uris, _uris)),
                pruneContainedObjects=pruneContainedObjects
            )
        )
        # logger.info(f"delete objects {response=:}")
        assert isinstance(response, DeleteDataObjectsResponse), "Expected DeleteDataObjectsResponse"

        return response.deleted_uris

    #
    # xtgeo
    #

    async def get_xtgeo_surface(self, epc_uri: DataObjectURI | str, gri_uri: DataObjectURI | str):
        gri, = await self.get_resqml_objects(gri_uri)

        # some checks
        assert isinstance(gri, ro.Grid2dRepresentation), "obj must be Grid2dRepresentation"
        assert isinstance(gri.grid2d_patch.geometry.points, ro.Point3dZValueArray), "Points must be Point3dZValueArray"
        assert isinstance(gri.grid2d_patch.geometry.points.zvalues, ro.DoubleHdf5Array), "Values must be DoubleHdf5Array"
        assert isinstance(gri.grid2d_patch.geometry.points.zvalues.values, ro.Hdf5Dataset), "Values must be Hdf5Dataset"
        assert isinstance(gri.grid2d_patch.geometry.points.supporting_geometry, ro.Point3dLatticeArray), "Points support_geo must be Point3dLatticeArray"

        sgeo = gri.grid2d_patch.geometry.points.supporting_geometry  # type: ignore
        assert isinstance(sgeo, ro.Point3dLatticeArray), "supporting_geometry must be Point3dLatticeArray"

        # get array
        array = await self.get_array(
            DataArrayIdentifier(
                uri=str(epc_uri), pathInResource=gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file
            )
        )

        return RegularSurface(
            ncol=array.shape[0], nrow=array.shape[1],
            xinc=sgeo.offset[0].spacing.value, yinc=sgeo.offset[1].spacing.value,
            xori=sgeo.origin.coordinate1, yori=sgeo.origin.coordinate2,
            values=array,
            masked=True
        )

    async def put_xtgeo_surface(self, surface: RegularSurface, epsg_code=23031, dataspace: DataspaceURI | str | None = None):
        """Returns (epc_uri, crs_uri, gri_uri)"""
        assert surface.values is not None, "cannot upload empty surface"

        epc, crs, gri = parse_xtgeo_surface_to_resqml_grid(surface, epsg_code)
        epc_uri, crs_uri, gri_uri = await self.put_resqml_objects(epc, crs, gri, dataspace=dataspace)

        response = await self.put_array(
            DataArrayIdentifier(
                uri=epc_uri.raw_uri if isinstance(epc_uri, DataObjectURI) else epc_uri,
                pathInResource=gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file
            ),
            surface.values.filled(np.nan).astype(np.float32)
        )

        return epc_uri, crs_uri, gri_uri

    #
    # array
    #

    async def get_array_metadata(self, *uids: DataArrayIdentifier):
        from etptypes.energistics.etp.v12.protocol.data_array.get_data_array_metadata import \
            GetDataArrayMetadata
        from etptypes.energistics.etp.v12.protocol.data_array.get_data_array_metadata_response import \
            GetDataArrayMetadataResponse

        response = await self.send(
            GetDataArrayMetadata(dataArrays={i.path_in_resource: i for i in uids})
        )
        assert isinstance(response, GetDataArrayMetadataResponse)

        if len(response.array_metadata) != len(uids):
            raise ETPError(f'Not all uids found ({uids})', 11)

        # return in same order as arguments
        return [response.array_metadata[i.path_in_resource] for i in uids]

    async def get_array(self, uid: DataArrayIdentifier):
        from etptypes.energistics.etp.v12.protocol.data_array.get_data_arrays import \
            GetDataArrays
        from etptypes.energistics.etp.v12.protocol.data_array.get_data_arrays_response import \
            GetDataArraysResponse

        # Check if we can upload the full array in one go.
        meta = (await self.get_array_metadata(uid))[0]
        if np.prod(np.array(meta.dimensions)) * np.float32().itemsize > self.max_array_size:
            return await self._get_array_chuncked(uid)

        response = await self.send(
            GetDataArrays(dataArrays={uid.path_in_resource: uid})
        )
        assert isinstance(response, GetDataArraysResponse), "Expected GetDataArraysResponse"

        arrays = list(response.data_arrays.values())
        return etp_data_array_to_numpy(arrays[0])

    async def put_array(self, uid: DataArrayIdentifier, data: np.ndarray):
        from etptypes.energistics.etp.v12.protocol.data_array.put_data_arrays import (
            PutDataArrays, PutDataArraysType)
        from etptypes.energistics.etp.v12.protocol.data_array.put_data_arrays_response import \
            PutDataArraysResponse

        # Check if we can upload the full array in one go.
        if data.nbytes > self.max_array_size:
            return await self._put_array_chuncked(uid, data)

        response = await self.send(
            PutDataArrays(
                dataArrays={uid.path_in_resource: PutDataArraysType(uid=uid, array=numpy_to_etp_data_array(data))})
        )
        assert isinstance(response, PutDataArraysResponse), "Expected PutDataArraysResponse"
        assert len(response.success) == 1, "expected one success from put_array"
        return response.success

    async def get_subarray(self, uid: DataArrayIdentifier, starts: np.ndarray | T.List[int], counts: np.ndarray | T.List[int]):
        starts = np.array(starts).astype(np.int64)
        counts = np.array(counts).astype(np.int64)

        from etptypes.energistics.etp.v12.protocol.data_array.get_data_subarrays import (
            GetDataSubarrays, GetDataSubarraysType)
        from etptypes.energistics.etp.v12.protocol.data_array.get_data_subarrays_response import \
            GetDataSubarraysResponse

        logger.debug(f"get_subarray {starts=:} {counts=:}")

        payload = GetDataSubarraysType(
            uid=uid,
            starts=starts.tolist(),
            counts=counts.tolist(),
        )
        response = await self.send(
            GetDataSubarrays(dataSubarrays={uid.path_in_resource: payload})
        )
        assert isinstance(response, GetDataSubarraysResponse), "Expected GetDataSubarraysResponse"

        arrays = list(response.data_subarrays.values())
        return etp_data_array_to_numpy(arrays[0])

    async def put_subarray(self, uid: DataArrayIdentifier, data: np.ndarray, starts: np.ndarray | T.List[int], counts: np.ndarray | T.List[int], put_uninitialized=False):
        from etptypes.energistics.etp.v12.protocol.data_array.put_data_subarrays import (
            PutDataSubarrays, PutDataSubarraysType)
        from etptypes.energistics.etp.v12.protocol.data_array.put_data_subarrays_response import \
            PutDataSubarraysResponse

        starts = np.array(starts).astype(np.int64)
        counts = np.array(counts).astype(np.int64)
        ends = starts + counts

        if put_uninitialized:
            transport_array_type = get_transfertype_from_dtype(data.dtype)
            await self._put_uninitialized_data_array(uid, data.shape, transport_array_type=transport_array_type)

        slices = tuple(map(lambda se: slice(se[0], se[1]), zip(starts, ends)))
        dataarray = numpy_to_etp_data_array(data[*slices])
        payload = PutDataSubarraysType(
            uid=uid,
            data=dataarray.data,
            starts=starts.tolist(),
            counts=counts.tolist(),
        )

        logger.debug(f"put_subarray {data.shape=:} {starts=:} {counts=:} {dataarray.data.item.__class__.__name__}")

        response = await self.send(
            PutDataSubarrays(dataSubarrays={uid.path_in_resource: payload})
        )
        assert isinstance(response, PutDataSubarraysResponse), "Expected PutDataSubarraysResponse"

        assert len(response.success) == 1, "expected one success"
        return response.success

    #
    # chuncked get array - ETP will not chunck response - so we need to do it manually
    #

    def _get_chunk_sizes(self, shape, dtype: np.dtype[T.Any] = np.dtype(np.float32)):
        shape = np.array(shape)

        # capsize blocksize
        max_items = self.max_array_size / dtype.itemsize  # remove 512 bytes for headers and body
        block_size = np.power(max_items, 1. / len(shape))
        block_size = min(2048, int(block_size // 2) * 2)

        assert block_size > 8, "computed blocksize unreasonable small"

        all_ranges = [range(s // block_size + 1) for s in shape]
        indexes = np.array(np.meshgrid(*all_ranges)).T.reshape(-1, len(shape))

        for ijk in indexes:
            starts = ijk * block_size
            ends = np.fmin(shape, starts + block_size)
            counts = ends - starts

            if any(counts == 0):
                continue

            yield starts, counts

    async def _get_array_chuncked(self, uid: DataArrayIdentifier):

        metadata = (await self.get_array_metadata(uid))[0]
        buffer_shape = np.array(metadata.dimensions).astype(np.int64)

        assert metadata.transport_array_type in SUPPORTED_ARRAY_TRANSPORTS, f"{metadata.transport_array_type} not supported transport type as of yet"
        dtype = SUPPORTED_ARRAY_TRANSPORTS[metadata.transport_array_type]
        buffer = np.zeros(buffer_shape, dtype=dtype)

        async def populate(starts, counts):
            array = await self.get_subarray(uid, starts, counts)
            ends = starts + counts
            slices = tuple(map(lambda se: slice(se[0], se[1]), zip(starts, ends)))
            buffer[slices] = array

        await asyncio.gather(*[
            populate(starts, counts)
            for starts, counts in self._get_chunk_sizes(buffer_shape, dtype)
        ])

        return buffer

    async def _put_array_chuncked(self, uid: DataArrayIdentifier, data: np.ndarray):

        transport_array_type = get_transfertype_from_dtype(data.dtype)
        await self._put_uninitialized_data_array(uid, data.shape, transport_array_type=transport_array_type)

        await asyncio.gather(*[
            self.put_subarray(uid, data, starts, counts)
            for starts, counts in self._get_chunk_sizes(data.shape, data.dtype)
        ])

        return {uid.uri: ''}

    async def _put_uninitialized_data_array(self, uid: DataArrayIdentifier, shape: T.Tuple[int, ...], transport_array_type=AnyArrayType.ARRAY_OF_FLOAT, logical_array_type=AnyLogicalArrayType.ARRAY_OF_BOOLEAN):
        from etptypes.energistics.etp.v12.datatypes.data_array_types.data_array_metadata import \
            DataArrayMetadata
        from etptypes.energistics.etp.v12.protocol.data_array.put_uninitialized_data_arrays import (
            PutUninitializedDataArrays, PutUninitializedDataArrayType)
        from etptypes.energistics.etp.v12.protocol.data_array.put_uninitialized_data_arrays_response import \
            PutUninitializedDataArraysResponse

        payload = PutUninitializedDataArrayType(
            uid=uid,
            metadata=(DataArrayMetadata(
                dimensions=list(shape),  # type: ignore
                transportArrayType=transport_array_type,
                logicalArrayType=logical_array_type,
                storeLastWrite=self.timestamp,
                storeCreated=self.timestamp,
            ))
        )
        response = await self.send(
            PutUninitializedDataArrays(dataArrays={uid.path_in_resource: payload})
        )
        assert isinstance(response, PutUninitializedDataArraysResponse), "Expected PutUninitializedDataArraysResponse"
        assert len(response.success) == 1, "expected one success"
        return response.success


# define an asynchronous context manager
class connect:

    def __init__(self, server_url: str, default_dataspace_uri: DataspaceURI | None = None, authorization: T.Optional[str] = None, timeout=10.):
        self.server_url = server_url
        self.headers = {"Authorization": authorization} if authorization else {}
        self.timeout = timeout
        self.default_dataspace_uri = default_dataspace_uri

    # enter the async context manager
    async def __aenter__(self):

        ws = await websockets.connect(
            self.server_url,
            subprotocols=[ETPClient.SUB_PROTOCOL],  # type: ignore
            extra_headers=self.headers,
            max_size=MAXPAYLOADSIZE
        )
        self.client = ETPClient(ws, default_dataspace_uri=self.default_dataspace_uri, timeout=self.timeout)
        await self.client.connect()

        return self.client

    # exit the async context manager
    async def __aexit__(self, exc_type, exc: Exception, tb: TracebackType):
        await self.client.close()
