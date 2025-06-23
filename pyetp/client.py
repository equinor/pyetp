import asyncio
import datetime
import logging
import sys
import typing as T
import uuid
from collections import defaultdict
from types import TracebackType
import time
import numpy as np
import websockets
from etpproto.connection import (CommunicationProtocol, ConnectionType,
                                 ETPConnection)
from etpproto.messages import Message, MessageFlags
from pydantic import SecretStr
from scipy.interpolate import griddata
from xtgeo import RegularSurface


import pyetp.resqml_objects as ro
#import energyml.resqml.v2_0_1.resqmlv2 as ro
#import energyml.eml.v2_0.commonv2 as roc
from pyetp import utils_arrays, utils_xml
from pyetp.config import SETTINGS
from pyetp.types import *
from pyetp.uri import DataObjectURI, DataspaceURI
from pyetp.utils import short_id, batched
#from asyncio import timeout

try:
    # for py >3.11, we can raise grouped exceptions
    from builtins import ExceptionGroup  # type: ignore
except ImportError:
    def ExceptionGroup(msg, errors):
        return errors[0]

try:
    from asyncio import timeout
except ImportError:
    import async_timeout
    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def timeout(delay: T.Optional[float]) -> T.Any:
        try:
            async with async_timeout.timeout(delay):
                yield None
        except asyncio.CancelledError as e:
            raise asyncio.TimeoutError(f'Timeout ({delay}s)') from e


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    pexec = ETPConnection.generic_transition_table["0"]["1000"]  # protocol exception

    for v in pddict.values():
        v["1000"] = pexec

    return pddict


class ETPClient(ETPConnection):

    generic_transition_table = get_all_etp_protocol_classes()

    _recv_events: T.Dict[int, asyncio.Event]
    _recv_buffer: T.Dict[int, T.List[ETPModel]]

    def __init__(self, ws: websockets.WebSocketClientProtocol, timeout=10.):
        super().__init__(connection_type=ConnectionType.CLIENT)
        self._recv_events = {}
        self._recv_buffer = defaultdict(lambda: list())  # type: ignore
        self.ws = ws

        self.timeout = timeout
        self.client_info.endpoint_capabilities['MaxWebSocketMessagePayloadSize'] = SETTINGS.MaxWebSocketMessagePayloadSize
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

        for msg_part in msg.encode_message_generator(self.max_size, self):
            await self.ws.send(msg_part)

        return msg.header.message_id

    async def _recv(self, correlation_id: int) -> ETPModel:
        assert correlation_id in self._recv_events, "trying to recv response on non-existing message"

        async with timeout(self.timeout):
            await self._recv_events[correlation_id].wait()

        # cleanup
        bodies = self._clear_msg_on_buffer(correlation_id)

        # error handling
        errors = self._parse_error_info(bodies)

        if len(errors) == 1:
            raise ETPError.from_proto(errors.pop())
        elif len(errors) > 1:
            raise ExceptionGroup("Server responded with ETPErrors:", ETPError.from_protos(errors))

        if len(bodies) > 1:
            logger.warning(f"Recived {len(bodies)} messages, but only expected one")

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

    async def close(self, reason=''):
        if self.ws.closed:
            self.__recvtask.cancel("stopped")
            # fast exit if already closed
            return

        try:
            await self._send(CloseSession(reason=reason))
        finally:
            await self.ws.close(reason=reason)
            self.is_connected = False
            self.__recvtask.cancel("stopped")

            if len(self._recv_buffer):
                logger.error(f"Closed connection - but had stuff left in buffers ({len(self._recv_buffer)})")
                # logger.warning(self._recv_buffer)  # may contain data so lets not flood logs

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
                T.cast(bytes, msg_data), ETPClient.generic_transition_table
            )

            if msg is None:
                logger.error(f"Could not parse {msg_data}")
                continue

            logger.debug(f"recv {msg.body.__class__.__name__} {repr(msg.header)}")
            self._add_msg_to_buffer(msg)

    #
    # session related
    #

    async def request_session(self):
        # Handshake protocol
        etp_version = Version(major=1, minor=2, revision=0, patch=0)
        msg = await self.send(
            RequestSession(
                applicationName=SETTINGS.application_name,
                applicationVersion=SETTINGS.application_version,
                clientInstanceId=uuid.uuid4(),  # type: ignore
                requestedProtocols=[
                    SupportedProtocol(protocol=p.value, protocolVersion=etp_version, role='store')
                    for p in CommunicationProtocol
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

    async def authorize(self, authorization: str, supplemental_authorization: T.Mapping[str, str] = {}):



        msg = await self.send(
            Authorize(
                authorization=authorization,
                supplementalAuthorization=supplemental_authorization
            )
        )
        assert msg and isinstance(msg, AuthorizeResponse)

        return msg

    #

    @property
    def max_size(self):
        return SETTINGS.MaxWebSocketMessagePayloadSize
        #return self.client_info.getCapability("MaxWebSocketMessagePayloadSize")

    @property
    def max_array_size(self):
        return self.max_size - 512  # maxsize - 512 bytes for header and body

    @property
    def timestamp(self):
        return int(datetime.datetime.now(datetime.timezone.utc).timestamp())


    def dataspace_uri(self, ds: str) -> DataspaceURI:
        if ds.count("/") > 1:
            raise Exception(f"Max one / in dataspace name")
        return DataspaceURI.from_name(ds)
    
    def list_objects(self, dataspace_uri: DataspaceURI, depth: int = 1) -> list:
        return self.send(GetResources(
                scope=ContextScopeKind.TARGETS_OR_SELF,
                context=ContextInfo(
                    uri=dataspace_uri.raw_uri,
                    depth=depth,
                    dataObjectTypes=[],
                    navigableEdges=RelationshipKind.PRIMARY,)
                )
            )
    #
    # dataspace
    #

    async def put_dataspaces(self, *dataspace_uris: DataspaceURI):
        _uris = list(map(DataspaceURI.from_any, dataspace_uris))
        for i in _uris:
            if i.raw_uri.count("/") > 4: # includes the 3 eml
                raise Exception(f"Max one / in dataspace name")
        time = self.timestamp
        response = await self.send(
            PutDataspaces(dataspaces={
                d.raw_uri: Dataspace(uri=d.raw_uri, storeCreated=time, storeLastWrite=time, path=d.dataspace)
                for d in _uris
            })
        )
        assert isinstance(response, PutDataspacesResponse), "Expected PutDataspacesResponse"

        assert len(response.success) == len(dataspace_uris), f"expected {len(dataspace_uris)} success's"

        return response.success

    async def put_dataspaces_no_raise(self, *dataspace_uris: DataspaceURI):
        try:
            return await self.put_dataspaces(*dataspace_uris)
        except ETPError:
            pass

    async def delete_dataspaces(self, *dataspace_uris: DataspaceURI):


        _uris = list(map(str, dataspace_uris))

        response = await self.send(DeleteDataspaces(uris=dict(zip(_uris, _uris))))
        assert isinstance(response, DeleteDataspacesResponse), "Expected DeleteDataspacesResponse"
        return response.success

    #
    # data objects
    #

    async def get_data_objects(self, *uris: T.Union[DataObjectURI, str]):

        _uris = list(map(str, uris))

        msg = await self.send(
            GetDataObjects(uris=dict(zip(_uris, _uris)))
        )
        assert isinstance(msg, GetDataObjectsResponse), "Expected dataobjectsresponse"
        assert len(msg.data_objects) == len(_uris), "Here we assume that all three objects fit in a single record"

        return [msg.data_objects[u] for u in _uris]

    async def put_data_objects(self, *objs: DataObject):



        response = await self.send(
            PutDataObjects(dataObjects={f"{p.resource.name}_{short_id()}": p for p in objs})
        )
        # logger.info(f"objects {response=:}")
        assert isinstance(response, PutDataObjectsResponse), "Expected PutDataObjectsResponse"
        # assert len(response.success) == len(objs)  # might be 0 if objects exists

        return response.success

    async def get_resqml_objects(self, *uris: T.Union[DataObjectURI, str]) -> T.List[ro.AbstractObject]:
        data_objects = await self.get_data_objects(*uris)
        return utils_xml.parse_resqml_objects(data_objects)

    async def put_resqml_objects(self, *objs: ro.AbstractObject, dataspace_uri: DataspaceURI):

        time = self.timestamp
        uris = [DataObjectURI.from_obj(dataspace_uri, obj) for obj in objs]
        dobjs = [DataObject(
            format="xml",
            data=utils_xml.resqml_to_xml(obj),
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

    async def delete_data_objects(self, *uris: T.Union[DataObjectURI, str], pruneContainedObjects=False):


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
    @staticmethod
    def check_inside(x: float, y: float, patch: ro.Grid2dPatch):
        xori = patch.geometry.points.supporting_geometry.origin.coordinate1
        yori = patch.geometry.points.supporting_geometry.origin.coordinate2
        xmax = xori + (patch.geometry.points.supporting_geometry.offset[0].spacing.value*patch.geometry.points.supporting_geometry.offset[0].spacing.count)
        ymax = yori + (patch.geometry.points.supporting_geometry.offset[1].spacing.value*patch.geometry.points.supporting_geometry.offset[1].spacing.count)
        if x < xori:
            return False
        if y < yori:
            return False
        if x > xmax:
            return False
        if y > ymax:
            return False
        return True

    @staticmethod
    def find_closest_index(x, y, patch: ro.Grid2dPatch):
        x_ind = (x-patch.geometry.points.supporting_geometry.origin.coordinate1)/patch.geometry.points.supporting_geometry.offset[0].spacing.value
        y_ind = (y-patch.geometry.points.supporting_geometry.origin.coordinate2)/patch.geometry.points.supporting_geometry.offset[1].spacing.value
        return round(x_ind), round(y_ind)

    async def get_surface_value_x_y(self, epc_uri: T.Union[DataObjectURI, str], gri_uri: T.Union[DataObjectURI, str],crs_uri: T.Union[DataObjectURI, str], x: T.Union[int, float], y: T.Union[int, float], method: T.Literal["bilinear", "nearest"]):
        gri, = await self.get_resqml_objects(gri_uri) # parallelized using subarray
        xori = gri.grid2d_patch.geometry.points.supporting_geometry.origin.coordinate1
        yori = gri.grid2d_patch.geometry.points.supporting_geometry.origin.coordinate2
        xinc = gri.grid2d_patch.geometry.points.supporting_geometry.offset[0].spacing.value
        yinc = gri.grid2d_patch.geometry.points.supporting_geometry.offset[1].spacing.value
        max_x_index_in_gri = gri.grid2d_patch.geometry.points.supporting_geometry.offset[0].spacing.count
        max_y_index_in_gri = gri.grid2d_patch.geometry.points.supporting_geometry.offset[1].spacing.count
        buffer = 4
        if not self.check_inside(x, y, gri.grid2d_patch):
            logger.info(f"Points not inside {x}:{y} {gri}")
            return
        uid = DataArrayIdentifier(
                    uri=str(epc_uri), pathInResource=gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file
                )
        if max_x_index_in_gri <= 10 or max_y_index_in_gri <= 10:
            surf = await self.get_xtgeo_surface(epc_uri, gri_uri, crs_uri)
            return surf.get_value_from_xy((x, y), sampling=method)

        x_ind, y_ind = self.find_closest_index(x, y, gri.grid2d_patch)
        if method == "nearest":
            arr = await self.get_subarray(uid, [x_ind, y_ind], [1, 1])
            return arr[0][0]
        min_x_ind = max(x_ind-(buffer/2), 0)
        min_y_ind = max(y_ind-(buffer/2), 0)
        count_x = min(max_x_index_in_gri-min_x_ind, buffer)
        count_y = min(max_y_index_in_gri-min_y_ind, buffer)
        # shift start index to left if not enough buffer on right
        if count_x < buffer:
            x_index_to_add = 3 - count_x
            min_x_ind_new = max(0, min_x_ind-x_index_to_add)
            count_x = count_x + min_x_ind-min_x_ind_new+1
            min_x_ind = min_x_ind_new
        if count_y < buffer:
            y_index_to_add = 3 - count_y
            min_y_ind_new = max(0, min_y_ind-y_index_to_add)
            count_y = count_y + min_y_ind-min_y_ind_new+1
            min_y_ind = min_y_ind_new
        arr = await self.get_subarray(uid, [min_x_ind, min_y_ind], [count_x, count_y])
        new_x_ori = xori+(min_x_ind*xinc)
        new_y_ori = yori+(min_y_ind*yinc)
        regridded = RegularSurface(
            ncol=arr.shape[0],
            nrow=arr.shape[1],
            xori=new_x_ori,
            yori=new_y_ori,
            xinc=xinc,
            yinc=yinc,
            rotation=0.0,
            values=arr.flatten(),
        )
        return regridded.get_value_from_xy((x, y))

    async def get_xtgeo_surface(self, epc_uri: T.Union[DataObjectURI, str], gri_uri: T.Union[DataObjectURI, str], crs_uri: T.Union[DataObjectURI, str]):
        gri, crs, = await self.get_resqml_objects(gri_uri, crs_uri)
        rotation = crs.areal_rotation.value
        # some checks

        assert isinstance(gri, ro.Grid2dRepresentation), "obj must be Grid2DRepresentation"
        sgeo = gri.grid2d_patch.geometry.points.supporting_geometry  # type: ignore
        if sys.version_info[1] != 10:
            assert isinstance(gri.grid2d_patch.geometry.points, ro.Point3dZValueArray), "Points must be Point3dZValueArray"
            assert isinstance(gri.grid2d_patch.geometry.points.zvalues, ro.DoubleHdf5Array), "Values must be DoubleHdf5Array"
            assert isinstance(gri.grid2d_patch.geometry.points.supporting_geometry, ro.Point3dLatticeArray), "Points support_geo must be Point3dLatticeArray"
            assert isinstance(sgeo, ro.Point3dLatticeArray), "supporting_geometry must be Point3dLatticeArray"
        assert isinstance(gri.grid2d_patch.geometry.points.zvalues.values, ro.Hdf5Dataset), "Values must be Hdf5Dataset"

        # get array
        array = await self.get_array(
            DataArrayIdentifier(
                uri=str(epc_uri), pathInResource=gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file
            )
        )

        return RegularSurface(
            ncol=array.shape[0], nrow=array.shape[1],
            xinc=sgeo.offset[0].spacing.value, yinc=sgeo.offset[1].spacing.value,  # type: ignore
            xori=sgeo.origin.coordinate1, yori=sgeo.origin.coordinate2,
            values=array,  # type: ignore
            rotation=rotation,
            masked=True
        )
    async def start_transaction(self, dataspace_uri: DataspaceURI, readOnly :bool= True) -> Uuid:
        trans_id = await self.send(StartTransaction(readOnly=readOnly, dataspaceUris=[dataspace_uri.raw_uri]))
        if trans_id.successful is False:
            raise Exception(f"Failed starting transaction {dataspace_uri.raw_uri}")
        return Uuid(trans_id.transaction_uuid) #uuid.UUID(bytes=trans_id.transaction_uuid)
    
    async def commit_transaction(self, transaction_id: Uuid):
        r = await self.send(CommitTransaction(transactionUuid=transaction_id))
        if r.successful is False:
            raise Exception(r.failure_reason)
        return r
    
    async def rollback_transaction(self, transaction_id: Uuid):
        return await self.send(RollbackTransaction(transactionUuid=transaction_id))
    
    async def put_xtgeo_surface(self, surface: RegularSurface, epsg_code: int, dataspace_uri: DataspaceURI):
        """Returns (epc_uri, crs_uri, gri_uri)"""
        assert surface.values is not None, "cannot upload empty surface"
        
        t_id = await self.start_transaction(dataspace_uri, False)
        epc, crs, gri = utils_xml.parse_xtgeo_surface_to_resqml_grid(surface, epsg_code)
        epc_uri, crs_uri, gri_uri = await self.put_resqml_objects(epc, crs, gri, dataspace_uri=dataspace_uri)

        await self.put_array(
            DataArrayIdentifier(
                uri=epc_uri.raw_uri if isinstance(epc_uri, DataObjectURI) else epc_uri,
                pathInResource=gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file  # type: ignore
            ),
            surface.values.filled(np.nan).astype(np.float32),
            t_id
        )

        return epc_uri, gri_uri, crs_uri

    #
    # resqpy meshes
    #

    async def get_epc_mesh(self, epc_uri: T.Union[DataObjectURI, str], uns_uri: T.Union[DataObjectURI, str]):
        uns, = await self.get_resqml_objects(uns_uri)

        # some checks
        assert isinstance(uns, ro.UnstructuredGridRepresentation), "obj must be UnstructuredGridRepresentation"
        assert isinstance(uns.geometry, ro.UnstructuredGridGeometry), "geometry must be UnstructuredGridGeometry"
        if sys.version_info[1] != 10:
            assert isinstance(uns.geometry.points, ro.Point3dHdf5Array), "points must be Point3dHdf5Array"
            assert isinstance(uns.geometry.faces_per_cell.elements, ro.IntegerHdf5Array), "faces_per_cell must be IntegerHdf5Array"
            assert isinstance(uns.geometry.faces_per_cell.cumulative_length, ro.IntegerHdf5Array), "faces_per_cell cl must be IntegerHdf5Array"
        assert isinstance(uns.geometry.points.coordinates, ro.Hdf5Dataset), "coordinates must be Hdf5Dataset"

        # # get array
        points = await self.get_array(
            DataArrayIdentifier(
                uri=str(epc_uri), pathInResource=uns.geometry.points.coordinates.path_in_hdf_file
            )
        )
        nodes_per_face = await self.get_array(
            DataArrayIdentifier(
                uri=str(epc_uri), pathInResource=uns.geometry.nodes_per_face.elements.values.path_in_hdf_file
            )
        )
        nodes_per_face_cl = await self.get_array(
            DataArrayIdentifier(
                uri=str(epc_uri), pathInResource=uns.geometry.nodes_per_face.cumulative_length.values.path_in_hdf_file
            )
        )
        faces_per_cell = await self.get_array(
            DataArrayIdentifier(
                uri=str(epc_uri), pathInResource=uns.geometry.faces_per_cell.elements.values.path_in_hdf_file
            )
        )
        faces_per_cell_cl = await self.get_array(
            DataArrayIdentifier(
                uri=str(epc_uri), pathInResource=uns.geometry.faces_per_cell.cumulative_length.values.path_in_hdf_file
            )
        )
        cell_face_is_right_handed = await self.get_array(
            DataArrayIdentifier(
                uri=str(epc_uri), pathInResource=uns.geometry.cell_face_is_right_handed.values.path_in_hdf_file
            )
        )

        return uns, points, nodes_per_face, nodes_per_face_cl, faces_per_cell, faces_per_cell_cl, cell_face_is_right_handed

    async def get_epc_mesh_property(self, epc_uri: T.Union[DataObjectURI, str], prop_uri: T.Union[DataObjectURI, str]):
        cprop0, = await self.get_resqml_objects(prop_uri)

        # some checks
        assert isinstance(cprop0, ro.ContinuousProperty) or isinstance(cprop0, ro.DiscreteProperty), "prop must be a Property"
        assert len(cprop0.patch_of_values) == 1, "property obj must have exactly one patch of values"

        # # get array
        values = await self.get_array(
            DataArrayIdentifier(
                uri=str(epc_uri), pathInResource=cprop0.patch_of_values[0].values.values.path_in_hdf_file,
            )
        )

        return cprop0, values

    @staticmethod
    def check_bound(points, x: float, y: float):
        if x > points[:, 0].max() or x < points[:, 0].min():
            return False
        if y > points[:, 1].max() or y < points[:, 1].min():
            return False
        return True

    async def get_epc_mesh_property_x_y(self, epc_uri: T.Union[DataObjectURI, str], uns_uri: T.Union[DataObjectURI, str], prop_uri: T.Union[DataObjectURI, str], x: float, y: float):
        uns, = await self.get_resqml_objects(uns_uri)
        points = await self.get_array(
                    DataArrayIdentifier(uri=str(epc_uri), pathInResource=uns.geometry.points.coordinates.path_in_hdf_file))
        chk = self.check_bound(points, x, y)
        if chk == False:
            return None
        unique_y = np.unique(points[:, 1])
        y_smaller_sorted = np.sort(unique_y[np.argwhere(unique_y < y).flatten()])
        if y_smaller_sorted.size > 1:
            y_floor = y_smaller_sorted[-2]
        elif y_smaller_sorted.size == 1:
            y_floor = y_smaller_sorted[-1]
        else:
            pass
        y_larger_sorted = np.sort(unique_y[np.argwhere(unique_y > y).flatten()])
        if y_larger_sorted.size > 1:
            y_ceil = y_larger_sorted[1]
        elif y_larger_sorted.size == 1:
            y_ceil = y_larger_sorted[0]
        else:
            pass
        start_new_row_idx = np.argwhere(np.diff(points[:, 1]) != 0).flatten() + 1

        to_fetch = []
        initial_result_arr_idx = 0
        for i in range(start_new_row_idx.size-1):
            sliced = points[start_new_row_idx[i]:start_new_row_idx[i+1], :]
            if sliced[0, 1] <= y_ceil and sliced[0, 1] >= y_floor:
                # Found slice that has same y
                x_diff = sliced[:, 0]-x
                if all([np.any((x_diff >= 0)), np.any((x_diff <= 0))]):  # y within this slice
                    first_idx = start_new_row_idx[i]
                    count = start_new_row_idx[i+1]-first_idx
                    to_fetch.append([start_new_row_idx[i], start_new_row_idx[i+1], count, initial_result_arr_idx])
                    initial_result_arr_idx += count

        total_points_filtered = sum([i[2] for i in to_fetch])

        cprop, = await self.get_resqml_objects(prop_uri)
        assert str(cprop.indexable_element) == 'IndexableElements.NODES'
        props_uid = DataArrayIdentifier(
                    uri=str(epc_uri), pathInResource=cprop.patch_of_values[0].values.values.path_in_hdf_file)
        meta, = await self.get_array_metadata(props_uid)
        filtered_points = np.zeros((total_points_filtered, 3), dtype=np.float64)
        all_values = np.empty(total_points_filtered, dtype=np.float64)

        async def populate(i):
            end_indx = i[2]+i[3]
            filtered_points[i[3]:end_indx] = points[i[0]:i[1]]
            if utils_arrays.get_nbytes(meta) * i[2]/points.shape[0] > self.max_array_size:
                all_values[i[3]:end_indx] = await self._get_array_chuncked(props_uid, i[0], i[2])
            else:
                all_values[i[3]:end_indx] = await self.get_subarray(props_uid, [i[0]], [i[2]])
            return

        r = await asyncio.gather(*[populate(i) for i in to_fetch])

        if isinstance(cprop, ro.DiscreteProperty):
            method = "nearest"
        else:
            method = "linear"

        # resolution= np.mean(np.diff(filtered[:,-1]))
        top = round(np.min(filtered_points[:, -1]), 1)
        base = round(np.max(filtered_points[:, -1]), 1)
        requested_depth = np.arange(top, base+1, 100)
        requested_depth = requested_depth[requested_depth > 0]
        request = np.tile([x, y, 0], (requested_depth.size, 1))
        request[:, 2] = requested_depth
        interpolated = griddata(filtered_points, all_values, request, method=method)
        response = np.vstack((requested_depth, interpolated))
        response_filtered = response[:, ~np.isnan(response[1])]
        return {"depth": response_filtered[0], "values": response_filtered[1]}

    async def put_rddms_property(self, epc_uri: T.Union[DataObjectURI , str], 
            cprop0: T.Union[ro.ContinuousProperty, ro.DiscreteProperty], 
            propertykind0: ro.PropertyKind, 
            array_ref: np.ndarray, 
            dataspace_uri: DataspaceURI ):

        assert isinstance(cprop0, ro.ContinuousProperty) or isinstance(cprop0, ro.DiscreteProperty), "prop must be a Property"
        assert len(cprop0.patch_of_values) == 1, "property obj must have exactly one patch of values"

        st = time.time()
        t_id = await self.start_transaction(dataspace_uri, False)
        propkind_uri = [""] if (propertykind0 is None) else (await self.put_resqml_objects(propertykind0, dataspace_uri=dataspace_uri))
        cprop_uri = await self.put_resqml_objects(cprop0, dataspace_uri=dataspace_uri)
        delay = time.time() - st
        logger.debug(f"pyetp: put_rddms_property: put objects took {delay} s")
        
        st = time.time()
        response = await self.put_array(
            DataArrayIdentifier(
                uri=epc_uri.raw_uri if isinstance(epc_uri, DataObjectURI) else epc_uri,
                pathInResource=cprop0.patch_of_values[0].values.values.path_in_hdf_file,
            ),
            array_ref,  # type: ignore
            t_id
        )
        delay = time.time() - st
        logger.debug(f"pyetp: put_rddms_property: put array ({array_ref.shape}) took {delay} s")
        return cprop_uri, propkind_uri

    async def put_epc_mesh(
        self, epc_filename: str, title_in: str, property_titles: T.List[str], projected_epsg: int, 
        dataspace_uri: DataspaceURI
    ):
        uns, crs, epc, timeseries, hexa = utils_xml.convert_epc_mesh_to_resqml_mesh(epc_filename, title_in, projected_epsg)
        t_id = await self.start_transaction(dataspace_uri, False)
        epc_uri, crs_uri, uns_uri = await self.put_resqml_objects(epc, crs, uns, dataspace_uri=dataspace_uri)
        timeseries_uri = ""
        if timeseries is not None:
            timeseries_uris = await self.put_resqml_objects(timeseries, dataspace_uri=dataspace_uri)
            timeseries_uri = list(timeseries_uris)[0] if (len(list(timeseries_uris)) > 0) else ""
        
        #
        # mesh geometry (six arrays)
        #
        response = await self.put_array(
            DataArrayIdentifier(
                uri=epc_uri.raw_uri if isinstance(epc_uri, DataObjectURI) else epc_uri,
                pathInResource=uns.geometry.points.coordinates.path_in_hdf_file
            ),
            hexa.points_cached  # type: ignore
        )

        response = await self.put_array(
            DataArrayIdentifier(
                uri=epc_uri.raw_uri if isinstance(epc_uri, DataObjectURI) else epc_uri,
                pathInResource=uns.geometry.nodes_per_face.elements.values.path_in_hdf_file
            ),
            hexa.nodes_per_face.astype(np.int32)  # type: ignore
        )

        response = await self.put_array(
            DataArrayIdentifier(
                uri=epc_uri.raw_uri if isinstance(epc_uri, DataObjectURI) else epc_uri,
                pathInResource=uns.geometry.nodes_per_face.cumulative_length.values.path_in_hdf_file
            ),
            hexa.nodes_per_face_cl  # type: ignore
        )

        response = await self.put_array(
            DataArrayIdentifier(
                uri=epc_uri.raw_uri if isinstance(epc_uri, DataObjectURI) else epc_uri,
                pathInResource=uns.geometry.faces_per_cell.elements.values.path_in_hdf_file
            ),
            hexa.faces_per_cell  # type: ignore
        )

        response = await self.put_array(
            DataArrayIdentifier(
                uri=epc_uri.raw_uri if isinstance(epc_uri, DataObjectURI) else epc_uri,
                pathInResource=uns.geometry.faces_per_cell.cumulative_length.values.path_in_hdf_file
            ),
            hexa.faces_per_cell_cl  # type: ignore
        )

        response = await self.put_array(
            DataArrayIdentifier(
                uri=epc_uri.raw_uri if isinstance(epc_uri, DataObjectURI) else epc_uri,
                pathInResource=uns.geometry.cell_face_is_right_handed.values.path_in_hdf_file
            ),
            hexa.cell_face_is_right_handed  # type: ignore
        )
        await self.commit_transaction(t_id)
        #
        # mesh properties: one Property, one array of values, and an optional PropertyKind per property
        #
        prop_rddms_uris = {}
        for propname in property_titles:
            if timeseries is not None:
                time_indices = list(range(len(timeseries.time)))
                cprop0s, props, propertykind0 = utils_xml.convert_epc_mesh_property_to_resqml_mesh(epc_filename, hexa, propname, uns, epc, timeseries=timeseries, time_indices=time_indices)
            else:
                time_indices = [-1]
                cprop0s, props, propertykind0 = utils_xml.convert_epc_mesh_property_to_resqml_mesh(epc_filename, hexa, propname, uns, epc)

            if cprop0s is None:
                continue

            cprop_uris = []
            for cprop0, prop, time_index in zip(cprop0s, props, time_indices):
                cprop_uri, propkind_uri = await self.put_rddms_property(epc_uri, cprop0, propertykind0, prop.array_ref(), dataspace_uri)
                cprop_uris.extend(cprop_uri)
            prop_rddms_uris[propname] = [propkind_uri, cprop_uris]

        return [epc_uri, crs_uri, uns_uri, timeseries_uri], prop_rddms_uris

    async def get_mesh_points(self, epc_uri: T.Union[DataObjectURI, str], uns_uri: T.Union[DataObjectURI, str]):
        uns, = await self.get_resqml_objects(uns_uri)
        points = await self.get_array(
                DataArrayIdentifier(
                    uri=str(epc_uri), pathInResource=uns.geometry.points.coordinates.path_in_hdf_file
                )
            )
        return points

    async def get_epc_property_surface_slice_node(self, epc_uri: T.Union[DataObjectURI, str], cprop0: ro.AbstractObject, points: np.ndarray, node_index: int, n_node_per_pos: int):
        # indexing_array = np.arange(0, points.shape[0], 1, dtype=np.int32)[node_index::n_node_per_pos]
        indexing_array = np.arange(node_index, points.shape[0], n_node_per_pos, dtype=np.int32)
        results = points[indexing_array, :]
        arr = await asyncio.gather(*[self.get_subarray(DataArrayIdentifier(
                uri=str(epc_uri), pathInResource=cprop0.patch_of_values[0].values.values.path_in_hdf_file,),
                [i], [1]) for i in indexing_array])
        arr = np.array(arr).flatten()
        assert results.shape[0] == arr.size
        results[:, 2] = arr
        return results

    async def get_epc_property_surface_slice_cell(self, epc_uri: T.Union[DataObjectURI, str], cprop0: ro.AbstractObject, points: np.ndarray, node_index: int, n_node_per_pos: int, get_cell_pos=True):
        m, = await self.get_array_metadata(DataArrayIdentifier(
                    uri=str(epc_uri), pathInResource=cprop0.patch_of_values[0].values.values.path_in_hdf_file,))
        n_cells = m.dimensions[0]
        layers_per_sediment_unit = 2
        n_cell_per_pos = n_node_per_pos - 1
        indexing_array = np.arange(node_index, n_cells, n_cell_per_pos, dtype=np.int32)
        if get_cell_pos:
            results = utils_arrays.get_cells_positions(points, n_cells, n_cell_per_pos, layers_per_sediment_unit, n_node_per_pos, node_index)
        else:
            results = np.zeros((int(n_cells/n_cell_per_pos), 3), dtype=np.float64)
        arr = await asyncio.gather(*[self.get_subarray(DataArrayIdentifier(
                uri=str(epc_uri), pathInResource=cprop0.patch_of_values[0].values.values.path_in_hdf_file,),
                [i], [1]) for i in indexing_array])
        arr = np.array(arr).flatten()
        assert results.shape[0] == arr.size
        results[:, 2] = arr
        return results

    async def get_epc_property_surface_slice(self, epc_uri: T.Union[DataObjectURI, str], uns_uri: T.Union[DataObjectURI, str], prop_uri: T.Union[DataObjectURI, str], node_index: int, n_node_per_pos: int):
        # n_node_per_pos number of nodes in a 1D location
        # node_index index of slice from top. Warmth has 2 nodes per sediment layer. E.g. top of second layer will have index 2
        points = await self.get_mesh_points(epc_uri, uns_uri)
        cprop0, = await self.get_resqml_objects(prop_uri)
        prop_at_node = False
        if str(cprop0.indexable_element) == 'IndexableElements.NODES':
            prop_at_node = True
        # node_per_sed = 2
        # n_sed_node = n_sed *node_per_sed
        # n_crust_node = 4
        # n_node_per_pos = n_sed_node + n_crust_node
        # start_idx_pos = sediment_id *node_per_sed
        if prop_at_node:
            return await self.get_epc_property_surface_slice_node(epc_uri, cprop0, points, node_index, n_node_per_pos)
        else:
            return await self.get_epc_property_surface_slice_cell(epc_uri, cprop0, points, node_index, n_node_per_pos)

    async def get_epc_property_surface_slice_xtgeo(self, epc_uri: T.Union[DataObjectURI, str], uns_uri: T.Union[DataObjectURI, str], prop_uri: T.Union[DataObjectURI, str], node_index: int, n_node_per_pos: int):
        data = await self.get_epc_property_surface_slice(epc_uri, uns_uri, prop_uri, node_index, n_node_per_pos)
        return utils_arrays.grid_xtgeo(data)

    #
    # array
    #

    async def get_array_metadata(self, *uids: DataArrayIdentifier):


        response = await self.send(
            GetDataArrayMetadata(dataArrays={i.path_in_resource: i for i in uids})
        )
        assert isinstance(response, GetDataArrayMetadataResponse)

        if len(response.array_metadata) != len(uids):
            raise ETPError(f'Not all uids found ({uids})', 11)

        # return in same order as arguments
        return [response.array_metadata[i.path_in_resource] for i in uids]

    async def get_array(self, uid: DataArrayIdentifier):


        # Check if we can upload the full array in one go.
        meta, = await self.get_array_metadata(uid)
        if utils_arrays.get_nbytes(meta) > self.max_array_size:
            return await self._get_array_chuncked(uid)

        response = await self.send(
            GetDataArrays(dataArrays={uid.path_in_resource: uid})
        )
        assert isinstance(response, GetDataArraysResponse), "Expected GetDataArraysResponse"

        arrays = list(response.data_arrays.values())
        return utils_arrays.to_numpy(arrays[0])

    async def put_array(self, uid: DataArrayIdentifier, data: np.ndarray, transaction_id: Uuid | None = None):


        # Check if we can upload the full array in one go.
        if data.nbytes > self.max_array_size:
            return await self._put_array_chuncked(uid, data, transaction_id)
        
        response = await self.send(
            PutDataArrays(
                dataArrays={uid.path_in_resource: PutDataArraysType(uid=uid, array=utils_arrays.to_data_array(data))})
        )

        assert isinstance(response, PutDataArraysResponse), "Expected PutDataArraysResponse"
        assert len(response.success) == 1, "expected one success from put_array"
        if isinstance(transaction_id, Uuid):
            await self.commit_transaction(transaction_id)
        return response.success


    async def get_subarray(self, uid: DataArrayIdentifier, starts: T.Union[np.ndarray, T.List[int]], counts: T.Union[np.ndarray, T.List[int]]):
        starts = np.array(starts).astype(np.int64)
        counts = np.array(counts).astype(np.int64)



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
        return utils_arrays.to_numpy(arrays[0])

    async def put_subarray(self, uid: DataArrayIdentifier, data: np.ndarray, starts: T.Union[np.ndarray, T.List[int]], counts: T.Union[np.ndarray, T.List[int]]):


        # starts [start_X, starts_Y]
        # counts [count_X, count_Y]
        starts = np.array(starts).astype(np.int64) # len = 2 [x_start_index, y_start_index]
        counts = np.array(counts).astype(np.int64) # len = 2
        ends = starts + counts # len = 2


        slices = tuple(map(lambda se: slice(se[0], se[1]), zip(starts, ends)))
        dataarray = utils_arrays.to_data_array(data[slices])
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

    def _get_chunk_sizes(self, shape, dtype: np.dtype[T.Any] = np.dtype(np.float32), offset=0):
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
            if offset != 0:
                starts = starts + offset
            ends = np.fmin(shape, starts + block_size)
            if offset != 0:
                ends = ends + offset
            counts = ends - starts
            if any(counts == 0):
                continue
            yield starts, counts

    async def _get_array_chuncked(self, uid: DataArrayIdentifier, offset: int = 0, total_count: T.Union[int, None] = None):

        metadata = (await self.get_array_metadata(uid))[0]
        if len(metadata.dimensions) != 1 and offset != 0:
            raise Exception("Offset is only implemented for 1D array")

        if isinstance(total_count, (int, float)):
            buffer_shape = np.array([total_count], dtype=np.int64)
        else:
            buffer_shape = np.array(metadata.dimensions, dtype=np.int64)
        dtype = utils_arrays.get_dtype(metadata.transport_array_type)
        buffer = np.zeros(buffer_shape, dtype=dtype)
        params = []

        async def populate(starts, counts):
            params.append([starts, counts])
            array = await self.get_subarray(uid, starts, counts)
            ends = starts + counts
            slices = tuple(map(lambda se: slice(se[0], se[1]), zip(starts-offset, ends-offset)))
            buffer[slices] = array
            return
        # coro = [populate(starts, counts) for starts, counts in self._get_chunk_sizes(buffer_shape, dtype, offset)]
        # logger.debug(f"Concurrent request: {self.max_concurrent_requests}")
        # for i in batched(coro, self.max_concurrent_requests):
        #     await asyncio.gather(*i)    
        r = await asyncio.gather(*[
            populate(starts, counts)
            for starts, counts in self._get_chunk_sizes(buffer_shape, dtype, offset)
        ])

        return buffer

    async def _put_array_chuncked(self, uid: DataArrayIdentifier, data: np.ndarray, transaction_id: Uuid | None = None):
        transport_array_type = utils_arrays.get_transport(data.dtype)

        await self._put_uninitialized_data_array(uid, data.shape, transport_array_type=transport_array_type)
        if isinstance(transaction_id, Uuid):
            await self.commit_transaction(transaction_id)

        ds_uri = DataspaceURI.from_any(uid.uri)
        t_id = None
        if isinstance(transaction_id, Uuid):
            t_id = await self.start_transaction(ds_uri, False)
        for starts, counts in self._get_chunk_sizes(data.shape, data.dtype):
            await self.put_subarray(uid, data, starts, counts)
        if isinstance(t_id, Uuid):
            await self.commit_transaction(t_id)
        
        return {uid.uri: ''}

    async def _put_uninitialized_data_array(self, uid: DataArrayIdentifier, shape: T.Tuple[int, ...], transport_array_type=AnyArrayType.ARRAY_OF_FLOAT, logical_array_type=AnyLogicalArrayType.ARRAY_OF_BOOLEAN):


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

        ws = await websockets.connect(
            self.server_url,
            subprotocols=[ETPClient.SUB_PROTOCOL],  # type: ignore
            extra_headers=headers,
            max_size=SETTINGS.MaxWebSocketMessagePayloadSize,
            ping_timeout=self.timeout,
            open_timeout=None,
        )

        self.client = ETPClient(ws, timeout=self.timeout)

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
