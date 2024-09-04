import asyncio
import sys
import datetime
import logging
import time
import typing as T
import uuid
from collections import defaultdict
from types import TracebackType
from async_timeout import timeout
import numpy as np
from pydantic import SecretStr
import websockets
from etpproto.connection import (CommunicationProtocol, ConnectionType,
                                 ETPConnection)
from etpproto.messages import Message, MessageFlags
from xtgeo import RegularSurface
import xtgeo
from scipy.interpolate import griddata
import pyetp.resqml_objects as ro
from pyetp.config import SETTINGS

from . import utils_arrays, utils_xml
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

    def __init__(self, ws: websockets.WebSocketClientProtocol, default_dataspace_uri: T.Union[DataspaceURI , None], timeout=10.):
        super().__init__(connection_type=ConnectionType.CLIENT)
        self._recv_events = {}
        self._recv_buffer = defaultdict(lambda: list())  # type: ignore
        self._default_duri = default_dataspace_uri
        self.ws = ws

        self.timeout = timeout
        self.client_info.endpoint_capabilities['MaxWebSocketMessagePayloadSize'] = MAXPAYLOADSIZE
        self.__recvtask = asyncio.create_task(self.__recv__())
        self.safeMode = False
        self.retryPause = 30

    #
    # client
    #

    async def send(self, body: ETPModel):
        if self.ws.closed:
            await self.connect()
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

        try:
            async with timeout(self.timeout):
                await self._recv_events[correlation_id].wait()
        except asyncio.CancelledError as e:
            raise TimeoutError(f'Timeout before reciving {correlation_id=}') from e

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
    def default_dataspace_uri(self, v: T.Union[DataspaceURI , str , None]):
        self._default_duri = None if v is None else DataspaceURI.from_any(v)

    def get_dataspace_or_default_uri(self, ds: T.Union[DataspaceURI , str , None]) -> DataspaceURI:
        """Returns default dataspace or user spefied one"""

        if ds is not None:
            return DataspaceURI.from_any(ds)

        if self._default_duri is None:
            raise ValueError("Could not get dataspace from userinput or default")

        return self._default_duri

    #
    # dataspace
    #

    async def put_dataspaces(self, *uris: T.Union[DataspaceURI , str]):
        from etptypes.energistics.etp.v12.protocol.dataspace.put_dataspaces import \
            PutDataspaces
        from etptypes.energistics.etp.v12.protocol.dataspace.put_dataspaces_response import \
            PutDataspacesResponse

        _uris = list(map(DataspaceURI.from_any, uris))

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

    async def put_dataspaces_no_raise(self, *uris: T.Union[DataspaceURI , str]):
        try:
            return await self.put_dataspaces(*uris)
        except ETPError:
            pass

    async def delete_dataspaces(self, *uris:T.Union[DataspaceURI , str]):
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

    async def get_data_objects(self, *uris: T.Union[DataObjectURI , str]):

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

    async def get_resqml_objects(self, *uris: T.Union[DataObjectURI , str]) -> T.List[ro.AbstractObject]:
        data_objects = await self.get_data_objects(*uris)
        return utils_xml.parse_resqml_objects(data_objects)

    async def put_resqml_objects(self, *objs: ro.AbstractObject, dataspace: T.Union[DataspaceURI ,str , None] = None):
        from etptypes.energistics.etp.v12.datatypes.object.resource import \
            Resource

        time = self.timestamp
        duri = self.get_dataspace_or_default_uri(dataspace)

        uris = [DataObjectURI.from_obj(duri, obj) for obj in objs]

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

    async def delete_data_objects(self, *uris: T.Union[DataObjectURI , str], pruneContainedObjects=False):
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
    @staticmethod
    def check_inside(x:float,y:float,patch:ro.Grid2dPatch):
        xori= patch.geometry.points.supporting_geometry.origin.coordinate1
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
    def find_closest_index(x,y,patch:ro.Grid2dPatch):
        x_ind = (x-patch.geometry.points.supporting_geometry.origin.coordinate1)/patch.geometry.points.supporting_geometry.offset[0].spacing.value
        y_ind = (y-patch.geometry.points.supporting_geometry.origin.coordinate2)/patch.geometry.points.supporting_geometry.offset[1].spacing.value
        return round(x_ind),round(y_ind)
    
    async def get_surface_value_x_y(self, epc_uri: T.Union[DataObjectURI , str] ,gri_uri: T.Union[DataObjectURI , str], x: T.Union[int,float], y: T.Union[int,float], method:T.Literal["bilinear","nearest"]):
        gri, = await self.get_resqml_objects(gri_uri) # parallelized using subarray
        xori= gri.grid2d_patch.geometry.points.supporting_geometry.origin.coordinate1
        yori = gri.grid2d_patch.geometry.points.supporting_geometry.origin.coordinate2 
        xinc = gri.grid2d_patch.geometry.points.supporting_geometry.offset[0].spacing.value
        yinc = gri.grid2d_patch.geometry.points.supporting_geometry.offset[1].spacing.value
        max_x_index_in_gri = gri.grid2d_patch.geometry.points.supporting_geometry.offset[0].spacing.count
        max_y_index_in_gri = gri.grid2d_patch.geometry.points.supporting_geometry.offset[1].spacing.count
        buffer = 4
        if not self.check_inside(x,y,gri.grid2d_patch):
            logger.info(f"Points not inside {x}:{y} {gri}")
            return
        uid=DataArrayIdentifier(
                    uri=str(epc_uri), pathInResource=gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file
                )
        if max_x_index_in_gri <= 10 or max_y_index_in_gri <= 10:
            surf = await self.get_xtgeo_surface(epc_uri,gri_uri)
            return surf.get_value_from_xy((x,y),sampling=method)
        
        x_ind,y_ind=self.find_closest_index(x,y,gri.grid2d_patch)
        if method == "nearest":
            arr = await self.get_subarray(uid,[x_ind,y_ind],[1,1])
            return arr[0][0]
        min_x_ind = max(x_ind-(buffer/2),0)
        min_y_ind = max(y_ind-(buffer/2),0)
        count_x = min(max_x_index_in_gri-min_x_ind,buffer)
        count_y = min(max_y_index_in_gri-min_y_ind,buffer)
        ## shift start index to left if not enough buffer on right
        if count_x < buffer:
            x_index_to_add = 3- count_x
            min_x_ind_new = max(0,min_x_ind-x_index_to_add)
            count_x=count_x + min_x_ind-min_x_ind_new+1
            min_x_ind = min_x_ind_new
        if count_y < buffer:
            y_index_to_add = 3- count_y
            min_y_ind_new = max(0,min_y_ind-y_index_to_add)
            count_y=count_y + min_y_ind-min_y_ind_new+1
            min_y_ind = min_y_ind_new
        arr = await self.get_subarray(uid,[min_x_ind,min_y_ind],[count_x,count_y])
        new_x_ori = xori+(min_x_ind*xinc)
        new_y_ori = yori+(min_y_ind*yinc) 
        regridded = xtgeo.RegularSurface(
            ncol=arr.shape[0],
            nrow=arr.shape[1],
            xori=new_x_ori,
            yori=new_y_ori,
            xinc=xinc,
            yinc=yinc,
            rotation=0.0,
            values=arr.flatten(),
        )
        return regridded.get_value_from_xy((x,y))
    
    async def get_xtgeo_surface(self, epc_uri: T.Union[DataObjectURI , str] ,gri_uri: T.Union[DataObjectURI , str]):
        gri, = await self.get_resqml_objects(gri_uri)

        # some checks
        
        assert isinstance(gri, ro.Grid2dRepresentation), "obj must be Grid2dRepresentation"
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
            masked=True
        )

    async def put_xtgeo_surface(self, surface: RegularSurface, epsg_code=23031, dataspace: T.Union[DataspaceURI , str , None] = None):
        """Returns (epc_uri, crs_uri, gri_uri)"""
        assert surface.values is not None, "cannot upload empty surface"

        epc, crs, gri = utils_xml.parse_xtgeo_surface_to_resqml_grid(surface, epsg_code)
        epc_uri, crs_uri, gri_uri = await self.put_resqml_objects(epc, crs, gri, dataspace=dataspace)
        response = await self.put_array(
            DataArrayIdentifier(
                uri=epc_uri.raw_uri if isinstance(epc_uri, DataObjectURI) else epc_uri,
                pathInResource=gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file  # type: ignore
            ),
            surface.values.filled(np.nan).astype(np.float32)
        )

        return epc_uri, crs_uri, gri_uri

    #
    # resqpy meshes
    #

    async def get_epc_mesh(self, epc_uri: T.Union[DataObjectURI , str], uns_uri: T.Union[DataObjectURI , str]):
        uns, = await self.get_resqml_objects(uns_uri)

        # some checks
        assert isinstance(uns, ro.UnstructuredGridRepresentation), "obj must be Grid2dRepresentation"
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

    async def get_epc_mesh_property(self, epc_uri: T.Union[DataObjectURI , str], prop_uri: T.Union[DataObjectURI , str]):
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
    def check_bound(points, x:float, y:float):
        if x > points[:,0].max() or x < points[:,0].min():
            return False
        if y > points[:,1].max() or y < points[:,1].min():
            return False
        return True
    
    async def get_epc_mesh_property_x_y(self, epc_uri: T.Union[DataObjectURI , str], uns_uri: T.Union[DataObjectURI , str], prop_uri: T.Union[DataObjectURI , str], x: float, y: float):
        uns, = await self.get_resqml_objects(uns_uri)
        points = await self.get_array(
        DataArrayIdentifier(
            uri=str(epc_uri), pathInResource=uns.geometry.points.coordinates.path_in_hdf_file
        ))
        chk = self.check_bound(points, x, y)
        if chk == False:
            return None
        x_lower_bound_index = np.argwhere(points[:,0]<x)[-1,0]
        first_x = np.argwhere(points[:,0]==points[x_lower_bound_index,0])[0,0]

        x_upper_bound_index = np.argwhere(points[:,0]>x)[0,0]
        last_x = np.argwhere(points[:,0]==points[x_upper_bound_index,0])[-1,0]

        y_lower_bound_index = np.argwhere(points[:,1]<y)[-1,0]
        first_y = np.argwhere(points[:,1]==points[y_lower_bound_index,1])[0,0]

        y_upper_bound_index = np.argwhere(points[:,1]>y)[0,0]
        last_y = np.argwhere(points[:,1]==points[y_upper_bound_index,1])[-1,0]


        first_idx = min(first_x,first_y)
        last_idx = max(last_x,last_y)
        filtered = points[first_idx:last_idx,:]
        cprop, = await self.get_resqml_objects(prop_uri)
        assert str(cprop.indexable_element) == 'IndexableElements.NODES'
        props_uid = DataArrayIdentifier(
                    uri=str(epc_uri), pathInResource=cprop.patch_of_values[0].values.values.path_in_hdf_file,
                )
        meta, = await self.get_array_metadata(props_uid)
        if utils_arrays.get_nbytes(meta)* filtered.shape[0]/points.shape[0] < self.max_array_size:
            values = await self._get_array_chuncked(props_uid,first_idx,filtered.shape[0])
        else:
            values = await self.get_subarray(props_uid, [first_idx], [filtered.shape[0]])
        if isinstance(cprop, ro.DiscreteProperty):
            method = "nearest"
        else:
            method = "linear"
        
        # resolution= np.mean(np.diff(filtered[:,-1]))
        top = np.min(filtered[:,-1])
        base = np.max(filtered[:,-1])
        requested_depth = np.arange(top,base+1,100)
        request = np.tile([x, y, 0], (requested_depth.size,1))
        request[:,2]=requested_depth
        interpolated = griddata(filtered, values, request, method=method)
        response = np.vstack((requested_depth,interpolated))
        response_filtered = response[:, ~np.isnan(response[1])]
        return {"depth": response_filtered[0], "values": response_filtered[1]}
       

    async def put_epc_mesh(
        self, epc_filename, title_in, property_titles, projected_epsg, dataspace
    ):
        uns, crs, epc, timeseries, hexa = utils_xml.convert_epc_mesh_to_resqml_mesh(epc_filename, title_in, projected_epsg)
        epc_uri, crs_uri, uns_uri = await self.put_resqml_objects(epc, crs, uns, dataspace=dataspace)
        timeseries_uri = ""
        if timeseries is not None:
            timeseries_uris = await self.put_resqml_objects(timeseries, dataspace=dataspace)
            timeseries_uri = list(timeseries_uris)[0] if (len(list(timeseries_uris)) > 0) else ""
        
        # print("put_epc_mesh", uns, crs, epc, timeseries)
        # print("put_epc_mesh", epc_uri, crs_uri, uns_uri, timeseries_uri)
        print("put_epc_mesh property_titles", property_titles)

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

        #
        # mesh properties: one Property, one array of values, and an optional PropertyKind per property
        #
        prop_rddms_uris = {}
        for propname in property_titles:
            if timeseries is not None:
                time_indices = list(range(len(timeseries.time)))
                # print(f"prop {propname}: len of timeseries: {len(timeseries.time)}")
                print(propname)
                cprop0s, props, propertykind0 = utils_xml.convert_epc_mesh_property_to_resqml_mesh(epc_filename, hexa, propname, uns, epc, timeseries=timeseries, time_indices=time_indices)
            else:
                time_indices = [-1]
                cprop0s, props, propertykind0 = utils_xml.convert_epc_mesh_property_to_resqml_mesh(epc_filename, hexa, propname, uns, epc)
            if isinstance(cprop0s, type(None)):
                continue
            cprop_uris = []
            for cprop0, prop, time_index in zip(cprop0s, props, time_indices):
                assert isinstance(cprop0, ro.ContinuousProperty) or isinstance(cprop0, ro.DiscreteProperty), "prop must be a Property"
                assert len(cprop0.patch_of_values) == 1, "property obj must have exactly one patch of values"

                propkind_uri = [""] if (propertykind0 is None) else (await self.put_resqml_objects(propertykind0, dataspace=dataspace))
                cprop_uri = await self.put_resqml_objects(cprop0, dataspace=dataspace)
                
                response = await self.put_array(
                    DataArrayIdentifier(
                        uri=epc_uri.raw_uri if isinstance(epc_uri, DataObjectURI) else epc_uri,
                        pathInResource=cprop0.patch_of_values[0].values.values.path_in_hdf_file,
                    ),
                    prop.array_ref(),  # type: ignore
                )
                cprop_uris.extend(cprop_uri)
            prop_rddms_uris[propname] = [propkind_uri, cprop_uris]

        return [epc_uri, crs_uri, uns_uri, timeseries_uri], prop_rddms_uris

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
        meta, = await self.get_array_metadata(uid)
        if utils_arrays.get_nbytes(meta) > self.max_array_size:
            return await self._get_array_chuncked(uid)

        response = await self.send(
            GetDataArrays(dataArrays={uid.path_in_resource: uid})
        )
        assert isinstance(response, GetDataArraysResponse), "Expected GetDataArraysResponse"

        arrays = list(response.data_arrays.values())
        return utils_arrays.to_numpy(arrays[0])

    async def put_array(self, uid: DataArrayIdentifier, data: np.ndarray):
        from etptypes.energistics.etp.v12.datatypes.data_array_types.put_data_arrays_type import \
            PutDataArraysType
        from etptypes.energistics.etp.v12.protocol.data_array.put_data_arrays import \
            PutDataArrays
        from etptypes.energistics.etp.v12.protocol.data_array.put_data_arrays_response import \
            PutDataArraysResponse

        # Check if we can upload the full array in one go.
        if data.nbytes > self.max_array_size:
            return await self._put_array_chuncked(uid, data)

        response = await self.send(
            PutDataArrays(
                dataArrays={uid.path_in_resource: PutDataArraysType(uid=uid, array=utils_arrays.to_data_array(data))})
        )
        assert isinstance(response, PutDataArraysResponse), "Expected PutDataArraysResponse"
        assert len(response.success) == 1, "expected one success from put_array"
        return response.success

    async def get_subarray(self, uid: DataArrayIdentifier, starts: T.Union[np.ndarray , T.List[int]], counts: T.Union[np.ndarray , T.List[int]]):
        starts = np.array(starts).astype(np.int64)
        counts = np.array(counts).astype(np.int64)

        from etptypes.energistics.etp.v12.datatypes.data_array_types.get_data_subarrays_type import \
            GetDataSubarraysType
        from etptypes.energistics.etp.v12.protocol.data_array.get_data_subarrays import \
            GetDataSubarrays
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
        return utils_arrays.to_numpy(arrays[0])

    async def put_subarray(self, uid: DataArrayIdentifier, data: np.ndarray, starts: T.Union[np.ndarray ,T.List[int]], counts: T.Union[np.ndarray , T.List[int]], put_uninitialized=False):
        from etptypes.energistics.etp.v12.datatypes.data_array_types.put_data_subarrays_type import \
            PutDataSubarraysType
        from etptypes.energistics.etp.v12.protocol.data_array.put_data_subarrays import \
            PutDataSubarrays
        from etptypes.energistics.etp.v12.protocol.data_array.put_data_subarrays_response import \
            PutDataSubarraysResponse
        # starts [start_X, starts_Y]
        # counts [count_X, count_Y]
        starts = np.array(starts).astype(np.int64) # len = 2 [x_start_index, y_start_index]
        counts = np.array(counts).astype(np.int64) # len = 2
        ends = starts + counts # len = 2
        if put_uninitialized:
            transport_array_type = utils_arrays.get_transport(data.dtype)
            await self._put_uninitialized_data_array(uid, data.shape, transport_array_type=transport_array_type)

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

    def _get_chunk_sizes(self, shape, dtype: np.dtype[T.Any] = np.dtype(np.float32), offset = 0):
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

    async def _get_array_chuncked(self, uid: DataArrayIdentifier, offset:int = 0, total_count: T.Union[int, None] = None):

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

        r = await asyncio.gather(*[
            populate(starts, counts)
            for starts, counts in self._get_chunk_sizes(buffer_shape, dtype, offset)
        ], return_exceptions=True)
        if any(r):
            time.sleep(self.retryPause)
            if self.ws.closed:
                await self.connect()

            [await populate(params[idx][0], params[idx][1]) for idx, res in enumerate(r) if isinstance(res, type(None)) == False]
        return buffer

    async def _put_array_chuncked(self, uid: DataArrayIdentifier, data: np.ndarray):

        transport_array_type = utils_arrays.get_transport(data.dtype)
        await self._put_uninitialized_data_array(uid, data.shape, transport_array_type=transport_array_type)
        params = []
        coro = []
        for starts, counts in self._get_chunk_sizes(data.shape, data.dtype):
            params.append([starts, counts])
            coro.append(self.put_subarray(uid, data, starts, counts))
        r = await asyncio.gather(*coro, return_exceptions=True)
        if any(r):
            time.sleep(self.retryPause)
            if self.ws.closed:
                await self.connect()
            [await self.put_subarray(uid, data, params[idx][0], params[idx][1]) for idx, res in enumerate(r) if isinstance(res, type(None)) == False]
        return {uid.uri: ''}

    async def _put_uninitialized_data_array(self, uid: DataArrayIdentifier, shape: T.Tuple[int, ...], transport_array_type=AnyArrayType.ARRAY_OF_FLOAT, logical_array_type=AnyLogicalArrayType.ARRAY_OF_BOOLEAN):
        from etptypes.energistics.etp.v12.datatypes.data_array_types.put_uninitialized_data_array_type import \
            PutUninitializedDataArrayType
        from etptypes.energistics.etp.v12.protocol.data_array.put_uninitialized_data_arrays import \
            PutUninitializedDataArrays
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

    def __init__(self, authorization: T.Optional[SecretStr] = None, timeout=10.):
        self.server_url = SETTINGS.etp_url
        self.authorization = authorization
        self.timeout = timeout
        self.default_dataspace_uri = DataspaceURI.from_name(SETTINGS.dataspace)

    # enter the async context manager
    async def __aenter__(self):
        if isinstance(self.authorization, str):
            token = self.authorization
        elif isinstance(self.authorization, SecretStr):
            token = self.authorization.get_secret_value()
        ws = await websockets.connect(
            self.server_url,
            subprotocols=[ETPClient.SUB_PROTOCOL],  # type: ignore
            extra_headers={"Authorization": token} if self.authorization else {},
            max_size=MAXPAYLOADSIZE,
            ping_timeout=self.timeout,
            open_timeout=None,
        )
        self.client = ETPClient(ws, default_dataspace_uri=self.default_dataspace_uri, timeout=self.timeout)
        await self.client.connect()

        return self.client

    # exit the async context manager
    async def __aexit__(self, exc_type, exc: Exception, tb: TracebackType):
        await self.client.close()
