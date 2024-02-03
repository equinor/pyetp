import datetime
import math
import uuid


from async_lru import alru_cache

from . import etp_helper
import map_api.resqml_objects as ro

import websockets
import lxml.etree as ET

import numpy as np

import typing as T

from xsdata.models.datatype import XmlDateTime
from xsdata.formats.dataclass.context import XmlContext
from xsdata.formats.dataclass.parsers import XmlParser
from xsdata.formats.dataclass.serializers import XmlSerializer
from xsdata.formats.dataclass.serializers.config import SerializerConfig

if T.TYPE_CHECKING:
    import xtgeo


# TODO: Is there a limit from pss-data-gateway?
# The websockets-library seems to have a limit that corresponds to the one from
# the ETP server.
MAX_WEBSOCKET_MESSAGE_SIZE = int(1.6e7)  # From the published ETP server


# TODO: Check pathing when the api is called
# with open("package.json", "r") as f:
#     jschema = json.load(f)
APPLICATION_NAME = "geomint"
APPLICATION_VERSION = "0.0.1"


async def upload_resqml_objects(
    ws, msg_id, max_payload_size, dataspace, resqml_objects
):
    serializer = XmlSerializer(config=SerializerConfig())

    dataspaces = []
    data_object_types = []
    uuids = []
    xmls = []
    for obj in resqml_objects:
        dataspaces.append(dataspace)
        data_object_types.append(get_data_object_type(obj))
        uuids.append(obj.uuid)
        xmls.append(str.encode(serializer.render(obj)))

    # FIXME: Handle possible errors from the ETP-server.
    records = await etp_helper.put_data_objects(
        ws, msg_id, dataspaces, data_object_types, uuids, xmls
    )
    assert all(key == "success" for record in records for key in record)

    rddms_uris = [
        etp_helper.get_data_object_uri(dataspace, dot, _uuid)
        for dataspace, dot, _uuid in zip(dataspaces, data_object_types, uuids)
    ]
    return rddms_uris


async def delete_resqml_objects(etp_server_url, rddms_uris, authorization):
    headers = {"Authorization": authorization}

    async with websockets.connect(
        etp_server_url,
        extra_headers=headers,
        subprotocols=["etp12.energistics.org"],
        max_size=MAX_WEBSOCKET_MESSAGE_SIZE,
    ) as ws:
        msg_id = etp_helper.ClientMessageId()
        records = await etp_helper.request_session(
            ws,
            msg_id,
            max_payload_size=MAX_WEBSOCKET_MESSAGE_SIZE,
            application_name=APPLICATION_NAME,
            application_version=APPLICATION_VERSION,
        )
        # TODO: Use the max_payload_size to ensure that data is uploaded in
        # chunks when needed.
        max_payload_size = records[0]["endpointCapabilities"][
            "MaxWebSocketMessagePayloadSize"
        ]["item"]

        records = await etp_helper.delete_data_objects(ws, msg_id, rddms_uris)
        assert len(records[0]["deletedUris"]) == len(rddms_uris)

        # Close session.
        await etp_helper.close_session(ws, msg_id, "Done deleting resqml objects")

    return records[0]["deletedUris"]


async def upload_array(
    ws, msg_id, max_payload_size, dataspace, epc, path_in_resource, array
):
    # This function uploads a single array that might potentially be oversized.

    # NOTE: For now the array should be in float32 as the ETP server does not
    # yet seem to support float64. The values will at least be converted to
    # float32, so take care when values are returned in case float64 is
    # uploaded.
    arr_size = array.size * array.dtype.itemsize

    # Check if we can upload the full array in one go.
    if arr_size < max_payload_size:
        # Upload the full array.

        # FIXME: Handle errors.
        records = await etp_helper.put_data_arrays(
            ws,
            msg_id,
            [dataspace],
            [get_data_object_type(epc)],
            [epc.uuid],
            [path_in_resource],
            [array],
        )
        # Verify that the upload was a success.
        assert len(records) == 1
        assert list(records[0])[0] == "success"

        return

    # Upload the array in several subarrays due to it being oversized.
    records = await etp_helper.put_uninitialized_data_arrays(
        ws,
        msg_id,
        dataspace,
        get_data_object_type(epc),
        epc.uuid,
        [path_in_resource],
        [array.shape],
    )
    # Verify that the upload was a success
    assert len(records) == 1
    assert len(list(records[0])) == 1
    assert list(records[0])[0] == "success"

    # Split matrix on axis 0, i.e., preserve the remaining shapes for each
    # block.
    blocks = np.array_split(
        array,
        int(np.ceil(arr_size / max_payload_size)),
    )

    # Upload each block separately.
    starts = [0] * len(array.shape)
    for block in blocks:
        records = await etp_helper.put_data_subarrays(
            ws,
            msg_id,
            dataspace,
            get_data_object_type(epc),
            epc.uuid,
            [path_in_resource],
            [block],
            [starts],
        )
        # Verify that the upload was a success
        assert len(records) == 1
        assert list(records[0])[0] == "success"

        # Increment row index to the next block.
        starts[0] += block.shape[0]


async def upload_resqml_surface(
    epc, crs, gri, surf_array, etp_server_url, dataspace, authorization
):
    headers = {"Authorization": authorization}

    async with websockets.connect(
        etp_server_url,
        extra_headers=headers,
        subprotocols=["etp12.energistics.org"],
        max_size=MAX_WEBSOCKET_MESSAGE_SIZE,
    ) as ws:
        msg_id = etp_helper.ClientMessageId()
        records = await etp_helper.request_session(
            ws,
            msg_id,
            max_payload_size=MAX_WEBSOCKET_MESSAGE_SIZE,
            application_name=APPLICATION_NAME,
            application_version=APPLICATION_VERSION,
        )

        # FIXME: Handle potential errors in records.
        # TODO: Use the max_payload_size to ensure that data is uploaded in
        # chunks or subarrays when needed.
        # TODO: Check if we need to include the message header size and the
        # metadata in the message body. That is, the max payload of uploaded data would then be:
        #   max_size = max_payload_size - sizeof(metadata)
        max_payload_size = records[0]["endpointCapabilities"][
            "MaxWebSocketMessagePayloadSize"
        ]["item"]

        # An alternate route is to first query for the dataspace and only
        # create it if it exists. However, we save a call to the server by just
        # trying to put the dataspace and ignore the error if it already
        # exists.
        records = await etp_helper.put_dataspaces(ws, msg_id, [dataspace])
        # The put_dataspaces returns a list of records, one for each dataspace.
        # However, as we are only adding a single dataspace there should only
        # be a single record.
        assert len(records) == 1
        # FIXME: We are here following the happy-path and assume that the only
        # "error message" we can get is if the dataspace exists. An alternative
        # would be to create more proper exception handling for the errors.

        rddms_urls = await upload_resqml_objects(
            ws, msg_id, max_payload_size, dataspace, [epc, crs, gri]
        )

        records = await upload_array(
            ws,
            msg_id,
            max_payload_size,
            dataspace,
            epc,
            gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file,
            surf_array,
        )

        await etp_helper.close_session(
            ws, msg_id, "Done uploading from upload_resqml_surface"
        )

    # Return the uri's of the three uploaded objects.
    return tuple(rddms_urls)


async def download_array(
    ws, msg_id, max_payload_size, dataspace, epc, path_in_resource
):
    # This function downloads a single array from the ETP-server connected to
    # the EpcExternalPartReference-object passed in with "path_in_resource" as
    # the key in hdf5-file. The point of this function is to use subarrays if
    # the request array is too large for a single websocket message.

    records = await etp_helper.get_data_array_metadata(
        ws,
        msg_id,
        dataspace,
        get_data_object_type(epc),
        epc.uuid,
        [path_in_resource],
    )
    assert len(records) == 1
    assert len(list(records[0]["arrayMetadata"])) == 1

    metadata = records[0]["arrayMetadata"].popitem()[1]
    arr_shape = metadata["dimensions"]
    # NOTE: We can in theory also retrieve the data type of the array, however,
    # the ETP-server seems for now to only support float32. Therefore, all data
    # will be interpreted as float32. That is, each element in the array will
    # be 4 bytes long.
    arr_size = math.prod(arr_shape) * 4

    # Check if we can retrieve the full array in one go.
    if arr_size < max_payload_size:
        # The array is small enough for a single request, so fetch everything
        # at once.
        records = await etp_helper.get_data_arrays(
            ws,
            msg_id,
            dataspace,
            get_data_object_type(epc),
            epc.uuid,
            [path_in_resource],
        )
        # Verify that we only got a single record with a single data array
        # entry.
        assert len(records) == 1
        assert len(list(records[0]["dataArrays"])) == 1

        return etp_helper.etp_data_array_to_numpy(records[0]["dataArrays"].popitem()[1])

    # We need to fetch the full array using several subarray-calls.
    num_blocks = int(np.ceil(arr_size / max_payload_size))
    # Split on axis 0, assuming that an element in axis 0 is not too large.
    # This code is based on the NumPy array_split-function here:
    # https://github.com/numpy/numpy/blob/d35cd07ea997f033b2d89d349734c61f5de54b0d/numpy/lib/shape_base.py#L731-L784
    # NOTE: If the array that is being fetched is multi-dimensional and we need
    # to split on several axes, then this code needs to generalized.
    num_each_section, extras = divmod(arr_shape[0], num_blocks)
    section_sizes = (
        [0]
        + extras * [num_each_section + 1]
        + (num_blocks - extras) * [num_each_section]
    )
    div_points = np.array(section_sizes, dtype=int).cumsum()

    # Create lists of starting indices and number of elements in each subarray.
    starts_list = []
    counts_list = []
    for i in range(num_blocks):
        starts_list.append([div_points[i], 0])
        counts_list.append(
            [div_points[i + 1] - div_points[i], arr_shape[1]],
        )

    blocks = []
    for starts, counts in zip(starts_list, counts_list):
        records = await etp_helper.get_data_subarrays(
            ws,
            msg_id,
            dataspace,
            get_data_object_type(epc),
            epc.uuid,
            [
                path_in_resource,
            ],
            [starts],
            [counts],
        )
        # Verify that we get a single record with a single subarray.
        assert len(records) == 1
        assert len(list(records[0]["dataSubarrays"])) == 1
        blocks.append(
            etp_helper.etp_data_array_to_numpy(records[0]["dataSubarrays"].popitem()[1])
        )

    # Check that we have received the requested number of blocks.
    assert len(blocks) == num_blocks

    # Assemble and return the full array.
    return np.concatenate(blocks, axis=0)


NT = T.TypeVar('NT')


def find_next_instance(data: T.List[T.Any], cls: T.Type[NT]) -> NT:
    return next(
        filter(
            lambda x: isinstance(x, cls),
            data
        )
    )


@alru_cache(maxsize=32)
async def download_resqml_surface(rddms_uris: T.Tuple[str, str, str], etp_server_url: str, dataspace: str, authorization: str):
    # NOTE: This assumes that a "resqml-surface" consists of a
    # Grid2dRepresentation, an EpcExternalPartReference, and a LocalDepth3dCrs
    assert len(rddms_uris) == 3

    headers = {"Authorization": authorization}

    async with websockets.connect(
        etp_server_url,
        extra_headers=headers,
        subprotocols=["etp12.energistics.org"],
        max_size=MAX_WEBSOCKET_MESSAGE_SIZE,
    ) as ws:
        msg_id = etp_helper.ClientMessageId()
        records = await etp_helper.request_session(
            ws,
            msg_id,
            max_payload_size=MAX_WEBSOCKET_MESSAGE_SIZE,
            application_name=APPLICATION_NAME,
            application_version=APPLICATION_VERSION,
        )

        # TODO: Use the max_payload_size to ensure that data is downloaded in
        # chunks when needed.
        max_payload_size = records[0]["endpointCapabilities"][
            "MaxWebSocketMessagePayloadSize"
        ]["item"]

        # Download the grid objects.
        records = await etp_helper.get_data_objects(ws, msg_id, rddms_uris)
        # NOTE: Here we assume that all three objects fit in a single record
        data_objects = records[0]["dataObjects"]
        # NOTE: This test will not work in case of too large objects as the
        # records will be chunks. If so these should be assembled before being
        # returned here.
        assert len(data_objects) == len(rddms_uris)

        returned_resqml = read_returned_resqml_objects(data_objects)

        # NOTE: In case there are multiple objects of a single type (not in
        # this call, but in other functions) we can replace "next" by "list"
        # (or just keep the generator from "filter" as-is) to sort out all the
        # relevant objects.
        epc = find_next_instance(returned_resqml, ro.EpcExternalPartReference)
        crs = find_next_instance(returned_resqml, ro.LocalDepth3dCrs)
        gri = find_next_instance(returned_resqml, ro.Grid2dRepresentation)

        # some checks
        assert isinstance(gri.grid2d_patch.geometry.points, ro.Point3dZValueArray), "Points must be Point3dZValueArray"
        assert isinstance(gri.grid2d_patch.geometry.points.zvalues, ro.DoubleHdf5Array), "Values must be DoubleHdf5Array"
        assert isinstance(gri.grid2d_patch.geometry.points.zvalues.values, ro.Hdf5Dataset), "Values must be Hdf5Dataset"

        gri_array = await download_array(
            ws,
            msg_id,
            max_payload_size,
            dataspace,
            epc,
            gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file,
        )

        # Close session.
        await etp_helper.close_session(ws, msg_id, "Done downloading surface array")

    # Return xml's and array
    return epc, crs, gri, gri_array


async def upload_xtgeo_surface_to_rddms(
    surf: 'xtgeo.RegularSurface', title, projected_epsg, etp_server_url, dataspace, authorization
):
    epc, crs, gri, surf_array = convert_xtgeo_surface_to_resqml_grid(
        surf, title, projected_epsg
    )
    return await upload_resqml_surface(
        epc, crs, gri, surf_array, etp_server_url, dataspace, authorization
    )


def get_data_object_type(obj):
    # This assumes that obj is instantiated.
    # Using auto-generated objects from xsdata with the object names preserved
    # from the schema-definition files we know that the class name is the same
    # as the RESQML-object name.
    return obj.__class__.__name__


def convert_xtgeo_surface_to_resqml_grid(surf: 'xtgeo.RegularSurface', title: str, projected_epsg):
    # Build the RESQML-objects "manually" from the generated dataclasses.
    # Their content is described also in the RESQML v2.0.1 standard that is
    # available for download here:
    # https://publications.opengroup.org/standards/energistics-standards/v231a
    common_citation_fields = dict(
        creation=XmlDateTime.from_string(
            datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
        ),
        originator=APPLICATION_NAME,
        format=f"equinor:{APPLICATION_NAME}:v{APPLICATION_VERSION}",
    )
    schema_version = "2.0"

    epc = ro.EpcExternalPartReference(
        citation=ro.Citation(
            title="Hdf Proxy",
            **common_citation_fields,
        ),
        schema_version=schema_version,
        uuid=str(uuid.uuid4()),
        mime_type="application/x-hdf5",
    )

    assert np.abs(surf.get_rotation()) < 1e-7

    crs = ro.LocalDepth3dCrs(
        citation=ro.Citation(
            title=f"CRS for {title}",
            **common_citation_fields,
        ),
        schema_version=schema_version,
        uuid=str(uuid.uuid4()),
        # NOTE: I assume that we let the CRS have no offset, and add any offset
        # in the grid instead.
        xoffset=0.0,
        yoffset=0.0,
        zoffset=0.0,
        areal_rotation=ro.PlaneAngleMeasure(
            # Here rotation should be zero!
            value=surf.get_rotation(),
            uom=ro.PlaneAngleUom.DEGA,
        ),
        # NOTE: Verify that this is the projected axis order
        projected_axis_order=ro.AxisOrder2d.EASTING_NORTHING,
        projected_uom=ro.LengthUom.M,
        vertical_uom=ro.LengthUom.M,
        zincreasing_downward=True,
        vertical_crs=ro.VerticalUnknownCrs(
            unknown="unknown",
        ),
        projected_crs=ro.ProjectedCrsEpsgCode(
            epsg_code=projected_epsg,
        ),
    )

    x0 = surf.xori
    y0 = surf.yori
    dx = surf.xinc
    dy = surf.yinc
    # NOTE: xtgeo uses nrow for axis 1 in the array, and ncol for axis 0.  This
    # means that surf.nrow is the fastest changing axis, and surf.ncol the
    # slowest changing axis, and we have surf.values.shape == (surf.ncol,
    # surf.nrow). The author of this note finds that confusing, but such is
    # life.
    nx = surf.ncol
    ny = surf.nrow

    gri = ro.Grid2dRepresentation(
        uuid=(grid_uuid := str(uuid.uuid4())),
        schema_version=schema_version,
        surface_role=ro.SurfaceRole.MAP,
        citation=ro.Citation(
            title=title,
            **common_citation_fields,
        ),
        grid2d_patch=ro.Grid2dPatch(
            # TODO: Perhaps we can use this for tiling?
            patch_index=0,
            # NumPy-arrays are C-ordered, meaning that the last index is
            # the index that changes most rapidly. However, xtgeo uses nrow for
            # axis 1 in the array, and ncol for axis 0. This means that
            # surf.nrow is the fastest changing axis, and surf.ncol the slowest
            # changing axis (as surf.values.shape == (surf.ncol, surf.nrow))
            fastest_axis_count=ny,
            slowest_axis_count=nx,
            geometry=ro.PointGeometry(
                local_crs=ro.DataObjectReference(
                    # NOTE: See Energistics Identifier Specification 4.0
                    # (it is downloaded alongside the RESQML v2.0.1
                    # standard) section 4.1 for an explanation on the
                    # format of content_type.
                    content_type=f"application/x-resqml+xml;version={schema_version};type={get_data_object_type(crs)}",
                    title=crs.citation.title,
                    uuid=crs.uuid,
                ),
                points=ro.Point3dZValueArray(
                    supporting_geometry=ro.Point3dLatticeArray(
                        origin=ro.Point3d(
                            coordinate1=x0,
                            coordinate2=y0,
                            coordinate3=0.0,
                        ),
                        # NOTE: The ordering in the offset-list should be
                        # preserved when the data is passed back and forth.
                        # However, _we_ need to ensure a consistent ordering
                        # for ourselves. In this setup I have set the slowest
                        # axis to come first, i.e., the x-axis or axis 0 in
                        # NumPy. The reason is so that it corresponds with the
                        # origin above where "coordinate1" is set to be the
                        # x0-coordinate, and "coordinate2" the y0-coordinate.
                        # However, we can change this as we see fit.
                        offset=[
                            # Offset for x-direction, i.e., the slowest axis
                            ro.Point3dOffset(
                                offset=ro.Point3d(
                                    coordinate1=1.0,
                                    coordinate2=0.0,
                                    coordinate3=0.0,
                                ),
                                spacing=ro.DoubleConstantArray(
                                    value=dx,
                                    count=nx - 1,
                                ),
                            ),
                            # Offset for y-direction, i.e., the fastest axis
                            ro.Point3dOffset(
                                offset=ro.Point3d(
                                    coordinate1=0.0,
                                    coordinate2=1.0,
                                    coordinate3=0.0,
                                ),
                                spacing=ro.DoubleConstantArray(
                                    value=dy,
                                    count=ny - 1,
                                ),
                            ),
                        ],
                    ),
                    zvalues=ro.DoubleHdf5Array(
                        values=ro.Hdf5Dataset(
                            path_in_hdf_file=f"/RESQML/{grid_uuid}/zvalues",
                            hdf_proxy=ro.DataObjectReference(
                                content_type=f"application/x-eml+xml;version={schema_version};type={get_data_object_type(epc)}",
                                title=epc.citation.title,
                                uuid=epc.uuid,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    # Use nan as the mask-value in the surface array
    surf_array: np.ndarray = surf.values.filled(np.nan)
    return epc, crs, gri, surf_array


def read_returned_resqml_objects(data_objects):
    # This function creates a list of resqml-objects from the returned xml from
    # the ETP-server. It dynamically finds the relevant resqml dataclass using
    # the object name found in the xml. Its intention is to be used after
    # calling the get_data_objects-protocol.

    # Set up an XML-parser from xsdata.
    parser = XmlParser(context=XmlContext())

    return [
        parser.from_bytes(
            data_object["data"],
            getattr(ro, ET.QName(ET.fromstring(data_object["data"]).tag).localname),
        )
        for data_object in data_objects.values()
    ]
