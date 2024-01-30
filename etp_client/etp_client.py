import datetime
import json
import tempfile
import zipfile
import os
import uuid
import re

from . import etp_helper
import map_api.resqml_objects as resqml_objects

import websockets
import resqpy
import resqpy.model
import lxml.etree as ET
import h5py
import numpy as np


from xsdata.models.datatype import XmlDateTime
from xsdata.formats.dataclass.context import XmlContext
from xsdata.formats.dataclass.parsers import XmlParser
from xsdata.formats.dataclass.serializers import XmlSerializer
from xsdata.formats.dataclass.serializers.config import SerializerConfig


# TODO: Is there a limit from pss-data-gateway?
# The websockets-library seems to a limit that corresponds to the one from
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
        ws, msg_id, max_payload_size, dataspaces, data_object_types, uuids, xmls
    )
    assert all(key == "success" for record in records for key in record)

    rddms_uris = [
        etp_helper.get_data_object_uri(dataspace, dot, _uuid)
        for dataspace, dot, _uuid in zip(dataspaces, data_object_types, uuids)
    ]
    return rddms_uris


async def upload_array(
    ws, msg_id, max_payload_size, dataspace, epc, path_in_hdf_file, array
):
    # NOTE: The intention of this function is to upload a single array that
    # might potentially be oversized. It should then upload the array using
    # several put_data_subarray-calls.

    # FIXME: Handle errors.
    records = await etp_helper.put_data_arrays(
        ws,
        msg_id,
        max_payload_size,
        [dataspace],
        [get_data_object_type(epc)],
        [epc.uuid],
        [path_in_hdf_file],
        [array],
    )

    return records


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
    return rddms_urls


async def download_array_data(ws, msg_id, max_payload_size, resqml_objects):
    # TODO: Use subarrays in case the arrays are larger than max_payload_size
    # TODO: Consider asking for data array metadata prior to fetching the array
    # either in one call or from subarrays.  Otherwise we don't know how large
    # the array will be in advance.

    # NOTE: We assume that there is a single EpcExternalPartReference in the
    # resqml_objects.

    epc_uri = next(
        filter(lambda x: "EpcExternalPartReference" in x, list(resqml_objects))
    )
    path_in_resources = {
        f"{k}{pir.text}": pir.text
        for k in list(resqml_objects)
        for pir in get_path_in_resource(resqml_objects[k]["xml"])
    }

    records = await etp_helper.get_data_arrays(
        ws, msg_id, max_payload_size, epc_uri, path_in_resources
    )
    assert len(records) == 1

    return {
        key: etp_helper.etp_data_array_to_numpy(val)
        for key, val in records[0]["dataArrays"].items()
    }


async def download_resqml_objects(ws, msg_id, max_payload_size, rddms_uris):
    # FIXME: Handle chunking if there are too many data-objects.
    records = await etp_helper.get_data_objects(
        ws, msg_id, max_payload_size, rddms_uris
    )
    data_objects = records[0]["dataObjects"]


async def download_resqml_surface(rddms_uris, etp_server_url, dataspace, authorization):
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
        records = await etp_helper.get_data_objects(
            ws, msg_id, max_payload_size, rddms_uris
        )
        # NOTE: Here we assume that all three objects fit in a single record
        data_objects = records[0]["dataObjects"]
        # NOTE: This test will not work in case of too large objects as the
        # records will be chunks. If so these should be assembled before being
        # returned here.
        assert len(data_objects) == len(rddms_uris)

        resqml_objects = {}
        for uri in rddms_uris:
            xml = ET.fromstring(data_objects[uri]["data"])
            resqml_objects[uri] = dict(
                xml=xml,
                uuid=xml.attrib["uuid"],
                data_object_type=ET.QName(xml).localname,
            )

        # Download array data.
        arrays = await download_array_data(
            ws,
            msg_id,
            max_payload_size,
            resqml_objects,
        )
        # NOTE: We assume that there is a single array connected to a resqml
        # surface
        assert len(arrays) == 1
        # Fetch array
        array = arrays.popitem()[1]

        # Close session.
        await etp_helper.close_session(ws, msg_id, "Done downloading surface array")

    # Return xml's and array
    return resqml_objects, array


async def download_xtgeo_surface_from_rddms(
    rddms_urls, etp_server_url, dataspace, authorization
):
    resqml_objects, surface_array = await download_resqml_surface(
        rddms_urls, etp_server_url, dataspace, authorization
    )

    # Convert resqml-objects and the surface array into an xtgeo
    # RegularSurface.

    # Return this surface


async def upload_xtgeo_surface_to_rddms(
    surface, title, projected_epsg, etp_server_url, dataspace, authorization
):
    epc, crs, gri, surf_array = convert_xtgeo_surface_to_resqml_grid(
        surface, title, projected_epsg
    )
    return await upload_resqml_surface(
        epc, crs, gri, surf_array, etp_server_url, dataspace, authorization
    )


def get_data_object_type(obj):
    # NOTE: The name of the xsdata-generated dataclasses uses Python naming
    # convention (captical camel-case for classes), and the proper
    # data-object-type as recognized by RESQML and ETP is kept in the
    # internal Meta-class of the objects (if the name has changed).
    # This means that the data-object-type of a RESQML-object is _either_
    # just the class-name (if the name remains unchanged from the
    # xsdata-generation as in the EpcExternalPartReference-case), or it is
    # kept in <object>.Meta.name (as in the both the Grid2dRepresentation
    # and LocalDepth3dCrs cases).
    # A robust way of fetch the right data-object-type irrespective of where the name is kept is to use
    #
    #   data_object_type = getattr(<object>.Meta, "name", "") or <object>.__class__.__name__
    #
    # This fetches the name from <object>.Meta.name if that exists,
    # otherwise we use the the class-name (which will be the same as in the
    # RESQML-standard).
    return getattr(obj.Meta, "name", "") or obj.__class__.__name__


def convert_xtgeo_surface_to_resqml_grid(surf, title, projected_epsg):
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

    epc = resqml_objects.EpcExternalPartReference(
        citation=resqml_objects.Citation(
            title="Hdf Proxy",
            **common_citation_fields,
        ),
        schema_version=schema_version,
        uuid=str(uuid.uuid4()),
        mime_type="application/x-hdf5",
    )

    assert np.abs(surf.get_rotation()) < 1e-7

    crs = resqml_objects.LocalDepth3DCrs(
        citation=resqml_objects.Citation(
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
        areal_rotation=resqml_objects.PlaneAngleMeasure(
            # Here rotation should be zero!
            value=surf.get_rotation(),
            uom=resqml_objects.PlaneAngleUom.DEGA,
        ),
        # NOTE: Verify that this is the projected axis order
        projected_axis_order=resqml_objects.AxisOrder2D.EASTING_NORTHING,
        projected_uom=resqml_objects.LengthUom.M,
        vertical_uom=resqml_objects.LengthUom.M,
        zincreasing_downward=True,
        vertical_crs=resqml_objects.VerticalUnknownCrs(
            unknown="unknown",
        ),
        projected_crs=resqml_objects.ProjectedCrsEpsgCode(
            epsg_code=projected_epsg,
        ),
    )

    x0 = surf.xori
    y0 = surf.yori
    dx = surf.xinc
    dy = surf.xinc
    # NOTE: xtgeo uses nrow for axis 1 in the array, and ncol for axis 0.  This
    # means that surf.nrow is the fastest changing axis, and surf.ncol the
    # slowest changing axis, and we have surf.values.shape == (surf.ncol,
    # surf.nrow). The author of this note finds that confusing, but such is
    # life.
    nx = surf.ncol
    ny = surf.nrow

    gri = resqml_objects.Grid2DRepresentation(
        uuid=(grid_uuid := str(uuid.uuid4())),
        schema_version=schema_version,
        surface_role=resqml_objects.SurfaceRole.MAP,
        citation=resqml_objects.Citation(
            title=title,
            **common_citation_fields,
        ),
        grid2d_patch=resqml_objects.Grid2DPatch(
            # TODO: Perhaps we can use this for tiling?
            patch_index=0,
            # NumPy-arrays are C-ordered, meaning that the last index is
            # the index that changes most rapidly. However, xtgeo uses nrow for
            # axis 1 in the array, and ncol for axis 0. This means that
            # surf.nrow is the fastest changing axis, and surf.ncol the slowest
            # changing axis (as surf.values.shape == (surf.ncol, surf.nrow))
            fastest_axis_count=ny,
            slowest_axis_count=nx,
            geometry=resqml_objects.PointGeometry(
                local_crs=resqml_objects.DataObjectReference(
                    # NOTE: See Energistics Identifier Specification 4.0
                    # (it is downloaded alongside the RESQML v2.0.1
                    # standard) section 4.1 for an explanation on the
                    # format of content_type.
                    content_type=f"application/x-resqml+xml;version={schema_version};type={get_data_object_type(crs)}",
                    title=crs.citation.title,
                    uuid=crs.uuid,
                ),
                points=resqml_objects.Point3DZvalueArray(
                    supporting_geometry=resqml_objects.Point3DLatticeArray(
                        origin=resqml_objects.Point3D(
                            coordinate1=x0,
                            coordinate2=y0,
                            coordinate3=0.0,
                        ),
                        offset=[
                            # Offset for the x-direction, i.e., the slowest axis
                            resqml_objects.Point3DOffset(
                                offset=resqml_objects.Point3D(
                                    coordinate1=1.0,
                                    coordinate2=0.0,
                                    coordinate3=0.0,
                                ),
                                spacing=resqml_objects.DoubleConstantArray(
                                    value=dx,
                                    count=nx - 1,
                                ),
                            ),
                            # Offset for the y-direction, i.e., the fastest axis
                            resqml_objects.Point3DOffset(
                                offset=resqml_objects.Point3D(
                                    coordinate1=0.0,
                                    coordinate2=1.0,
                                    coordinate3=0.0,
                                ),
                                spacing=resqml_objects.DoubleConstantArray(
                                    value=dy,
                                    count=ny - 1,
                                ),
                            ),
                        ],
                    ),
                    zvalues=resqml_objects.DoubleHdf5Array(
                        values=resqml_objects.Hdf5Dataset(
                            path_in_hdf_file=f"/RESQML/{grid_uuid}/zvalues",
                            hdf_proxy=resqml_objects.DataObjectReference(
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
    surf_array = surf.values.filled(np.nan)

    return epc, crs, gri, surf_array


# async def upload_array_data(
#     ws, msg_id, max_payload_size, dataspace, resqml_objects, h5_filename
# ):
#     # NOTE: We assume that there is a single EpcExternalPartReference (i.e., a single hdf5-file)
#     # connected to the surface arrays. This is the encouraged solution from the RESQML-standard,
#     # but it is not enforced. It also looks like resqpy has support for multiple hdf5-files.
#     epc_key = next(filter(lambda x: "EpcExternal" in x, list(resqml_objects)))
#     epc_object_type = resqml_objects[epc_key]["data_object_type"]
#     epc_uuid = resqml_objects[epc_key]["uuid"]
#
#     paths_in_resources = []
#     arrays = []
#     with h5py.File(h5_filename, "r") as f:
#         for key in resqml_objects:
#             for pir in resqml_objects[key]["pirs"]:
#                 paths_in_resources.append(pir)
#                 arrays.append(np.array(f[pir]))
#
#     records = await etp_helper.put_data_arrays(
#         ws,
#         msg_id,
#         max_payload_size,
#         [dataspace],
#         [epc_object_type],
#         [epc_uuid],
#         paths_in_resources,
#         arrays,
#     )
#
#     return records


# async def upload_xtgeo_surface_to_rddms(
#     surface, title, etp_server_url, dataspace, authorization
# ):
#     with tempfile.TemporaryDirectory() as tmpdirname:
#         # Note, resqpy does not seem to construct the correct xml-objects
#         # before they are written to disk. The problem has to do with namespace
#         # prefixing in XML attributes. I think the problem is on the
#         # open-etp-server and their use of rapidxml_ns which I think does not
#         # support the usage of the full namespace in attributes. As such, we
#         # have to write to disk, then read in again to get the correct values.
#         # Here we do that using a temporary directory to ensure that data on
#         # disk is deleted after the context manager finishes.
#
#         # TODO: Pass in CRS info, project-id (as a title?), description, units,
#         # and type
#         epc_filename, h5_filename = convert_xtgeo_surface_to_resqml(
#             surface, title, tmpdirname
#         )
#         records = await upload_resqml_surface(
#             epc_filename, h5_filename, etp_server_url, dataspace, authorization
#         )
#         # TODO: Check if this is the case when chunking is included
#         assert len(records) == 1
#         # NOTE: I think this should be valid as long as an xtgeo surface is
#         # represented by a single Grid2dRepresentation.
#         rddms_urls = list(records[0]["success"])
#
#     return rddms_urls


# def convert_xtgeo_surface_to_resqml(surf, title, directory):
#     # Random file name to (hopefully) avoid collisions.
#     # TODO: Find a thread-safe solution
#     filename = str(uuid.uuid4().hex) + ".epc"
#     # Set up model for test data
#     with resqpy.model.ModelContext(
#         os.path.join(directory, filename), mode="c"
#     ) as model:
#         # This needs to be done in order to get a uuid
#         # TODO: Is there any CRS-relevant information in the xtgeo surface?
#         # TODO: Use CRS-info from the metadata
#         crs = resqpy.crs.Crs(model)
#         crs.create_xml()
#
#         # TODO: Test map shape before and after
#         # Verify that no rotation/transposition is done
#         mesh = resqpy.surface.Mesh(
#             model,
#             crs_uuid=model.crs_uuid,
#             mesh_flavour="reg&z",
#             ni=surf.nrow,
#             nj=surf.ncol,
#             origin=(surf.xori, surf.yori, 0.0),
#             dxyz_dij=np.array([[surf.xinc, 0.0, 0.0], [0.0, surf.yinc, 0.0]]),
#             # TODO: Consider setting a specific value for the masked data.
#             # surf.values is a masked array, surf.values.data inserts some pre-defined
#             # value for the masked elements.
#             z_values=surf.values.data,
#             originator="pss-data-gateway",
#             title=title,
#         )
#         mesh.create_xml()
#         # Write to disk (the hdf5-file is constructed already in the
#         # model-constructor)
#         mesh.write_hdf5()
#         model.store_epc()
#
#         # NOTE: It looks like resqpy only supports a single h5-file, and hence a
#         # single EpcExternalPartReference in the epc-file.
#         return model.epc_file, model.h5_file_name()
#
#
# def read_epc_file(epc_filename):
#     # Read epc-file from disk
#     dat = {}
#     with zipfile.ZipFile(epc_filename, "r") as zfile:
#         for zinfo in filter(lambda x: x.filename.startswith("obj_"), zfile.infolist()):
#             with zfile.open(zinfo.filename) as f:
#                 dot, _uuid = get_object_type_and_uuid(zinfo.filename)
#                 xml = f.read()
#                 # NOTE: In general there can be multiple references to a Hdf-file (see
#                 # page 31 of the RESQML v2.0.1 standard), but we only support a
#                 # single reference so far (we would need to locate the relevant
#                 # EpcExternalPartReference along with the path-in-resource)
#                 pirs = [pir.text for pir in get_path_in_resource(xml)]
#                 if len(pirs) > 1:
#                     raise NotImplementedError(
#                         "We do not yet support storage of data in multiple Hdf-files"
#                     )
#                 dat[zinfo.filename] = dict(
#                     data_object_type=dot,
#                     uuid=_uuid,
#                     xml=xml,
#                     pirs=pirs,
#                 )
#
#     return dat


# def get_path_in_resource(xml):
#     if type(xml) in [str, bytes]:
#         xml = ET.fromstring(xml)
#
#     return xml.xpath("//*[starts-with(local-name(), 'PathInHdfFile')]")
#
#
# def get_object_type_and_uuid(filename):
#     uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
#     obj_pattern = r"obj_[a-zA-Z0-9]+"
#     pattern = r"(" + obj_pattern + r")_(" + uuid_pattern + r")"
#     m = re.match(pattern, filename)
#
#     return m.group(1), m.group(2)
