import json
import tempfile
import zipfile
import os
import uuid
import re

from . import etp_helper

import websockets
import resqpy
import resqpy.model
import lxml.etree as ET
import h5py
import numpy as np


# TODO: Is there a limit from pss-data-gateway?
# The websockets-library seems to a limit that corresponds to the one from
# the ETP server.
MAX_WEBSOCKET_MESSAGE_SIZE = int(1.6e7)  # From the published ETP server


with open("package.json", "r") as f:
    jschema = json.load(f)
    APPLICATION_NAME = jschema["name"]
    APPLICATION_VERSION = jschema["version"]


async def upload_resqml_objects(
    ws, msg_id, max_payload_size, dataspace, resqml_objects
):
    dataspaces = []
    data_object_types = []
    uuids = []
    xmls = []
    for values in resqml_objects.values():
        dataspaces.append(dataspace)
        data_object_types.append(values["data_object_type"])
        uuids.append(values["uuid"])
        xmls.append(values["xml"])

    # TODO: Handle possible errors
    records = await etp_helper.put_data_objects(
        ws, msg_id, max_payload_size, dataspaces, data_object_types, uuids, xmls
    )
    return records


async def upload_array_data(
    ws, msg_id, max_payload_size, dataspace, resqml_objects, h5_filename
):
    # NOTE: We assume that there is a single EpcExternalPartReference (i.e., a single hdf5-file)
    # connected to the surface arrays. This is the encouraged solution from the RESQML-standard,
    # but it is not enforced. It also looks like resqpy has support for multiple hdf5-files.
    epc_key = next(filter(lambda x: "EpcExternal" in x, list(resqml_objects)))
    epc_object_type = resqml_objects[epc_key]["data_object_type"]
    epc_uuid = resqml_objects[epc_key]["uuid"]

    paths_in_resources = []
    arrays = []
    with h5py.File(h5_filename, "r") as f:
        for key in resqml_objects:
            for pir in resqml_objects[key]["pirs"]:
                paths_in_resources.append(pir)
                arrays.append(np.array(f[pir]))

    records = await etp_helper.put_data_arrays(
        ws,
        msg_id,
        max_payload_size,
        [dataspace],
        [epc_object_type],
        [epc_uuid],
        paths_in_resources,
        arrays,
    )

    return records


async def upload_resqml_surface(
    epc_filename, h5_filename, url, dataspace, authorization
):
    headers = {"Authorization": authorization}

    async with websockets.connect(
        url,
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

        # An alternate route is to first query for the dataspace and
        # only create it if it exists. However, we save a call to the server by just trying
        # to put the dataspace and ignore the error if it already exists.
        records = await etp_helper.put_dataspaces(ws, msg_id, [dataspace])
        # The put_dataspaces returns a list of records, one for each dataspace.
        # However, as we are only adding a single dataspace there should only be a single
        # record.
        assert len(records) == 1
        # Note that we are here following the happy-path and assume that the only "error
        # message" we can get is if the dataspace exists. An alternative would be to create
        # more proper exception handling for the errors.

        resqml_objects = read_epc_file(epc_filename)

        records = await upload_resqml_objects(
            ws, msg_id, max_payload_size, dataspace, resqml_objects
        )

        await upload_array_data(
            ws, msg_id, max_payload_size, dataspace, resqml_objects, h5_filename
        )

        await etp_helper.close_session(
            ws, msg_id, "Done uploading from upload_resqml_surface"
        )

    return records


async def upload_xtgeo_surface_to_rddms(surface, title, url, dataspace, authorization):
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Note, resqpy does not seem to construct the correct xml-objects
        # before they are written to disk. As such, we have to write to
        # disk, then read in again to get the correct values. Here we do
        # that using a temporary directory to ensure that data on disk is
        # deleted after the context manager finishes.

        # TODO: Pass in CRS info, project-id (as a title?), description, units, and type
        epc_filename, h5_filename = convert_xtgeo_surface_to_resqml(
            surface, title, tmpdirname
        )
        records = await upload_resqml_surface(
            epc_filename, h5_filename, url, dataspace, authorization
        )
        # TODO: Check if this is the case when chunking is included
        assert len(records) == 1
        # NOTE: I think this should be valid as long as we are following the happy-path
        # and that an xtgeo surface will always be represented by a single
        # Grid2dRepresentation. This also only includes the url to the grid-object, and
        # not the crs nor the external reference (h5-link). However, these should be
        # recoverable from the grid-object.
        record = records[0]
        rddms_url = next(filter(lambda x: "Grid2d" in x, list(record["success"])))

    return rddms_url


def convert_xtgeo_surface_to_resqml(surf, title, directory):
    # Random file name to (hopefully) avoid collisions.
    # TODO: Find a thread-safe solution
    filename = str(uuid.uuid4().hex) + ".epc"
    # Set up model for test data
    with resqpy.model.ModelContext(
        os.path.join(directory, filename), mode="c"
    ) as model:
        # This needs to be done in order to get a uuid
        # TODO: Is there any CRS-relevant information in the xtgeo surface?
        # TODO: Use CRS-info from the metadata
        crs = resqpy.crs.Crs(model)
        crs.create_xml()

        # TODO: Test map shape before and after
        # Verify that no rotation/transposition is done
        mesh = resqpy.surface.Mesh(
            model,
            crs_uuid=model.crs_uuid,
            mesh_flavour="reg&z",
            ni=surf.nrow,
            nj=surf.ncol,
            origin=(surf.xori, surf.yori, 0.0),
            dxyz_dij=np.array([[surf.xinc, 0.0, 0.0], [0.0, surf.yinc, 0.0]]),
            # TODO: Consider setting a specific value for the masked data.
            # surf.values is a masked array, surf.values.data inserts some pre-defined
            # value for the masked elements.
            z_values=surf.values.data,
            originator="pss-data-gateway",
            title=title,
        )
        mesh.create_xml()
        # Write to disk (the hdf5-file is constructed already in the
        # model-constructor)
        mesh.write_hdf5()
        model.store_epc()

        # NOTE: It looks like resqpy only supports a single h5-file, and hence a
        # single EpcExternalPartReference in the epc-file.
        return model.epc_file, model.h5_file_name()


def read_epc_file(epc_filename):
    # Read epc-file from disk
    dat = {}
    with zipfile.ZipFile(epc_filename, "r") as zfile:
        for zinfo in filter(lambda x: x.filename.startswith("obj_"), zfile.infolist()):
            with zfile.open(zinfo.filename) as f:
                dot, id = get_object_type_and_uuid(zinfo.filename)
                xml = f.read()
                # NOTE: In general there can be multiple references to a Hdf-file (see
                # page 31 of the RESQML v2.0.1 standard), but we only support a
                # single reference so far (we would need to locate the relevant
                # EpcExternalPartReference along with the path-in-resource)
                pirs = [pir.text for pir in get_path_in_resource(xml)]
                if len(pirs) > 1:
                    raise NotImplementedError(
                        "We do not yet support storage of data in multiple Hdf-files"
                    )
                dat[zinfo.filename] = dict(
                    data_object_type=dot,
                    uuid=id,
                    xml=xml,
                    pirs=pirs,
                )

    return dat


def get_path_in_resource(xml):
    if type(xml) in [str, bytes]:
        xml = ET.fromstring(xml)

    return xml.xpath("//*[starts-with(local-name(), 'PathInHdfFile')]")


def get_object_type_and_uuid(filename):
    uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    obj_pattern = r"obj_[a-zA-Z0-9]+"
    pattern = r"(" + obj_pattern + r")_(" + uuid_pattern + r")"
    m = re.match(pattern, filename)

    return m.group(1), m.group(2)
