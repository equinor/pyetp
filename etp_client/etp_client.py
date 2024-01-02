import json
import pprint

from . import etp_helper

import websockets


ETP_SERVER_URL = "wss://interop-rddms.azure-api.net"
PSS_DATASPACE = "demo/pss-data-gateway"
MAX_WEBSOCKET_MESSAGE_SIZE = int(1.6e7)  # From the published ETP server

# TODO: Check pathing when the api is called
with open("package.json", "r") as f:
    jschema = json.load(f)
    APPLICATION_NAME = jschema["name"]
    APPLICATION_VERSION = jschema["version"]


async def create_dataspace(ws, msg_id, dataspace):
    # An alternate route in this function is to first query for the dataspace and
    # only create it if it exists. However, we save a call to the server by just trying
    # to put the dataspace and handle the error if it already exists.

    # The put_dataspaces returns a list of records, one for each dataspace.
    # However, as we are only adding a single dataspace there should only be a single
    # record.
    record = (await etp_helper.put_dataspaces(ws, msg_id, [dataspace]))[0]

    # Get the dataspace URI if the dataspace was created successfully (the leftmost
    # part of the 'or') or from the 'errors'-attribute if the dataspace already exists.
    # Note that we are here following the happy-path and assume that the only "error
    # message" we can get is if the dataspace exists. An alternative would be to create
    # more proper exception handling for the errors.
    ds_uri = list(record.get("success") or record["errors"])[0]

    return ds_uri


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


async def upload_resqml_surface(resqml_objects, surface_values, authorization):
    # NOTE: This assumes that there is a single surface with values, and the
    # appropriate amount of RESQML-objects. An alternative is to create an
    # uploader for .epc-files. This would then need an hdf5-file for the
    # array-data.
    headers = {"Authorization": authorization}

    async with websockets.connect(
        ETP_SERVER_URL,
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

        ds_uri = await create_dataspace(ws, msg_id, PSS_DATASPACE)

        await upload_resqml_objects(
            ws, msg_id, max_payload_size, PSS_DATASPACE, resqml_objects
        )

        await etp_helper.close_session(
            ws, msg_id, "Done uploading from upload_resqml_surface"
        )

    return "RDDMSURL, but which one?"
