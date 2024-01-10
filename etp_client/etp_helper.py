import json
import datetime
import asyncio
import uuid
import io
import enum
import pathlib

import fastavro
import lxml.etree as ET


# Read and store ETP-schemas in a global dictionary
# The file etp.avpr can be downloaded from Energistics here:
# https://publications.opengroup.org/standards/energistics-standards/energistics-transfer-protocol/v234
with open(pathlib.Path("map_api/etp_client/etp.avpr"), "r") as foo:
    jschema = json.load(foo)


def parse_func(js, named_schemas=dict()):
    ps = fastavro.schema.parse_schema(js, named_schemas)
    return fastavro.schema.fullname(ps), ps


# These are now avro schemas for ETP
ETP_SCHEMAS = dict(parse_func(js) for js in jschema["types"])


def serialize_message(header_record, body_record, body_schema_key):
    # TODO: Possibly support compression?
    fo = io.BytesIO()
    fastavro.write.schemaless_writer(
        fo, ETP_SCHEMAS["Energistics.Etp.v12.Datatypes.MessageHeader"], header_record
    )
    fastavro.write.schemaless_writer(fo, ETP_SCHEMAS[body_schema_key], body_record)

    return fo.getvalue()


def get_data_object_uri(dataspace, data_object_type, _uuid):
    if not data_object_type.startswith("resqml20") or not data_object_type.startswith(
        "eml20"
    ):
        data_object_type = (
            f"resqml20.{data_object_type}"
            if "EpcExternalPart" not in data_object_type
            else f"eml20.{data_object_type}"
        )

    return f"eml:///dataspace('{dataspace}')/{data_object_type}({_uuid})"


def numpy_to_etp_data_array(array):
    return dict(
        dimensions=list(array.shape),
        # See Energistics.Etp.v12.Datatypes.AnyArray for the "item"-key, and
        # Energistics.Etp.v12.Datatypes.ArrayOfDouble for the "values"-key.
        data=dict(item=dict(values=array.ravel().tolist())),
    )


class MHFlags(enum.Enum):
    # Flags in MessageHeader, see section 23.25 in the ETP 1.2 standard
    FIN = 0x2
    COMPRESSED = 0x8
    ACK = 0x10
    HEADER_EXTENSION = 0x20


class ClientMessageId:
    def __init__(self):
        self.lock = asyncio.Lock()
        # Unique, positive, increasing, even integers for the client
        self.message_id = 2

    async def __call__(self):
        async with self.lock:
            ret_id = self.message_id
            self.message_id += 2
            return ret_id


async def handle_multipart_response(ws, get_msg_id, schema_key):
    # Note that we only handle ProtocolException errors for now
    records = []
    while True:
        response = await ws.recv()

        fo = io.BytesIO(response)
        mh_record = fastavro.read.schemaless_reader(
            fo,
            ETP_SCHEMAS["Energistics.Etp.v12.Datatypes.MessageHeader"],
        )

        records.append(
            fastavro.read.schemaless_reader(
                fo,
                ETP_SCHEMAS[
                    schema_key
                    if mh_record["messageType"] != 1000
                    else "Energistics.Etp.v12.Protocol.Core.ProtocolException"
                ],
            )
        )

        if (mh_record["messageFlags"] & MHFlags.FIN.value) != 0:
            # We have received a FIN-bit, i.e., the last reponse has been
            # read.
            break

    return records


async def request_session(
    ws,
    get_msg_id,
    # Max size of a websocket message payload. Note that this value will most
    # likely be updated from the server. The user should check the returned
    # value instead. The default value is the listed value from the vanilla
    # open-etp-server. I think this is also the _total limit_ for a websocket
    # message from the client. Note that we also need to set the max_size in
    # the websocket connection for incoming messages. These should be the
    # same.
    max_payload_size=int(1.6e7),
    application_name="pss-client",
    application_version="0.0.0",
    additional_supported_protocols=[],
    additional_supported_data_objects=[],
):
    # Request session
    mh_record = dict(
        protocol=0,  # Core protocol
        messageType=1,  # RequestSession
        correlationId=0,  # Ignored for RequestSession
        messageId=await get_msg_id(),
        messageFlags=MHFlags.FIN.value,  # FIN-bit
    )

    rs_record = dict(
        applicationName=application_name,
        applicationVersion=application_version,
        clientInstanceId=uuid.uuid4().bytes,
        requestedProtocols=[  # [SupportedProtocol]
            dict(
                protocol=3,  # Discovery
                protocolVersion=dict(
                    major=1,
                    minor=2,
                ),
                role="store",
            ),
            dict(
                protocol=4,  # Store
                protocolVersion=dict(
                    major=1,
                    minor=2,
                ),
                role="store",
            ),
            dict(
                protocol=9,  # DataArray
                protocolVersion=dict(
                    major=1,
                    minor=2,
                ),
                role="store",
            ),
            dict(
                protocol=24,  # Dataspace
                protocolVersion=dict(
                    major=1,
                    minor=2,
                ),
                role="store",
            ),
            *additional_supported_protocols,
        ],
        supportedDataObjects=[
            dict(  # SupportedDataObject
                qualifiedType="resqml20.*",
            ),
            dict(
                qualifiedType="eml20.*",
            ),
            *additional_supported_data_objects,
        ],
        currentDateTime=datetime.datetime.now(datetime.timezone.utc).timestamp(),
        earliestRetainedChangeTime=0,
        endpointCapabilities=dict(
            MaxWebSocketMessagePayloadSize=dict(
                item=max_payload_size,
            ),
        ),
    )

    await ws.send(
        serialize_message(
            mh_record,
            rs_record,
            "Energistics.Etp.v12.Protocol.Core.RequestSession",
        ),
    )

    # Note, OpenSession is a single message, but the
    # handle_multipart_response-function works just as well for single
    # messages.
    return await handle_multipart_response(
        ws, get_msg_id, "Energistics.Etp.v12.Protocol.Core.OpenSession"
    )


async def close_session(ws, get_msg_id, reason):
    # Close session
    mh_record = dict(
        protocol=0,  # Core
        messageType=5,  # CloseSession
        correlationId=0,  # Ignored
        messageId=await get_msg_id(),
        messageFlags=MHFlags.FIN.value,
    )

    cs_record = dict(reason=reason)

    await ws.send(
        serialize_message(
            mh_record, cs_record, "Energistics.Etp.v12.Protocol.Core.CloseSession"
        )
    )

    await ws.wait_closed()
    assert ws.closed


async def put_dataspaces(ws, get_msg_id, dataspaces):
    uris = list(map(lambda dataspace: f"eml:///dataspace('{dataspace}')", dataspaces))

    mh_record = dict(
        protocol=24,  # Dataspace
        messageType=3,  # PutDataspaces
        correlationId=0,  # Ignored
        messageId=await get_msg_id(),
        messageFlags=MHFlags.FIN.value,
    )
    time = datetime.datetime.now(datetime.timezone.utc).timestamp()
    pds_record = dict(
        dataspaces=dict(
            (
                uri,
                dict(
                    uri=uri,
                    path=dataspace,
                    # Here we create the dataspace for the first time, hence last write
                    # and created are the same
                    storeLastWrite=time,
                    storeCreated=time,
                ),
            )
            for uri, dataspace in zip(uris, dataspaces)
        )
    )

    await ws.send(
        serialize_message(
            mh_record,
            pds_record,
            "Energistics.Etp.v12.Protocol.Dataspace.PutDataspaces",
        )
    )

    return await handle_multipart_response(
        ws,
        get_msg_id,
        "Energistics.Etp.v12.Protocol.Dataspace.PutDataspacesResponse",
    )


async def put_data_objects(
    ws,
    get_msg_id,
    max_payload_size,
    dataspaces,
    data_object_types,
    uuids,
    xmls,
    titles=None,
):
    # TODO: Use chunks if the data objects are too large for the websocket payload

    uris = [
        get_data_object_uri(ds, dot, _uuid)
        for ds, dot, _uuid in zip(dataspaces, data_object_types, uuids)
    ]

    if not titles:
        titles = uris

    xmls = [xml if type(xml) in [str, bytes] else ET.tostring(xml) for xml in xmls]

    time = datetime.datetime.now(datetime.timezone.utc).timestamp()

    mh_record = dict(
        protocol=4,  # Store
        messageType=2,  # PutDataObjects
        correlationId=0,  # Ignored
        messageId=await get_msg_id(),
        # This is in general a multi-part message, but we will here only send
        # one.
        messageFlags=MHFlags.FIN.value,
    )

    pdo_record = dict(
        dataObjects={
            title: dict(
                data=xml,
                format="xml",
                blobId=None,
                resource=dict(
                    uri=uri,
                    name=title,
                    lastChanged=time,
                    storeCreated=time,
                    storeLastWrite=time,
                    activeStatus="Inactive",
                ),
            )
            for title, uri, xml in zip(titles, uris, xmls)
        },
    )

    await ws.send(
        serialize_message(
            mh_record,
            pdo_record,
            "Energistics.Etp.v12.Protocol.Store.PutDataObjects",
        )
    )

    return await handle_multipart_response(
        ws,
        get_msg_id,
        "Energistics.Etp.v12.Protocol.Store.PutDataObjectsResponse",
    )


async def put_data_arrays(
    ws,
    get_msg_id,
    max_payload_size,
    dataspaces,
    data_object_types,
    uuids,
    paths_in_resources,
    arrays,
):
    uris = [
        get_data_object_uri(ds, dot, _uuid)
        for ds, dot, _uuid in zip(dataspaces, data_object_types, uuids)
    ]

    mh_record = dict(
        protocol=9,  # DataArray
        messageType=4,  # PutDataArrays
        correlationId=0,  # Ignored
        messageId=await get_msg_id(),
        messageFlags=MHFlags.FIN.value,  # Multi-part=False
    )
    pda_record = dict(
        dataArrays={
            uri: dict(
                uid=dict(
                    uri=uri,
                    pathInResource=pir,
                ),
                array=numpy_to_etp_data_array(array),
            )
            for uri, pir, array in zip(uris, paths_in_resources, arrays)
        }
    )

    await ws.send(
        serialize_message(
            mh_record,
            pda_record,
            "Energistics.Etp.v12.Protocol.DataArray.PutDataArrays",
        )
    )

    return await handle_multipart_response(
        ws,
        get_msg_id,
        "Energistics.Etp.v12.Protocol.DataArray.PutDataArraysResponse",
    )
