import json
import datetime
import asyncio
import uuid
import io
import enum
import pathlib

import fastavro
import lxml.etree as ET
import numpy as np


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
    # FIXME: Fetch prefix from the namespaces in the resqml-objects.
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


def numpy_to_etp_data_subarray(subarray):
    return dict(item=dict(values=subarray.ravel().tolist()))


def etp_data_array_to_numpy(data_array):
    return np.asarray(data_array["data"]["item"]["values"]).reshape(
        data_array["dimensions"]
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


async def handle_multipart_response(ws, schema_key):
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
        ws, "Energistics.Etp.v12.Protocol.Core.OpenSession"
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
        "Energistics.Etp.v12.Protocol.Dataspace.PutDataspacesResponse",
    )


async def put_data_objects(
    ws,
    get_msg_id,
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
        "Energistics.Etp.v12.Protocol.Store.PutDataObjectsResponse",
    )


async def get_data_objects(ws, get_msg_id, uris):
    # Note, the uris contain the dataspace name, the data object type, and the
    # uuid. An alternative to passing the complete uris would be to pass in
    # each part separately. I am unsure what is easiest down the line.
    # Note also that we use the uri as the name of the data object to ensure
    # uniqueness.

    # Get data objects
    # If the number of uris is larger than the websocket size, we must send
    # multiple GetDataObjects-requests. The returned data objects can also be
    # too large, and would then require chunking.
    mh_record = dict(
        protocol=4,  # Store
        messageType=1,  # GetDataObjects
        correlationId=0,  # Ignored
        messageId=await get_msg_id(),
        messageFlags=MHFlags.FIN.value,  # Multi-part=False
    )
    # Assuming that all uris fit in a single record, for now.
    gdo_record = dict(
        uris=dict((uri, uri) for uri in uris),
        format="xml",
    )

    await ws.send(
        serialize_message(
            mh_record,
            gdo_record,
            "Energistics.Etp.v12.Protocol.Store.GetDataObjects",
        )
    )

    return await handle_multipart_response(
        ws,
        "Energistics.Etp.v12.Protocol.Store.GetDataObjectsResponse",
    )


async def delete_data_objects(ws, get_msg_id, uris, pruneContainedObjects=False):
    mh_record = dict(
        protocol=4,  # Store
        messageType=3,  # DeleteDataObjects
        correlationId=0,  # Ignored
        messageId=await get_msg_id(),
        messageFlags=MHFlags.FIN.value,  # Multi-part=False
    )

    ddo_record = dict(
        uris=dict((uri, uri) for uri in uris),
        # -- pruneContainedObjects:
        # let ETP server delete contained or related objects that are
        # not contained by any other data objects
        # Consider to set this to always be True when deleting
        # a map
        # FIXME: doesn't seem to be working correctly yet; consider filing
        # an issue for ETP server
        pruneContainedObjects=pruneContainedObjects,
    )

    await ws.send(
        serialize_message(
            mh_record,
            ddo_record,
            "Energistics.Etp.v12.Protocol.Store.DeleteDataObjects",
        )
    )
    return await handle_multipart_response(
        ws, "Energistics.Etp.v12.Protocol.Store.DeleteDataObjectsResponse"
    )


async def put_uninitialized_data_arrays(
    ws,
    get_msg_id,
    dataspace,
    epc_object_type,
    epc_uuid,
    paths_in_resources,
    arr_shapes,
    transport_array_type="",
    logical_array_type="",
):
    epc_uri = get_data_object_uri(dataspace, epc_object_type, epc_uuid)
    time = datetime.datetime.now(datetime.timezone.utc).timestamp()

    mh_record = dict(
        protocol=9,  # DataArray
        messageType=9,  # PutUninitializedDataArrays
        correlationId=0,  # Ignored
        messageId=await get_msg_id(),
        messageFlags=MHFlags.FIN.value,  # Multi-part=False
    )
    puda_record = dict(
        dataArrays={
            pir: dict(
                uid=dict(
                    uri=epc_uri,
                    pathInResource=pir,
                ),
                metadata=dict(
                    dimensions=list(arr_shape),
                    # NOTE: Using arrayOfDouble64LE with arrayOfDouble did not
                    # work even though we used np.float64-arrays.
                    # These fields seem not to work as intended. The ETP-server
                    # responds with (invalid) combinations (according to
                    # section 13.2.2.1 in the ETP v1.2 spec), but we are able
                    # to reconstruct the full data from the returned values.
                    transportArrayType=transport_array_type or "arrayOfFloat",
                    logicalArrayType=logical_array_type or "arrayOfBoolean",
                    storeLastWrite=time,
                    storeCreated=time,
                ),
            )
            for pir, arr_shape in zip(paths_in_resources, arr_shapes)
        }
    )

    await ws.send(
        serialize_message(
            mh_record,
            puda_record,
            "Energistics.Etp.v12.Protocol.DataArray.PutUninitializedDataArrays",
        )
    )

    return await handle_multipart_response(
        ws,
        "Energistics.Etp.v12.Protocol.DataArray.PutUninitializedDataArraysResponse",
    )


async def put_data_arrays(
    ws,
    get_msg_id,
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
        "Energistics.Etp.v12.Protocol.DataArray.PutDataArraysResponse",
    )


async def put_data_subarrays(
    ws,
    get_msg_id,
    dataspace,
    epc_object_type,
    epc_uuid,
    paths_in_resources,
    subarrays,
    starts,
):
    epc_uri = get_data_object_uri(dataspace, epc_object_type, epc_uuid)

    mh_record = dict(
        protocol=9,  # DataArray
        messageType=5,  # PutDataSubarrays
        correlationId=0,  # Ignored
        messageId=await get_msg_id(),
        messageFlags=MHFlags.FIN.value,  # Multi-part=False
    )
    pds_record = dict(
        dataSubarrays={
            pir: dict(
                uid=dict(
                    uri=epc_uri,
                    pathInResource=pir,
                ),
                data=numpy_to_etp_data_subarray(subarray),
                starts=start,
                counts=list(subarray.shape),
            )
            for subarray, start, pir in zip(subarrays, starts, paths_in_resources)
        }
    )

    await ws.send(
        serialize_message(
            mh_record,
            pds_record,
            "Energistics.Etp.v12.Protocol.DataArray.PutDataSubarrays",
        )
    )

    return await handle_multipart_response(
        ws,
        "Energistics.Etp.v12.Protocol.DataArray.PutDataSubarraysResponse",
    )


async def get_data_array_metadata(
    ws, get_msg_id, dataspace, epc_object_type, epc_uuid, paths_in_resources
):
    # This function can fetch multiple array metadata (multiple
    # paths_in_resources), but only from a single epc-object.
    epc_uri = get_data_object_uri(dataspace, epc_object_type, epc_uuid)

    # Get data array metadata
    mh_record = dict(
        protocol=9,  # DataArray
        messageType=6,  # GetDataArrayMetadata
        correlationId=0,  # Ignored
        messageId=await get_msg_id(),
        messageFlags=MHFlags.FIN.value,  # Multi-part=False
    )
    gdam_record = dict(
        dataArrays={
            # Note that we use the pathInResource (pathInHdfFile) from the
            # Grid2dRepresentation, but the uri is for
            # EpcExternalPartReference!
            pir: dict(
                uri=epc_uri,
                pathInResource=pir,
            )
            for pir in paths_in_resources
        },
    )

    await ws.send(
        serialize_message(
            mh_record,
            gdam_record,
            "Energistics.Etp.v12.Protocol.DataArray.GetDataArrayMetadata",
        )
    )

    return await handle_multipart_response(
        ws,
        "Energistics.Etp.v12.Protocol.DataArray.GetDataArrayMetadataResponse",
    )


async def get_data_arrays(
    ws, get_msg_id, dataspace, epc_object_type, epc_uuid, paths_in_resources
):
    # This function can fetch multiple arrays (multiple paths_in_resources),
    # but only from a single epc-object.
    epc_uri = get_data_object_uri(dataspace, epc_object_type, epc_uuid)

    # Get full data array
    mh_record = dict(
        protocol=9,  # DataArray
        messageType=2,  # GetDataArrays
        correlationId=0,  # Ignored
        messageId=await get_msg_id(),
        messageFlags=MHFlags.FIN.value,  # Multi-part=False
    )
    gda_record = dict(
        dataArrays={
            pir: dict(
                uri=epc_uri,
                pathInResource=pir,
            )
            for pir in paths_in_resources
        },
    )

    await ws.send(
        serialize_message(
            mh_record,
            gda_record,
            "Energistics.Etp.v12.Protocol.DataArray.GetDataArrays",
        )
    )

    return await handle_multipart_response(
        ws,
        "Energistics.Etp.v12.Protocol.DataArray.GetDataArraysResponse",
    )


async def get_data_subarrays(
    ws,
    get_msg_id,
    dataspace,
    epc_object_type,
    epc_uuid,
    paths_in_resources,
    starts_list,
    counts_list,
):
    # This function can fetch multiple arrays (multiple paths_in_resources),
    # but only from a single epc-object. In practice we will mostly be using it
    # to fetch a single subarray that is as large as possible as part of a too
    # large array.
    epc_uri = get_data_object_uri(dataspace, epc_object_type, epc_uuid)

    mh_record = dict(
        protocol=9,  # DataArray
        messageType=3,  # GetDataSubarrays
        correlationId=0,  # Ignored
        messageId=await get_msg_id(),
        messageFlags=MHFlags.FIN.value,  # Multi-part=False
    )
    gds_record = dict(
        dataSubarrays={
            pir: dict(
                uid=dict(
                    uri=epc_uri,
                    pathInResource=pir,
                ),
                starts=starts,
                counts=counts,
            )
            for starts, counts, pir in zip(starts_list, counts_list, paths_in_resources)
        },
    )

    await ws.send(
        serialize_message(
            mh_record,
            gds_record,
            "Energistics.Etp.v12.Protocol.DataArray.GetDataSubarrays",
        )
    )

    return await handle_multipart_response(
        ws, "Energistics.Etp.v12.Protocol.DataArray.GetDataSubarraysResponse"
    )


async def get_data_subarray(
    ws,
    get_msg_id,
    epc_uri,
    path_in_resource,
    starts,
    counts,
    key=None,
):
    # This method only supports the request of a single subarray.
    # The protocol from ETP can support the request of multiple subarrays.

    mh_record = dict(
        protocol=9,  # DataArray
        messageType=3,  # GetDataSubarrays
        correlationId=0,  # Ignored
        messageId=await get_msg_id(),
        messageFlags=MHFlags.FIN.value,  # Multi-part=False
    )
    gds_record = dict(
        dataSubarrays={
            key
            or "0": dict(
                uid=dict(
                    uri=epc_uri,
                    pathInResource=path_in_resource,
                ),
                starts=starts,
                counts=counts,
            )
        },
    )

    await ws.send(
        serialize_message(
            mh_record,
            gds_record,
            "Energistics.Etp.v12.Protocol.DataArray.GetDataSubarrays",
        )
    )

    return await handle_multipart_response(
        ws, "Energistics.Etp.v12.Protocol.DataArray.GetDataSubarraysResponse"
    )
