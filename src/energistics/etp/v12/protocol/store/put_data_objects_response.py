import typing

import energistics.base
import energistics.uris
from energistics.etp.v12.datatypes.object.put_response import PutResponse


@energistics.base.add_protocol_avro_metadata
class PutDataObjectsResponse(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Store",
        "name": "PutDataObjectsResponse",
        "protocol": "4",
        "messageType": "9",
        "senderRole": "store",
        "protocolRoles": "store,customer",
        "multipartFlag": True,
        "fields": [
            {
                "name": "success",
                "type": {
                    "type": "map",
                    "values": "Energistics.Etp.v12.Datatypes.Object.PutResponse",
                },
            }
        ],
        "fullName": "Energistics.Etp.v12.Protocol.Store.PutDataObjectsResponse",
        "depends": ["Energistics.Etp.v12.Datatypes.Object.PutResponse"],
    }

    success: typing.Mapping[str, PutResponse]
