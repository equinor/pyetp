import typing

import energistics.base
from energistics.etp.v12.datatypes.object.edge import Edge


@energistics.base.add_protocol_avro_metadata
class GetResourcesEdgesResponse(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Discovery",
        "name": "GetResourcesEdgesResponse",
        "protocol": "3",
        "messageType": "7",
        "senderRole": "store",
        "protocolRoles": "store,customer",
        "multipartFlag": True,
        "fields": [
            {
                "name": "edges",
                "type": {
                    "type": "array",
                    "items": "Energistics.Etp.v12.Datatypes.Object.Edge",
                },
            }
        ],
        "fullName": "Energistics.Etp.v12.Protocol.Discovery.GetResourcesEdgesResponse",
        "depends": ["Energistics.Etp.v12.Datatypes.Object.Edge"],
    }

    edges: list[Edge]
