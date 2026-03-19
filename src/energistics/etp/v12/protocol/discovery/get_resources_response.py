import typing

from pydantic import Field

import energistics.base
from energistics.etp.v12.datatypes.object.resource import Resource


@energistics.base.add_protocol_avro_metadata
class GetResourcesResponse(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Discovery",
        "name": "GetResourcesResponse",
        "protocol": "3",
        "messageType": "4",
        "senderRole": "store",
        "protocolRoles": "store,customer",
        "multipartFlag": True,
        "fields": [
            {
                "name": "resources",
                "type": {
                    "type": "array",
                    "items": "Energistics.Etp.v12.Datatypes.Object.Resource",
                },
                "default": [],
            }
        ],
        "fullName": "Energistics.Etp.v12.Protocol.Discovery.GetResourcesResponse",
        "depends": ["Energistics.Etp.v12.Datatypes.Object.Resource"],
    }

    resources: list[Resource] = Field(default_factory=list)
