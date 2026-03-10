import typing

from pydantic import Field

import energistics.base
from energistics.etp.v12.datatypes.object.deleted_resource import DeletedResource


@energistics.base.add_protocol_avro_metadata
class GetDeletedResourcesResponse(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Discovery",
        "name": "GetDeletedResourcesResponse",
        "protocol": "3",
        "messageType": "6",
        "senderRole": "store",
        "protocolRoles": "store,customer",
        "multipartFlag": True,
        "fields": [
            {
                "name": "deletedResources",
                "type": {
                    "type": "array",
                    "items": "Energistics.Etp.v12.Datatypes.Object.DeletedResource",
                },
                "default": [],
            }
        ],
        "fullName": "Energistics.Etp.v12.Protocol.Discovery.GetDeletedResourcesResponse",
        "depends": ["Energistics.Etp.v12.Datatypes.Object.DeletedResource"],
    }

    deleted_resources: list[DeletedResource] = Field(
        alias="deletedResources", default_factory=list
    )
