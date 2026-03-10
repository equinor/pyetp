import typing

from pydantic import Field

import energistics.base
from energistics.etp.v12.datatypes.object import Dataspace


@energistics.base.add_protocol_avro_metadata
class GetDataspacesResponse(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Dataspace",
        "name": "GetDataspacesResponse",
        "protocol": "24",
        "messageType": "2",
        "senderRole": "store",
        "protocolRoles": "store,customer",
        "multipartFlag": True,
        "fields": [
            {
                "name": "dataspaces",
                "type": {
                    "type": "array",
                    "items": "Energistics.Etp.v12.Datatypes.Object.Dataspace",
                },
                "default": [],
            }
        ],
        "fullName": "Energistics.Etp.v12.Protocol.Dataspace.GetDataspacesResponse",
        "depends": ["Energistics.Etp.v12.Datatypes.Object.Dataspace"],
    }

    dataspaces: list[Dataspace] = Field(default_factory=list)
