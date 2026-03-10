import typing

import energistics.base
from energistics.etp.v12.datatypes.object import Dataspace


@energistics.base.add_protocol_avro_metadata
class PutDataspaces(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Dataspace",
        "name": "PutDataspaces",
        "protocol": "24",
        "messageType": "3",
        "senderRole": "customer",
        "protocolRoles": "store,customer",
        "multipartFlag": False,
        "fields": [
            {
                "name": "dataspaces",
                "type": {
                    "type": "map",
                    "values": "Energistics.Etp.v12.Datatypes.Object.Dataspace",
                },
            }
        ],
        "fullName": "Energistics.Etp.v12.Protocol.Dataspace.PutDataspaces",
        "depends": ["Energistics.Etp.v12.Datatypes.Object.Dataspace"],
    }

    dataspaces: typing.Mapping[str, Dataspace]
