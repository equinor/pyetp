import typing

import energistics.base


@energistics.base.add_protocol_avro_metadata
class Acknowledge(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Core",
        "name": "Acknowledge",
        "protocol": "0",
        "messageType": "1001",
        "senderRole": "*",
        "protocolRoles": "client, server",
        "multipartFlag": False,
        "fields": [],
        "fullName": "Energistics.Etp.v12.Protocol.Core.Acknowledge",
        "depends": [],
    }
