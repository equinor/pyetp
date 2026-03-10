import typing

import energistics.base


@energistics.base.add_protocol_avro_metadata
class CloseSession(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Core",
        "name": "CloseSession",
        "protocol": "0",
        "messageType": "5",
        "senderRole": "client,server",
        "protocolRoles": "client, server",
        "multipartFlag": False,
        "fields": [{"name": "reason", "type": "string", "default": ""}],
        "fullName": "Energistics.Etp.v12.Protocol.Core.CloseSession",
        "depends": [],
    }

    reason: str = ""
