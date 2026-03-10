import typing

import energistics.base


@energistics.base.add_protocol_avro_metadata
class PutDataspacesResponse(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Dataspace",
        "name": "PutDataspacesResponse",
        "protocol": "24",
        "messageType": "6",
        "senderRole": "store",
        "protocolRoles": "store,customer",
        "multipartFlag": True,
        "fields": [{"name": "success", "type": {"type": "map", "values": "string"}}],
        "fullName": "Energistics.Etp.v12.Protocol.Dataspace.PutDataspacesResponse",
        "depends": [],
    }

    success: typing.Mapping[str, str]
