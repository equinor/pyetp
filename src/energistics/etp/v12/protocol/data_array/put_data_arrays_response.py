import typing

import energistics.base


@energistics.base.add_protocol_avro_metadata
class PutDataArraysResponse(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.DataArray",
        "name": "PutDataArraysResponse",
        "protocol": "9",
        "messageType": "10",
        "senderRole": "store",
        "protocolRoles": "store,customer",
        "multipartFlag": True,
        "fields": [{"name": "success", "type": {"type": "map", "values": "string"}}],
        "fullName": "Energistics.Etp.v12.Protocol.DataArray.PutDataArraysResponse",
        "depends": [],
    }

    success: typing.Mapping[str, str]
