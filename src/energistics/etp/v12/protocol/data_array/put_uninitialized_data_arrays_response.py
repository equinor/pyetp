import typing

import energistics.base


@energistics.base.add_protocol_avro_metadata
class PutUninitializedDataArraysResponse(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.DataArray",
        "name": "PutUninitializedDataArraysResponse",
        "protocol": "9",
        "messageType": "12",
        "senderRole": "store",
        "protocolRoles": "store,customer",
        "multipartFlag": True,
        "fields": [{"name": "success", "type": {"type": "map", "values": "string"}}],
        "fullName": "Energistics.Etp.v12.Protocol.DataArray.PutUninitializedDataArraysResponse",
        "depends": [],
    }

    success: typing.Mapping[str, str]
