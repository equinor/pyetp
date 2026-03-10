import typing

import energistics.base


@energistics.base.add_protocol_avro_metadata
class DeleteDataspacesResponse(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Dataspace",
        "name": "DeleteDataspacesResponse",
        "protocol": "24",
        "messageType": "5",
        "senderRole": "store",
        "protocolRoles": "store,customer",
        "multipartFlag": True,
        "fields": [{"name": "success", "type": {"type": "map", "values": "string"}}],
        "fullName": "Energistics.Etp.v12.Protocol.Dataspace.DeleteDataspacesResponse",
        "depends": [],
    }

    success: typing.Mapping[str, str]
