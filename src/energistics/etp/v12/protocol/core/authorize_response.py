import typing

from pydantic import Field

import energistics.base


@energistics.base.add_protocol_avro_metadata
class AuthorizeResponse(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Core",
        "name": "AuthorizeResponse",
        "protocol": "0",
        "messageType": "7",
        "senderRole": "client,server",
        "protocolRoles": "client, server",
        "multipartFlag": False,
        "fields": [
            {"name": "success", "type": "boolean"},
            {"name": "challenges", "type": {"type": "array", "items": "string"}},
        ],
        "fullName": "Energistics.Etp.v12.Protocol.Core.AuthorizeResponse",
        "depends": [],
    }

    success: bool
    challenges: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: typing.Any) -> None:
        if self.success and self.challenges:
            # This is not permitted according to the ETP v1.2 spec, but it
            # might be ignored in the wild and so possibly this should be a
            # warning or ignored instead.
            raise ValueError(
                f"Authorization was successful, but challenges {self.challenges} is "
                "non-empty"
            )
