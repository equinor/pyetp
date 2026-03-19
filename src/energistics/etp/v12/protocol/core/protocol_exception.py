import typing

from pydantic import Field

import energistics.base
from energistics.etp.v12.datatypes.error_info import ErrorInfo


@energistics.base.add_protocol_avro_metadata
class ProtocolException(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Core",
        "name": "ProtocolException",
        "protocol": "0",
        "messageType": "1000",
        "senderRole": "*",
        "protocolRoles": "client, server",
        "multipartFlag": True,
        "fields": [
            {
                "name": "error",
                "type": ["null", "Energistics.Etp.v12.Datatypes.ErrorInfo"],
            },
            {
                "name": "errors",
                "type": {
                    "type": "map",
                    "values": "Energistics.Etp.v12.Datatypes.ErrorInfo",
                },
                "default": {},
            },
        ],
        "fullName": "Energistics.Etp.v12.Protocol.Core.ProtocolException",
        "depends": [
            "Energistics.Etp.v12.Datatypes.ErrorInfo",
            "Energistics.Etp.v12.Datatypes.ErrorInfo",
        ],
    }

    error: ErrorInfo | None
    errors: typing.Mapping[str, ErrorInfo] = Field(default_factory=dict)

    def model_post_init(self, __context: typing.Any) -> None:
        if self.error is None and not self.errors:
            raise ValueError("No error-fields were populated")
        if self.error is not None and self.errors:
            raise ValueError("Both error-fields were populated")
