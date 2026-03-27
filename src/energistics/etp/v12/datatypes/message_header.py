import enum
import typing

from pydantic import Field, field_validator, model_validator

import energistics.base


class MessageHeaderFlags(enum.IntFlag):
    NOFLAG = 0x00
    FIN = 0x02
    COMPRESSED = 0x08
    ACK = 0x10
    EXTENSION = 0x20


@energistics.base.add_avro_metadata
class MessageHeader(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes",
        "name": "MessageHeader",
        "fields": [
            {"name": "protocol", "type": "int"},
            {"name": "messageType", "type": "int"},
            {"name": "correlationId", "type": "long"},
            {"name": "messageId", "type": "long"},
            {"name": "messageFlags", "type": "int"},
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.MessageHeader",
        "depends": [],
    }

    protocol: energistics.base.Protocol
    message_type: int = Field(alias="messageType")
    correlation_id: int = Field(alias="correlationId")
    message_id: int = Field(alias="messageId")
    message_flags: MessageHeaderFlags = Field(alias="messageFlags")

    @model_validator(mode="after")
    def check_allowed_compression(self) -> typing.Self:
        if (
            self.protocol == energistics.base.Protocol.CORE
            or self.message_type in [1000, 1001]
        ) and self.is_compressed():
            raise ValueError("Messages from core must not be compressed")

        return self

    @field_validator("message_id", mode="after")
    @classmethod
    def validate_message_id(cls, value: int) -> int:
        if value == 0:
            raise ValueError("A message id of 0 is invalid")

        return value

    @classmethod
    def from_etp_protocol_body(
        cls,
        body: typing.Type[energistics.base.ETPBaseProtocolModel]
        | energistics.base.ETPBaseProtocolModel,
        message_id: int,
        message_flags: MessageHeaderFlags,
        correlation_id: int = 0,
    ) -> typing.Self:
        return cls(
            protocol=body._protocol,
            message_type=body._message_type,
            correlation_id=correlation_id,
            message_id=message_id,
            message_flags=message_flags,
        )

    def is_final_message(self) -> bool:
        return (self.message_flags & MessageHeaderFlags.FIN) != 0

    def is_compressed(self) -> bool:
        return (self.message_flags & MessageHeaderFlags.COMPRESSED) != 0

    def requests_acknowledgement(self) -> bool:
        return (self.message_flags & MessageHeaderFlags.ACK) != 0

    def uses_extension_header(self) -> bool:
        return (self.message_flags & MessageHeaderFlags.EXTENSION) != 0
