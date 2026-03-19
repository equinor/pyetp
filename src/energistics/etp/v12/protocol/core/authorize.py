import typing

from pydantic import Field

import energistics.base


@energistics.base.add_protocol_avro_metadata
class Authorize(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Core",
        "name": "Authorize",
        "protocol": "0",
        "messageType": "6",
        "senderRole": "client,server",
        "protocolRoles": "client, server",
        "multipartFlag": False,
        "fields": [
            {"name": "authorization", "type": "string"},
            {
                "name": "supplementalAuthorization",
                "type": {"type": "map", "values": "string"},
            },
        ],
        "fullName": "Energistics.Etp.v12.Protocol.Core.Authorize",
        "depends": [],
    }

    authorization: str
    # The supplemental authorization does not have a default defined in the
    # schema, but we do not yet use it for anything. So we pass in an empty
    # dictionary by default.
    supplemental_authorization: typing.Mapping[str, str] = Field(
        alias="supplementalAuthorization", default_factory=dict
    )
