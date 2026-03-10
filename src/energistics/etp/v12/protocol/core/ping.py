import datetime
import typing

from pydantic import Field

import energistics.base


@energistics.base.add_protocol_avro_metadata
class Ping(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Core",
        "name": "Ping",
        "protocol": "0",
        "messageType": "8",
        "senderRole": "client,server",
        "protocolRoles": "client, server",
        "multipartFlag": False,
        "fields": [{"name": "currentDateTime", "type": "long"}],
        "fullName": "Energistics.Etp.v12.Protocol.Core.Ping",
        "depends": [],
    }

    # We add the current time as a default value
    current_date_time: int = Field(
        alias="currentDateTime",
        default_factory=lambda: int(
            # Return the current (UTC) time in microseconds
            datetime.datetime.now(datetime.timezone.utc).timestamp() * 1e6
        ),
    )
