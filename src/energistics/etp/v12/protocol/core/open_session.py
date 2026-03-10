import datetime
import typing

from pydantic import Field

import energistics.base
from energistics.etp.v12.datatypes.data_value import DataValue
from energistics.etp.v12.datatypes.endpoint_capability_kind import (
    EndpointCapabilityKind,
)
from energistics.etp.v12.datatypes.supported_data_object import SupportedDataObject
from energistics.etp.v12.datatypes.supported_protocol import SupportedProtocol
from energistics.etp.v12.datatypes.uuid import Uuid


@energistics.base.add_protocol_avro_metadata
class OpenSession(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Core",
        "name": "OpenSession",
        "protocol": "0",
        "messageType": "2",
        "senderRole": "server",
        "protocolRoles": "client, server",
        "multipartFlag": False,
        "fields": [
            {"name": "applicationName", "type": "string"},
            {"name": "applicationVersion", "type": "string"},
            {"name": "serverInstanceId", "type": "Energistics.Etp.v12.Datatypes.Uuid"},
            {
                "name": "supportedProtocols",
                "type": {
                    "type": "array",
                    "items": "Energistics.Etp.v12.Datatypes.SupportedProtocol",
                },
            },
            {
                "name": "supportedDataObjects",
                "type": {
                    "type": "array",
                    "items": "Energistics.Etp.v12.Datatypes.SupportedDataObject",
                },
            },
            {"name": "supportedCompression", "type": "string", "default": ""},
            {
                "name": "supportedFormats",
                "type": {"type": "array", "items": "string"},
                "default": ["xml"],
            },
            {"name": "currentDateTime", "type": "long"},
            {"name": "earliestRetainedChangeTime", "type": "long"},
            {"name": "sessionId", "type": "Energistics.Etp.v12.Datatypes.Uuid"},
            {
                "name": "endpointCapabilities",
                "type": {
                    "type": "map",
                    "values": "Energistics.Etp.v12.Datatypes.DataValue",
                },
                "default": {},
            },
        ],
        "fullName": "Energistics.Etp.v12.Protocol.Core.OpenSession",
        "depends": [
            "Energistics.Etp.v12.Datatypes.Uuid",
            "Energistics.Etp.v12.Datatypes.SupportedProtocol",
            "Energistics.Etp.v12.Datatypes.SupportedDataObject",
            "Energistics.Etp.v12.Datatypes.Uuid",
            "Energistics.Etp.v12.Datatypes.DataValue",
        ],
    }

    application_name: str = Field(alias="applicationName")
    application_version: str = Field(alias="applicationVersion")
    server_instance_id: Uuid = Field(alias="serverInstanceId")
    supported_protocols: list[SupportedProtocol] = Field(alias="supportedProtocols")
    supported_data_objects: list[SupportedDataObject] = Field(
        alias="supportedDataObjects"
    )
    supported_compression: str = Field(
        alias="supportedCompression",
        default="",
    )
    supported_formats: list[str] = Field(
        alias="supportedFormats", default_factory=lambda: ["xml"]
    )
    current_date_time: int = Field(
        alias="currentDateTime",
        default_factory=lambda: int(
            # Return the current (UTC) time in microseconds
            datetime.datetime.now(datetime.timezone.utc).timestamp() * 1e6
        ),
    )
    # This field is only set by stores
    earliest_retained_change_time: int = Field(alias="earliestRetainedChangeTime")
    session_id: Uuid = Field(alias="sessionId")
    endpoint_capabilities: typing.Mapping[EndpointCapabilityKind, DataValue] = Field(
        alias="endpointCapabilities", default_factory=dict
    )
