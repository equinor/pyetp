import typing

from pydantic import Field

import energistics.base
from energistics.etp.v12.datatypes.contact import Contact
from energistics.etp.v12.datatypes.data_value import DataValue
from energistics.etp.v12.datatypes.endpoint_capability_kind import (
    EndpointCapabilityKind,
)
from energistics.etp.v12.datatypes.supported_data_object import SupportedDataObject
from energistics.etp.v12.datatypes.supported_protocol import SupportedProtocol


@energistics.base.add_avro_metadata
class ServerCapabilities(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes",
        "name": "ServerCapabilities",
        "fields": [
            {"name": "applicationName", "type": "string"},
            {"name": "applicationVersion", "type": "string"},
            {
                "name": "contactInformation",
                "type": "Energistics.Etp.v12.Datatypes.Contact",
            },
            {
                "name": "supportedCompression",
                "type": {"type": "array", "items": "string"},
                "default": [],
            },
            {
                "name": "supportedEncodings",
                "type": {"type": "array", "items": "string"},
                "default": ["binary"],
            },
            {
                "name": "supportedFormats",
                "type": {"type": "array", "items": "string"},
                "default": ["xml"],
            },
            {
                "name": "supportedDataObjects",
                "type": {
                    "type": "array",
                    "items": "Energistics.Etp.v12.Datatypes.SupportedDataObject",
                },
            },
            {
                "name": "supportedProtocols",
                "type": {
                    "type": "array",
                    "items": "Energistics.Etp.v12.Datatypes.SupportedProtocol",
                },
            },
            {
                "name": "endpointCapabilities",
                "type": {
                    "type": "map",
                    "values": "Energistics.Etp.v12.Datatypes.DataValue",
                },
                "default": {},
            },
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.ServerCapabilities",
        "depends": [
            "Energistics.Etp.v12.Datatypes.Contact",
            "Energistics.Etp.v12.Datatypes.SupportedDataObject",
            "Energistics.Etp.v12.Datatypes.SupportedProtocol",
            "Energistics.Etp.v12.Datatypes.DataValue",
        ],
    }

    application_name: str = Field(alias="applicationName")
    application_version: str = Field(alias="applicationVersion")
    contact_information: Contact = Field(alias="contactInformation")
    supported_compression: list[str] = Field(
        alias="supportedCompression", default_factory=list
    )
    supported_encodings: list[str] = Field(
        alias="supportedEncodings", default_factory=lambda: ["binary"]
    )
    supported_formats: list[str] = Field(
        alias="supportedFormats", default_factory=lambda: ["xml"]
    )
    supported_data_objects: list[SupportedDataObject] = Field(
        alias="supportedDataObjects"
    )
    supported_protocols: list[SupportedProtocol] = Field(alias="supportedProtocols")
    endpoint_capabilities: typing.Mapping[EndpointCapabilityKind, DataValue] = Field(
        alias="endpointCapabilities", default_factory=dict
    )
