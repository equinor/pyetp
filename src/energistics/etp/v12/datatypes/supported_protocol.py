import typing

from pydantic import Field, model_validator

import energistics.base
from energistics.base import Protocol, Role
from energistics.etp.v12.datatypes.data_value import DataValue
from energistics.etp.v12.datatypes.protocol_capability_kind import (
    ProtocolCapabilityKind,
)
from energistics.etp.v12.datatypes.version import Version


@energistics.base.add_avro_metadata
class SupportedProtocol(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes",
        "name": "SupportedProtocol",
        "fields": [
            {"name": "protocol", "type": "int"},
            {
                "name": "protocolVersion",
                "type": "Energistics.Etp.v12.Datatypes.Version",
            },
            {"name": "role", "type": "string"},
            {
                "name": "protocolCapabilities",
                "type": {
                    "type": "map",
                    "values": "Energistics.Etp.v12.Datatypes.DataValue",
                },
                "default": {},
            },
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.SupportedProtocol",
        "depends": [
            "Energistics.Etp.v12.Datatypes.Version",
            "Energistics.Etp.v12.Datatypes.DataValue",
        ],
    }

    protocol: Protocol
    protocol_version: Version = Field(
        alias="protocolVersion",
        validate_default=True,
        default=Version(major=1, minor=2, revision=0, patch=0),
    )
    role: Role
    protocol_capabilities: typing.Mapping[ProtocolCapabilityKind, DataValue] = Field(
        alias="protocolCapabilities",
        validate_default=True,
        default_factory=dict,
    )

    @model_validator(mode="after")
    def check_protocol_and_role(self) -> typing.Self:
        if self.protocol == Protocol.CORE and self.role not in [
            Role.CLIENT,
            Role.SERVER,
        ]:
            raise ValueError(
                f"Protocol {self.protocol} must have a role of either '{Role.CLIENT}' "
                f"or '{Role.SERVER}'"
            )

        elif self.protocol == Protocol.CHANNEL_STREAMING and self.role not in [
            Role.PRODUCER,
            Role.CONSUMER,
        ]:
            raise ValueError(
                f"Protocol {self.protocol} must have a role of either "
                f"'{Role.CONSUMER}' or '{Role.PRODUCER}'"
            )

        elif self.protocol in [
            Protocol.CHANNEL_DATA_FRAME,
            Protocol.DISCOVERY,
            Protocol.STORE,
            Protocol.STORE_NOTIFICATION,
            Protocol.GROWING_OBJECT,
            Protocol.GROWING_OBJECT_NOTIFICATION,
            Protocol.DATA_ARRAY,
            Protocol.DISCOVERY_QUERY,
            Protocol.STORE_QUERY,
            Protocol.GROWING_OBJECT_QUERY,
            Protocol.TRANSACTION,
            Protocol.CHANNEL_SUBSCRIBE,
            Protocol.CHANNEL_DATA_LOAD,
            Protocol.DATASPACE,
            Protocol.SUPPORTED_TYPES,
        ] and self.role not in [
            Role.STORE,
            Role.CUSTOMER,
        ]:
            raise ValueError(
                f"Protocol {self.protocol} must have a role of either "
                f"'{Role.CUSTOMER}' or '{Role.STORE}'"
            )

        for k, v in self.protocol_capabilities.items():
            if not isinstance(v.item, k.get_valid_type()):
                raise TypeError(
                    f"The '{k}' ProtocolCapabilityKind must be of type "
                    f"{k.get_valid_type()}"
                )

        return self
