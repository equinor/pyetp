import typing

from pydantic import Field, model_validator

import energistics.base
from energistics.etp.v12.datatypes.data_object_capability_kind import (
    DataObjectCapabilityKind,
)
from energistics.etp.v12.datatypes.data_value import DataValue


@energistics.base.add_avro_metadata
class SupportedDataObject(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes",
        "name": "SupportedDataObject",
        "fields": [
            {"name": "qualifiedType", "type": "string"},
            {
                "name": "dataObjectCapabilities",
                "type": {
                    "type": "map",
                    "values": "Energistics.Etp.v12.Datatypes.DataValue",
                },
                "default": {},
            },
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.SupportedDataObject",
        "depends": ["Energistics.Etp.v12.Datatypes.DataValue"],
    }

    qualified_type: str = Field(alias="qualifiedType")
    data_object_capabilities: typing.Mapping[DataObjectCapabilityKind, DataValue] = (
        Field(
            alias="dataObjectCapabilities",
            validate_default=True,
            default={},
        )
    )

    @model_validator(mode="after")
    def check_qualified_type_and_capabilities(self) -> typing.Self:
        # These qualified types correspond to the ones supported by the
        # open-etp-server from OSDU.
        valid_qualified_types = [
            "eml20.",
            "resqml20.",
            "resqml22.",
            "eml23.",
            "witsml21.",
            "prodml22.",
        ]
        for vqt in valid_qualified_types:
            if self.qualified_type.startswith(vqt):
                break
        else:
            raise TypeError(
                "Valid qualified types must start with one of: "
                f"{valid_qualified_types}. Append '*' for all types, or specific "
                "objects under each category."
            )

        for k, v in self.data_object_capabilities.items():
            if not isinstance(v.item, k.get_valid_type()):
                raise TypeError(
                    f"The '{k}' DataObjectCapabilityKind must be of type "
                    f"{k.get_valid_type()}"
                )

        return self
