import typing

from pydantic import Field, model_validator

import energistics.base
import energistics.uris
from energistics.etp.v12.datatypes.data_value import DataValue
from energistics.etp.v12.datatypes.object.active_status_kind import ActiveStatusKind


@energistics.base.add_avro_metadata
class Resource(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes.Object",
        "name": "Resource",
        "fields": [
            {"name": "uri", "type": "string"},
            {
                "name": "alternateUris",
                "type": {"type": "array", "items": "string"},
                "default": [],
            },
            {"name": "name", "type": "string"},
            {"name": "sourceCount", "type": ["null", "int"], "default": None},
            {"name": "targetCount", "type": ["null", "int"], "default": None},
            {"name": "lastChanged", "type": "long"},
            {"name": "storeLastWrite", "type": "long"},
            {"name": "storeCreated", "type": "long"},
            {
                "name": "activeStatus",
                "type": "Energistics.Etp.v12.Datatypes.Object.ActiveStatusKind",
            },
            {
                "name": "customData",
                "type": {
                    "type": "map",
                    "values": "Energistics.Etp.v12.Datatypes.DataValue",
                },
                "default": {},
            },
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.Object.Resource",
        "depends": [
            "Energistics.Etp.v12.Datatypes.Object.ActiveStatusKind",
            "Energistics.Etp.v12.Datatypes.DataValue",
        ],
    }

    uri: str
    alternate_uris: list[str] = Field(alias="alternateUris", default_factory=list)
    name: str
    source_count: int | None = Field(alias="sourceCount", default=None)
    target_count: int | None = Field(alias="targetCount", default=None)
    last_changed: int = Field(alias="lastChanged")
    store_last_write: int = Field(alias="storeLastWrite")
    store_created: int = Field(alias="storeCreated")
    active_status: ActiveStatusKind = Field(alias="activeStatus")
    custom_data: typing.Mapping[str, DataValue] = Field(
        alias="customData", default_factory=dict
    )

    @model_validator(mode="after")
    def check_uri(self) -> typing.Self:
        energistics.uris.DataObjectURI.validate_uri(self.uri)

        return self
