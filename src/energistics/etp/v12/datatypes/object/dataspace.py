import typing

from pydantic import Field, model_validator

import energistics.base
import energistics.uris
from energistics.etp.v12.datatypes.data_value import DataValue


@energistics.base.add_avro_metadata
class Dataspace(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes.Object",
        "name": "Dataspace",
        "fields": [
            {"name": "uri", "type": "string"},
            {"name": "path", "type": "string", "default": ""},
            {"name": "storeLastWrite", "type": "long"},
            {"name": "storeCreated", "type": "long"},
            {
                "name": "customData",
                "type": {
                    "type": "map",
                    "values": "Energistics.Etp.v12.Datatypes.DataValue",
                },
                "default": {},
            },
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.Object.Dataspace",
        "depends": ["Energistics.Etp.v12.Datatypes.DataValue"],
    }

    uri: str
    path: str = Field(default="")
    store_last_write: int = Field(alias="storeLastWrite")
    store_created: int = Field(alias="storeCreated")
    custom_data: typing.Mapping[str, DataValue] = Field(
        alias="customData", default_factory=dict
    )

    @model_validator(mode="after")
    def check_uri_and_path(self) -> typing.Self:
        dataspace_uri = energistics.uris.DataspaceURI.from_uri(self.uri)
        path = dataspace_uri.dataspace or ""

        if path != self.path:
            raise ValueError(
                f"The path ({path}) in the dataspace uri ({self.uri}) does not match "
                f"the provided path ({self.path})"
            )

        return self
