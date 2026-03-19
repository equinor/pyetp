import typing

from pydantic import AfterValidator, Field

import energistics.base
from energistics.etp.v12.datatypes.data_value import DataValue
from energistics.validators import check_data_object_uri


@energistics.base.add_avro_metadata
class DeletedResource(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes.Object",
        "name": "DeletedResource",
        "fields": [
            {"name": "uri", "type": "string"},
            {"name": "deletedTime", "type": "long"},
            {
                "name": "customData",
                "type": {
                    "type": "map",
                    "values": "Energistics.Etp.v12.Datatypes.DataValue",
                },
                "default": {},
            },
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.Object.DeletedResource",
        "depends": ["Energistics.Etp.v12.Datatypes.DataValue"],
    }

    uri: typing.Annotated[str, AfterValidator(check_data_object_uri)]
    deleted_time: int = Field(alias="deletedTime")
    custom_data: typing.Mapping[str, DataValue] = Field(
        alias="customData", default_factory=dict
    )
