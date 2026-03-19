import typing

from pydantic import AfterValidator, Field

import energistics.base
from energistics.validators import check_data_object_uri


@energistics.base.add_avro_metadata
class DataArrayIdentifier(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes.DataArrayTypes",
        "name": "DataArrayIdentifier",
        "fields": [
            {"name": "uri", "type": "string"},
            {"name": "pathInResource", "type": "string"},
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.DataArrayTypes.DataArrayIdentifier",
        "depends": [],
    }

    uri: typing.Annotated[
        str,
        AfterValidator(check_data_object_uri),
    ]
    path_in_resource: str = Field(alias="pathInResource")
