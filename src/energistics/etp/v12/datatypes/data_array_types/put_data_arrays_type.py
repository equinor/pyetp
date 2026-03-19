import typing

from pydantic import Field

import energistics.base
from energistics.etp.v12.datatypes.data_array_types.data_array import (
    DataArray,
)
from energistics.etp.v12.datatypes.data_array_types.data_array_identifier import (
    DataArrayIdentifier,
)
from energistics.etp.v12.datatypes.data_value import DataValue


@energistics.base.add_avro_metadata
class PutDataArraysType(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes.DataArrayTypes",
        "name": "PutDataArraysType",
        "fields": [
            {
                "name": "uid",
                "type": "Energistics.Etp.v12.Datatypes.DataArrayTypes.DataArrayIdentifier",
            },
            {
                "name": "array",
                "type": "Energistics.Etp.v12.Datatypes.DataArrayTypes.DataArray",
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
        "fullName": "Energistics.Etp.v12.Datatypes.DataArrayTypes.PutDataArraysType",
        "depends": [
            "Energistics.Etp.v12.Datatypes.DataArrayTypes.DataArrayIdentifier",
            "Energistics.Etp.v12.Datatypes.DataArrayTypes.DataArray",
            "Energistics.Etp.v12.Datatypes.DataValue",
        ],
    }

    uid: DataArrayIdentifier
    array: DataArray
    custom_data: typing.Mapping[str, DataValue] = Field(
        alias="customData", default_factory=dict
    )
