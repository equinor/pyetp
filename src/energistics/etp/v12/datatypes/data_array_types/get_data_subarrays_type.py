import typing

from pydantic import Field

import energistics.base
from energistics.etp.v12.datatypes.data_array_types.data_array_identifier import (
    DataArrayIdentifier,
)


@energistics.base.add_avro_metadata
class GetDataSubarraysType(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes.DataArrayTypes",
        "name": "GetDataSubarraysType",
        "fields": [
            {
                "name": "uid",
                "type": "Energistics.Etp.v12.Datatypes.DataArrayTypes.DataArrayIdentifier",
            },
            {
                "name": "starts",
                "type": {"type": "array", "items": "long"},
                "default": [],
            },
            {
                "name": "counts",
                "type": {"type": "array", "items": "long"},
                "default": [],
            },
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.DataArrayTypes.GetDataSubarraysType",
        "depends": ["Energistics.Etp.v12.Datatypes.DataArrayTypes.DataArrayIdentifier"],
    }

    uid: DataArrayIdentifier
    starts: list[int] = Field(default_factory=list)
    counts: list[int] = Field(default_factory=list)
