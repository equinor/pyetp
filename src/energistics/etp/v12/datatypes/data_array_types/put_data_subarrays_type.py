import typing

import energistics.base
from energistics.etp.v12.datatypes.any_array import (
    AnyArray,
)
from energistics.etp.v12.datatypes.data_array_types.data_array_identifier import (
    DataArrayIdentifier,
)


@energistics.base.add_avro_metadata
class PutDataSubarraysType(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes.DataArrayTypes",
        "name": "PutDataSubarraysType",
        "fields": [
            {
                "name": "uid",
                "type": "Energistics.Etp.v12.Datatypes.DataArrayTypes.DataArrayIdentifier",
            },
            {"name": "data", "type": "Energistics.Etp.v12.Datatypes.AnyArray"},
            {"name": "starts", "type": {"type": "array", "items": "long"}},
            {"name": "counts", "type": {"type": "array", "items": "long"}},
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.DataArrayTypes.PutDataSubarraysType",
        "depends": [
            "Energistics.Etp.v12.Datatypes.DataArrayTypes.DataArrayIdentifier",
            "Energistics.Etp.v12.Datatypes.AnyArray",
        ],
    }

    uid: DataArrayIdentifier
    data: AnyArray
    starts: list[int]
    counts: list[int]
