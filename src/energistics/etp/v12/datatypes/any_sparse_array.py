import typing

import energistics.base
from energistics.etp.v12.datatypes.any_subarray import AnySubarray


@energistics.base.add_avro_metadata
class AnySparseArray(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes",
        "name": "AnySparseArray",
        "fields": [
            {
                "name": "slices",
                "type": {
                    "type": "array",
                    "items": "Energistics.Etp.v12.Datatypes.AnySubarray",
                },
            }
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.AnySparseArray",
        "depends": ["Energistics.Etp.v12.Datatypes.AnySubarray"],
    }

    slices: list[AnySubarray]
