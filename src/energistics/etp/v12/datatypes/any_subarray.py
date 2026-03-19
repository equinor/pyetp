import typing

import energistics.base
from energistics.etp.v12.datatypes import AnyArray


@energistics.base.add_avro_metadata
class AnySubarray(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes",
        "name": "AnySubarray",
        "fields": [
            {"name": "start", "type": "long"},
            {"name": "slice", "type": "Energistics.Etp.v12.Datatypes.AnyArray"},
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.AnySubarray",
        "depends": ["Energistics.Etp.v12.Datatypes.AnyArray"],
    }

    start: int
    slice: AnyArray
