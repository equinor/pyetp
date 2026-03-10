import typing

import energistics.base
from energistics.etp.v12.datatypes.any_array import AnyArray


@energistics.base.add_avro_metadata
class DataArray(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes.DataArrayTypes",
        "name": "DataArray",
        "fields": [
            {"name": "dimensions", "type": {"type": "array", "items": "long"}},
            {"name": "data", "type": "Energistics.Etp.v12.Datatypes.AnyArray"},
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.DataArrayTypes.DataArray",
        "depends": ["Energistics.Etp.v12.Datatypes.AnyArray"],
    }

    dimensions: list[int]
    data: AnyArray
