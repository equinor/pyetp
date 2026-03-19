import typing

import numpy as np
import numpy.typing as npt

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

    def to_numpy_array(
        self,
    ) -> npt.NDArray[
        np.bool_ | np.int32 | np.int64 | np.float32 | np.float64 | np.str_ | np.int8
    ]:
        if isinstance(self.data.item, bytes):
            return np.array(np.frombuffer(self.data.item, dtype=np.int8)).reshape(
                self.dimensions
            )
        return self.data.item.values.reshape(self.dimensions)
