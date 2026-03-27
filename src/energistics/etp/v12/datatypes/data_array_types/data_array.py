import typing

import numpy as np
import numpy.typing as npt

import energistics.base
from energistics.array_mapping import TransportArrayTypeMapping
from energistics.etp.v12.datatypes.any_array import AnyArray
from energistics.types import (
    ETPBasicArrayType,
    ETPBasicNumpyArrayType,
    ETPNumpyArrayType,
)


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
    ) -> npt.NDArray[ETPBasicNumpyArrayType | np.int8]:
        # This method just returns the array in the transport array type.
        # Afterwards, the array must be converted to the relevant logical array
        # type.
        if isinstance(self.data.item, bytes):
            return np.array(np.frombuffer(self.data.item, dtype=np.int8)).reshape(
                self.dimensions
            )
        return self.data.item.values.reshape(self.dimensions)

    @classmethod
    def from_numpy_array(
        cls,
        data_array: npt.NDArray[ETPNumpyArrayType],
    ) -> typing.Self:
        if not issubclass(data_array.dtype.type, ETPNumpyArrayType):
            raise TypeError(
                f"Array type {data_array.dtype} is not included in the valid transport "
                f"array types {ETPNumpyArrayType}. The data must be cast to one "
                "of the valid types first."
            )

        array_cls = TransportArrayTypeMapping.get_etp_array_class(data_array.dtype)

        if array_cls is bytes:
            itemsize = data_array.dtype.itemsize
            item = np.ravel(data_array).tobytes()
            dimensions = list(data_array.shape)
            # Adjust final dimension to include the number of bytes in the data
            # type.
            dimensions[-1] = dimensions[-1] * itemsize

            return cls(dimensions=dimensions, data=AnyArray(item=item))

        # Here `array_cls` can not be `bytes`, but we need to explicitly cast
        # it to any of the ETP-array types for the static type checker.
        assert array_cls is not bytes
        array_cls = typing.cast(typing.Type[ETPBasicArrayType], array_cls)

        return cls(
            dimensions=data_array.shape,
            data=AnyArray(item=array_cls(values=np.ravel(data_array))),
        )
