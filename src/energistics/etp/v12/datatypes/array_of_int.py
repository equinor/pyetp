import typing

import numpy as np
import numpy.typing as npt
from pydantic import PlainSerializer, PlainValidator

import energistics.base
from energistics.serializers import get_array_serializer


def validate_array(values: typing.Any) -> typing.Any:
    if isinstance(values, np.ndarray):
        if (
            not issubclass(values.dtype.type, np.signedinteger)
            or values.dtype.itemsize > 4
        ):
            raise ValueError(
                f"Got array with dtype '{values.dtype}', expected a dtype of "
                f"'np.int8', 'np.int16' or 'np.int32'"
            )

    varr = np.array(values, dtype=np.int32)

    if len(varr.shape) != 1:
        raise ValueError(
            "Values must be an object that can be coerced into a 1-d array of "
            "'np.int8', 'np.int16' or 'np.int32' numbers"
        )

    return varr


@energistics.base.add_avro_metadata
class ArrayOfInt(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes",
        "name": "ArrayOfInt",
        "fields": [{"name": "values", "type": {"type": "array", "items": "int"}}],
        "fullName": "Energistics.Etp.v12.Datatypes.ArrayOfInt",
        "depends": [],
    }

    # values: list[int]
    values: typing.Annotated[
        npt.NDArray[np.int32],
        PlainValidator(validate_array),
        PlainSerializer(get_array_serializer(np.dtype(np.int32))),
    ]
