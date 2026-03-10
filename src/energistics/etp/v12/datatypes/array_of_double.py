import typing

import numpy as np
import numpy.typing as npt
from pydantic import PlainSerializer, PlainValidator

import energistics.base
from energistics.serializers import get_array_serializer
from energistics.validators import get_array_validator


@energistics.base.add_avro_metadata
class ArrayOfDouble(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes",
        "name": "ArrayOfDouble",
        "fields": [{"name": "values", "type": {"type": "array", "items": "double"}}],
        "fullName": "Energistics.Etp.v12.Datatypes.ArrayOfDouble",
        "depends": [],
    }

    values: typing.Annotated[
        npt.NDArray[np.float64],
        PlainValidator(get_array_validator(np.dtype(np.float64))),
        PlainSerializer(get_array_serializer(np.dtype(np.float64))),
    ]
