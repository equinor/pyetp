import typing

import numpy as np
import numpy.typing as npt
from pydantic import PlainSerializer, PlainValidator

import energistics.base
from energistics.serializers import get_array_serializer
from energistics.validators import get_array_validator


@energistics.base.add_avro_metadata
class ArrayOfBoolean(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes",
        "name": "ArrayOfBoolean",
        "fields": [{"name": "values", "type": {"type": "array", "items": "boolean"}}],
        "fullName": "Energistics.Etp.v12.Datatypes.ArrayOfBoolean",
        "depends": [],
    }

    # values: list[bool]
    values: typing.Annotated[
        npt.NDArray[np.bool_],
        PlainValidator(get_array_validator(np.dtype(np.bool_))),
        PlainSerializer(get_array_serializer(np.dtype(np.bool_))),
    ]
