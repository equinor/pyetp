import typing

import numpy as np
from pydantic import PlainSerializer, PlainValidator

import energistics.base
from energistics.serializers import get_masked_array_serializer
from energistics.validators import get_masked_array_validator


@energistics.base.add_avro_metadata
class ArrayOfNullableInt(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes",
        "name": "ArrayOfNullableInt",
        "fields": [
            {"name": "values", "type": {"type": "array", "items": ["null", "int"]}}
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.ArrayOfNullableInt",
        "depends": [],
    }

    # values: list[int | None]
    values: typing.Annotated[
        np.ma.MaskedArray[tuple[int], np.dtype[np.int32]],
        PlainValidator(get_masked_array_validator(np.dtype(np.int32))),
        PlainSerializer(get_masked_array_serializer(np.dtype(np.int32))),
    ]
