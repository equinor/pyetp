import typing

import numpy as np
from pydantic import PlainSerializer, PlainValidator

import energistics.base
from energistics.serializers import get_masked_array_serializer
from energistics.validators import get_masked_array_validator


@energistics.base.add_avro_metadata
class ArrayOfNullableBoolean(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes",
        "name": "ArrayOfNullableBoolean",
        "fields": [
            {
                "name": "values",
                "type": {"type": "array", "items": ["null", "boolean"]},
            }
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.ArrayOfNullableBoolean",
        "depends": [],
    }

    # values: list[bool | None]
    values: typing.Annotated[
        np.ma.MaskedArray[tuple[int], np.dtype[np.bool_]],
        PlainValidator(get_masked_array_validator(np.dtype(np.bool_))),
        PlainSerializer(get_masked_array_serializer(np.dtype(np.bool_))),
    ]
