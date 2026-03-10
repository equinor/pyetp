import typing

import numpy as np
from pydantic import PlainSerializer, PlainValidator

import energistics.base
from energistics.serializers import get_masked_array_serializer
from energistics.validators import get_masked_array_validator


@energistics.base.add_avro_metadata
class ArrayOfNullableLong(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes",
        "name": "ArrayOfNullableLong",
        "fields": [
            {"name": "values", "type": {"type": "array", "items": ["null", "long"]}}
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.ArrayOfNullableLong",
        "depends": [],
    }

    # values: list[int | None]
    values: typing.Annotated[
        np.ma.MaskedArray[tuple[int], np.dtype[np.int64]],
        PlainValidator(get_masked_array_validator(np.dtype(np.int64))),
        PlainSerializer(get_masked_array_serializer(np.dtype(np.int64))),
    ]
