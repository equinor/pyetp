import typing

import numpy as np
import numpy.typing as npt
from pydantic import PlainSerializer, PlainValidator

import energistics.base
from energistics.serializers import get_array_serializer
from energistics.validators import get_array_validator


@energistics.base.add_avro_metadata
class ArrayOfBytes(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes",
        "name": "ArrayOfBytes",
        "fields": [{"name": "values", "type": {"type": "array", "items": "bytes"}}],
        "fullName": "Energistics.Etp.v12.Datatypes.ArrayOfBytes",
        "depends": [],
    }

    # values: list[bytes]
    values: typing.Annotated[
        npt.NDArray[np.bytes_],
        PlainValidator(get_array_validator(np.dtype(np.bytes_))),
        PlainSerializer(get_array_serializer(np.dtype(np.bytes_))),
    ]
