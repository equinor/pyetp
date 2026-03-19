import typing

import numpy as np
import numpy.typing as npt
from pydantic import PlainSerializer, PlainValidator

import energistics.base
from energistics.serializers import get_array_serializer
from energistics.validators import get_array_validator


@energistics.base.add_avro_metadata
class ArrayOfString(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes",
        "name": "ArrayOfString",
        "fields": [{"name": "values", "type": {"type": "array", "items": "string"}}],
        "fullName": "Energistics.Etp.v12.Datatypes.ArrayOfString",
        "depends": [],
    }

    values: typing.Annotated[
        npt.NDArray[np.str_],
        PlainValidator(get_array_validator(np.dtypes.StringDType())),
        PlainSerializer(get_array_serializer(np.dtypes.StringDType())),
    ]
