import typing

import numpy as np

from energistics.etp.v12.datatypes.array_of_boolean import ArrayOfBoolean
from energistics.etp.v12.datatypes.array_of_bytes import ArrayOfBytes
from energistics.etp.v12.datatypes.array_of_double import ArrayOfDouble
from energistics.etp.v12.datatypes.array_of_float import ArrayOfFloat
from energistics.etp.v12.datatypes.array_of_int import ArrayOfInt
from energistics.etp.v12.datatypes.array_of_long import ArrayOfLong
from energistics.etp.v12.datatypes.array_of_nullable_boolean import (
    ArrayOfNullableBoolean,
)
from energistics.etp.v12.datatypes.array_of_nullable_int import ArrayOfNullableInt
from energistics.etp.v12.datatypes.array_of_nullable_long import ArrayOfNullableLong
from energistics.etp.v12.datatypes.array_of_string import ArrayOfString

ETPBasicArrayType: typing.TypeAlias = (
    ArrayOfBoolean
    | ArrayOfInt
    | ArrayOfLong
    | ArrayOfFloat
    | ArrayOfDouble
    | ArrayOfString
)

ETPArrayType: typing.TypeAlias = (
    ETPBasicArrayType
    | ArrayOfBytes
    | ArrayOfNullableBoolean
    | ArrayOfNullableInt
    | ArrayOfNullableLong
)

ETPBasicNumpyArrayType: typing.TypeAlias = (
    np.bool_ | np.int32 | np.int64 | np.float32 | np.float64 | np.str_ | np.int8
)

ETPNumpyArrayType: typing.TypeAlias = (
    ETPBasicNumpyArrayType
    | np.int16
    | np.uint8
    | np.uint16
    | np.uint32
    | np.uint64
    | np.bytes_
)

ETPBasicNumpyArrayDType: typing.TypeAlias = np.dtype[ETPBasicNumpyArrayType]
ETPNumpyArrayDType: typing.TypeAlias = np.dtype[ETPNumpyArrayType]
