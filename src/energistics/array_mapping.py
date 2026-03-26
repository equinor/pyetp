import math
import typing

import numpy as np
import numpy.typing as npt

from energistics.etp.v12.datatypes.any_array_type import AnyArrayType
from energistics.etp.v12.datatypes.any_logical_array_type import AnyLogicalArrayType
from energistics.etp.v12.datatypes.array_of_boolean import ArrayOfBoolean
from energistics.etp.v12.datatypes.array_of_double import ArrayOfDouble
from energistics.etp.v12.datatypes.array_of_float import ArrayOfFloat
from energistics.etp.v12.datatypes.array_of_int import ArrayOfInt
from energistics.etp.v12.datatypes.array_of_long import ArrayOfLong
from energistics.etp.v12.datatypes.array_of_string import ArrayOfString
from energistics.types import ETPBasicArrayType, ETPNumpyArrayDType

# See section 13.2.2.1 for the allowed mapping between logical array types and
# transport array types.
# NOTE: Currently the logical-array-mapping does not work on the
# open-etp-server. We write the relevant logical array type, but we only get
# AnyLogicalArrayType.ARRAY_OF_BOOLEAN in return from the server.
_ANY_LOGICAL_ARRAY_TYPE_MAP: dict[ETPNumpyArrayDType, AnyLogicalArrayType] = {
    np.dtype(np.bool_): AnyLogicalArrayType.ARRAY_OF_BOOLEAN,
    np.dtype(np.int8): AnyLogicalArrayType.ARRAY_OF_INT8,
    np.dtype(np.uint8): AnyLogicalArrayType.ARRAY_OF_UINT8,
    np.dtype("<i2"): AnyLogicalArrayType.ARRAY_OF_INT16_LE,
    np.dtype("<i4"): AnyLogicalArrayType.ARRAY_OF_INT32_LE,
    np.dtype("<i8"): AnyLogicalArrayType.ARRAY_OF_INT64_LE,
    np.dtype("<u2"): AnyLogicalArrayType.ARRAY_OF_UINT16_LE,
    np.dtype("<u4"): AnyLogicalArrayType.ARRAY_OF_UINT32_LE,
    np.dtype("<u8"): AnyLogicalArrayType.ARRAY_OF_UINT64_LE,
    np.dtype("<f4"): AnyLogicalArrayType.ARRAY_OF_FLOAT32_LE,
    np.dtype("<f8"): AnyLogicalArrayType.ARRAY_OF_DOUBLE64_LE,
    np.dtype(">i2"): AnyLogicalArrayType.ARRAY_OF_INT16_BE,
    np.dtype(">i4"): AnyLogicalArrayType.ARRAY_OF_INT32_BE,
    np.dtype(">i8"): AnyLogicalArrayType.ARRAY_OF_INT64_BE,
    np.dtype(">u2"): AnyLogicalArrayType.ARRAY_OF_UINT16_BE,
    np.dtype(">u4"): AnyLogicalArrayType.ARRAY_OF_UINT32_BE,
    np.dtype(">u8"): AnyLogicalArrayType.ARRAY_OF_UINT64_BE,
    np.dtype(">f4"): AnyLogicalArrayType.ARRAY_OF_FLOAT32_BE,
    np.dtype(">f8"): AnyLogicalArrayType.ARRAY_OF_DOUBLE64_BE,
}

_INV_ANY_LOGICAL_ARRAY_TYPE_MAP: dict[AnyLogicalArrayType, ETPNumpyArrayDType] = {
    v: k for k, v in _ANY_LOGICAL_ARRAY_TYPE_MAP.items()
}

# NOTE: The commented dtypes should be included once the logical array field is
# supported from the server side.
_ANY_ARRAY_TYPE_MAP: dict[ETPNumpyArrayDType, AnyArrayType] = {
    np.dtype(np.bool_): AnyArrayType.ARRAY_OF_BOOLEAN,
    np.dtype(np.int8): AnyArrayType.BYTES,
    # np.dtype(np.uint8): AnyArrayType.BYTES,
    # np.dtype("<i2"): AnyArrayType.BYTES,
    np.dtype("<i4"): AnyArrayType.ARRAY_OF_INT,
    np.dtype("<i8"): AnyArrayType.ARRAY_OF_LONG,
    # np.dtype("<u2"): AnyArrayType.BYTES,
    # np.dtype("<u4"): AnyArrayType.BYTES,
    # np.dtype("<u8"): AnyArrayType.BYTES,
    np.dtype("<f4"): AnyArrayType.ARRAY_OF_FLOAT,
    np.dtype("<f8"): AnyArrayType.ARRAY_OF_DOUBLE,
    # np.dtype(">i2"): AnyArrayType.BYTES,
    # np.dtype(">i4"): AnyArrayType.BYTES,
    # np.dtype(">i8"): AnyArrayType.BYTES,
    # np.dtype(">u2"): AnyArrayType.BYTES,
    # np.dtype(">u4"): AnyArrayType.BYTES,
    # np.dtype(">u8"): AnyArrayType.BYTES,
    # np.dtype(">f4"): AnyArrayType.BYTES,
    # np.dtype(">f8"): AnyArrayType.BYTES,
    np.dtype(np.str_): AnyArrayType.ARRAY_OF_STRING,
}

_INV_ANY_ARRAY_TYPE_MAP: dict[
    AnyArrayType,
    ETPNumpyArrayDType,
] = {
    AnyArrayType.ARRAY_OF_BOOLEAN: np.dtype(np.bool_),
    # The BYTES-arrays are converted to the proper dtype using the logical
    # array type. We can therefore interpret the bytes as np.int8, before we
    # combine the byte strings to the proper type.
    AnyArrayType.BYTES: np.dtype(np.int8),
    AnyArrayType.ARRAY_OF_INT: np.dtype(np.int32),
    AnyArrayType.ARRAY_OF_LONG: np.dtype(np.int64),
    AnyArrayType.ARRAY_OF_FLOAT: np.dtype(np.float32),
    AnyArrayType.ARRAY_OF_DOUBLE: np.dtype(np.float64),
    AnyArrayType.ARRAY_OF_STRING: np.dtype(np.str_),
}


_ANY_ARRAY_MAP: dict[
    AnyArrayType, typing.Type[ETPBasicArrayType] | typing.Type[bytes]
] = {
    AnyArrayType.ARRAY_OF_FLOAT: ArrayOfFloat,
    AnyArrayType.ARRAY_OF_DOUBLE: ArrayOfDouble,
    AnyArrayType.ARRAY_OF_INT: ArrayOfInt,
    AnyArrayType.ARRAY_OF_LONG: ArrayOfLong,
    AnyArrayType.ARRAY_OF_BOOLEAN: ArrayOfBoolean,
    AnyArrayType.ARRAY_OF_STRING: ArrayOfString,
    AnyArrayType.BYTES: bytes,
}

_INV_ANY_ARRAY_MAP: dict[
    typing.Type[ETPBasicArrayType] | typing.Type[bytes], AnyArrayType
] = {v: k for k, v in _ANY_ARRAY_MAP.items()}


class LogicalArrayTypeMapping:
    numpy_to_etp_map: dict[ETPNumpyArrayDType, AnyLogicalArrayType] = (
        _ANY_LOGICAL_ARRAY_TYPE_MAP
    )
    etp_to_numpy_map: dict[AnyLogicalArrayType, ETPNumpyArrayDType] = (
        _INV_ANY_LOGICAL_ARRAY_TYPE_MAP
    )

    @staticmethod
    def get_etp_array_type(dtype: ETPNumpyArrayDType) -> AnyLogicalArrayType:
        dtype = np.dtype(dtype)

        if dtype.type is str or dtype.type is np.str_:
            dtype = np.dtype(np.str_)

        if dtype not in list(LogicalArrayTypeMapping.numpy_to_etp_map):
            # We ignore the AnyLogicalArrayType.ARRAY_OF_CUSTOM for now.
            raise KeyError(
                f"Data type {dtype} does not have a valid corresponding ETP v1.2 "
                "logical array type"
            )

        return LogicalArrayTypeMapping.numpy_to_etp_map[dtype]

    @staticmethod
    def get_dtype(array_type: AnyLogicalArrayType | str) -> ETPNumpyArrayDType:
        enum_name = AnyLogicalArrayType(array_type)
        return LogicalArrayTypeMapping.etp_to_numpy_map[enum_name]

    @staticmethod
    def check_if_array_is_valid_dtype(array: npt.NDArray[typing.Any]) -> bool:
        dtype = array.dtype

        if dtype.type is str or dtype.type is np.str_:
            dtype = np.dtype(np.str_)

        return dtype in list(LogicalArrayTypeMapping.numpy_to_etp_map)

    @staticmethod
    def get_array_size(
        dimensions: list[int] | tuple[int], array_type: AnyLogicalArrayType | str
    ) -> int:
        dtype = LogicalArrayTypeMapping.get_dtype(array_type)
        return math.prod(dimensions) * dtype.itemsize


class TransportArrayTypeMapping:
    numpy_to_etp_map: dict[ETPNumpyArrayDType, AnyArrayType] = _ANY_ARRAY_TYPE_MAP
    etp_to_numpy_map: dict[AnyArrayType, ETPNumpyArrayDType] = _INV_ANY_ARRAY_TYPE_MAP

    @staticmethod
    def get_etp_array_type(dtype: ETPNumpyArrayDType) -> AnyArrayType:
        dtype = np.dtype(dtype)

        if dtype.type is str or dtype.type is np.str_:
            dtype = np.dtype(np.str_)

        if dtype not in list(TransportArrayTypeMapping.numpy_to_etp_map):
            raise KeyError(
                f"Data type {dtype} does not have a valid map to an ETP v1.2 "
                f"transport array type. Valid types are: {list(_ANY_ARRAY_TYPE_MAP)}"
            )

        return TransportArrayTypeMapping.numpy_to_etp_map[dtype]

    @staticmethod
    def get_etp_array_class(
        dtype: ETPNumpyArrayDType,
    ) -> typing.Type[ETPBasicArrayType] | typing.Type[bytes]:
        array_type = TransportArrayTypeMapping.get_etp_array_type(dtype)
        return _ANY_ARRAY_MAP[array_type]

    @staticmethod
    def get_dtype(array_type: AnyArrayType | str) -> ETPNumpyArrayDType:
        enum_name = AnyArrayType(array_type)
        return TransportArrayTypeMapping.etp_to_numpy_map[enum_name]

    @staticmethod
    def check_if_array_is_valid_dtype(array: npt.NDArray[typing.Any]) -> bool:
        dtype = array.dtype

        if dtype.type is str or dtype.type is np.str_:
            dtype = np.dtype(np.str_)

        return dtype in list(TransportArrayTypeMapping.numpy_to_etp_map)

    @staticmethod
    def get_array_size(
        dimensions: list[int] | tuple[int], array_type: AnyArrayType | str
    ) -> int:
        dtype = TransportArrayTypeMapping.get_dtype(array_type)
        return math.prod(dimensions) * dtype.itemsize

    @staticmethod
    def get_valid_dtype_cast(array: npt.NDArray[typing.Any]) -> ETPNumpyArrayDType:
        if TransportArrayTypeMapping.check_if_array_is_valid_dtype(array):
            return array.dtype

        if array.dtype == np.dtype(np.uint8):
            return np.dtype(np.int8)
        elif array.dtype == np.dtype("<u2"):
            return np.dtype("<i2")
        elif array.dtype == np.dtype(">u2"):
            return np.dtype("<i2")
        elif array.dtype == np.dtype("<u4"):
            return np.dtype("<i4")
        elif array.dtype == np.dtype(">u4"):
            return np.dtype("<i4")
        elif array.dtype == np.dtype("<u8"):
            return np.dtype("<i8")
        elif array.dtype == np.dtype(">u8"):
            return np.dtype("<i8")
        elif array.dtype.type is str:
            return np.dtype(np.str_)

        raise TypeError(f"DType '{array.dtype}' does not have a valid cast")


def get_logical_and_transport_array_types(
    dtype: ETPNumpyArrayDType,
) -> tuple[AnyLogicalArrayType, AnyArrayType]:
    # See section 13.2.2.1 in the ETP v1.2 specification for the allowed
    # mappings between the logical and transport types.
    # Using this function ensures that the combination of the logical and
    # transport array types are valid (it is set up in valid combinations in
    # the mapping dictionaries at the top).

    return (
        LogicalArrayTypeMapping.get_etp_array_type(dtype),
        TransportArrayTypeMapping.get_etp_array_type(dtype),
    )
