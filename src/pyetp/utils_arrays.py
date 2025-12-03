import typing as T

import numpy as np
import numpy.typing as npt
from etptypes.energistics.etp.v12.datatypes.any_array import AnyArray
from etptypes.energistics.etp.v12.datatypes.any_array_type import AnyArrayType
from etptypes.energistics.etp.v12.datatypes.any_logical_array_type import (
    AnyLogicalArrayType,
)
from etptypes.energistics.etp.v12.datatypes.array_of_boolean import ArrayOfBoolean
from etptypes.energistics.etp.v12.datatypes.array_of_double import ArrayOfDouble
from etptypes.energistics.etp.v12.datatypes.array_of_float import ArrayOfFloat
from etptypes.energistics.etp.v12.datatypes.array_of_int import ArrayOfInt
from etptypes.energistics.etp.v12.datatypes.array_of_long import ArrayOfLong
from etptypes.energistics.etp.v12.datatypes.data_array_types.data_array import (
    DataArray,
)
from etptypes.energistics.etp.v12.datatypes.data_array_types.data_array_metadata import (
    DataArrayMetadata,
)

SUPPORTED_ARRAY_TYPES: T.TypeAlias = (
    ArrayOfFloat | ArrayOfBoolean | ArrayOfInt | ArrayOfLong | ArrayOfDouble
)


# See section 13.2.2.1 for the allowed mapping between logical array types and
# transport array types.
# NOTE: Currently the logical-array-mapping does not work on the
# open-etp-server. We write the relevant logical array type, but we only get
# AnyLogicalArrayType.ARRAY_OF_BOOLEAN in return from the server.
_ANY_LOGICAL_ARRAY_TYPE_MAP: dict[npt.DTypeLike, AnyLogicalArrayType] = {
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

valid_logical_array_dtypes = list(_ANY_LOGICAL_ARRAY_TYPE_MAP)
LogicalArrayDTypes: T.TypeAlias = T.Union[tuple(valid_logical_array_dtypes)]

_INV_ANY_LOGICAL_ARRAY_TYPE_MAP: dict[AnyLogicalArrayType, npt.DTypeLike] = {
    v: k for k, v in _ANY_LOGICAL_ARRAY_TYPE_MAP.items()
}


# TODO: This map should be used once the logical-array-type is supported in the
# open-etp-server.
# _ANY_ARRAY_TYPE_MAP: dict[npt.DTypeLike, AnyArrayType] = {
#     np.dtype(np.bool_): AnyArrayType.ARRAY_OF_BOOLEAN,
#     np.dtype(np.int8): AnyArrayType.BYTES,
#     np.dtype(np.uint8): AnyArrayType.BYTES,
#     np.dtype("<i2"): AnyArrayType.BYTES,
#     np.dtype("<i4"): AnyArrayType.BYTES,
#     np.dtype("<i8"): AnyArrayType.BYTES,
#     np.dtype("<u2"): AnyArrayType.BYTES,
#     np.dtype("<u4"): AnyArrayType.BYTES,
#     np.dtype("<u8"): AnyArrayType.BYTES,
#     np.dtype("<f4"): AnyArrayType.ARRAY_OF_FLOAT,
#     np.dtype("<f8"): AnyArrayType.ARRAY_OF_DOUBLE,
#     np.dtype(">i2"): AnyArrayType.BYTES,
#     np.dtype(">i4"): AnyArrayType.BYTES,
#     np.dtype(">i8"): AnyArrayType.BYTES,
#     np.dtype(">u2"): AnyArrayType.BYTES,
#     np.dtype(">u4"): AnyArrayType.BYTES,
#     np.dtype(">u8"): AnyArrayType.BYTES,
#     np.dtype(">f4"): AnyArrayType.BYTES,
#     np.dtype(">f8"): AnyArrayType.BYTES,
# }


# This AnyArrayType-map is used until the logical-array-type is properly
# implemented for the open-etp-server. In this case we
_ANY_ARRAY_TYPE_MAP: dict[npt.DTypeLike, AnyArrayType] = {
    np.dtype(np.bool_): AnyArrayType.ARRAY_OF_BOOLEAN,
    np.dtype(np.int8): AnyArrayType.BYTES,
    np.dtype("<i4"): AnyArrayType.ARRAY_OF_INT,
    np.dtype("<i8"): AnyArrayType.ARRAY_OF_LONG,
    np.dtype("<f4"): AnyArrayType.ARRAY_OF_FLOAT,
    np.dtype("<f8"): AnyArrayType.ARRAY_OF_DOUBLE,
}
valid_any_array_dtypes = list(_ANY_ARRAY_TYPE_MAP)

_INV_ANY_ARRAY_TYPE_MAP: dict[AnyArrayType, npt.DTypeLike] = {
    AnyArrayType.ARRAY_OF_BOOLEAN: np.dtype(np.bool_),
    # The BYTES-arrays are converted to the proper dtype using the logical
    # array type. We can therefore interpret the bytes as np.int8, before we
    # combine the byte strings to the proper type.
    AnyArrayType.BYTES: np.dtype(np.int8),
    AnyArrayType.ARRAY_OF_INT: np.dtype("<i4"),
    AnyArrayType.ARRAY_OF_LONG: np.dtype("<i8"),
    AnyArrayType.ARRAY_OF_FLOAT: np.dtype("<f4"),
    AnyArrayType.ARRAY_OF_DOUBLE: np.dtype("<f8"),
}


_ANY_ARRAY_MAP: dict[AnyArrayType, SUPPORTED_ARRAY_TYPES] = {
    AnyArrayType.ARRAY_OF_FLOAT: ArrayOfFloat,
    AnyArrayType.ARRAY_OF_DOUBLE: ArrayOfDouble,
    AnyArrayType.ARRAY_OF_INT: ArrayOfInt,
    AnyArrayType.ARRAY_OF_LONG: ArrayOfLong,
    AnyArrayType.ARRAY_OF_BOOLEAN: ArrayOfBoolean,
    AnyArrayType.BYTES: bytes,
}

_INV_ANY_ARRAY_MAP: dict[SUPPORTED_ARRAY_TYPES, AnyArrayType] = {
    v: k for k, v in _ANY_ARRAY_MAP.items()
}


def check_if_array_is_valid_dtype(array: npt.NDArray[T.Any]) -> bool:
    return array.dtype in valid_any_array_dtypes


def get_valid_dtype_cast(array: npt.NDArray[T.Any]) -> npt.DTypeLike:
    if check_if_array_is_valid_dtype(array):
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

    raise TypeError(f"Dtype {array.dtype} does not have a valid cast")


def get_logical_array_type(dtype: npt.DTypeLike) -> AnyLogicalArrayType:
    logical_array_type = _ANY_LOGICAL_ARRAY_TYPE_MAP.get(dtype)

    if logical_array_type is not None:
        return logical_array_type

    # Here we might be taking a chance by not caring about the endianess of the
    # string.
    if dtype.type == np.str_:
        return AnyLogicalArrayType.ARRAY_OF_STRING

    # We ignore the AnyLogicalArrayType.ARRAY_OF_CUSTOM for now.
    raise KeyError(f"Data type {dtype} is not a valid ETP v1.2 logical array type")


def get_transport_array_type(dtype: npt.DTypeLike) -> AnyArrayType:
    transport_array_type = _ANY_ARRAY_TYPE_MAP.get(dtype)

    if transport_array_type is not None:
        return transport_array_type

    # Here we might be taking a chance by not caring about the endianess of the
    # string.
    if dtype.type == np.str_:
        return AnyArrayType.ARRAY_OF_STRING

    raise KeyError(
        f"Data type {dtype} does not have a valid map to an ETP v1.2 transport array "
        f"type. Valid types are: {list(_ANY_ARRAY_TYPE_MAP)}"
    )


def get_logical_and_transport_array_types(
    dtype: npt.DTypeLike,
) -> tuple[AnyLogicalArrayType, AnyArrayType]:
    # See section 13.2.2.1 in the ETP v1.2 specification for the allowed
    # mappings between the logical and transport types.
    # Using this function ensures that the combination of the logical and
    # transport array types are valid (it is set up in valid combinations in
    # the mapping dictionaries at the top).

    return get_logical_array_type(dtype), get_transport_array_type(dtype)


def get_etp_data_array_from_numpy(data: npt.NDArray) -> DataArray:
    transport_array_type = get_transport_array_type(data.dtype)
    cls = _ANY_ARRAY_MAP[transport_array_type]

    if cls is bytes:
        # In the current implementation we only support 1-byte sized dtype's
        # when using "bytes". In the future, with logical array types, this can
        # cover multiple dtypes, but then the dimensions must be adjusted.
        itemsize = data.dtype.itemsize
        item = np.ravel(data).tobytes()
        dimensions = list(data.shape)
        dimensions[-1] = dimensions[-1] * itemsize

        return DataArray(dimensions=dimensions, data=AnyArray(item=item))

    return DataArray(
        dimensions=data.shape, data=AnyArray(item=cls(values=np.ravel(data).tolist()))
    )


def get_transport_array_size(metadata: DataArrayMetadata) -> int:
    dtype = _INV_ANY_ARRAY_TYPE_MAP[metadata.transport_array_type]
    return int(np.prod(metadata.dimensions) * dtype.itemsize)


def get_dtype_from_any_array_class(cls: AnyArray) -> npt.DTypeLike:
    if cls is bytes:
        return np.dtype(np.int8)
    elif cls is ArrayOfBoolean:
        return np.dtype(np.bool_)
    elif cls is ArrayOfInt:
        return np.dtype("<i4")
    elif cls is ArrayOfLong:
        return np.dtype("<i8")
    elif cls is ArrayOfFloat:
        return np.dtype("<f4")
    elif cls is ArrayOfDouble:
        return np.dtype("<f8")
    # TODO: Update NumPy to >= 2.0, and import the ArrayOfString-class
    # elif cls == ArrayOfString:
    #     return np.StringDType()

    raise TypeError(f"Class {cls} is not a valid array class")


def get_dtype_from_any_array_type(_type: AnyArrayType | str) -> npt.DTypeLike:
    enum_name = AnyArrayType(_type)
    return _INV_ANY_ARRAY_TYPE_MAP[enum_name]


def get_dtype_from_any_logical_array_type(
    _type: AnyLogicalArrayType | str,
) -> npt.DTypeLike:
    enum_name = AnyLogicalArrayType(_type)
    return _INV_ANY_LOGICAL_ARRAY_TYPE_MAP[enum_name]


def get_numpy_array_from_etp_data_array(
    data_array: DataArray,
) -> npt.NDArray[
    # The types used here do not tell which endianess is used for the returned
    # arrays, but until we can use np.dtype("<f4")-like syntax (Python > 3.10),
    # this will do.
    np.int8 | np.bool_ | np.int32 | np.int64 | np.float32 | np.float64
]:
    dtype = get_dtype_from_any_array_class(type(data_array.data.item))

    if type(data_array.data.item) is bytes:
        return np.array(np.frombuffer(data_array.data.item, dtype=dtype)).reshape(
            data_array.dimensions
        )

    return np.array(data_array.data.item.values, dtype=dtype).reshape(
        data_array.dimensions
    )
