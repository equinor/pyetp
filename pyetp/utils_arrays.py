
import typing as T
import numpy as np

from .types import (AnyArray, AnyArrayType, ArrayOfBoolean, ArrayOfDouble,
                    ArrayOfFloat, ArrayOfInt, ArrayOfLong, DataArray,
                    DataArrayMetadata)

SUPPORED_ARRAY_TYPES = T.Union[ArrayOfFloat , ArrayOfBoolean, ArrayOfInt, ArrayOfLong , ArrayOfDouble]

_ARRAY_MAP_TYPES: dict[AnyArrayType, np.dtype[T.Any]] = {
    AnyArrayType.ARRAY_OF_FLOAT: np.dtype(np.float32),
    AnyArrayType.ARRAY_OF_DOUBLE: np.dtype(np.float64),
    AnyArrayType.ARRAY_OF_INT: np.dtype(np.int32),
    AnyArrayType.ARRAY_OF_LONG: np.dtype(np.int64),
    AnyArrayType.ARRAY_OF_BOOLEAN: np.dtype(np.bool_)
}

_ARRAY_MAP: dict[AnyArrayType, T.Type[SUPPORED_ARRAY_TYPES]] = {
    AnyArrayType.ARRAY_OF_FLOAT: ArrayOfFloat,
    AnyArrayType.ARRAY_OF_DOUBLE: ArrayOfDouble,
    AnyArrayType.ARRAY_OF_INT: ArrayOfInt,
    AnyArrayType.ARRAY_OF_LONG: ArrayOfLong,
    AnyArrayType.ARRAY_OF_BOOLEAN: ArrayOfBoolean
}


def get_transport_from_name(k: str):
    return AnyArrayType(k[0].lower() + k[1:])


def get_transport(dtype: np.dtype):

    arraytype = [item[0] for item in _ARRAY_MAP_TYPES.items() if item[1] == dtype]
    if not len(arraytype):
        raise TypeError(f"Not {type(dtype)} supported")

    return arraytype[0]


def get_cls(dtype: np.dtype):
    return _ARRAY_MAP[get_transport(dtype)]


def get_dtype(item: T.Union[AnyArray ,AnyArrayType]):
    atype = item if isinstance(item, AnyArrayType) else get_transport_from_name(item.item.__class__.__name__)

    if atype not in _ARRAY_MAP_TYPES:
        raise TypeError(f"Not {atype} supported")

    return _ARRAY_MAP_TYPES[atype]


def get_nbytes(md: DataArrayMetadata):
    dtype = get_dtype(md.transport_array_type)
    return int(np.prod(np.array(md.dimensions)) * dtype.itemsize)


def to_numpy(data_array: DataArray):
    dims: T.Tuple[int, ...] = tuple(map(int, data_array.dimensions))
    return np.asarray(
        data_array.data.item.values,  # type: ignore
        dtype=get_dtype(data_array.data)
    ).reshape(dims)


def to_data_array(data: np.ndarray):
    cls = get_cls(data.dtype)
    return DataArray(
        dimensions=data.shape,  # type: ignore
        data=AnyArray(item=cls(values=data.flatten().tolist()))
    )
