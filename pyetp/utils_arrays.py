
import typing as T

import numpy as np
from scipy import interpolate
from xtgeo import RegularSurface

from pyetp.types import (AnyArray, AnyArrayType, ArrayOfBoolean, ArrayOfDouble,
                         ArrayOfFloat, ArrayOfInt, ArrayOfLong, DataArray,
                         DataArrayMetadata)

SUPPORED_ARRAY_TYPES = T.Union[ArrayOfFloat, ArrayOfBoolean, ArrayOfInt, ArrayOfLong, ArrayOfDouble]

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


def get_dtype(item: T.Union[AnyArray, AnyArrayType]):
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


def mid_point_rectangle(arr: np.ndarray):
    all_x = arr[:, 0]
    all_y = arr[:, 1]
    min_x = np.min(all_x)
    min_y = np.min(all_y)
    mid_x = ((np.max(all_x)-min_x)/2)+min_x
    mid_y = ((np.max(all_y)-min_y)/2)+min_y
    return np.array([mid_x, mid_y])


def grid_xtgeo(data: np.ndarray):
    max_x = np.nanmax(data[:, 0])
    max_y = np.nanmax(data[:, 1])
    min_x = np.nanmin(data[:, 0])
    min_y = np.nanmin(data[:, 1])
    u_x = np.sort(np.unique(data[:, 0]))
    u_y = np.sort(np.unique(data[:, 1]))
    xinc = u_x[1] - u_x[0]
    yinc = u_y[1] - u_y[0]
    grid_x, grid_y = np.mgrid[
        min_x: max_x + xinc: xinc,
        min_y: max_y + yinc: yinc,
    ]

    interp = interpolate.LinearNDInterpolator(data[:, :-1], data[:, -1], fill_value=np.nan, rescale=False)
    z = interp(np.array([grid_x.flatten(), grid_y.flatten()]).T)
    zz = np.reshape(z, grid_x.shape)

    return RegularSurface(
        ncol=grid_x.shape[0],
        nrow=grid_x.shape[1],
        xori=min_x,
        yori=min_y,
        xinc=xinc,
        yinc=yinc,
        rotation=0.0,
        values=zz,
    )


def get_cells_positions(points: np.ndarray, n_cells: int, n_cell_per_pos: int, layers_per_sediment_unit: int, n_node_per_pos: int, node_index: int):
    results = np.zeros((int(n_cells/n_cell_per_pos), 3), dtype=np.float64)
    grid_x_pos = np.unique(points[:, 0])
    grid_y_pos = np.unique(points[:, 1])
    counter = 0
    # find cell index and location

    for y_ind in range(0, len(grid_y_pos)-1):
        for x_ind in range(0, len(grid_x_pos)-1):
            top_depth = []
            for corner_x in range(layers_per_sediment_unit):
                for corner_y in range(layers_per_sediment_unit):
                    node_indx = (((y_ind+corner_y)*len(grid_x_pos) + (x_ind+corner_x)) * n_node_per_pos) + node_index
                    top_depth.append(points[node_indx])
            results[counter, 0:2] = mid_point_rectangle(np.array(top_depth))
            counter += 1
    return results
