import typing
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

ArrayTypes: typing.TypeAlias = npt.NDArray[
    np.bool_ | np.bytes_ | np.int32 | np.int64 | np.float32 | np.float64 | np.str_
]


def get_array_serializer(
    dtype: npt.DTypeLike,
) -> Callable[[ArrayTypes], list[bool | bytes | int | float | str]]:
    def serialize_array(
        values: ArrayTypes,
    ) -> list[bool | bytes | int | float | str]:
        return typing.cast(
            list[bool | bytes | int | float | str],
            values.tolist(),
        )

    return serialize_array


def get_masked_array_serializer(
    dtype: npt.DTypeLike,
) -> Callable[
    [np.ma.MaskedArray[tuple[int], np.dtype[np.bool_ | np.int32 | np.int64]]],
    list[bool | None] | list[int | None],
]:
    def serialize_masked_array(
        values: np.ma.MaskedArray[tuple[int], np.dtype[np.bool_ | np.int32 | np.int64]],
    ) -> list[bool | None] | list[int | None]:
        return typing.cast(
            list[bool | None] | list[int | None],
            values.tolist(fill_value=None),
        )

    return serialize_masked_array
