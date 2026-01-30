import random

import numpy as np
import numpy.typing as npt
import pytest

import pyetp.utils_arrays
from energistics.etp.v12.datatypes import (
    AnyArrayType,
    AnyLogicalArrayType,
)


@pytest.mark.parametrize(
    "dtype",
    [
        # We only add the supported dtypes for now. See comments in
        # pyetp.utils_arrays.py.
        np.dtype(np.bool_),
        np.dtype(np.int8),
        np.dtype("<i4"),
        np.dtype("<i8"),
        np.dtype("<f4"),
        np.dtype("<f8"),
    ],
)
def test_allowed_mappings(dtype: npt.DTypeLike) -> None:
    # See section 13.2.2.1 in the ETP v1.2 specification.
    logical_array_type, transport_array_type = (
        pyetp.utils_arrays.get_logical_and_transport_array_types(dtype)
    )

    match logical_array_type:
        case AnyLogicalArrayType.ARRAY_OF_BOOLEAN:
            assert transport_array_type == AnyArrayType.ARRAY_OF_BOOLEAN
        case (
            AnyLogicalArrayType.ARRAY_OF_INT8
            | AnyLogicalArrayType.ARRAY_OF_UINT8
            | AnyLogicalArrayType.ARRAY_OF_INT16_LE
            | AnyLogicalArrayType.ARRAY_OF_INT32_LE
            | AnyLogicalArrayType.ARRAY_OF_UINT16_LE
            | AnyLogicalArrayType.ARRAY_OF_UINT32_LE
        ):
            assert transport_array_type in [
                AnyArrayType.BYTES,
                AnyArrayType.ARRAY_OF_INT,
                AnyArrayType.ARRAY_OF_LONG,
            ]
        case (
            AnyLogicalArrayType.ARRAY_OF_INT64_LE
            | AnyLogicalArrayType.ARRAY_OF_UINT64_LE
        ):
            assert transport_array_type in [
                AnyArrayType.BYTES,
                AnyArrayType.ARRAY_OF_LONG,
            ]
        case AnyLogicalArrayType.ARRAY_OF_FLOAT32_LE:
            assert transport_array_type == AnyArrayType.ARRAY_OF_FLOAT
        case AnyLogicalArrayType.ARRAY_OF_DOUBLE64_LE:
            assert transport_array_type == AnyArrayType.ARRAY_OF_DOUBLE
        case _:
            assert False, (
                f"Invalid combination/unsupported: {logical_array_type} and "
                f"{transport_array_type}"
            )


@pytest.mark.parametrize(
    "dtype",
    [
        # We only add the supported dtypes for now. See comments in
        # pyetp.utils_arrays.py.
        np.dtype(np.bool_),
        np.dtype(np.int8),
        np.dtype("<i4"),
        np.dtype("<i8"),
        np.dtype("<f4"),
        np.dtype("<f8"),
    ],
)
def test_transport_array_size(dtype: npt.DTypeLike) -> None:
    for i in range(10):
        shape = tuple(random.randint(1, 15) for i in range(random.randint(1, 5)))
        data = np.random.random(shape).astype(dtype)

        transport_array_type = pyetp.utils_arrays.get_transport_array_type(dtype)
        transport_array_size = pyetp.utils_arrays.get_transport_array_size(
            transport_array_type, shape
        )

        assert data.nbytes == transport_array_size


@pytest.mark.parametrize(
    "dtype",
    [
        # We only add the supported dtypes for now. See comments in
        # pyetp.utils_arrays.py.
        np.dtype(np.bool_),
        np.dtype(np.int8),
        np.dtype("<i4"),
        np.dtype("<i8"),
        np.dtype("<f4"),
        np.dtype("<f8"),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (10, 21),
        (2563, 302),
        (150, 550),
        (123, 234),
        (234, 39, 104),
    ],
)
def test_array_block_sizes(dtype: npt.DTypeLike, shape: tuple[int]) -> None:
    max_array_size = 10_000 - 512

    data = np.random.random(shape).astype(dtype)
    data_buffer = np.zeros_like(data)

    block_starts, block_counts = pyetp.utils_arrays.get_array_block_sizes(
        data.shape,
        data.dtype,
        max_array_size,
    )

    # This number is only achievable if the array is flat, or we can use a
    # single block.
    optimal_number_of_blocks = int(np.ceil(data.nbytes / max_array_size))

    # This test might be too optimistic, but we have yet to encounter a case
    # where it breaks.
    assert optimal_number_of_blocks <= len(block_starts)
    assert optimal_number_of_blocks > len(block_starts) / 2

    total_size = 0
    for starts, counts in zip(block_starts, block_counts):
        slices = tuple(
            map(
                lambda s, c: slice(s, s + c),
                np.array(starts).astype(int),
                np.array(counts).astype(int),
            )
        )

        data_buffer[slices] = data[slices]

        slice_size = data[slices].nbytes

        assert slice_size == int(np.prod(counts) * dtype.itemsize)
        assert slice_size <= max_array_size

        total_size += data[slices].nbytes

    assert total_size == data.nbytes
    np.testing.assert_equal(data, data_buffer)
