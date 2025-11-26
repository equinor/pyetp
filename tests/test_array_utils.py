import random
import pytest
import numpy as np
import numpy.typing as npt

import pyetp.utils_arrays

from etptypes.energistics.etp.v12.datatypes.any_logical_array_type import (
    AnyLogicalArrayType,
)
from etptypes.energistics.etp.v12.datatypes.any_array_type import AnyArrayType


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
