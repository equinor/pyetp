import typing
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

import energistics.uris


def get_array_validator(dtype: npt.DTypeLike) -> Callable[[typing.Any], typing.Any]:
    def validate_array(values: typing.Any) -> typing.Any:
        if isinstance(values, np.ndarray):
            if values.dtype != dtype:
                # TODO: Consider just coercing to the right dtype.
                raise ValueError(
                    f"Got array with dtype '{values.dtype}', expected a dtype of "
                    f"'{dtype}'"
                )
            varr = values
        else:
            varr = np.array(values, dtype=dtype)

        if len(varr.shape) != 1:
            raise ValueError(
                "Values must be an object that can be coerced into a 1-d array of "
                f"'{dtype}' numbers"
            )

        return varr

    return validate_array


def get_masked_array_validator(
    dtype: npt.DTypeLike,
) -> Callable[[typing.Any], typing.Any]:
    def validate_masked_array(values: typing.Any) -> typing.Any:
        if isinstance(values, np.ma.MaskedArray):
            if values.dtype != dtype:
                # TODO: Consider just coercing to the right dtype.
                raise ValueError(
                    f"Got masked array with dtype '{values.dtype}', expected a dtype "
                    f"of '{dtype}'"
                )
            if len(values.shape) != 1:
                raise ValueError(
                    f"Values must be a flat 1-d array of masked '{dtype}' numbers"
                )
            return values

        obj_values = np.array(values)

        if len(obj_values.shape) != 1:
            raise ValueError(
                "Values must be an object that can be coerced into a 1-d array of "
                f"masked '{dtype}' numbers"
            )

        # Find nulled out values (assuming Python 'None').
        mask = obj_values == None  # noqa: E711
        # Set these values to zero to allow coercion into the right dtype.
        obj_values[mask] = 0

        return np.ma.masked_array(
            obj_values.astype(dtype),
            mask=mask,
        )

    return validate_masked_array


def check_data_object_types(data_object_types: list[str]) -> list[str]:
    # These qualified types correspond to the ones supported by the
    # open-etp-server from OSDU.
    valid_qualified_types = [
        "eml20.",
        "resqml20.",
        "resqml22.",
        "eml23.",
        "witsml21.",
        "prodml22.",
    ]

    errors = []
    for dot in data_object_types:
        for vqt in valid_qualified_types:
            if dot.startswith(vqt):
                break
        else:
            errors.append(
                TypeError(
                    "Valid data object types must start with one of: "
                    f"{valid_qualified_types}. Append '*' for all types, or add "
                    "specific object after the dot."
                )
            )
    if len(errors) > 0:
        raise ExceptionGroup(
            f"There were {len(errors)} invalid data object types",
            errors,
        )

    return data_object_types


def check_dataspace_uri(uri: str) -> str:
    if energistics.uris.DataspaceURI.is_valid_uri(uri):
        return uri

    raise ValueError(f"Uri '{uri}' is not a valid dataspace uri")


def check_data_object_uri(uri: str) -> str:
    if energistics.uris.DataObjectURI.is_valid_uri(uri):
        return uri

    raise ValueError(f"Uri '{uri}' is not a valid data object uri")


def check_dataspace_or_data_object_uri(uri: str) -> str:
    # TODO: Support also data object query uris
    if energistics.uris.DataspaceURI.is_valid_uri(
        uri
    ) or energistics.uris.DataObjectURI.is_valid_uri(uri):
        return uri

    raise ValueError(
        f"Uri '{uri}' is not a valid dataspace uri nor a valid data object uri"
    )
