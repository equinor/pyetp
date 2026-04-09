import typing
from dataclasses import fields

import resqml_objects.v201 as ro

resqml_schema_version = "2.0.1"
common_schema_version = "2.0"


def find_hdf5_datasets(
    obj: ro.AbstractCitedDataObject,
) -> list[ro.Hdf5Dataset]:
    return _find_hdf5_datasets(obj)


def _find_hdf5_datasets(obj: typing.Any) -> list[ro.Hdf5Dataset]:
    hds = []

    if isinstance(obj, list):
        for _obj in obj:
            hds.extend(_find_hdf5_datasets(_obj))
        return hds

    try:
        _fields = fields(obj)
    except TypeError:
        return hds

    for f in _fields:
        if isinstance(getattr(obj, f.name), ro.Hdf5Dataset):
            hds.append(getattr(obj, f.name))
        else:
            hds.extend(_find_hdf5_datasets(getattr(obj, f.name)))

    return hds


def find_data_object_references(
    obj: ro.AbstractCitedDataObject,
) -> list[ro.DataObjectReference]:
    return _find_data_object_references(obj)


def _find_data_object_references(obj: typing.Any) -> list[ro.DataObjectReference]:
    dors = []

    if isinstance(obj, list):
        for _obj in obj:
            dors.extend(_find_data_object_references(_obj))
        return dors

    try:
        _fields = fields(obj)
    except TypeError:
        return dors

    for f in _fields:
        if isinstance(getattr(obj, f.name), ro.DataObjectReference):
            dors.append(getattr(obj, f.name))
        else:
            dors.extend(_find_data_object_references(getattr(obj, f.name)))

    return dors


def replace_data_object_references(
    obj: ro.AbstractCitedDataObject,
    uuid_to_obj: dict[str, ro.AbstractCitedDataObject],
) -> None:
    """Recursively walk *obj* and replace ``DataObjectReference`` fields
    in-place when a matching UUID is found in *uuid_to_obj*."""
    _replace_data_object_references(obj, uuid_to_obj)


def _replace_data_object_references(
    obj: typing.Any,
    uuid_to_obj: dict[str, ro.AbstractCitedDataObject],
) -> None:
    if isinstance(obj, list):
        for item in obj:
            _replace_data_object_references(item, uuid_to_obj)
        return

    try:
        _fields = fields(obj)
    except TypeError:
        return

    for f in _fields:
        value = getattr(obj, f.name)
        if isinstance(value, ro.DataObjectReference):
            if value.uuid in uuid_to_obj:
                setattr(obj, f.name, uuid_to_obj[value.uuid])
        else:
            _replace_data_object_references(value, uuid_to_obj)
