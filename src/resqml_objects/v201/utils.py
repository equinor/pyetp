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
