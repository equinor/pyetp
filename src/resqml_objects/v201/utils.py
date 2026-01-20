import typing
import warnings
from dataclasses import fields

import resqml_objects.v201 as ro

resqml_schema_version = "2.0.1"
common_schema_version = "2.0"


def get_content_type_string(
    obj: ro.AbstractObject,
    resqml_schema_version: str = resqml_schema_version,
    common_schema_version: str = common_schema_version,
) -> str:
    warnings.warn(
        "The 'get_content_type_string'-function is deprecated and will be removed in "
        "a future version of pyetp. Either use it directly from "
        "'DataObjectReference.get_content_type_string', or let 'DataObjectReference' "
        "handle it where it is needed (under 'resqml_object.v201')."
    )
    # See Energistics Identifier Specification 4.0 (it is downloaded alongside
    # the RESQML v2.0.1 standard) section 4.1 for an explanation on the format
    # of content_type.

    namespace = getattr(obj.Meta, "namespace", None) or getattr(
        obj.Meta, "target_namespace"
    )

    if namespace == "http://www.energistics.org/energyml/data/resqmlv2":
        return (
            f"application/x-resqml+xml;version={resqml_schema_version};"
            f"type={obj.__class__.__name__}"
        )
    elif namespace == "http://www.energistics.org/energyml/data/commonv2":
        return (
            f"application/x-eml+xml;version={common_schema_version};"
            f"type={obj.__class__.__name__}"
        )

    raise NotImplementedError(
        f"Namespace {namespace} from object {obj} is not supported"
    )


def get_data_object_reference(
    obj: ro.AbstractCitedDataObject,
) -> ro.DataObjectReference:
    content_type = get_content_type_string(obj)

    return ro.DataObjectReference(
        content_type=content_type,
        title=obj.citation.title,
        uuid=obj.uuid,
        version_string=obj.citation.version_string,
    )


def find_hdf5_datasets(
    obj: ro.AbstractCitedDataObject,
) -> list[ro.Hdf5Dataset]:
    return _find_hdf5_datasets(obj)


def _find_hdf5_datasets(obj: typing.Any) -> list[ro.Hdf5Dataset]:
    hds = []

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


def get_qualified_type(
    data_object: typing.Type[ro.AbstractCitedDataObject] | ro.AbstractCitedDataObject,
) -> str:
    # Get class object instead of the instance.
    if type(data_object) is not type:
        data_object = type(data_object)

    namespace = getattr(data_object.Meta, "namespace", None) or getattr(
        data_object.Meta, "target_namespace"
    )

    if namespace == "http://www.energistics.org/energyml/data/resqmlv2":
        return f"resqml20.{data_object.__name__}"
    elif namespace == "http://www.energistics.org/energyml/data/commonv2":
        return f"eml20.{data_object.__name__}"

    raise NotImplementedError(
        f"Namespace {namespace} from object {data_object} is not supported"
    )
