import resqml_objects.v201 as ro

resqml_schema_version = "2.0.1"
common_schema_version = "2.0"


def get_content_type_string(
    obj: ro.AbstractObject,
    resqml_schema_version: str = resqml_schema_version,
    common_schema_version: str = common_schema_version,
) -> str:
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
