from uuid import UUID

from etpproto.uri import DataObjectURI as _DataObjectURI
from etpproto.uri import DataspaceUri as _DataspaceURI
from pydantic import BaseConfig

import map_api.resqml_objects as ro


class _Mixin:
    # print full url as default
    raw_uri: str

    def __str__(self):
        return self.raw_uri

    def __hash__(self):
        return hash(self.raw_uri)

    # pydantic validators

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="url", example="eml:///dataspace('my/space')/resqml20.ObjectName(5fe90ad4-6d34-4f73-a72d-992b26f8442e)")

    @classmethod
    def __get_validators__(cls):
        yield cls.__validate__

    @classmethod
    def __validate__(cls, v):
        if isinstance(v, cls):
            return v
        try:
            return cls(v)
        except AttributeError:
            raise TypeError(f"URL '{v}' is not a valid {cls.__name__}")


class DataspaceURI(_DataspaceURI, _Mixin):

    @classmethod
    def from_name(cls, name: str):
        return cls(f"eml:///dataspace('{name}')")

    @classmethod
    def from_any(cls, v):
        if isinstance(v, DataspaceURI):
            return v
        if isinstance(v, DataObjectURI):
            return cls.from_name(v.dataspace)
        if isinstance(v, str):
            return cls(v) if v.startswith('eml://') else cls.from_name(v)

        raise TypeError(f"Type {type(v)} not supported dataspace uri")


class DataObjectURI(_DataObjectURI, _Mixin):

    @classmethod
    def from_parts(cls, duri: DataspaceURI | str, domain_and_version: str, obj_type: str, uuid: UUID | str):
        duri = DataspaceURI.from_any(duri)
        return cls(f"{duri}/{domain_and_version}.{obj_type}({uuid})")

    @classmethod
    def from_obj(cls, dataspace: DataspaceURI | str, obj: ro.AbstractObject):

        objname = obj.__class__.__name__
        namespace: str = getattr(obj.Meta, 'namespace', None) or getattr(obj.Meta, 'target_namespace')
        namespace = namespace.lower()

        # TODO: we can rather look at citation.format - which could be used for xmlformat ? - however to be backward capatiable we check namespaces instead
        if namespace.endswith('resqmlv2'):
            domain = "resqml20"
        elif namespace.endswith('data/commonv2'):
            domain = "eml20"
        else:
            raise TypeError(f"Could not parse domain from namespace ({namespace})")

        return cls.from_parts(dataspace, domain, objname, obj.uuid)


BaseConfig.json_encoders = {
    DataspaceURI: lambda v: str(v),
    DataObjectURI: lambda v: str(v),
    **BaseConfig.json_encoders
}
