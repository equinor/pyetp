from etpproto.uri import DataObjectURI as _DataObjectURI
from etpproto.uri import DataspaceUri as _DataspaceUri

import map_api.resqml_objects as ro


class DataspaceUri(_DataspaceUri):

    @classmethod
    def from_name(cls, name: str):
        return cls(f"eml:///dataspace('{name}')")

    def __str__(self):
        return self.raw_uri


class DataObjectURI(_DataObjectURI):

    @classmethod
    def from_parts(cls, duri: DataspaceUri | str, domain_and_version: str, obj_type: str, uuid: str):

        # lets be bit more leanent here - allow for incorrect input in form of dataspace name too
        if isinstance(duri, str) and not duri.startswith('eml://'):
            duri = DataspaceUri.from_name(duri)

        return cls(f"{duri}/{domain_and_version}.{obj_type}({uuid})")

    @classmethod
    def from_obj(cls, dataspace: DataspaceUri | str, obj: ro.AbstractObject):

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

    def __str__(self):
        return self.raw_uri
