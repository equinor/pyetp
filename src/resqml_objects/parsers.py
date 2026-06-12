import os

from lxml import etree
from xsdata.exceptions import ConverterError
from xsdata.formats.dataclass.models.generics import DerivedElement
from xsdata.formats.dataclass.parsers import XmlParser

import resqml_objects.v201 as ro_201
from resqml_objects.serializers import RO201Obj, RO201SubObj

xsi_type_key = "{http://www.w3.org/2001/XMLSchema-instance}type"
_PATCH_FLAG_ENV = "PYETP_PATCH_MISSING_XSD_NAMESPACE"
_XSD_DECL = b'xmlns:xsd="http://www.w3.org/2001/XMLSchema"'
_XSI_DECL = b'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'


def _patch_missing_xsd_namespace_enabled() -> bool:
    raw = os.environ.get(_PATCH_FLAG_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _inject_xsd_namespace(raw_data: bytes) -> bytes:
    """Inject ``xmlns:xsd="..."`` into the root element by piggy-backing on
    the existing ``xmlns:xsi`` declaration.

    Returns the bytes unchanged when there is nothing to patch (no ``xsi``
    declaration to hook onto, or ``xsd`` is already declared).
    """

    if _XSD_DECL in raw_data:
        return raw_data

    return raw_data.replace(
        _XSI_DECL,
        _XSI_DECL + b" " + _XSD_DECL,
        1,
    )


def _parse(raw_data: bytes) -> RO201Obj | RO201SubObj:
    parser = XmlParser()

    xml_obj = etree.fromstring(raw_data)
    obj_type = xml_obj.get(xsi_type_key) or etree.QName(str(xml_obj.tag)).localname

    if ":" in obj_type:
        obj_type = obj_type.split(":")[1]

    parsed_obj = parser.from_bytes(raw_data, getattr(ro_201, obj_type))
    ret_obj: ro_201.AbstractObject = (
        parsed_obj if not isinstance(parsed_obj, DerivedElement) else parsed_obj.value
    )

    return ret_obj


def parse_resqml_v201_object(raw_data: bytes) -> RO201Obj | RO201SubObj:
    if not _patch_missing_xsd_namespace_enabled():
        return _parse(raw_data)

    try:
        return _parse(raw_data)
    except ConverterError:
        patched = _inject_xsd_namespace(raw_data)
        if patched == raw_data:
            # Nothing to patch — this is something different error, propagate it.
            raise
        return _parse(patched)
