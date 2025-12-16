from lxml import etree
from xsdata.formats.dataclass.models.generics import DerivedElement
from xsdata.formats.dataclass.parsers import XmlParser

import resqml_objects.v201 as ro_201

xsi_type_key = "{http://www.w3.org/2001/XMLSchema-instance}type"


def parse_resqml_v201_object(
    raw_data: bytes,
) -> ro_201.AbstractObject:
    parser = XmlParser()

    xml_obj = etree.fromstring(raw_data)
    obj_type = xml_obj.get(xsi_type_key) or etree.QName(xml_obj.tag).localname

    if ":" in obj_type:
        obj_type = obj_type.split(":")[1]

    parsed_obj = parser.from_bytes(raw_data, getattr(ro_201, obj_type))

    return (
        parsed_obj if not isinstance(parsed_obj, DerivedElement) else parsed_obj.value
    )
