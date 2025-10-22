from lxml import etree
from xsdata.formats.dataclass.context import XmlContext
from xsdata.formats.dataclass.parsers import XmlParser

import resqml_objects.v201 as ro_201


def parse_resqml_v201_object(raw_data: bytes) -> ro_201.AbstractObject:
    parser = XmlParser(context=XmlContext())

    obj_name = etree.QName(etree.fromstring(raw_data).tag).localname
    return parser.from_bytes(raw_data, getattr(ro_201, obj_name))
