import datetime
import uuid

from lxml import etree
from xsdata.models.datatype import XmlDateTime

import resqml_objects.v201 as ro
from resqml_objects.parsers import parse_resqml_v201_object
from resqml_objects.serializers import serialize_resqml_v201_object


def test_epc_external_part_reference() -> None:
    epc = ro.EpcExternalPartReference(
        schema_version="2.0",
        uuid=str(uuid.uuid4()),
        citation=ro.Citation(
            title="foo",
            originator="bar",
            format="baz",
            creation=XmlDateTime.from_datetime(datetime.datetime.now()),
        ),
        mime_type="application/x-hdf5",
    )

    obj_epc = ro.obj_EpcExternalPartReference(
        schema_version="2.0",
        uuid=str(uuid.uuid4()),
        citation=ro.Citation(
            title="foo",
            originator="bar",
            format="baz",
            creation=XmlDateTime.from_datetime(datetime.datetime.now()),
        ),
        mime_type="application/x-hdf5",
    )

    epc_b = serialize_resqml_v201_object(epc)
    obj_epc_b = serialize_resqml_v201_object(obj_epc)

    xml_obj = etree.fromstring(obj_epc_b)
    xsi_type_key = "{http://www.w3.org/2001/XMLSchema-instance}type"

    assert xml_obj.get(xsi_type_key) == "eml:obj_EpcExternalPartReference"

    ret_epc = parse_resqml_v201_object(epc_b)
    ret_obj_epc = parse_resqml_v201_object(obj_epc_b)

    assert epc == ret_epc
    assert obj_epc == ret_obj_epc
