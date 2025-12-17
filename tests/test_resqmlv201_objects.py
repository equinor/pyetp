
from lxml import etree

import resqml_objects.v201 as ro
from resqml_objects.parsers import parse_resqml_v201_object
from resqml_objects.serializers import serialize_resqml_v201_object


def test_default_citation() -> None:
    cit = ro.Citation.init_default(title="foo", originator="pyetp-tester")
    cit_b = serialize_resqml_v201_object(cit)
    ret_cit = parse_resqml_v201_object(cit_b)
    assert cit == ret_cit


def test_default_hdf5_epc_external_part_reference() -> None:
    cit = ro.Citation.init_default(title="foo", originator="pyetp-tester")

    epc = ro.EpcExternalPartReference.init_default_hdf5(citation=cit)

    assert epc.schema_version == "2.0"
    assert epc.mime_type == "application/x-hdf5"

    obj_epc = ro.obj_EpcExternalPartReference.init_default_hdf5(citation=cit, uuid=epc.uuid)

    assert epc.schema_version == obj_epc.schema_version
    assert epc.mime_type == obj_epc.mime_type
    assert epc.citation == obj_epc.citation
    assert epc.uuid == obj_epc.uuid

    epc_b = serialize_resqml_v201_object(epc)
    obj_epc_b = serialize_resqml_v201_object(obj_epc)

    xml_obj = etree.fromstring(obj_epc_b)
    xsi_type_key = "{http://www.w3.org/2001/XMLSchema-instance}type"

    assert xml_obj.get(xsi_type_key) == "eml:obj_EpcExternalPartReference"

    ret_epc = parse_resqml_v201_object(epc_b)
    ret_obj_epc = parse_resqml_v201_object(obj_epc_b)

    assert epc == ret_epc
    assert obj_epc == ret_obj_epc
