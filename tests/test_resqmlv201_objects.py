import dataclasses
import datetime
import typing

from lxml import etree

import resqml_objects.v201 as ro
from pyetp._version import version
from resqml_objects.parsers import parse_resqml_v201_object
from resqml_objects.serializers import serialize_resqml_v201_object

R = typing.TypeVar(
    "R", bound=ro.AbstractObject | ro.AbstractObject_1 | ro.AbstractObject_Type
)


def compare_serialization_parsing_roundtrip(obj: R) -> tuple[R, bytes]:
    obj_b = serialize_resqml_v201_object(obj)
    ret_obj = parse_resqml_v201_object(obj_b)

    assert obj == ret_obj

    return ret_obj, obj_b


def test_default_citation() -> None:
    cit = ro.Citation(title="foo", originator="pyetp-tester")
    assert cit.format == f"equinor:pyetp:{version}"

    _, _ = compare_serialization_parsing_roundtrip(cit)

    now = datetime.datetime.now()
    cit = ro.Citation(title="foo", originator="pyetp-tester", creation=now)

    assert cit.creation.to_datetime() == now


def test_default_hdf5_epc_external_part_reference() -> None:
    cit = ro.Citation(title="foo", originator="pyetp-tester")

    epc = ro.EpcExternalPartReference(citation=cit)

    assert epc.schema_version == "2.0"
    assert epc.mime_type == "application/x-hdf5"

    obj_epc = ro.obj_EpcExternalPartReference(citation=cit, uuid=epc.uuid)

    assert dataclasses.asdict(epc) == dataclasses.asdict(obj_epc)

    ret_epc, epc_b_ = compare_serialization_parsing_roundtrip(epc)
    ret_obj_epc, obj_epc_b = compare_serialization_parsing_roundtrip(obj_epc)

    xml_obj = etree.fromstring(obj_epc_b)
    xsi_type_key = "{http://www.w3.org/2001/XMLSchema-instance}type"

    assert xml_obj.get(xsi_type_key) == "eml:obj_EpcExternalPartReference"


def test_timestamp() -> None:
    now = datetime.datetime.now()
    timestamp = ro.Timestamp(date_time=now)

    assert timestamp.date_time.to_datetime() == now
    assert timestamp.year_offset is None

    _, _ = compare_serialization_parsing_roundtrip(timestamp)


def test_datetime() -> None:
    now = datetime.datetime.now()
    dt = ro.DateTime(value=now)

    assert dt.value.to_datetime() == now

    _, _ = compare_serialization_parsing_roundtrip(dt)


def test_local_depth_3d_crs() -> None:
    cit = ro.Citation(title="foo CRS", originator="pyetp-tester")

    crs = ro.obj_LocalDepth3dCrs(
        citation=cit,
        vertical_crs=ro.VerticalCrsEpsgCode(epsg_code=1234),
        projected_crs=ro.ProjectedCrsEpsgCode(epsg_code=23031),
    )

    assert crs.schema_version == "2.0.1"

    ret_crs, crs_b = compare_serialization_parsing_roundtrip(crs)

    assert crs == ret_crs

    crs2 = ro.LocalDepth3dCrs(
        citation=cit,
        uuid=crs.uuid,
        vertical_crs=crs.vertical_crs,
        projected_crs=crs.projected_crs,
    )

    _, _ = compare_serialization_parsing_roundtrip(crs2)

    assert dataclasses.asdict(crs) == dataclasses.asdict(crs2)
