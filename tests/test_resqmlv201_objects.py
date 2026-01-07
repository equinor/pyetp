import dataclasses
import datetime
import typing

import numpy as np
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

    dor = ro.DataObjectReference.from_object(epc)
    obj_dor = ro.DataObjectReference.from_object(obj_epc)

    assert dor.uuid == epc.uuid == obj_dor.uuid
    assert dor.title == epc.citation.title == obj_dor.title
    assert (
        dor.content_type
        == "application/x-eml+xml;version=2.0;type=EpcExternalPartReference"
    )
    assert (
        obj_dor.content_type
        == "application/x-eml+xml;version=2.0;type=obj_EpcExternalPartReference"
    )

    dataspace_path = "bar/baz"

    assert dor.get_etp_data_object_uri(dataspace_path) == (
        f"eml:///dataspace('{dataspace_path}')/eml20.EpcExternalPartReference("
        f"{dor.uuid})"
    )
    assert obj_dor.get_etp_data_object_uri(dataspace_path) == (
        f"eml:///dataspace('{dataspace_path}')/eml20.obj_EpcExternalPartReference("
        f"{obj_dor.uuid})"
    )


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


def test_regular_grid_2d_representation() -> None:
    shape = tuple(np.random.randint(10, 123, size=2).tolist())

    x = np.linspace(0, 1, shape[0])
    y = np.linspace(1, 2, shape[1])

    origin = np.array([x[0], y[0]])
    spacing = np.array([x[1] - x[0], y[1] - y[0]])
    unit_vectors = np.eye(2)

    crs = ro.obj_LocalDepth3dCrs(
        citation=ro.Citation(title="Grid CRS", originator="pyetp-tester"),
        vertical_crs=ro.VerticalCrsEpsgCode(epsg_code=1234),
        projected_crs=ro.ProjectedCrsEpsgCode(epsg_code=23031),
    )

    epc = ro.obj_EpcExternalPartReference(
        citation=ro.Citation(title="Grid epc", originator="pyetp-tester"),
    )

    gri = ro.obj_Grid2dRepresentation.from_regular_surface(
        citation=ro.Citation(title="Grid", originator="pyetp-tester"),
        crs=crs,
        epc_external_part_reference=epc,
        shape=shape,
        origin=origin,
        spacing=spacing,
        unit_vec_1=unit_vectors[:, 0],
        unit_vec_2=unit_vectors[:, 1],
    )

    ret_gri, _ = compare_serialization_parsing_roundtrip(gri)

    dor = ro.DataObjectReference.from_object(gri)
    assert (
        dor.content_type
        == "application/x-resqml+xml;version=2.0.1;type=obj_Grid2dRepresentation"
    )
    assert dor.uuid == gri.uuid
    assert dor.title == gri.citation.title
    assert (
        dor.get_etp_data_object_uri("")
        == f"eml:///resqml20.obj_Grid2dRepresentation({gri.uuid})"
    )

    ret_shape = (
        ret_gri.grid2d_patch.fastest_axis_count,
        ret_gri.grid2d_patch.slowest_axis_count,
    )

    assert ret_shape == shape

    sg = ret_gri.grid2d_patch.geometry.points.supporting_geometry

    ret_origin = np.array(
        [
            sg.origin.coordinate1,
            sg.origin.coordinate2,
            sg.origin.coordinate3,
        ],
    )

    np.testing.assert_equal(ret_origin[:2], origin)
    np.testing.assert_equal(ret_origin[2], 0.0)

    ret_spacing = np.array(
        [
            sg.offset[0].spacing.value,
            sg.offset[1].spacing.value,
        ]
    )

    np.testing.assert_equal(ret_spacing, spacing)

    ret_spacing_count = (
        sg.offset[0].spacing.count,
        sg.offset[1].spacing.count,
    )

    assert tuple(rsc + 1 for rsc in ret_spacing_count) == shape

    assert sg.offset[0].offset.coordinate3 == 0.0
    assert sg.offset[1].offset.coordinate3 == 0.0

    ret_unit_vectors = np.array(
        [
            [sg.offset[0].offset.coordinate1, sg.offset[1].offset.coordinate1],
            [sg.offset[0].offset.coordinate2, sg.offset[1].offset.coordinate2],
        ],
    )

    np.testing.assert_equal(ret_unit_vectors, unit_vectors)

    X, Y = gri.get_xy_grid()

    assert X.shape == shape == Y.shape

    _X, _Y = np.meshgrid(x, y, indexing="ij")
    np.testing.assert_allclose(X, _X)
    np.testing.assert_allclose(Y, _Y)

    _X, _Y = gri.get_xy_grid(crs=crs)

    np.testing.assert_allclose(X, _X)
    np.testing.assert_allclose(Y, _Y)


        ),
        (
            sg.offset[1].offset.coordinate1,
            sg.offset[1].offset.coordinate2,
            sg.offset[1].offset.coordinate3,
        ),
    )

    assert ret_unit_vectors == unit_vectors


def test_rotated_regular_grid_2d_representation() -> None:
    pass


def test_double_rotated_regular_grid_2d_representation() -> None:
    # This test should test a grid that is rotated in both the CRS and the
    # grid.
    pass
