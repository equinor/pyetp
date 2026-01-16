import dataclasses
import datetime
import typing

import numpy as np
from lxml import etree

import resqml_objects.v201 as ro
from pyetp._version import version
from resqml_objects.parsers import parse_resqml_v201_object
from resqml_objects.serializers import serialize_resqml_v201_object
from resqml_objects.surface_helpers import RegularGridParameters

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
        ret_gri.grid2d_patch.slowest_axis_count,
        ret_gri.grid2d_patch.fastest_axis_count,
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


def test_rotated_regular_grid_2d_representation() -> None:
    # Here we compare the results of a rotated surface in three different
    # CRS's:
    #
    #  1. The surface in the global CRS.
    #  2. The surface in a rotated and shifted local CRS.
    #  3. The surface in a surface aligned local CRS.
    #
    # All three cases should return the same surface when constructing the grid
    # in the global CRS.

    shape = tuple(np.random.randint(10, 123, size=2).tolist())

    x = np.linspace(0, 1, shape[0])
    y = np.linspace(0, 2, shape[1])

    origin = 2 * 20 * (np.random.random(2) - 0.5)
    spacing = np.array([x[1] - x[0], y[1] - y[0]])

    grid_angle = 2 * np.pi * (np.random.random() - 0.5)

    grid_unit_vectors = RegularGridParameters.angle_to_unit_vectors(grid_angle)

    crs_angle = 2 * np.pi * (np.random.random() - 0.5)
    crs_offset = 2 * 10 * (np.random.random(2) - 0.5)

    rt_origin = origin - crs_offset
    rt_angle = grid_angle - crs_angle

    rt_grid_unit_vectors = RegularGridParameters.angle_to_unit_vectors(rt_angle)

    # Case 1: An untransformed local CRS on top of the global CRS.
    uu_crs = ro.obj_LocalDepth3dCrs(
        citation=ro.Citation(
            title="Unrotated and untranslated CRS", originator="pyetp-tester"
        ),
        vertical_crs=ro.VerticalUnknownCrs(unknown="MSL"),
        projected_crs=ro.ProjectedCrsEpsgCode(epsg_code=12345),
    )

    # Case 2: An arbitrarily transformed local CRS on top of the global CRS.
    rt_crs = ro.obj_LocalDepth3dCrs(
        citation=ro.Citation(
            title="Rotated and translated CRS", originator="pyetp-tester"
        ),
        vertical_crs=ro.VerticalUnknownCrs(unknown="MSL"),
        projected_crs=ro.ProjectedCrsEpsgCode(epsg_code=12345),
        areal_rotation=ro.PlaneAngleMeasure(
            value=crs_angle,
            uom=ro.PlaneAngleUom.RAD,
        ),
        xoffset=float(crs_offset[0]),
        yoffset=float(crs_offset[1]),
    )

    # Case 3: A surface-aligned local CRS on top of the global CRS.
    aligned_crs = ro.obj_LocalDepth3dCrs(
        citation=ro.Citation(title="Surface-aligned CRS", originator="pyetp-tester"),
        vertical_crs=ro.VerticalUnknownCrs(unknown="MSL"),
        projected_crs=ro.ProjectedCrsEpsgCode(epsg_code=12345),
        areal_rotation=ro.PlaneAngleMeasure(
            value=grid_angle,
            uom=ro.PlaneAngleUom.RAD,
        ),
        xoffset=float(origin[0]),
        yoffset=float(origin[1]),
    )

    # We share the same epc-object across all grids.
    epc = ro.obj_EpcExternalPartReference(
        citation=ro.Citation(title="Grid epc", originator="pyetp-tester"),
    )

    uu_gri = ro.obj_Grid2dRepresentation.from_regular_surface(
        citation=ro.Citation(
            title="Grid in untransformed CRS", originator="pyetp-tester"
        ),
        crs=uu_crs,
        epc_external_part_reference=epc,
        shape=shape,
        origin=origin,
        spacing=spacing,
        unit_vec_1=grid_unit_vectors[:, 0],
        unit_vec_2=grid_unit_vectors[:, 1],
    )

    rt_gri = ro.obj_Grid2dRepresentation.from_regular_surface(
        citation=ro.Citation(
            title="Grid in transformed local CRS", originator="pyetp-tester"
        ),
        crs=rt_crs,
        epc_external_part_reference=epc,
        shape=shape,
        origin=rt_origin,
        spacing=spacing,
        unit_vec_1=rt_grid_unit_vectors[:, 0],
        unit_vec_2=rt_grid_unit_vectors[:, 1],
    )

    aligned_gri = ro.obj_Grid2dRepresentation.from_regular_surface(
        citation=ro.Citation(
            title="Grid in surface-aligned local CRS", originator="pyetp-tester"
        ),
        crs=aligned_crs,
        epc_external_part_reference=epc,
        shape=shape,
        origin=np.zeros_like(origin),
        spacing=spacing,
        unit_vec_1=np.array([1.0, 0.0]),
        unit_vec_2=np.array([0.0, 1.0]),
    )

    dor = ro.DataObjectReference.from_object(epc)

    assert uu_gri.grid2d_patch.geometry.points.zvalues.values.hdf_proxy == dor
    assert rt_gri.grid2d_patch.geometry.points.zvalues.values.hdf_proxy == dor
    assert aligned_gri.grid2d_patch.geometry.points.zvalues.values.hdf_proxy == dor

    # Check that all grids evaulate to the same global points, irrespective of
    # their local CRS.
    uu_X, uu_Y = uu_gri.get_xy_grid(crs=uu_crs)
    rt_X, rt_Y = rt_gri.get_xy_grid(crs=rt_crs)
    aligned_X, aligned_Y = aligned_gri.get_xy_grid(crs=aligned_crs)

    np.testing.assert_allclose(uu_X, rt_X)
    np.testing.assert_allclose(uu_X, aligned_X)
    np.testing.assert_allclose(uu_Y, rt_Y)
    np.testing.assert_allclose(uu_Y, aligned_Y)

    # Test that the untransformed crs does not alter the grid.
    _uu_X, _uu_Y = uu_gri.get_xy_grid()
    np.testing.assert_equal(uu_X, _uu_X)
    np.testing.assert_equal(uu_Y, _uu_Y)

    # Test that the grids from the transformed local CRS (in the local
    # reference system) have the expected origin and angle on the surface.
    rt_X, rt_Y = rt_gri.get_xy_grid()

    np.testing.assert_equal(np.array([rt_X[0, 0], rt_Y[0, 0]]), rt_origin)

    vec_1 = np.array([rt_X[1, 0] - rt_X[0, 0], rt_Y[1, 0] - rt_Y[0, 0]])
    vec_2 = np.array([rt_X[0, 1] - rt_X[0, 0], rt_Y[0, 1] - rt_Y[0, 0]])

    unit_vectors = np.column_stack([vec_1, vec_2])

    tot_angle = rt_angle

    # Force angle to be in the range [-pi, pi]
    if tot_angle < -np.pi:
        tot_angle += 2 * np.pi
    elif tot_angle > np.pi:
        tot_angle -= 2 * np.pi

    np.testing.assert_allclose(
        RegularGridParameters.unit_vectors_to_angle(unit_vectors), tot_angle
    )

    # Test that the grids in the surface-aligned local CRS correspond to
    # meshgrids from the `x` and `y` edges.
    aligned_X, aligned_Y = aligned_gri.get_xy_grid()

    X, Y = np.meshgrid(x, y, indexing="ij")

    np.testing.assert_allclose(aligned_X, X)
    np.testing.assert_allclose(aligned_Y, Y)
