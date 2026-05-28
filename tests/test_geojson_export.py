from __future__ import annotations

import uuid as uuid_lib
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import resqml_objects.v201 as ro


def polyline_set_to_geojson(
    polyline_set: ro.obj_PolylineSetRepresentation,
    arrays: dict[str, npt.NDArray[Any]],
    crs: ro.AbstractLocal3dCrs | None = None,
    *,
    extra_properties: dict[str, Any] | None = None,
    name: str | None = None,
) -> Any:
    return polyline_set.get_geojson(
        arrays, crs, extra_properties=extra_properties, name=name
    )


def get_random_crs(
    *,
    domain: str = "depth",
    projected_crs: ro.AbstractProjectedCrs | None = None,
    vertical_crs: ro.AbstractVerticalCrs | None = None,
    title: str = "Random fault CRS",
    extra_metadata: list[ro.NameValuePair] | None = None,
) -> ro.AbstractLocal3dCrs:
    """Build a LocalDepth3dCrs (default) or LocalTime3dCrs for tests.

    All keyword arguments optional — override them when a
    test needs a specific CRS shape (e.g. EPSG-in-extra-metadata).
    """

    base_kwargs: dict[str, Any] = dict(
        citation=ro.Citation(title=title, originator="geojson-tester"),
        projected_crs=projected_crs
        or ro.ProjectedUnknownCrs(unknown="No EPSG code specified"),
        vertical_crs=vertical_crs
        or ro.VerticalUnknownCrs(unknown="No EPSG code specified"),
        vertical_uom=ro.LengthUom.M,
    )
    if extra_metadata is not None:
        base_kwargs["extra_metadata"] = extra_metadata

    if domain == "depth":
        return ro.obj_LocalDepth3dCrs(**base_kwargs)
    return ro.obj_LocalTime3dCrs(**base_kwargs, time_uom=ro.TimeUom.MS)


def get_random_polyline_set(
    *,
    crs: ro.AbstractLocal3dCrs | None = None,
    node_counts: list[int] | None = None,
    closed: bool | list[bool] = False,
    points: npt.NDArray[np.float64] | None = None,
    title: str = "Random fault polyline-set",
) -> tuple[
    ro.AbstractLocal3dCrs,
    ro.obj_EpcExternalPartReference,
    ro.obj_PolylineSetRepresentation,
    dict[str, npt.NDArray[Any]],
]:
    """Build a random synthetic polyline-set for tests.

    All keyword arguments optional:

    * ``crs``         — pass to use a specific CRS shape; defaults to a fresh
                       epsg-proper CRS.
    * ``node_counts`` — list giving the size of each polyline; defaults to a
                       random number of polylines, all the same random length.
    * ``closed``      — single bool for all, or a per-polyline list; defaults
                       to all-False (typical fault sticks).
    * ``points``      — pre-built ``(N_total, 3)`` array; defaults to random.
                       Pass when an assertion needs deterministic coordinates.
    * ``title``       — citation title of the polyline-set.

    Returns ``(crs, epc, polyline_set, arrays)``.
    """
    # Decide polyline counts.
    if node_counts is None:
        n_polylines = int(np.random.randint(2, 6))
        nodes_per = int(np.random.randint(3, 8))
        node_counts_arr = np.full(n_polylines, nodes_per, dtype=np.int64)
    else:
        node_counts_arr = np.asarray(node_counts, dtype=np.int64)
        n_polylines = int(node_counts_arr.shape[0])
    n_total = int(node_counts_arr.sum())

    # Decide points.
    if points is None:
        points = np.random.rand(n_total, 3) * 1000.0
    else:
        assert points.shape == (n_total, 3), (
            f"points shape {points.shape} doesn't match node_counts sum {n_total}"
        )

    # Decide closed flags.
    if isinstance(closed, bool):
        closed_arr = np.full(n_polylines, closed, dtype=np.bool_)
    else:
        closed_arr = np.asarray(closed, dtype=np.bool_)
        assert closed_arr.shape == (n_polylines,)

    # Default CRS: epsg-proper.
    if crs is None:
        crs = get_random_crs(
            projected_crs=ro.ProjectedCrsEpsgCode(epsg_code=23031),
            vertical_crs=ro.VerticalCrsEpsgCode(epsg_code=5715),
        )

    epc = ro.obj_EpcExternalPartReference(
        citation=ro.Citation(title="Random epc", originator="geojson-tester"),
    )

    arrays: dict[str, npt.NDArray[Any]] = {}

    # node_count_per_polyline: use ConstantArray when all values agree, else HDF5.
    if len(set(node_counts_arr.tolist())) == 1:
        node_count_array: ro.AbstractIntegerArray = ro.IntegerConstantArray(
            value=int(node_counts_arr[0]), count=n_polylines
        )
    else:
        nc_path = f"/RESQML/{uuid_lib.uuid4()}/node_counts"
        node_count_array = ro.IntegerHdf5Array(
            null_value=-1,
            values=ro.Hdf5Dataset(
                path_in_hdf_file=nc_path,
                hdf_proxy=ro.DataObjectReference.from_object(epc),
            ),
        )
        arrays[nc_path] = node_counts_arr

    # closed_polylines: use ConstantArray when all values agree, else HDF5.
    if len(set(closed_arr.tolist())) == 1:
        closed_array: ro.AbstractBooleanArray = ro.BooleanConstantArray(
            value=bool(closed_arr[0]), count=n_polylines
        )
    else:
        cl_path = f"/RESQML/{uuid_lib.uuid4()}/closed"
        closed_array = ro.BooleanHdf5Array(
            values=ro.Hdf5Dataset(
                path_in_hdf_file=cl_path,
                hdf_proxy=ro.DataObjectReference.from_object(epc),
            ),
        )
        arrays[cl_path] = closed_arr

    # Build the patch.
    points_path = f"/RESQML/{uuid_lib.uuid4()}/points"
    patch = ro.PolylineSetPatch(
        patch_index=0,
        closed_polylines=closed_array,
        node_count_per_polyline=node_count_array,
        geometry=ro.PointGeometry(
            local_crs=ro.DataObjectReference.from_object(crs),
            points=ro.Point3dHdf5Array(
                coordinates=ro.Hdf5Dataset(
                    path_in_hdf_file=points_path,
                    hdf_proxy=ro.DataObjectReference.from_object(epc),
                ),
            ),
        ),
    )

    pls = ro.obj_PolylineSetRepresentation(
        citation=ro.Citation(title=title, originator="geojson-tester"),
        line_role=ro.LineRole.INTERPRETATION_LINE,
        line_patch=[patch],
    )
    arrays[points_path] = points

    return crs, epc, pls, arrays


def test_resolve_crs_epsg_proper() -> None:
    crs = get_random_crs(
        projected_crs=ro.ProjectedCrsEpsgCode(epsg_code=23031),
        vertical_crs=ro.VerticalCrsEpsgCode(epsg_code=5715),
    )
    info = crs._resolve_crs()
    assert info["projected_epsg_code"] == 23031
    assert info["vertical_epsg_code"] == 5715
    assert info["source"] == "projected_crs"
    assert info["name"] == "urn:ogc:def:crs:EPSG::23031"
    assert info["z_domain"] == "depth"


# Synthetic CRS identifiers used throughout the variant tests below.
# They preserve the structural patterns the regexes need (P<n>_T<n>,
# BoundProjected:EPSG::P_EPSG::V, PROJCS["…"])
_TEST_PROJ_EPSG = 32633
_TEST_VERT_EPSG = 5773
_TEST_CRS_TITLE_WITH_PT = f"TEST_CRS_P{_TEST_PROJ_EPSG}_T{_TEST_VERT_EPSG}"
_TEST_BOUND_PROJECTED = (
    f"BoundProjected:EPSG::{_TEST_PROJ_EPSG}_EPSG::{_TEST_VERT_EPSG}"
)


def test_resolve_crs_epsg_in_extra_metadata() -> None:
    """ProjectedUnknownCrs + extra_metadata carrying a
    ``BoundProjected:EPSG::P_EPSG::V`` string — proper CRS fields are
    unknown, but the EPSG codes are recoverable from the extra_metadata.
    """
    crs = get_random_crs(
        domain="time",
        title=_TEST_CRS_TITLE_WITH_PT,
        extra_metadata=[ro.NameValuePair(name="crs", value=_TEST_BOUND_PROJECTED)],
    )
    info = crs._resolve_crs()
    assert info["projected_epsg_code"] == _TEST_PROJ_EPSG
    assert info["vertical_epsg_code"] == _TEST_VERT_EPSG
    assert info["source"] == "extra_metadata"
    assert info["name"] == f"urn:ogc:def:crs:EPSG::{_TEST_PROJ_EPSG}"
    assert info["z_domain"] == "time"


def test_resolve_crs_epsg_only_in_citation_title() -> None:
    """If extra_metadata is absent but the citation title carries the
    ``P<digits>_T<digits>`` pattern, we should still recover it.
    """
    crs = get_random_crs(title=_TEST_CRS_TITLE_WITH_PT)
    info = crs._resolve_crs()
    assert info["projected_epsg_code"] == _TEST_PROJ_EPSG
    assert info["vertical_epsg_code"] == _TEST_VERT_EPSG
    assert info["source"] == "citation_title"


def test_resolve_crs_wkt_string() -> None:
    """WKT supplied as the projected_crs.unknown text."""
    wkt = (
        'PROJCS["Test Projected CRS",GEOGCS["Test Geographic CRS",'
        'DATUM["Test_Datum",SPHEROID["Test Spheroid",6378137,298.257223563]]],'
        'PROJECTION["Transverse_Mercator"],UNIT["metre",1]]'
    )
    crs = get_random_crs(projected_crs=ro.ProjectedUnknownCrs(unknown=wkt))
    info = crs._resolve_crs()
    assert info["wkt"] == wkt
    assert info["source"] == "projected_crs.unknown"
    assert info["name"] == "Test Projected CRS"


def test_resolve_crs_wkt_in_extra_metadata_captures_just_value() -> None:
    """When WKT is in extra_metadata alongside unrelated entries (project
    metadata, ingest tool tags, etc.), the captured ``wkt`` must be just
    the value of the WKT entry — no leading junk from the joined blob.
    """
    wkt = (
        'PROJCS["Test UTM Zone",GEOGCS["Test Geo CRS",'
        'DATUM["Test_Datum",SPHEROID["Test Spheroid",6378137,298.257223563]]],'
        'PROJECTION["Transverse_Mercator"],UNIT["metre",1]]'
    )
    unrelated_value = str(uuid_lib.uuid4())
    crs = get_random_crs(
        projected_crs=ro.ProjectedUnknownCrs(unknown="WKT"),
        extra_metadata=[
            # Unrelated metadata entry — the extractor must ignore it.
            ro.NameValuePair(name="unrelated_metadata", value=unrelated_value),
            # The real WKT entry — the extractor must pick THIS value.
            ro.NameValuePair(name="projected_crs_wkt", value=wkt),
        ],
    )
    info = crs._resolve_crs()
    assert info["wkt"] == wkt
    assert "unrelated_metadata" not in info["wkt"]
    assert unrelated_value not in info["wkt"]
    assert info["source"] == "extra_metadata"
    assert info["name"] == "Test UTM Zone"


def test_resolve_crs_ow_style_name() -> None:
    """ProjectedUnknownCrs whose ``unknown`` holds a name-like string
    with no EPSG digits or WKT pattern. We just surface it as
    ``unknown_name``.
    """
    crs = get_random_crs(
        projected_crs=ro.ProjectedUnknownCrs(unknown="TEST_CRS_NAME_OW_STYLE"),
        title="some-crs",
    )
    info = crs._resolve_crs()
    assert info["unknown_name"] == "TEST_CRS_NAME_OW_STYLE"
    assert info["name"] == "TEST_CRS_NAME_OW_STYLE"


# -----------------------------------------------------------------------------
# Naive helpers — get_epsg_code / get_wkt / is_time_domain / is_depth_domain
# -----------------------------------------------------------------------------


def test_get_epsg_code_epsg_proper() -> None:
    crs = get_random_crs(projected_crs=ro.ProjectedCrsEpsgCode(epsg_code=23031))
    assert crs.get_epsg_code() == 23031


def test_get_epsg_code_bound_projected_in_extra_metadata() -> None:
    crs = get_random_crs(
        extra_metadata=[ro.NameValuePair(name="crs", value=_TEST_BOUND_PROJECTED)],
    )
    assert crs.get_epsg_code() == _TEST_PROJ_EPSG


def test_get_epsg_code_from_citation_title_pattern() -> None:
    crs = get_random_crs(title=_TEST_CRS_TITLE_WITH_PT)
    assert crs.get_epsg_code() == _TEST_PROJ_EPSG


def test_get_epsg_code_none_when_truly_unknown() -> None:
    crs = get_random_crs()  # ProjectedUnknownCrs, default title, no extras
    assert crs.get_epsg_code() is None


def test_get_wkt_from_projected_unknown() -> None:
    wkt = 'PROJCS["TestCRS",GEOGCS["g",DATUM["d",SPHEROID["s",6378137,298.257]]]]'
    crs = get_random_crs(projected_crs=ro.ProjectedUnknownCrs(unknown=wkt))
    assert crs.get_wkt() == wkt


def test_get_wkt_from_extra_metadata() -> None:
    wkt = 'PROJCS["Other",GEOGCS["g",DATUM["d",SPHEROID["s",6378137,298.257]]]]'
    crs = get_random_crs(
        extra_metadata=[
            ro.NameValuePair(name="unrelated", value="just a tag"),
            ro.NameValuePair(name="projected_crs_wkt", value=wkt),
        ],
    )
    assert crs.get_wkt() == wkt


def test_get_wkt_none_when_only_epsg_code() -> None:
    """Naive accessor: do NOT synthesise WKT from an EPSG code."""
    crs = get_random_crs(projected_crs=ro.ProjectedCrsEpsgCode(epsg_code=23031))
    assert crs.get_epsg_code() == 23031
    assert crs.get_wkt() is None


def test_is_time_and_depth_domain_are_exclusive() -> None:
    depth = get_random_crs(domain="depth")
    time = get_random_crs(domain="time")

    assert depth.is_depth_domain() is True
    assert depth.is_time_domain() is False

    assert time.is_time_domain() is True
    assert time.is_depth_domain() is False


def test_polyline_set_single_polyline_no_crs() -> None:
    _, _, pls, arrays = get_random_polyline_set(node_counts=[5])
    fc = polyline_set_to_geojson(pls, arrays, crs=None)

    assert fc["type"] == "FeatureCollection"
    assert "crs" not in fc
    assert fc["properties"]["polyline_count"] == 1
    assert fc["name"] == "Random fault polyline-set"

    feat = fc["features"][0]
    assert feat["geometry"]["type"] == "LineString"
    assert len(feat["geometry"]["coordinates"]) == 5
    points_arr = pls.line_patch[0].geometry.points
    assert isinstance(points_arr, ro.Point3dHdf5Array)
    points_path = points_arr.coordinates.path_in_hdf_file
    np.testing.assert_allclose(
        feat["geometry"]["coordinates"], arrays[points_path], atol=1e-5
    )
    assert feat["properties"]["polyline_index"] == 0
    assert feat["properties"]["node_count"] == 5
    assert feat["properties"]["closed"] is False


def test_polyline_set_multiple_polylines_with_constant_node_count() -> None:
    _, _, pls, arrays = get_random_polyline_set(node_counts=[3, 3, 3])
    fc = polyline_set_to_geojson(pls, arrays, crs=None)

    assert fc["properties"]["polyline_count"] == 3
    assert [f["properties"]["node_count"] for f in fc["features"]] == [3, 3, 3]
    assert [f["properties"]["polyline_index"] for f in fc["features"]] == [0, 1, 2]
    for f in fc["features"]:
        assert len(f["geometry"]["coordinates"]) == 3


def test_polyline_set_emits_crs_block_for_epsg_in_extra_metadata_shape() -> None:
    """End-to-end on the epsg-in-extra-metadata shape: Time CRS with
    BoundProjected EPSG codes hidden in extra_metadata. The resulting
    FeatureCollection should carry the resolved EPSG codes at the top
    level under ``crs.properties``.
    """
    crs = get_random_crs(
        domain="time",
        title=_TEST_CRS_TITLE_WITH_PT,
        extra_metadata=[ro.NameValuePair(name="crs", value=_TEST_BOUND_PROJECTED)],
    )
    _, _, pls, arrays = get_random_polyline_set(crs=crs, node_counts=[4])

    fc = polyline_set_to_geojson(
        pls,
        arrays,
        crs=crs,
        extra_properties={"fault_interpretation_title": "Test-Fault"},
    )

    assert fc["crs"]["type"] == "name"
    crs_props = fc["crs"]["properties"]
    assert crs_props["projected_epsg_code"] == _TEST_PROJ_EPSG
    assert crs_props["vertical_epsg_code"] == _TEST_VERT_EPSG
    assert crs_props["z_domain"] == "time"
    assert crs_props["name"] == f"urn:ogc:def:crs:EPSG::{_TEST_PROJ_EPSG}"

    for f in fc["features"]:
        assert f["properties"]["fault_interpretation_title"] == "Test-Fault"


def test_polyline_set_skips_degenerate_single_point_polylines() -> None:
    _, _, pls, arrays = get_random_polyline_set(node_counts=[1, 3])
    fc = polyline_set_to_geojson(pls, arrays, crs=None)

    # Only the 3-point polyline should survive — LineString needs >=2 nodes.
    assert fc["properties"]["polyline_count"] == 1
    assert fc["features"][0]["properties"]["node_count"] == 3
    assert fc["features"][0]["properties"]["polyline_index"] == 1


def test_polyline_set_unknown_points_path_raises() -> None:
    _, _, pls, _ = get_random_polyline_set(node_counts=[3])
    with pytest.raises(KeyError):
        polyline_set_to_geojson(pls, arrays={}, crs=None)


def test_polyline_set_inconsistent_node_count_sum_raises() -> None:
    _, _, pls, arrays = get_random_polyline_set(node_counts=[3, 3])
    # Tamper with the points array so the sum no longer matches.
    points_arr = pls.line_patch[0].geometry.points
    assert isinstance(points_arr, ro.Point3dHdf5Array)
    points_path = points_arr.coordinates.path_in_hdf_file
    arrays = dict(arrays)
    arrays[points_path] = arrays[points_path][:5]
    with pytest.raises(ValueError, match="node_count_per_polyline sums to"):
        polyline_set_to_geojson(pls, arrays, crs=None)


def test_get_geojson_method_matches_free_function() -> None:
    """``polyline.get_geojson(arrays, crs)`` must produce the same output as
    calling ``polyline_set_to_geojson(polyline, arrays, crs)`` directly.
    """
    crs = get_random_crs(
        domain="time",
        title=_TEST_CRS_TITLE_WITH_PT,
        extra_metadata=[ro.NameValuePair(name="crs", value=_TEST_BOUND_PROJECTED)],
    )
    _, _, pls, arrays = get_random_polyline_set(crs=crs, node_counts=[4])

    via_method = pls.get_geojson(arrays, crs)
    via_function = polyline_set_to_geojson(pls, arrays, crs)

    assert dict(via_method) == dict(via_function)


def test_get_geojson_method_accepts_extra_properties_and_name() -> None:
    _, _, pls, arrays = get_random_polyline_set(node_counts=[3])
    fc = pls.get_geojson(
        arrays,
        crs=None,
        extra_properties={"fault_interpretation_title": "TestFault-1"},
        name="my-fault",
    )

    assert fc["name"] == "my-fault"
    assert (
        fc["features"][0]["properties"]["fault_interpretation_title"] == "TestFault-1"
    )


def test_patch_decode_returns_arrays() -> None:
    _, _, pls, arrays = get_random_polyline_set(node_counts=[3, 3, 3])

    points, node_counts, closed = pls.line_patch[0].decode(arrays)

    assert points.shape == (9, 3)
    assert points.dtype == np.float64
    assert node_counts.tolist() == [3, 3, 3]
    assert node_counts.dtype == np.int64
    assert closed.tolist() == [False, False, False]
    assert closed.dtype == np.bool_


def test_patch_decode_xy_z_access_via_points() -> None:
    """``points[:, 0]`` etc. is how you get raw x / y / z columns."""
    _, _, pls, arrays = get_random_polyline_set(node_counts=[5])
    points, _, _ = pls.line_patch[0].decode(arrays)

    # x/y/z columns must round-trip from the original input array.
    points_arr = pls.line_patch[0].geometry.points
    assert isinstance(points_arr, ro.Point3dHdf5Array)
    points_path = points_arr.coordinates.path_in_hdf_file
    expected = arrays[points_path]
    np.testing.assert_array_equal(points[:, 0], expected[:, 0])
    np.testing.assert_array_equal(points[:, 1], expected[:, 1])
    np.testing.assert_array_equal(points[:, 2], expected[:, 2])
