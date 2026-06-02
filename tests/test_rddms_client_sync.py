import numpy as np
import numpy.typing as npt
import pytest

import resqml_objects.v201 as ro
from rddms_io.sync_client import RDDMSClientSync
from resqml_objects.surface_helpers import RegularGridParameters
from tests.conftest import etp_server_url, skip_decorator


def get_random_surface() -> tuple[
    tuple[
        ro.obj_EpcExternalPartReference,
        ro.obj_LocalDepth3dCrs,
        ro.obj_Grid2dRepresentation,
    ],
    dict[str, npt.NDArray[np.float64]],
]:
    shape = tuple(np.random.randint(10, 123, size=2).tolist())

    x = np.linspace(
        -20 * (np.random.random() + 0.1), 20 * (np.random.random() + 0.1), shape[0]
    )
    y = np.linspace(
        -20 * (np.random.random() + 0.1), 20 * (np.random.random() + 0.1), shape[1]
    )
    Z = np.exp(
        -(np.linspace(-1, 1, shape[0])[:, None] ** 2)
        - np.linspace(-1, 1, shape[1]) ** 2
    )

    origin = np.array([x[0], y[0]])
    spacing = np.array([x[1] - x[0], y[1] - y[0]])

    grid_angle = 2 * np.pi * (np.random.random() - 0.5)
    grid_unit_vectors = RegularGridParameters.angle_to_unit_vectors(grid_angle)

    crs_angle = 2 * np.pi * (np.random.random() - 0.5)
    crs_offset = 2 * 10 * (np.random.random(2) - 0.5)

    crs = ro.obj_LocalDepth3dCrs(
        citation=ro.Citation(
            title="Random crs",
            originator="rddms-io-tester",
        ),
        vertical_crs=ro.VerticalUnknownCrs(unknown="MSL"),
        projected_crs=ro.ProjectedCrsEpsgCode(epsg_code=23031),
        areal_rotation=ro.PlaneAngleMeasure(
            value=crs_angle,
            uom=ro.PlaneAngleUom.RAD,
        ),
        xoffset=float(crs_offset[0]),
        yoffset=float(crs_offset[1]),
    )

    epc = ro.obj_EpcExternalPartReference(
        citation=ro.Citation(title="Random epc", originator="rddms-io-tester"),
    )

    gri = ro.obj_Grid2dRepresentation.from_regular_surface(
        citation=ro.Citation(title="Random grid", originator="rddms-io-tester"),
        crs=crs,
        epc_external_part_reference=epc,
        shape=shape,
        origin=origin,
        spacing=spacing,
        unit_vec_1=grid_unit_vectors[:, 0],
        unit_vec_2=grid_unit_vectors[:, 1],
    )
    assert isinstance(gri.grid2d_patch.geometry.points, ro.Point3dZValueArray)
    assert isinstance(gri.grid2d_patch.geometry.points.zvalues, ro.DoubleHdf5Array)
    key = gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file

    return (epc, crs, gri), {key: Z}


@skip_decorator
def test_upload_and_download_surface() -> None:
    (epc, crs, gri), data_arrays = get_random_surface()
    assert isinstance(gri.grid2d_patch.geometry.points, ro.Point3dZValueArray)
    assert isinstance(gri.grid2d_patch.geometry.points.zvalues, ro.DoubleHdf5Array)
    key = gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file

    rddms_client = RDDMSClientSync(uri=etp_server_url)

    dataspace_path = "rddms-io-sync/test-upload-and-download-surface"

    rddms_client.create_dataspace(dataspace_path, ignore_if_exists=True)
    dataspaces = rddms_client.list_dataspaces()
    assert dataspace_path in [d.path for d in dataspaces]

    epc_uri, crs_uri, gri_uri = rddms_client.upload_model(
        dataspace_uri=dataspace_path,
        ml_objects=[epc, crs, gri],
        data_arrays=data_arrays,
        debounce=True,
    )

    resources = rddms_client.list_objects_under_dataspace(dataspace_path)
    uris = [r.uri for r in resources]

    assert epc_uri in uris
    assert crs_uri in uris
    assert gri_uri in uris

    gri_lo = rddms_client.list_linked_objects(start_uri=gri_uri)
    assert gri_uri == gri_lo.start_uri
    assert crs_uri == gri_lo.target_edges[0].target_uri

    array_metadata = rddms_client.list_array_metadata(
        ml_uris=[epc_uri, crs_uri, gri_uri]
    )
    assert len(array_metadata) == 1
    assert tuple(array_metadata[gri_uri][key].dimensions) == data_arrays[key].shape
    assert array_metadata == rddms_client.list_object_array_metadata(
        dataspace_uri=dataspace_path,
        ml_objects=[gri],
    )

    ret_models = rddms_client.download_models(
        ml_uris=[epc_uri, crs_uri, gri_uri],
        download_arrays=True,
        download_linked_objects=True,
    )
    assert ret_models[0].obj == epc
    assert ret_models[1].obj == crs
    assert ret_models[2].obj == gri
    assert ret_models[2].linked_models[0].obj == crs

    np.testing.assert_equal(ret_models[2].arrays[key], data_arrays[key])

    rddms_client.delete_model(ml_uris=uris)
    rddms_client.delete_dataspace(dataspace_path)


def create_random_polyline_set() -> tuple[
    tuple[
        ro.obj_EpcExternalPartReference,
        ro.obj_LocalDepth3dCrs,
        ro.obj_PolylineSetRepresentation,
    ],
    dict[str, npt.NDArray[np.float64]],
    list[npt.NDArray[np.float64]],
    list[bool],
]:
    n_polylines = int(np.random.randint(2, 6))
    node_counts_in = [int(np.random.randint(2, 8)) for _ in range(n_polylines)]
    # Guarantee at least one differing length so node_count_per_polyline
    # becomes an IntegerHdf5Array (and not an IntegerConstantArray).
    if all(n == node_counts_in[0] for n in node_counts_in):
        node_counts_in[0] = (
            node_counts_in[0] + 1 if node_counts_in[0] < 7 else node_counts_in[0] - 1
        )
    polylines = [
        np.random.rand(n, 3).astype(np.float64) * 1000.0 for n in node_counts_in
    ]
    closed_in = [bool(np.random.randint(0, 2)) for _ in range(n_polylines)]
    # Guarantee at least one differing flag so closed becomes an Hdf5 array.
    if all(c == closed_in[0] for c in closed_in):
        closed_in[0] = not closed_in[0]

    crs = ro.obj_LocalDepth3dCrs(
        citation=ro.Citation(title="Random crs", originator="rddms-io-tester"),
        vertical_crs=ro.VerticalCrsEpsgCode(epsg_code=5715),
        projected_crs=ro.ProjectedCrsEpsgCode(epsg_code=23031),
    )

    epc = ro.obj_EpcExternalPartReference(
        citation=ro.Citation(title="Random epc", originator="rddms-io-tester"),
    )

    pls, data_arrays = ro.obj_PolylineSetRepresentation.from_polylines(
        citation=ro.Citation(title="Random polyline-set", originator="rddms-io-tester"),
        crs=crs,
        epc_external_part_reference=epc,
        polylines=polylines,
        closed=closed_in,
        line_role=ro.LineRole.INTERPRETATION_LINE,
    )

    return (epc, crs, pls), data_arrays, polylines, closed_in


@skip_decorator
def test_upload_and_download_polyline_set() -> None:
    (epc, crs, pls), data_arrays, polylines_in, closed_in = create_random_polyline_set()

    rddms_client = RDDMSClientSync(uri=etp_server_url)

    dataspace_path = "rddms-io-sync/test-upload-and-download-polyline-set"

    rddms_client.create_dataspace(dataspace_path, ignore_if_exists=True)
    dataspaces = rddms_client.list_dataspaces()
    assert dataspace_path in [d.path for d in dataspaces]

    epc_uri, crs_uri, pls_uri = rddms_client.upload_model(
        dataspace_uri=dataspace_path,
        ml_objects=[epc, crs, pls],
        data_arrays=data_arrays,
        debounce=True,
    )

    resources = rddms_client.list_objects_under_dataspace(dataspace_path)
    uris = [r.uri for r in resources]

    assert epc_uri in uris
    assert crs_uri in uris
    assert pls_uri in uris

    pls_lo = rddms_client.list_linked_objects(start_uri=pls_uri)
    assert pls_uri == pls_lo.start_uri
    assert crs_uri == pls_lo.target_edges[0].target_uri

    ret_models = rddms_client.download_models(
        ml_uris=[epc_uri, crs_uri, pls_uri],
        download_arrays=True,
        download_linked_objects=True,
    )
    assert ret_models[0].obj == epc
    assert ret_models[1].obj == crs
    assert ret_models[2].obj == pls
    assert ret_models[2].linked_models[0].obj == crs

    # The HDF5 payload should round-trip exactly for every path we uploaded.
    for path, arr in data_arrays.items():
        np.testing.assert_equal(ret_models[2].arrays[path], arr)

    # Decode the patch back to (points, node_counts, closed) and confirm
    # we recovered the original inputs.
    pls_read = ret_models[2].obj
    assert isinstance(pls_read, ro.obj_PolylineSetRepresentation)
    points, node_counts, closed_out = pls_read.line_patch[0].decode(
        ret_models[2].arrays
    )
    np.testing.assert_allclose(points, np.concatenate(polylines_in, axis=0))
    assert node_counts.tolist() == [int(p.shape[0]) for p in polylines_in]
    assert closed_out.tolist() == closed_in

    rddms_client.delete_model(ml_uris=uris)
    rddms_client.delete_dataspace(dataspace_path)


@skip_decorator
@pytest.mark.asyncio
async def test_upload_and_download_surface_async() -> None:
    (epc, crs, gri), data_arrays = get_random_surface()
    assert isinstance(gri.grid2d_patch.geometry.points, ro.Point3dZValueArray)
    assert isinstance(gri.grid2d_patch.geometry.points.zvalues, ro.DoubleHdf5Array)
    key = gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file

    rddms_client = RDDMSClientSync(uri=etp_server_url)

    dataspace_path = "rddms-io-sync/test-upload-and-download-surface-async"

    rddms_client.create_dataspace(dataspace_path, ignore_if_exists=True)
    dataspaces = rddms_client.list_dataspaces()
    assert dataspace_path in [d.path for d in dataspaces]

    epc_uri, crs_uri, gri_uri = rddms_client.upload_model(
        dataspace_uri=dataspace_path,
        ml_objects=[epc, crs, gri],
        data_arrays=data_arrays,
        debounce=True,
    )

    resources = rddms_client.list_objects_under_dataspace(dataspace_path)
    uris = [r.uri for r in resources]

    assert epc_uri in uris
    assert crs_uri in uris
    assert gri_uri in uris

    gri_lo = rddms_client.list_linked_objects(start_uri=gri_uri)
    assert gri_uri == gri_lo.start_uri
    assert crs_uri == gri_lo.target_edges[0].target_uri

    array_metadata = rddms_client.list_array_metadata(
        ml_uris=[epc_uri, crs_uri, gri_uri]
    )
    assert len(array_metadata) == 1
    assert tuple(array_metadata[gri_uri][key].dimensions) == data_arrays[key].shape
    assert array_metadata == rddms_client.list_object_array_metadata(
        dataspace_uri=dataspace_path,
        ml_objects=[gri],
    )

    ret_models = rddms_client.download_models(
        ml_uris=[epc_uri, crs_uri, gri_uri],
        download_arrays=True,
        download_linked_objects=True,
    )
    assert ret_models[0].obj == epc
    assert ret_models[1].obj == crs
    assert ret_models[2].obj == gri
    assert ret_models[2].linked_models[0].obj == crs

    np.testing.assert_equal(ret_models[2].arrays[key], data_arrays[key])

    rddms_client.delete_model(ml_uris=uris)
    rddms_client.delete_dataspace(dataspace_path)
