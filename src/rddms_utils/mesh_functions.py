import sys
import time
import logging
import typing
import asyncio
import datetime
import uuid
import pathlib

import numpy as np

import resqpy.model as rq
import resqpy.property as rqp
import resqpy.time_series as rts
import resqpy.unstructured as rug

from scipy.interpolate import griddata
import pyetp.utils_arrays

from pyetp import ETPClient, utils_arrays
from pyetp.config import SETTINGS
from pyetp.uri import DataObjectURI, DataspaceURI
from resqml_objects.epc_readers import (
    get_resqml_v201_objects,
    get_arrays_and_paths_in_hdf_file,
)
from resqml_objects.v201.utils import (
    resqml_schema_version,
    common_schema_version,
    get_content_type_string,
    get_data_object_reference,
)
from xsdata.models.datatype import XmlDateTime

from etptypes.energistics.etp.v12.datatypes.data_array_types.data_array_identifier import (
    DataArrayIdentifier,
)

import resqml_objects.v201 as ro


def get_cells_positions(
    points: np.ndarray,
    n_cells: int,
    n_cell_per_pos: int,
    layers_per_sediment_unit: int,
    n_node_per_pos: int,
    node_index: int,
):
    def mid_point_rectangle(arr: np.ndarray):
        all_x = arr[:, 0]
        all_y = arr[:, 1]
        min_x = np.min(all_x)
        min_y = np.min(all_y)
        mid_x = ((np.max(all_x) - min_x) / 2) + min_x
        mid_y = ((np.max(all_y) - min_y) / 2) + min_y
        return np.array([mid_x, mid_y])

    results = np.zeros((int(n_cells / n_cell_per_pos), 3), dtype=np.float64)
    grid_x_pos = np.unique(points[:, 0])
    grid_y_pos = np.unique(points[:, 1])
    counter = 0
    # find cell index and location

    for y_ind in range(0, len(grid_y_pos) - 1):
        for x_ind in range(0, len(grid_x_pos) - 1):
            top_depth = []
            for corner_x in range(layers_per_sediment_unit):
                for corner_y in range(layers_per_sediment_unit):
                    node_indx = (
                        ((y_ind + corner_y) * len(grid_x_pos) + (x_ind + corner_x))
                        * n_node_per_pos
                    ) + node_index
                    top_depth.append(points[node_indx])
            results[counter, 0:2] = mid_point_rectangle(np.array(top_depth))
            counter += 1
    return results


async def put_rddms_property(
    etp_client: ETPClient,
    epc_uri: typing.Union[DataObjectURI, str],
    cprop0: typing.Union[ro.ContinuousProperty, ro.DiscreteProperty],
    propertykind0: ro.PropertyKind,
    array_ref: np.ndarray,
    dataspace_uri: DataspaceURI,
):
    assert isinstance(cprop0, ro.ContinuousProperty) or isinstance(
        cprop0, ro.DiscreteProperty
    ), "prop must be a Property"
    assert len(cprop0.patch_of_values) == 1, (
        "property obj must have exactly one patch of values"
    )

    st = time.time()
    propkind_uri = (
        [""]
        if (propertykind0 is None)
        else (
            await etp_client.put_resqml_objects(
                propertykind0, dataspace_uri=dataspace_uri
            )
        )
    )
    cprop_uri = await etp_client.put_resqml_objects(cprop0, dataspace_uri=dataspace_uri)
    delay = time.time() - st
    logging.debug(f"pyetp: put_rddms_property: put objects took {delay} s")

    st = time.time()
    _ = await etp_client.put_array(
        DataArrayIdentifier(
            uri=epc_uri.raw_uri if isinstance(epc_uri, DataObjectURI) else epc_uri,
            pathInResource=cprop0.patch_of_values[0].values.values.path_in_hdf_file,
        ),
        array_ref,  # type: ignore
    )
    delay = time.time() - st
    logging.debug(
        f"pyetp: put_rddms_property: put array ({array_ref.shape}) took {delay} s"
    )
    return cprop_uri, propkind_uri


async def get_epc_mesh(
    etp_client: ETPClient,
    epc_uri: typing.Union[DataObjectURI, str],
    uns_uri: typing.Union[DataObjectURI, str],
):
    (uns,) = await etp_client.get_resqml_objects(uns_uri)

    # some checks
    assert isinstance(uns, ro.UnstructuredGridRepresentation), (
        "obj must be UnstructuredGridRepresentation"
    )
    assert isinstance(uns.geometry, ro.UnstructuredGridGeometry), (
        "geometry must be UnstructuredGridGeometry"
    )
    if sys.version_info[1] != 10:
        assert isinstance(uns.geometry.points, ro.Point3dHdf5Array), (
            "points must be Point3dHdf5Array"
        )
        assert isinstance(uns.geometry.faces_per_cell.elements, ro.IntegerHdf5Array), (
            "faces_per_cell must be IntegerHdf5Array"
        )
        assert isinstance(
            uns.geometry.faces_per_cell.cumulative_length, ro.IntegerHdf5Array
        ), "faces_per_cell cl must be IntegerHdf5Array"
    assert isinstance(uns.geometry.points.coordinates, ro.Hdf5Dataset), (
        "coordinates must be Hdf5Dataset"
    )

    # # get array
    points = await etp_client.get_array(
        DataArrayIdentifier(
            uri=str(epc_uri),
            pathInResource=uns.geometry.points.coordinates.path_in_hdf_file,
        )
    )
    nodes_per_face = await etp_client.get_array(
        DataArrayIdentifier(
            uri=str(epc_uri),
            pathInResource=uns.geometry.nodes_per_face.elements.values.path_in_hdf_file,
        )
    )
    nodes_per_face_cl = await etp_client.get_array(
        DataArrayIdentifier(
            uri=str(epc_uri),
            pathInResource=uns.geometry.nodes_per_face.cumulative_length.values.path_in_hdf_file,
        )
    )
    faces_per_cell = await etp_client.get_array(
        DataArrayIdentifier(
            uri=str(epc_uri),
            pathInResource=uns.geometry.faces_per_cell.elements.values.path_in_hdf_file,
        )
    )
    faces_per_cell_cl = await etp_client.get_array(
        DataArrayIdentifier(
            uri=str(epc_uri),
            pathInResource=uns.geometry.faces_per_cell.cumulative_length.values.path_in_hdf_file,
        )
    )
    cell_face_is_right_handed = await etp_client.get_array(
        DataArrayIdentifier(
            uri=str(epc_uri),
            pathInResource=uns.geometry.cell_face_is_right_handed.values.path_in_hdf_file,
        )
    )

    return (
        uns,
        points,
        nodes_per_face,
        nodes_per_face_cl,
        faces_per_cell,
        faces_per_cell_cl,
        cell_face_is_right_handed,
    )


async def get_epc_mesh_property(
    etp_client: ETPClient,
    epc_uri: typing.Union[DataObjectURI, str],
    prop_uri: typing.Union[DataObjectURI, str],
):
    (cprop0,) = await etp_client.get_resqml_objects(prop_uri)

    # some checks
    assert isinstance(cprop0, ro.ContinuousProperty) or isinstance(
        cprop0, ro.DiscreteProperty
    ), "prop must be a Property"
    assert len(cprop0.patch_of_values) == 1, (
        "property obj must have exactly one patch of values"
    )

    # # get array
    values = await etp_client.get_array(
        DataArrayIdentifier(
            uri=str(epc_uri),
            pathInResource=cprop0.patch_of_values[0].values.values.path_in_hdf_file,
        )
    )

    return cprop0, values


async def get_epc_mesh_property_x_y(
    etp_client: ETPClient,
    epc_uri: typing.Union[DataObjectURI, str],
    uns_uri: typing.Union[DataObjectURI, str],
    prop_uri: typing.Union[DataObjectURI, str],
    x: float,
    y: float,
):
    (uns,) = await etp_client.get_resqml_objects(uns_uri)
    points = await etp_client.get_array(
        DataArrayIdentifier(
            uri=str(epc_uri),
            pathInResource=uns.geometry.points.coordinates.path_in_hdf_file,
        )
    )

    def check_bound(points, x: float, y: float):
        if x > points[:, 0].max() or x < points[:, 0].min():
            return False
        if y > points[:, 1].max() or y < points[:, 1].min():
            return False
        return True

    chk = check_bound(points, x, y)
    if not chk:
        return None
    unique_y = np.unique(points[:, 1])
    y_smaller_sorted = np.sort(unique_y[np.argwhere(unique_y < y).flatten()])
    if y_smaller_sorted.size > 1:
        y_floor = y_smaller_sorted[-2]
    elif y_smaller_sorted.size == 1:
        y_floor = y_smaller_sorted[-1]
    else:
        pass
    y_larger_sorted = np.sort(unique_y[np.argwhere(unique_y > y).flatten()])
    if y_larger_sorted.size > 1:
        y_ceil = y_larger_sorted[1]
    elif y_larger_sorted.size == 1:
        y_ceil = y_larger_sorted[0]
    else:
        pass
    start_new_row_idx = np.argwhere(np.diff(points[:, 1]) != 0).flatten() + 1

    to_fetch = []
    initial_result_arr_idx = 0
    for i in range(start_new_row_idx.size - 1):
        sliced = points[start_new_row_idx[i] : start_new_row_idx[i + 1], :]
        if sliced[0, 1] <= y_ceil and sliced[0, 1] >= y_floor:
            # Found slice that has same y
            x_diff = sliced[:, 0] - x
            if all(
                [np.any((x_diff >= 0)), np.any((x_diff <= 0))]
            ):  # y within this slice
                first_idx = start_new_row_idx[i]
                count = start_new_row_idx[i + 1] - first_idx
                to_fetch.append(
                    [
                        start_new_row_idx[i],
                        start_new_row_idx[i + 1],
                        count,
                        initial_result_arr_idx,
                    ]
                )
                initial_result_arr_idx += count

    total_points_filtered = sum([i[2] for i in to_fetch])

    (cprop,) = await etp_client.get_resqml_objects(prop_uri)
    assert str(cprop.indexable_element) == "IndexableElements.NODES"
    props_uid = DataArrayIdentifier(
        uri=str(epc_uri),
        pathInResource=cprop.patch_of_values[0].values.values.path_in_hdf_file,
    )
    (meta,) = await etp_client.get_array_metadata(props_uid)
    filtered_points = np.zeros((total_points_filtered, 3), dtype=np.float64)
    all_values = np.empty(total_points_filtered, dtype=np.float64)

    async def populate(i):
        end_indx = i[2] + i[3]
        filtered_points[i[3] : end_indx] = points[i[0] : i[1]]
        if (
            utils_arrays.get_transport_array_size(meta) * i[2] / points.shape[0]
            > etp_client.max_array_size
        ):
            all_values[i[3] : end_indx] = await etp_client._get_array_chunked(
                props_uid, i[0], i[2]
            )
        else:
            all_values[i[3] : end_indx] = await etp_client.get_subarray(
                props_uid, [i[0]], [i[2]]
            )
        return

    _ = await asyncio.gather(*[populate(i) for i in to_fetch])

    if isinstance(cprop, ro.DiscreteProperty):
        method = "nearest"
    else:
        method = "linear"

    # resolution= np.mean(np.diff(filtered[:,-1]))
    top = round(np.min(filtered_points[:, -1]), 1)
    base = round(np.max(filtered_points[:, -1]), 1)
    requested_depth = np.arange(top, base + 1, 100)
    requested_depth = requested_depth[requested_depth > 0]
    request = np.tile([x, y, 0], (requested_depth.size, 1))
    request[:, 2] = requested_depth
    interpolated = griddata(filtered_points, all_values, request, method=method)
    response = np.vstack((requested_depth, interpolated))
    response_filtered = response[:, ~np.isnan(response[1])]
    return {"depth": response_filtered[0], "values": response_filtered[1]}


async def put_epc_mesh_new(
    etp_client: ETPClient,
    epc_filename: str | pathlib.Path,
    dataspace_uri: DataspaceURI,
) -> list[DataObjectURI, ...]:
    path = pathlib.Path(epc_filename)
    mesh_objects = get_resqml_v201_objects(path)
    mesh_arrays = get_arrays_and_paths_in_hdf_file(path.with_suffix(".h5"))

    transaction_uuid = await etp_client.start_transaction(
        dataspace_uri=dataspace_uri, read_only=False
    )

    uris = await etp_client.put_resqml_objects(
        *mesh_objects, dataspace_uri=dataspace_uri
    )
    epc_uris = list(filter(lambda e: "EpcExternalPartReference" in str(e), uris))
    assert len(epc_uris) == 1
    epc_uri = str(epc_uris[0])

    tasks = []
    for pir, arr in mesh_arrays.items():
        dai = DataArrayIdentifier(uri=epc_uri, path_in_resource=pir)
        arr = arr.astype(pyetp.utils_arrays.get_valid_dtype_cast(arr))
        tasks.append(etp_client.put_array(dai, arr))

    _ = await asyncio.gather(*tasks)

    await etp_client.commit_transaction(transaction_uuid=transaction_uuid)

    return uris


async def put_epc_mesh(
    etp_client: ETPClient,
    epc_filename: str,
    title_in: str,
    property_titles: typing.List[str],
    projected_epsg: int,
    dataspace_uri: DataspaceURI,
):
    uns, crs, epc, timeseries, hexa = convert_epc_mesh_to_resqml_mesh(
        epc_filename, title_in, projected_epsg
    )

    transaction_uuid = await etp_client.start_transaction(
        dataspace_uri=dataspace_uri, read_only=False
    )

    epc_uri, crs_uri, uns_uri = await etp_client.put_resqml_objects(
        epc, crs, uns, dataspace_uri=dataspace_uri
    )
    timeseries_uri = ""
    if timeseries is not None:
        timeseries_uris = await etp_client.put_resqml_objects(
            timeseries, dataspace_uri=dataspace_uri
        )
        timeseries_uri = (
            list(timeseries_uris)[0] if (len(list(timeseries_uris)) > 0) else ""
        )

    #
    # mesh geometry (six arrays)
    #
    put_jobs = []

    p = etp_client.put_array(
        DataArrayIdentifier(
            uri=epc_uri.raw_uri if isinstance(epc_uri, DataObjectURI) else epc_uri,
            pathInResource=uns.geometry.points.coordinates.path_in_hdf_file,
        ),
        hexa.points_cached,  # type: ignore
    )
    put_jobs.append(p)

    p = etp_client.put_array(
        DataArrayIdentifier(
            uri=epc_uri.raw_uri if isinstance(epc_uri, DataObjectURI) else epc_uri,
            pathInResource=uns.geometry.nodes_per_face.elements.values.path_in_hdf_file,
        ),
        hexa.nodes_per_face.astype(np.int32),  # type: ignore
    )
    put_jobs.append(p)

    p = etp_client.put_array(
        DataArrayIdentifier(
            uri=epc_uri.raw_uri if isinstance(epc_uri, DataObjectURI) else epc_uri,
            pathInResource=uns.geometry.nodes_per_face.cumulative_length.values.path_in_hdf_file,
        ),
        hexa.nodes_per_face_cl,  # type: ignore
    )
    put_jobs.append(p)

    p = etp_client.put_array(
        DataArrayIdentifier(
            uri=epc_uri.raw_uri if isinstance(epc_uri, DataObjectURI) else epc_uri,
            pathInResource=uns.geometry.faces_per_cell.elements.values.path_in_hdf_file,
        ),
        hexa.faces_per_cell,  # type: ignore
    )
    put_jobs.append(p)

    p = etp_client.put_array(
        DataArrayIdentifier(
            uri=epc_uri.raw_uri if isinstance(epc_uri, DataObjectURI) else epc_uri,
            pathInResource=uns.geometry.faces_per_cell.cumulative_length.values.path_in_hdf_file,
        ),
        hexa.faces_per_cell_cl,  # type: ignore
    )
    put_jobs.append(p)

    p = etp_client.put_array(
        DataArrayIdentifier(
            uri=epc_uri.raw_uri if isinstance(epc_uri, DataObjectURI) else epc_uri,
            pathInResource=uns.geometry.cell_face_is_right_handed.values.path_in_hdf_file,
        ),
        hexa.cell_face_is_right_handed,  # type: ignore
    )
    put_jobs.append(p)

    _ = await asyncio.gather(*put_jobs)

    #
    # mesh properties: one Property, one array of values, and an optional PropertyKind per property
    #
    prop_rddms_uris = {}
    for propname in property_titles:
        if timeseries is not None:
            time_indices = list(range(len(timeseries.time)))
            cprop0s, props, propertykind0 = convert_epc_mesh_property_to_resqml_mesh(
                epc_filename,
                hexa,
                propname,
                uns,
                epc,
                timeseries=timeseries,
                time_indices=time_indices,
            )
        else:
            time_indices = [-1]
            cprop0s, props, propertykind0 = convert_epc_mesh_property_to_resqml_mesh(
                epc_filename, hexa, propname, uns, epc
            )

        if cprop0s is None:
            continue

        cprop_uris = []
        for cprop0, prop, time_index in zip(cprop0s, props, time_indices):
            cprop_uri, propkind_uri = await put_rddms_property(
                etp_client,
                epc_uri,
                cprop0,
                propertykind0,
                prop.array_ref(),
                dataspace_uri,
            )
            cprop_uris.extend(cprop_uri)
        prop_rddms_uris[propname] = [propkind_uri, cprop_uris]

    await etp_client.commit_transaction(transaction_uuid=transaction_uuid)

    return [epc_uri, crs_uri, uns_uri, timeseries_uri], prop_rddms_uris


async def get_mesh_points(
    etp_client: ETPClient,
    epc_uri: typing.Union[DataObjectURI, str],
    uns_uri: typing.Union[DataObjectURI, str],
):
    (uns,) = await etp_client.get_resqml_objects(uns_uri)
    points = await etp_client.get_array(
        DataArrayIdentifier(
            uri=str(epc_uri),
            pathInResource=uns.geometry.points.coordinates.path_in_hdf_file,
        )
    )
    return points


async def get_epc_property_surface_slice_node(
    etp_client: ETPClient,
    epc_uri: typing.Union[DataObjectURI, str],
    cprop0: ro.AbstractObject,
    points: np.ndarray,
    node_index: int,
    n_node_per_pos: int,
):
    # indexing_array = np.arange(0, points.shape[0], 1, dtype=np.int32)[node_index::n_node_per_pos]
    indexing_array = np.arange(
        node_index, points.shape[0], n_node_per_pos, dtype=np.int32
    )
    results = points[indexing_array, :]
    arr = await asyncio.gather(
        *[
            etp_client.get_subarray(
                DataArrayIdentifier(
                    uri=str(epc_uri),
                    pathInResource=cprop0.patch_of_values[
                        0
                    ].values.values.path_in_hdf_file,
                ),
                [i],
                [1],
            )
            for i in indexing_array
        ]
    )
    arr = np.array(arr).flatten()
    assert results.shape[0] == arr.size
    results[:, 2] = arr
    return results


async def get_epc_property_surface_slice_cell(
    etp_client: ETPClient,
    epc_uri: typing.Union[DataObjectURI, str],
    cprop0: ro.AbstractObject,
    points: np.ndarray,
    node_index: int,
    n_node_per_pos: int,
    get_cell_pos=True,
):
    (m,) = await etp_client.get_array_metadata(
        DataArrayIdentifier(
            uri=str(epc_uri),
            pathInResource=cprop0.patch_of_values[0].values.values.path_in_hdf_file,
        )
    )
    n_cells = m.dimensions[0]
    layers_per_sediment_unit = 2
    n_cell_per_pos = n_node_per_pos - 1
    indexing_array = np.arange(node_index, n_cells, n_cell_per_pos, dtype=np.int32)
    if get_cell_pos:
        results = get_cells_positions(
            points,
            n_cells,
            n_cell_per_pos,
            layers_per_sediment_unit,
            n_node_per_pos,
            node_index,
        )
    else:
        results = np.zeros((int(n_cells / n_cell_per_pos), 3), dtype=np.float64)
    arr = await asyncio.gather(
        *[
            etp_client.get_subarray(
                DataArrayIdentifier(
                    uri=str(epc_uri),
                    pathInResource=cprop0.patch_of_values[
                        0
                    ].values.values.path_in_hdf_file,
                ),
                [i],
                [1],
            )
            for i in indexing_array
        ]
    )
    arr = np.array(arr).flatten()
    assert results.shape[0] == arr.size
    results[:, 2] = arr
    return results


async def get_epc_property_surface_slice(
    etp_client: ETPClient,
    epc_uri: typing.Union[DataObjectURI, str],
    uns_uri: typing.Union[DataObjectURI, str],
    prop_uri: typing.Union[DataObjectURI, str],
    node_index: int,
    n_node_per_pos: int,
):
    # n_node_per_pos number of nodes in a 1D location
    # node_index index of slice from top. Warmth has 2 nodes per sediment layer. E.g. top of second layer will have index 2
    points = await get_mesh_points(etp_client, epc_uri, uns_uri)
    (cprop0,) = await etp_client.get_resqml_objects(prop_uri)
    prop_at_node = False
    if str(cprop0.indexable_element) == "IndexableElements.NODES":
        prop_at_node = True
    # node_per_sed = 2
    # n_sed_node = n_sed *node_per_sed
    # n_crust_node = 4
    # n_node_per_pos = n_sed_node + n_crust_node
    # start_idx_pos = sediment_id *node_per_sed
    if prop_at_node:
        return await get_epc_property_surface_slice_node(
            etp_client, epc_uri, cprop0, points, node_index, n_node_per_pos
        )
    else:
        return await get_epc_property_surface_slice_cell(
            etp_client, epc_uri, cprop0, points, node_index, n_node_per_pos
        )


def create_common_citation(title: str):
    return ro.Citation(
        title=title,
        creation=XmlDateTime.from_string(
            datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
        ),
        originator=SETTINGS.application_name,
        format=f"{SETTINGS.application_name}:v{SETTINGS.application_version}",
    )


def convert_epc_mesh_to_resqml_mesh(
    epc_filename: str,
    title_in: str,
    projected_epsg: int,
    resqml_schema_version: str = resqml_schema_version,
    common_schema_version: str = common_schema_version,
):
    title = title_in or "hexamesh"

    model = rq.Model(epc_filename)
    assert model is not None

    #
    # read mesh:  vertex positions and cell definitions
    #
    hexa_uuid = model.uuid(obj_type="UnstructuredGridRepresentation", title=title_in)
    assert hexa_uuid is not None
    hexa = rug.HexaGrid(model, uuid=hexa_uuid)
    assert hexa is not None
    assert hexa.cell_shape == "hexahedral"
    hexa.check_hexahedral()

    ts_uuid = model.uuid(obj_type="TimeSeries")
    # ts_uuid_2 = model.uuid(obj_type='GeologicTimeSeries')
    # logging.debug(f"TS UUIDs: {ts_uuid} {ts_uuid_2}")
    gts = rts.GeologicTimeSeries(model, uuid=ts_uuid)
    logging.debug(f"gts: {gts}")
    timeseries = None
    if (ts_uuid is not None) and (gts is not None):
        ro_timestamps = []
        for i in gts.iter_timestamps(as_string=False):
            ro_timestamps.append(
                ro.Timestamp(
                    date_time=XmlDateTime.from_string("0001-01-01T00:00:00.00+00:00"),
                    year_offset=int(i),
                )
            )
        logging.info(
            f"Generating time series with {len(ro_timestamps)} indices, year offsets: {ro_timestamps[0].year_offset} -- {ro_timestamps[-1].year_offset}."
        )
        timeseries = ro.TimeSeries(
            citation=create_common_citation(str(gts.citation_title)),
            schema_version=resqml_schema_version,
            uuid=str(gts.uuid),
            time=ro_timestamps,
        )

    crs = ro.LocalDepth3dCrs(
        citation=create_common_citation(f"CRS for {title}"),
        schema_version=resqml_schema_version,
        uuid=str(uuid.uuid4()),
        xoffset=0.0,
        yoffset=0.0,
        zoffset=0.0,
        areal_rotation=ro.PlaneAngleMeasure(
            value=0.0,
            uom=ro.PlaneAngleUom.DEGA,
        ),
        projected_axis_order=ro.AxisOrder2d.EASTING_NORTHING,
        projected_uom=ro.LengthUom.M,
        vertical_uom=ro.LengthUom.M,
        zincreasing_downward=True,
        vertical_crs=ro.VerticalCrsEpsgCode(epsg_code=projected_epsg),
        projected_crs=ro.ProjectedCrsEpsgCode(
            epsg_code=projected_epsg,
        ),
    )

    epc = ro.EpcExternalPartReference(
        citation=create_common_citation("Hdf Proxy"),
        schema_version=common_schema_version,
        uuid=str(uuid.uuid4()),
        mime_type="application/x-hdf5",
    )

    cellshape = (
        ro.CellShape.HEXAHEDRAL
        if (hexa.cell_shape == "hexahedral")
        else ro.CellShape.TETRAHEDRAL
    )

    geom = ro.UnstructuredGridGeometry(
        local_crs=get_data_object_reference(crs),
        node_count=hexa.node_count or -1,
        face_count=hexa.face_count or -1,
        cell_shape=cellshape,
        points=ro.Point3dHdf5Array(
            coordinates=ro.Hdf5Dataset(
                path_in_hdf_file=f"/RESQML/{str(hexa_uuid)}/points",
                hdf_proxy=get_data_object_reference(epc),
            )
        ),
        nodes_per_face=ro.ResqmlJaggedArray(
            elements=ro.IntegerHdf5Array(
                null_value=-1,
                values=ro.Hdf5Dataset(
                    path_in_hdf_file=f"/RESQML/{str(hexa_uuid)}/nodes_per_face",
                    hdf_proxy=get_data_object_reference(epc),
                ),
            ),
            cumulative_length=ro.IntegerHdf5Array(
                null_value=-1,
                values=ro.Hdf5Dataset(
                    path_in_hdf_file=f"/RESQML/{str(hexa_uuid)}/nodes_per_face_cl",
                    hdf_proxy=get_data_object_reference(epc),
                ),
            ),
        ),
        faces_per_cell=ro.ResqmlJaggedArray(
            elements=ro.IntegerHdf5Array(
                null_value=-1,
                values=ro.Hdf5Dataset(
                    path_in_hdf_file=f"/RESQML/{str(hexa_uuid)}/faces_per_cell",
                    hdf_proxy=get_data_object_reference(epc),
                ),
            ),
            cumulative_length=ro.IntegerHdf5Array(
                null_value=-1,
                values=ro.Hdf5Dataset(
                    path_in_hdf_file=f"/RESQML/{str(hexa_uuid)}/faces_per_cell_cl",
                    hdf_proxy=get_data_object_reference(epc),
                ),
            ),
        ),
        cell_face_is_right_handed=ro.BooleanHdf5Array(
            values=ro.Hdf5Dataset(
                path_in_hdf_file=(
                    f"/RESQML/{str(hexa_uuid)}/cell_face_is_right_handed"
                ),
                hdf_proxy=get_data_object_reference(epc),
            )
        ),
    )

    #
    uns = ro.UnstructuredGridRepresentation(
        uuid=str(hexa.uuid),
        schema_version=resqml_schema_version,
        # surface_role=resqml_objects.SurfaceRole.MAP,
        citation=create_common_citation(hexa.title),
        cell_count=hexa.cell_count or -1,
        geometry=geom,
    )

    return uns, crs, epc, timeseries, hexa


def convert_epc_mesh_property_to_resqml_mesh(
    epc_filename,
    hexa,
    prop_title,
    uns: ro.UnstructuredGridRepresentation,
    epc: ro.EpcExternalPartReference,
    timeseries=None,
    time_indices: list[int] = [],
):
    model = rq.Model(epc_filename)
    assert model is not None
    prop_types = [
        "obj_ContinuousProperty",
        "obj_DiscreteProperty",
        "obj_CategoricalProperty",
        "obj_PointsProperty",
    ]
    p = []
    for i in prop_types:
        p1 = model.uuids(title=prop_title, obj_type=i)
        p.extend(p1)
    p_test = rqp.Property(model, uuid=p[0])

    use_timeseries = isinstance(p_test.time_index(), int)
    if not use_timeseries:
        prop_uuid0 = p[0]
        prop0 = rqp.Property(model, uuid=prop_uuid0)
    else:
        prop_uuids = p
        prop_uuid0 = prop_uuids[time_indices[0]]
        prop0 = rqp.Property(
            model, uuid=prop_uuid0
        )  # a prop representative of all in the timeseries

    continuous = prop0.is_continuous()

    def uom_for_prop_title(pt: str):
        if pt == "Age":
            return ro.ResqmlUom.A_1
        if pt == "Temperature":
            return ro.ResqmlUom.DEG_C
        if pt == "LayerID":
            return ro.ResqmlUom.EUC
        if pt == "Porosity_initial":
            return ro.ResqmlUom.M3_M3
        if pt == "Porosity_decay":
            return ro.ResqmlUom.VALUE_1_M
        if pt == "Density_solid":
            return ro.ResqmlUom.KG_M3
        if pt == "insulance_thermal":
            return ro.ThermalInsulanceUom.DELTA_K_M2_W
        if pt == "Radiogenic_heat_production":
            return ro.ResqmlUom.U_W_M3
        if (pt == "dynamic nodes") or (pt == "points"):
            return ro.ResqmlUom.M
        if pt == "thermal_conductivity":
            return ro.ResqmlUom.W_M_K
        if pt == "Vitrinite reflectance" or pt == "%Ro":
            return ro.ResqmlUom.VALUE
        if "Expelled" in pt:
            return ro.ResqmlUom.KG_M3
        if "Transformation" in pt:
            return ro.ResqmlUom.VALUE
        return ro.ResqmlUom.EUC

    if prop0.local_property_kind_uuid() is None:
        propertykind0 = None
    else:
        pk = rqp.PropertyKind(model, uuid=prop0.local_property_kind_uuid())
        propertykind0 = ro.PropertyKind(
            schema_version=resqml_schema_version,
            citation=create_common_citation(f"{prop_title}"),
            naming_system="urn:resqml:bp.com:resqpy",
            is_abstract=False,
            representative_uom=uom_for_prop_title(prop_title),
            parent_property_kind=ro.StandardPropertyKind(
                kind=ro.ResqmlPropertyKind.CONTINUOUS
                if continuous
                else ro.ResqmlPropertyKind.DISCRETE
            ),
            uuid=str(pk.uuid),
        )

    cprop0s, props = [], []

    for i in range(len(time_indices) if use_timeseries else 1):
        if not use_timeseries:
            prop_uuid = prop_uuid0
            prop = prop0
        else:
            prop_uuid = prop_uuids[time_indices[i]]
            prop = rqp.Property(model, uuid=prop_uuid)

        pov = ro.PatchOfValues(
            values=ro.DoubleHdf5Array(
                values=ro.Hdf5Dataset(
                    path_in_hdf_file=f"/RESQML/{str(prop_uuid)}/values",
                    hdf_proxy=get_data_object_reference(epc),
                )
            )
            if continuous
            else ro.IntegerHdf5Array(
                values=ro.Hdf5Dataset(
                    path_in_hdf_file=f"/RESQML/{str(prop_uuid)}/values",
                    hdf_proxy=get_data_object_reference(epc),
                ),
                null_value=int(1e30),
            )
        )

        timeindex_ref = None
        if use_timeseries:
            time_index = time_indices[i]
            timeindex_ref = ro.TimeIndex(
                index=time_index,
                time_series=get_data_object_reference(timeseries),
            )

        r_uom = (
            ro.ResqmlUom(value=uom_for_prop_title(prop_title))
            if (prop.uom() is None)
            else prop.uom()
        )

        if continuous:
            cprop0 = ro.ContinuousProperty(
                schema_version=resqml_schema_version,
                citation=create_common_citation(f"{prop_title}"),
                uuid=str(prop.uuid),
                uom=r_uom,
                count=1,
                indexable_element=prop.indexable_element(),
                supporting_representation=get_data_object_reference(uns),
                property_kind=ro.LocalPropertyKind(
                    local_property_kind=get_data_object_reference(propertykind0),
                )
                if (propertykind0 is not None)
                else ro.StandardPropertyKind(kind=prop.property_kind()),
                minimum_value=[prop.minimum_value() or 0.0],
                maximum_value=[prop.maximum_value() or 1.0],
                facet=[
                    ro.PropertyKindFacet(
                        facet=ro.Facet.WHAT,
                        value=prop_title,  # prop.facet(),
                    )
                ],
                patch_of_values=[pov],
                time_index=timeindex_ref,
            )
        else:
            cprop0 = ro.DiscreteProperty(
                schema_version=resqml_schema_version,
                citation=create_common_citation(f"{prop_title}"),
                uuid=str(prop.uuid),
                # uom = prop.uom(),
                count=1,
                indexable_element=prop.indexable_element(),
                supporting_representation=get_data_object_reference(uns),
                property_kind=ro.LocalPropertyKind(
                    local_property_kind=get_data_object_reference(propertykind0),
                )
                if (propertykind0 is not None)
                else ro.StandardPropertyKind(kind=prop.property_kind()),
                minimum_value=[int(prop.minimum_value() or 0)],
                maximum_value=[int(prop.maximum_value() or 1)],
                facet=[
                    ro.PropertyKindFacet(
                        facet=ro.Facet.WHAT,
                        value=prop_title,  # prop.facet(),
                    )
                ],
                patch_of_values=[pov],
                time_index=timeindex_ref,
            )
        cprop0s.append(cprop0)
        props.append(prop)

    return cprop0s, props, propertykind0
