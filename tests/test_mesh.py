import asyncio
import pathlib
import re

import numpy as np
import pytest
import resqpy.model as rq
import resqpy.property as rqp
import resqpy.unstructured as rug
from etptypes.energistics.etp.v12.datatypes.data_array_types.data_array_identifier import (
    DataArrayIdentifier,
)

import pyetp.utils_arrays
import resqml_objects.v201 as ro
from pyetp.client import ETPClient
from pyetp.uri import DataspaceURI
from rddms_utils.mesh_functions import (
    get_epc_mesh,
    get_epc_mesh_property,
    put_epc_mesh,
)
from resqml_objects.epc_readers import (
    get_arrays_and_paths_in_hdf_file,
    get_resqml_v201_objects,
)

data_path = pathlib.Path("data")


@pytest.mark.parametrize(
    "input_mesh_file",
    [
        data_path / "model_hexa_0.epc",
        data_path / "model_hexa_ts_0_new.epc",
    ],
)
@pytest.mark.asyncio
async def test_mesh_raw(
    eclient: ETPClient, duri: DataspaceURI, input_mesh_file: pathlib.Path
):
    robjs = get_resqml_v201_objects(input_mesh_file)
    input_hdf_file = input_mesh_file.with_suffix(".h5")
    arr_data = get_arrays_and_paths_in_hdf_file(input_hdf_file)

    original_dtypes = {}
    casted_arr_data = {}
    for k, v in arr_data.items():
        original_dtypes[k] = v.dtype
        casted_arr_data[k] = v.astype(pyetp.utils_arrays.get_valid_dtype_cast(v))

    transaction_uuid = await eclient.start_transaction(
        dataspace_uri=duri, read_only=False
    )

    uris = await eclient.put_resqml_objects(*robjs, dataspace_uri=duri)
    epc_uris = list(filter(lambda u: "EpcExternalPartReference" in str(u), uris))
    assert len(epc_uris) == 1
    epc_uri = str(epc_uris[0])

    tasks = []
    for pir, data in casted_arr_data.items():
        uid = DataArrayIdentifier(uri=epc_uri, path_in_resource=pir)
        tasks.append(eclient.put_array(uid=uid, data=data))

    await asyncio.gather(*tasks)

    await eclient.commit_transaction(transaction_uuid=transaction_uuid)

    ret_robjs = await eclient.get_resqml_objects(*uris)

    for robj, ret_robj in zip(robjs, ret_robjs):
        assert robj == ret_robj

    ret_paths = []
    pattern = re.compile(r"path_in_hdf_file='([a-zA-Z/\-0-9_]+)',")

    for ret_robj in ret_robjs:
        m = re.findall(pattern, str(ret_robj))
        ret_paths.extend(m)

    assert len(ret_paths) == len(list(arr_data))
    assert sorted(ret_paths) == sorted(arr_data)

    tasks = []
    for ret_pir in ret_paths:
        tasks.append(
            eclient.get_array(
                uid=DataArrayIdentifier(
                    uri=epc_uri,
                    path_in_resource=ret_pir,
                )
            )
        )

    ret_arrays = await asyncio.gather(*tasks)
    ret_arr = dict(zip(ret_paths, ret_arrays))

    for k, v in ret_arr.items():
        assert v.dtype == casted_arr_data[k].dtype
        np.testing.assert_equal(v, casted_arr_data[k])

    casted_ret_arr = {k: v.astype(original_dtypes[k]) for k, v in ret_arr.items()}

    for k, v in casted_ret_arr.items():
        assert v.dtype == arr_data[k].dtype
        np.testing.assert_equal(v, arr_data[k])


@pytest.mark.parametrize(
    "input_mesh_file", ["./data/model_hexa_0.epc", "./data/model_hexa_ts_0_new.epc"]
)
@pytest.mark.asyncio
async def test_mesh(eclient: ETPClient, duri: DataspaceURI, input_mesh_file: str):
    model = rq.Model(input_mesh_file)
    assert model is not None

    hexa_uuids = model.uuids(obj_type="UnstructuredGridRepresentation")
    hexa = rug.HexaGrid(model, uuid=hexa_uuids[0])
    assert hexa is not None
    assert hexa.nodes_per_face is not None, "hexamesh object is incomplete"
    assert hexa.nodes_per_face_cl is not None, "hexamesh object is incomplete"
    assert hexa.faces_per_cell is not None, "hexamesh object is incomplete"
    assert hexa.faces_per_cell_cl is not None, "hexamesh object is incomplete"
    assert hexa.cell_face_is_right_handed is not None, "hexamesh object is incomplete"

    uuids = model.uuids(obj_type="ContinuousProperty")
    assert len(uuids) == len(set(uuids))

    prop_titles = list(set([rqp.Property(model, uuid=u).title for u in uuids]))
    uuids = model.uuids(obj_type="DiscreteProperty")

    prop_titles = list(
        set(prop_titles + [rqp.Property(model, uuid=u).title for u in uuids])
    )

    # The optional "points" (dynamic nodes) property is neither
    # ContinuousProperty nor DiscreteProperty: special treatment
    node_uuids = model.uuids(title="points")
    special_prop_titles = list(
        set([rqp.Property(model, uuid=u).title for u in node_uuids])
    )
    prop_titles = prop_titles + special_prop_titles
    rddms_uris, prop_uris = await put_epc_mesh(
        eclient, str(input_mesh_file), hexa.title, prop_titles, 23031, duri
    )

    (
        uns,
        points,
        nodes_per_face,
        nodes_per_face_cl,
        faces_per_cell,
        faces_per_cell_cl,
        cell_face_is_right_handed,
    ) = await get_epc_mesh(eclient, rddms_uris[0], rddms_uris[2])

    mesh_has_timeseries = len(rddms_uris) > 3 and len(str(rddms_uris[3])) > 0

    assert str(hexa.uuid) == str(uns.uuid), "returned mesh uuid must match"
    np.testing.assert_allclose(points, hexa.points_ref())  # type: ignore

    np.testing.assert_allclose(nodes_per_face, hexa.nodes_per_face)
    np.testing.assert_allclose(nodes_per_face_cl, hexa.nodes_per_face_cl)
    np.testing.assert_allclose(faces_per_cell, hexa.faces_per_cell)
    np.testing.assert_allclose(faces_per_cell_cl, hexa.faces_per_cell_cl)
    np.testing.assert_allclose(
        cell_face_is_right_handed, hexa.cell_face_is_right_handed
    )

    for key, value in prop_uris.items():
        found_indices = set()
        for prop_uri in value[1]:
            prop0, values = await get_epc_mesh_property(
                eclient, rddms_uris[0], prop_uri
            )
            assert prop0.supporting_representation.uuid == str(uns.uuid), (
                "property support must match the mesh"
            )
            time_index = prop0.time_index.index if prop0.time_index else -1
            assert time_index not in found_indices, f"Duplicate time index {time_index}"
            if mesh_has_timeseries:
                prop_uuids = model.uuids(title=key)
                prop_uuid = prop_uuids[time_index]
            else:
                prop_uuid = model.uuid(title=key)
            prop = rqp.Property(model, uuid=prop_uuid)

            continuous = prop.is_continuous()
            assert isinstance(prop0, ro.ContinuousProperty) == continuous, (
                "property types must match"
            )
            assert isinstance(prop0, ro.DiscreteProperty) == (not continuous), (
                "property types must match"
            )
            np.testing.assert_allclose(
                prop.array_ref(),
                values,
                err_msg=f"property {key} at time_index {time_index} does not match",
            )  # type: ignore
            found_indices.add(time_index)
