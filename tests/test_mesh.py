import asyncio
import pathlib
import re

import numpy as np
import pytest
from etptypes.energistics.etp.v12.datatypes.data_array_types.data_array_identifier import (
    DataArrayIdentifier,
)

import pyetp.utils_arrays
from pyetp.client import ETPClient
from pyetp.uri import DataspaceURI
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
