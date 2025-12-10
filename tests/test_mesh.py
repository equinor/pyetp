import asyncio
import pathlib
import re
import sys

import numpy as np
import pytest

import pyetp.utils_arrays
from pyetp.client import ETPClient
from pyetp.uri import DataspaceURI
from resqml_objects.epc_readers import (
    get_arrays_and_paths_in_hdf_file,
    get_resqml_v201_objects,
)

data_path = pathlib.Path("data")


@pytest.mark.skipif(
    (sys.version_info.major, sys.version_info.minor) == (3, 10),
    reason="This test requires Python 3.11 or higher",
)
@pytest.mark.parametrize(
    "input_mesh_file",
    [
        data_path / "model_hexa_0.epc",
        data_path / "model_hexa_ts_0_new.epc",
    ],
)
@pytest.mark.asyncio
async def test_mesh_raw(
    etp_client: ETPClient, dataspace_uri: DataspaceURI, input_mesh_file: pathlib.Path
):
    robjs = get_resqml_v201_objects(input_mesh_file)
    input_hdf_file = input_mesh_file.with_suffix(".h5")
    arr_data = get_arrays_and_paths_in_hdf_file(input_hdf_file)

    original_dtypes = {}
    casted_arr_data = {}
    for k, v in arr_data.items():
        original_dtypes[k] = v.dtype
        casted_arr_data[k] = v.astype(pyetp.utils_arrays.get_valid_dtype_cast(v))

    transaction_uuid = await etp_client.start_transaction(
        dataspace_uri=dataspace_uri, read_only=False
    )

    uris = await etp_client.put_resqml_objects(*robjs, dataspace_uri=dataspace_uri)
    epc_uris = list(filter(lambda u: "EpcExternalPartReference" in str(u), uris))
    assert len(epc_uris) == 1
    epc_uri = str(epc_uris[0])

    tasks = []
    for pir, data in casted_arr_data.items():
        tasks.append(
            etp_client.upload_array(epc_uri=epc_uri, path_in_resource=pir, data=data)
        )

    await asyncio.gather(*tasks)

    await etp_client.commit_transaction(transaction_uuid=transaction_uuid)

    ret_robjs = await etp_client.get_resqml_objects(*uris)

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
            etp_client.download_array(epc_uri=epc_uri, path_in_resource=ret_pir)
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
