import uuid

import numpy as np

import energistics.base
from energistics.etp.v12.datatypes import (
    AnyArray,
    AnyArrayType,
    AnyLogicalArrayType,
    ArrayOfFloat,
)
from energistics.etp.v12.datatypes.data_array_types import (
    DataArray,
    DataArrayIdentifier,
    DataArrayMetadata,
    GetDataSubarraysType,
    PutDataArraysType,
    PutDataSubarraysType,
    PutUninitializedDataArrayType,
)
from energistics.etp.v12.protocol.data_array import (
    GetDataArrayMetadata,
    GetDataArrayMetadataResponse,
    GetDataArrays,
    GetDataArraysResponse,
    GetDataSubarrays,
    GetDataSubarraysResponse,
    PutDataArrays,
    PutDataArraysResponse,
    PutDataSubarrays,
    PutDataSubarraysResponse,
    PutUninitializedDataArrays,
    PutUninitializedDataArraysResponse,
)
from energistics.types import ETPBasicArrayType
from tests.test_etp_objects.conftest import avro_roundtrip


def test_get_data_array_metadata() -> None:
    gdam = GetDataArrayMetadata(
        data_arrays={
            "foowell": DataArrayIdentifier(
                uri=(
                    "eml:///dataspace('foo/bar')/eml20.EpcExternalPartReference"
                    f"({uuid.uuid4()!s})"
                ),
                path_in_resource=f"/RESQML/{uuid.uuid4()!s}/points",
            ),
        },
    )

    assert gdam._protocol == energistics.base.Protocol.DATA_ARRAY
    assert gdam._message_type == 6
    assert not gdam._is_multipart

    assert gdam == avro_roundtrip(gdam)


def test_get_data_array_metadata_response() -> None:
    gdamr = GetDataArrayMetadataResponse(
        array_metadata={
            "foowell": DataArrayMetadata(
                dimensions=(1, 2, 3),
                transport_array_type=AnyArrayType.ARRAY_OF_FLOAT,
                logical_array_type=AnyLogicalArrayType.ARRAY_OF_FLOAT32_LE,
                store_last_write=250,
                store_created=150,
            ),
        },
    )

    assert gdamr._protocol == energistics.base.Protocol.DATA_ARRAY
    assert gdamr._message_type == 7
    assert gdamr._is_multipart

    assert gdamr == avro_roundtrip(gdamr)


def test_get_data_arrays() -> None:
    gda = GetDataArrays(
        data_arrays={
            "foowell": DataArrayIdentifier(
                uri=(
                    "eml:///dataspace('foo/bar')/eml20.EpcExternalPartReference"
                    f"({uuid.uuid4()!s})"
                ),
                path_in_resource=f"/RESQML/{uuid.uuid4()!s}/points",
            ),
        },
    )

    assert gda._protocol == energistics.base.Protocol.DATA_ARRAY
    assert gda._message_type == 2
    assert not gda._is_multipart

    assert gda == avro_roundtrip(gda)


def test_get_data_arrays_response() -> None:
    arr = np.random.random((123, 37)).reshape(123, 37).astype(np.float32)
    arrb = arr.tobytes()

    gdar = GetDataArraysResponse(
        data_arrays={
            "foowell": DataArray(
                dimensions=arr.shape,
                data=AnyArray(
                    item=ArrayOfFloat(values=np.ravel(arr)),
                ),
            ),
            "foobytes": DataArray(
                dimensions=[len(arrb)],
                data=AnyArray(
                    item=arrb,
                ),
            ),
        },
    )

    assert gdar._protocol == energistics.base.Protocol.DATA_ARRAY
    assert gdar._message_type == 1
    assert gdar._is_multipart

    ret_gdar = avro_roundtrip(gdar)
    assert type(ret_gdar) is type(gdar)
    assert len(ret_gdar.data_arrays) == len(gdar.data_arrays) and list(
        ret_gdar.data_arrays
    ) == list(gdar.data_arrays)
    for k in list(ret_gdar.data_arrays):
        assert ret_gdar.data_arrays[k].dimensions == gdar.data_arrays[k].dimensions
        assert type(ret_gdar.data_arrays[k].data.item) is type(
            gdar.data_arrays[k].data.item
        )
        ret_array = ret_gdar.data_arrays[k].data.item
        array = gdar.data_arrays[k].data.item

        if type(ret_array) is bytes:
            assert ret_array == array
        else:
            assert isinstance(ret_array, ETPBasicArrayType)
            assert isinstance(array, ETPBasicArrayType)

            np.testing.assert_equal(ret_array.values, array.values)


def test_get_data_subarrays() -> None:
    gds = GetDataSubarrays(
        data_subarrays={
            "foowell": GetDataSubarraysType(
                uid=DataArrayIdentifier(
                    uri=(
                        "eml:///dataspace('foo/bar')/eml20.EpcExternalPartReference"
                        f"({uuid.uuid4()!s})"
                    ),
                    path_in_resource=f"/RESQML/{uuid.uuid4()!s}/points",
                ),
                starts=[0, 0],
                counts=[1, 25],
            ),
        },
    )

    assert gds._protocol == energistics.base.Protocol.DATA_ARRAY
    assert gds._message_type == 3
    assert not gds._is_multipart

    assert gds == avro_roundtrip(gds)


def test_get_data_subarrays_response() -> None:
    arr = np.random.random((123, 37)).reshape(123, 37).astype(np.float32)
    arrb = arr.tobytes()

    gdsr = GetDataSubarraysResponse(
        data_subarrays={
            "foowell": DataArray(
                dimensions=arr.shape,
                data=AnyArray(
                    item=ArrayOfFloat(values=np.ravel(arr)),
                ),
            ),
            "foobytes": DataArray(
                dimensions=[len(arrb)],
                data=AnyArray(
                    item=arrb,
                ),
            ),
        },
    )

    assert gdsr._protocol == energistics.base.Protocol.DATA_ARRAY
    assert gdsr._message_type == 8
    assert gdsr._is_multipart

    ret_gdsr = avro_roundtrip(gdsr)
    assert type(ret_gdsr) is type(gdsr)
    assert len(ret_gdsr.data_subarrays) == len(gdsr.data_subarrays) and list(
        ret_gdsr.data_subarrays
    ) == list(gdsr.data_subarrays)
    for k in list(ret_gdsr.data_subarrays):
        assert (
            ret_gdsr.data_subarrays[k].dimensions == gdsr.data_subarrays[k].dimensions
        )
        assert type(ret_gdsr.data_subarrays[k].data.item) is type(
            gdsr.data_subarrays[k].data.item
        )
        ret_array = ret_gdsr.data_subarrays[k].data.item
        array = gdsr.data_subarrays[k].data.item

        if type(ret_array) is bytes:
            assert ret_array == array
        else:
            assert isinstance(ret_array, ETPBasicArrayType)
            assert isinstance(array, ETPBasicArrayType)

            np.testing.assert_equal(ret_array.values, array.values)


def test_put_data_arrays() -> None:
    arr = np.random.random((12, 7)).reshape(12, 7).astype(np.float32)
    arrb = arr.tobytes()

    pda = PutDataArrays(
        data_arrays={
            "foowell": PutDataArraysType(
                uid=DataArrayIdentifier(
                    uri=(
                        "eml:///dataspace('foo/bar')/eml20.EpcExternalPartReference"
                        f"({uuid.uuid4()!s})"
                    ),
                    path_in_resource=f"/RESQML/{uuid.uuid4()!s}/points",
                ),
                array=DataArray(
                    dimensions=arr.shape,
                    data=AnyArray(
                        item=ArrayOfFloat(values=np.ravel(arr)),
                    ),
                ),
            ),
            "foobytes": PutDataArraysType(
                uid=DataArrayIdentifier(
                    uri=(
                        "eml:///dataspace('foo/bar')/eml20.EpcExternalPartReference"
                        f"({uuid.uuid4()!s})"
                    ),
                    path_in_resource=f"/RESQML/{uuid.uuid4()!s}/points",
                ),
                array=DataArray(
                    dimensions=[len(arrb)],
                    data=AnyArray(
                        item=arrb,
                    ),
                ),
            ),
        },
    )

    assert pda._protocol == energistics.base.Protocol.DATA_ARRAY
    assert pda._message_type == 4
    assert not pda._is_multipart

    ret_pda = avro_roundtrip(pda)
    assert type(ret_pda) is type(pda)
    assert len(ret_pda.data_arrays) == len(pda.data_arrays) and list(
        ret_pda.data_arrays
    ) == list(pda.data_arrays)
    for k in list(ret_pda.data_arrays):
        ret_da = ret_pda.data_arrays[k].array
        da = pda.data_arrays[k].array

        assert ret_da.dimensions == da.dimensions
        assert type(ret_da.data.item) is type(da.data.item)
        ret_array = ret_da.data.item
        array = da.data.item

        if type(ret_array) is bytes:
            assert ret_array == array
        else:
            assert isinstance(ret_array, ETPBasicArrayType)
            assert isinstance(array, ETPBasicArrayType)

            np.testing.assert_equal(ret_array.values, array.values)


def test_put_data_arrays_response() -> None:
    pdar = PutDataArraysResponse(
        success={"foowell": ""},
    )

    assert pdar._protocol == energistics.base.Protocol.DATA_ARRAY
    assert pdar._message_type == 10
    assert pdar._is_multipart

    assert pdar == avro_roundtrip(pdar)


def test_put_data_subarrays() -> None:
    arr = np.random.random((12, 7)).reshape(12, 7).astype(np.float32)
    arrb = arr.tobytes()

    pds = PutDataSubarrays(
        data_subarrays={
            "foowell": PutDataSubarraysType(
                uid=DataArrayIdentifier(
                    uri=(
                        "eml:///dataspace('foo/bar')/eml20.EpcExternalPartReference"
                        f"({uuid.uuid4()!s})"
                    ),
                    path_in_resource=f"/RESQML/{uuid.uuid4()!s}/points",
                ),
                data=AnyArray(
                    item=ArrayOfFloat(values=np.ravel(arr)),
                ),
                starts=[0, 1],
                counts=[10, 5],
            ),
            "foobytes": PutDataSubarraysType(
                uid=DataArrayIdentifier(
                    uri=(
                        "eml:///dataspace('foo/bar')/eml20.EpcExternalPartReference"
                        f"({uuid.uuid4()!s})"
                    ),
                    path_in_resource=f"/RESQML/{uuid.uuid4()!s}/points",
                ),
                data=AnyArray(
                    item=arrb,
                ),
                starts=[16],
                counts=[32],
            ),
        },
    )

    assert pds._protocol == energistics.base.Protocol.DATA_ARRAY
    assert pds._message_type == 5
    assert not pds._is_multipart

    ret_pds = avro_roundtrip(pds)
    assert type(ret_pds) is type(pds)
    assert len(ret_pds.data_subarrays) == len(pds.data_subarrays) and list(
        ret_pds.data_subarrays
    ) == list(pds.data_subarrays)
    for k in list(ret_pds.data_subarrays):
        assert ret_pds.data_subarrays[k].starts == pds.data_subarrays[k].starts
        assert ret_pds.data_subarrays[k].counts == pds.data_subarrays[k].counts
        ret_ds = ret_pds.data_subarrays[k].data
        ds = pds.data_subarrays[k].data

        assert type(ret_ds.item) is type(ds.item)
        ret_array = ret_ds.item
        array = ds.item

        if type(ret_array) is bytes:
            assert ret_array == array
        else:
            assert isinstance(ret_array, ETPBasicArrayType)
            assert isinstance(array, ETPBasicArrayType)

            np.testing.assert_equal(ret_array.values, array.values)


def test_put_data_subarrays_response() -> None:
    pdsr = PutDataSubarraysResponse(
        success={"foowell": ""},
    )

    assert pdsr._protocol == energistics.base.Protocol.DATA_ARRAY
    assert pdsr._message_type == 11
    assert pdsr._is_multipart

    assert pdsr == avro_roundtrip(pdsr)


def test_put_uninitialized_data_arrays() -> None:
    arr_shape = (123, 64)
    arrb_shape = int(np.prod(arr_shape))

    puda = PutUninitializedDataArrays(
        data_arrays={
            "foowell": PutUninitializedDataArrayType(
                uid=DataArrayIdentifier(
                    uri=(
                        "eml:///dataspace('foo/bar')/eml20.EpcExternalPartReference"
                        f"({uuid.uuid4()!s})"
                    ),
                    path_in_resource=f"/RESQML/{uuid.uuid4()!s}/points",
                ),
                metadata=DataArrayMetadata(
                    dimensions=arr_shape,
                    transport_array_type="arrayOfFloat",
                    logical_array_type="arrayOfFloat32LE",
                    store_last_write=12345,
                    store_created=1234,
                ),
            ),
            "foobytes": PutUninitializedDataArrayType(
                uid=DataArrayIdentifier(
                    uri=(
                        "eml:///dataspace('foo/bar')/eml20.EpcExternalPartReference"
                        f"({uuid.uuid4()!s})"
                    ),
                    path_in_resource=f"/RESQML/{uuid.uuid4()!s}/points",
                ),
                metadata=DataArrayMetadata(
                    dimensions=[arrb_shape],
                    transport_array_type="bytes",
                    logical_array_type="arrayOfFloat32LE",
                    store_last_write=12345,
                    store_created=1234,
                ),
            ),
        },
    )

    assert puda._protocol == energistics.base.Protocol.DATA_ARRAY
    assert puda._message_type == 9
    assert not puda._is_multipart

    assert puda == avro_roundtrip(puda)


def test_put_uninitialized_data_arrays_response() -> None:
    pudar = PutUninitializedDataArraysResponse(
        success={"foowell": ""},
    )

    assert pudar._protocol == energistics.base.Protocol.DATA_ARRAY
    assert pudar._message_type == 12
    assert pudar._is_multipart

    assert pudar == avro_roundtrip(pudar)
