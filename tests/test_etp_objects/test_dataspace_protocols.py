import datetime

import pytest

import energistics.base
from energistics.etp.v12.datatypes.object import Dataspace
from energistics.etp.v12.protocol.dataspace import (
    DeleteDataspaces,
    DeleteDataspacesResponse,
    GetDataspaces,
    GetDataspacesResponse,
    PutDataspaces,
    PutDataspacesResponse,
)
from tests.test_etp_objects.conftest import avro_roundtrip


def test_get_dataspaces() -> None:
    gd = GetDataspaces(store_last_write_filter=10)

    assert gd._protocol == energistics.base.Protocol.DATASPACE
    assert gd._message_type == 1
    assert not gd._is_multipart

    ret_gd = avro_roundtrip(gd)
    assert gd == ret_gd


def test_get_dataspaces_response() -> None:
    now_stamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1e6)
    path = "foo/bar"
    uri = f"eml:///dataspace('{path}')"

    gdr = GetDataspacesResponse(
        dataspaces=[
            Dataspace(
                uri=uri,
                path=path,
                store_last_write=now_stamp,
                store_created=now_stamp,
            ),
        ],
    )

    assert gdr._protocol == energistics.base.Protocol.DATASPACE
    assert gdr._message_type == 2
    assert gdr._is_multipart

    ret_gdr = avro_roundtrip(gdr)
    assert gdr == ret_gdr


def test_put_dataspaces() -> None:
    now_stamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1e6)
    path = "foo/bar"
    uri = f"eml:///dataspace('{path}')"

    pd = PutDataspaces(
        dataspaces={
            "foo": Dataspace(
                uri=uri,
                path=path,
                store_last_write=now_stamp,
                store_created=now_stamp,
            ),
        },
    )

    assert pd._protocol == energistics.base.Protocol.DATASPACE
    assert pd._message_type == 3
    assert not pd._is_multipart

    ret_pd = avro_roundtrip(pd)
    assert pd == ret_pd


def test_put_dataspaces_response() -> None:
    pdr = PutDataspacesResponse(success=dict(foo="", bar=""))

    assert pdr._protocol == energistics.base.Protocol.DATASPACE
    assert pdr._message_type == 6
    assert pdr._is_multipart

    ret_pdr = avro_roundtrip(pdr)
    assert pdr == ret_pdr


def test_delete_dataspaces() -> None:
    dd = DeleteDataspaces(
        uris={
            "foo": "eml:///dataspace('foo/bar')",
            "bar": "eml:///dataspace('bar/foo')",
        }
    )

    assert dd._protocol == energistics.base.Protocol.DATASPACE
    assert dd._message_type == 4
    assert not dd._is_multipart

    ret_dd = avro_roundtrip(dd)
    assert dd == ret_dd

    with pytest.raises(ExceptionGroup):
        DeleteDataspaces(
            uris={
                "foo": "eml://",
                "bar": "eml:///dataspace(bar/foo)",
            },
        )


def test_delete_dataspaces_response() -> None:
    ddr = DeleteDataspacesResponse(success=dict(foo="", bar=""))

    assert ddr._protocol == energistics.base.Protocol.DATASPACE
    assert ddr._message_type == 5
    assert ddr._is_multipart

    ret_ddr = avro_roundtrip(ddr)
    assert ddr == ret_ddr
