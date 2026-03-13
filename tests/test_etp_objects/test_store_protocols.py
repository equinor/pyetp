import uuid

import numpy as np

import energistics.base
from energistics.etp.v12.datatypes import ArrayOfString
from energistics.etp.v12.datatypes.object import DataObject, PutResponse, Resource
from energistics.etp.v12.protocol.store import (
    Chunk,
    DeleteDataObjects,
    DeleteDataObjectsResponse,
    GetDataObjects,
    GetDataObjectsResponse,
    PutDataObjects,
    PutDataObjectsResponse,
)
from tests.test_etp_objects.conftest import avro_roundtrip


def test_get_data_objects() -> None:
    gdo = GetDataObjects(
        uris={
            "foowell": f"eml:///dataspace('foo/bar')/witsml21.Well({uuid.uuid4()!s})",
            "foogrid": (
                "eml:///dataspace('bar/foo')/resqml20.Grid2dRepresentation"
                f"({uuid.uuid4()!s})"
            ),
        },
    )

    assert gdo._protocol == energistics.base.Protocol.STORE
    assert gdo._message_type == 1
    assert not gdo._is_multipart

    assert gdo == avro_roundtrip(gdo)


def test_get_data_objects_response() -> None:
    gdor = GetDataObjectsResponse(
        data_objects={
            "foowell": DataObject(
                resource=Resource(
                    uri=f"eml:///dataspace('foo/bar')/witsml21.Well({uuid.uuid4()!s})",
                    name="Welly",
                    last_changed=10,
                    store_last_write=2000,
                    store_created=120,
                    active_status="Active",
                ),
                data=b"not-a-valid-object",
            ),
            "foogrid": DataObject(
                resource=Resource(
                    uri=(
                        "eml:///dataspace('bar/foo')/resqml20.Grid2dRepresentation"
                        f"({uuid.uuid4()!s})"
                    ),
                    name="Griddy",
                    last_changed=10,
                    store_last_write=2000,
                    store_created=120,
                    active_status="Active",
                ),
                blob_id=uuid.uuid4(),
            ),
        }
    )

    assert gdor._protocol == energistics.base.Protocol.STORE
    assert gdor._message_type == 4
    assert gdor._is_multipart

    assert gdor == avro_roundtrip(gdor)


def test_put_data_objects() -> None:
    pdo = PutDataObjects(
        data_objects={
            "foowell": DataObject(
                resource=Resource(
                    uri=f"eml:///dataspace('foo/bar')/witsml21.Well({uuid.uuid4()!s})",
                    name="Welly",
                    last_changed=10,
                    store_last_write=2000,
                    store_created=120,
                    active_status="Active",
                ),
                data=b"not-a-valid-object",
            ),
            "foogrid": DataObject(
                resource=Resource(
                    uri=(
                        "eml:///dataspace('bar/foo')/resqml20.Grid2dRepresentation"
                        f"({uuid.uuid4()!s})"
                    ),
                    name="Griddy",
                    last_changed=10,
                    store_last_write=2000,
                    store_created=120,
                    active_status="Active",
                ),
                blob_id=uuid.uuid4(),
            ),
        },
    )

    assert pdo._protocol == energistics.base.Protocol.STORE
    assert pdo._message_type == 2
    assert pdo._is_multipart

    assert pdo == avro_roundtrip(pdo)


def test_put_data_objects_response() -> None:
    pdor = PutDataObjectsResponse(
        success={
            "foowell": PutResponse(),
            "foogrid": PutResponse(
                created_contained_object_uris=["yes"],
                deleted_contained_object_uris=["possibly"],
                joined_contained_object_uris=["not likely"],
                unjoined_contained_object_uris=["no"],
            ),
        },
    )

    assert pdor._protocol == energistics.base.Protocol.STORE
    assert pdor._message_type == 9
    assert pdor._is_multipart

    assert pdor == avro_roundtrip(pdor)


def test_delete_data_objects() -> None:
    ddo = DeleteDataObjects(
        uris={
            "foowell": f"eml:///dataspace('foo/bar')/witsml21.Well({uuid.uuid4()!s})",
            "foogrid": (
                "eml:///dataspace('bar/foo')/resqml20.Grid2dRepresentation"
                f"({uuid.uuid4()!s})"
            ),
        },
    )

    assert ddo._protocol == energistics.base.Protocol.STORE
    assert ddo._message_type == 3
    assert not ddo._is_multipart

    assert ddo == avro_roundtrip(ddo)


def test_delete_data_objects_response() -> None:
    ddor = DeleteDataObjectsResponse(
        deleted_uris={
            "foowell": ArrayOfString(
                values=[
                    f"eml:///dataspace('foo/bar')/witsml21.Well({uuid.uuid4()!s})",
                    (
                        "eml:///dataspace('foo/bar')/eml20.EpcExternalPartReference"
                        f"({uuid.uuid4()!s})"
                    ),
                ],
            ),
            "foogrid": ArrayOfString(
                values=[
                    (
                        "eml:///dataspace('bar/foo')/resqml20.Grid2dRepresentation"
                        f"({uuid.uuid4()!s})"
                    ),
                ]
            ),
        },
    )

    assert ddor._protocol == energistics.base.Protocol.STORE
    assert ddor._message_type == 10
    assert ddor._is_multipart

    ret_ddor = avro_roundtrip(ddor)

    for k, v in ddor.deleted_uris.items():
        assert k in ret_ddor.deleted_uris
        np.testing.assert_equal(
            ddor.deleted_uris[k].values, ret_ddor.deleted_uris[k].values
        )


def test_chunk() -> None:
    c = Chunk(
        blob_id=uuid.uuid1(),
        data=b"foobar",
        final=False,
    )

    assert c._protocol == energistics.base.Protocol.STORE
    assert c._message_type == 8
    assert c._is_multipart

    assert c == avro_roundtrip(c)
