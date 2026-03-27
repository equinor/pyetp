import datetime
import random
import uuid

import numpy as np
import pydantic
import pytest

from energistics.etp.v12.datatypes import (
    AnyArray,
    AnySparseArray,
    AnySubarray,
    ArrayOfBoolean,
    ArrayOfBytes,
    ArrayOfDouble,
    ArrayOfFloat,
    ArrayOfInt,
    ArrayOfLong,
    ArrayOfNullableBoolean,
    ArrayOfNullableInt,
    ArrayOfNullableLong,
    ArrayOfString,
    Contact,
    DataValue,
    MessageHeader,
    ServerCapabilities,
    SupportedDataObject,
    SupportedProtocol,
    Uuid,
    Version,
)
from energistics.etp.v12.datatypes.data_value import (
    ItemType,
)
from energistics.etp.v12.datatypes.message_header import MessageHeaderFlags
from energistics.etp.v12.datatypes.object import DataObject, Dataspace, Resource
from energistics.etp.v12.protocol.core import CloseSession
from energistics.types import ETPArrayType, ETPBasicArrayType
from tests.test_etp_objects.conftest import avro_roundtrip, avro_roundtrip_uuid


@pytest.mark.parametrize(
    "array_instance",
    [
        ArrayOfBoolean(values=[True, True, False]),
        ArrayOfBytes(values=[b"\x00\x01\x04\xa4", b"\x76"]),
        ArrayOfDouble(values=[1.238532134e102, -1.2384932e-50]),
        ArrayOfFloat(values=[1.18e8, 13458.19, -0.000013]),
        ArrayOfInt(values=[1, -5, 102, 2**29 + 11]),
        ArrayOfLong(values=[2**54 - 1023, -(2**36) + 985, 1, -2]),
        ArrayOfNullableBoolean(values=[True, None, True, False, None]),
        ArrayOfNullableInt(values=[1, -5, -23, None, 2**13 - 27]),
        ArrayOfNullableLong(values=[2**34 - 23987, -12378596238, None]),
        ArrayOfString(values=["Hello", "goodbye"]),
    ],
)
def test_etp_array_types(
    array_instance: ETPArrayType,
) -> None:
    ret_array_instance = avro_roundtrip(array_instance)
    np.testing.assert_equal(array_instance.values, ret_array_instance.values)


@pytest.mark.parametrize(
    "item",
    [
        None,
        True,
        random.randint(a=-(2**31), b=(2**31 - 1)),
        random.randint(a=-(2**63), b=(2**63 - 1)),
        (random.random() - 0.5) * 2e15,
        (random.random() - 0.5) * 2e53,
        "foobar string",
        ArrayOfBoolean(values=[True, True, False]),
        ArrayOfBytes(values=[b"\x00\x01\x04\xa4", b"\x76"]),
        ArrayOfDouble(values=[1.238532134e102, -1.2384932e-50]),
        ArrayOfFloat(values=[1.18e8, 13458.19, -0.000013]),
        ArrayOfInt(values=[1, -5, 102, 2**29 + 11]),
        ArrayOfLong(values=[2**54 - 1023, -(2**36) + 985, 1, -2]),
        ArrayOfNullableBoolean(values=[True, None, True, False, None]),
        ArrayOfNullableInt(values=[1, -5, -23, None, 2**13 - 27]),
        ArrayOfNullableLong(values=[2**34 - 23987, -12378596238, None]),
        ArrayOfString(values=["Hello", "goodbye"]),
        b"foobarbytes",
        AnySparseArray(
            slices=[
                AnySubarray(
                    start=10,
                    slice=AnyArray(item=ArrayOfDouble(values=np.random.random(10))),
                ),
                AnySubarray(
                    start=23,
                    slice=AnyArray(
                        item=ArrayOfBoolean(values=[True, False, True, False, False])
                    ),
                ),
                AnySubarray(
                    start=1,
                    slice=AnyArray(
                        item=ArrayOfInt(
                            values=np.random.randint(low=-10, high=10, size=20).astype(
                                np.int32
                            )
                        ),
                    ),
                ),
                AnySubarray(
                    start=-5,
                    slice=AnyArray(
                        item=ArrayOfLong(
                            values=2**51 * np.random.randint(low=-10, high=10, size=20)
                        ),
                    ),
                ),
                AnySubarray(
                    start=523,
                    slice=AnyArray(
                        item=ArrayOfFloat(
                            values=np.random.random(10).astype(np.float32)
                        ),
                    ),
                ),
                AnySubarray(
                    start=0,
                    slice=AnyArray(
                        item=ArrayOfString(values=["foo", "bar", "baz", "hello"]),
                    ),
                ),
                AnySubarray(
                    start=0,
                    slice=AnyArray(
                        item=b"123adbnjkdsfhsd",
                    ),
                ),
            ],
        ),
    ],
)
def test_data_value(item: ItemType) -> None:
    dv = DataValue(item=item)
    ret_dv = avro_roundtrip(dv)

    assert type(ret_dv.item) is type(dv.item)

    if isinstance(
        dv.item,
        ETPArrayType,
    ):
        assert isinstance(ret_dv.item, ETPArrayType)
        assert isinstance(dv.item, ETPArrayType)
        np.testing.assert_equal(ret_dv.item.values, dv.item.values)
    elif isinstance(dv.item, AnySparseArray):
        assert isinstance(dv.item, AnySparseArray)
        assert isinstance(ret_dv.item, AnySparseArray)
        for ret_s, s in zip(ret_dv.item.slices, dv.item.slices):
            assert isinstance(s, AnySubarray) and isinstance(ret_s, AnySubarray)
            assert ret_s.start == s.start
            if isinstance(s.slice.item, bytes):
                assert s.slice.item == ret_s.slice.item
            else:
                assert isinstance(ret_s.slice.item, ETPBasicArrayType)
                assert isinstance(s.slice.item, ETPBasicArrayType)
                np.testing.assert_equal(ret_s.slice.item.values, s.slice.item.values)
    else:
        assert dv.item == ret_dv.item


def test_supported_protocol() -> None:
    sp_core = SupportedProtocol(
        protocol=0,
        role="server",
        protocol_capabilities={
            "MaxDataArraySize": DataValue(item=10),
        },
    )
    assert sp_core.protocol_version == Version(major=1, minor=2)

    ret_sp_core = avro_roundtrip(sp_core)
    assert sp_core == ret_sp_core

    with pytest.raises(pydantic.ValidationError):
        SupportedProtocol(
            protocol=30,
            role="server",
        )
    with pytest.raises(pydantic.ValidationError):
        SupportedProtocol(
            protocol=0,
            role="producer",
        )
    with pytest.raises(pydantic.ValidationError):
        SupportedProtocol(
            protocol=0,
            role="server",
            protocol_capabilities={
                "MinDataArraySize": DataValue(item=10),
            },
        )


def test_dataspace() -> None:
    now = datetime.datetime.now(datetime.timezone.utc)
    now_stamp = int(now.timestamp() * 1e6)

    path = "foo/bar"
    uri = f"eml:///dataspace('{path}')"

    ds = Dataspace(
        uri=uri,
        path=path,
        store_last_write=now_stamp,
        store_created=now_stamp,
    )

    ret_ds = avro_roundtrip(ds)
    assert ret_ds == ds
    assert (
        datetime.datetime.fromtimestamp(
            ret_ds.store_last_write / 1e6, tz=datetime.timezone.utc
        )
        == now
    )

    with pytest.raises(pydantic.ValidationError):
        Dataspace(
            uri=uri,
            store_last_write=now_stamp,
            store_created=now_stamp,
        )

    with pytest.raises(pydantic.ValidationError):
        Dataspace(
            uri=uri,
            path="foo/baz",
            store_last_write=now_stamp,
            store_created=now_stamp,
        )

    with pytest.raises(pydantic.ValidationError):
        Dataspace(
            uri="invalid-dataspace",
            store_last_write=now_stamp,
            store_created=now_stamp,
        )


def test_data_object() -> None:
    do = DataObject(
        resource=Resource(
            uri=f"eml:///dataspace('foo/bar')/witsml21.Well({uuid.uuid4()!s})",
            name="Welly",
            last_changed=10,
            store_last_write=2000,
            store_created=120,
            active_status="Active",
        ),
        data=b"not-a-valid-object",
    )

    assert do == avro_roundtrip(do)

    with pytest.raises(pydantic.ValidationError):
        DataObject(
            resource=Resource(
                uri=f"eml:///dataspace('foo/bar')/witsml21.Well({uuid.uuid4()!s})",
                name="Welly",
                last_changed=10,
                store_last_write=2000,
                store_created=120,
                active_status="Active",
            ),
        )

    with pytest.raises(pydantic.ValidationError):
        DataObject(
            resource=Resource(
                uri=f"eml:///dataspace('foo/bar')/witsml21.Well({uuid.uuid4()!s})",
                name="Welly",
                last_changed=10,
                store_last_write=2000,
                store_created=120,
                active_status="Active",
            ),
            blob_id=uuid.uuid4(),
            data=b"not-a-valid-object",
        )


def test_server_capabilites() -> None:
    sc = ServerCapabilities(
        application_name="test",
        application_version="1.2.3",
        contact_information=Contact(
            organization_name="org",
            contact_name="name",
            contact_phone="phone",
            contact_email="email",
        ),
        supported_compression=["gzip"],
        supported_encodings=["binary"],
        supported_formats=["xml"],
        supported_data_objects=[
            SupportedDataObject(
                qualified_type="eml20.*",
                data_object_capabilities={
                    "MaxDataObjectSize": DataValue(item=10),
                },
            ),
        ],
        supported_protocols=[
            SupportedProtocol(
                protocol=3,
                role="store",
            ),
            SupportedProtocol(
                protocol=9,
                role="store",
            ),
        ],
    )

    assert sc == avro_roundtrip(sc)


def test_messsage_header() -> None:
    mh = MessageHeader(
        protocol=1,
        message_type=2,
        correlation_id=0,
        message_id=2,
        message_flags=MessageHeaderFlags.FIN | MessageHeaderFlags.COMPRESSED,
    )

    assert mh == avro_roundtrip(mh)

    assert mh.is_final_message()
    assert mh.is_compressed()
    assert not mh.requests_acknowledgement()
    assert not mh.uses_extension_header()

    mh = MessageHeader.from_etp_protocol_body(
        CloseSession(reason="test"),
        message_flags=MessageHeaderFlags.FIN,
        message_id=46,
    )

    assert mh.protocol == CloseSession._protocol
    assert mh.message_type == CloseSession._message_type

    assert mh == avro_roundtrip(mh)

    assert mh.is_final_message()
    assert not mh.is_compressed()
    assert not mh.requests_acknowledgement()
    assert not mh.uses_extension_header()

    with pytest.raises(pydantic.ValidationError):
        MessageHeader.from_etp_protocol_body(
            CloseSession(),
            message_id=12,
            message_flags=MessageHeaderFlags.FIN | MessageHeaderFlags.COMPRESSED,
        )


def test_uuid() -> None:
    u = uuid.uuid4()
    etp_u_1 = Uuid(u)
    etp_u_2 = Uuid(u.bytes)
    etp_u_3 = Uuid(str(u))
    etp_u_4 = Uuid(etp_u_1)

    assert u == uuid.UUID(str(etp_u_1.root))
    assert etp_u_1 == etp_u_2 == etp_u_3 == etp_u_4

    assert etp_u_1 == avro_roundtrip_uuid(etp_u_1)
