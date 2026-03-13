import uuid

import pydantic
import pytest

import energistics.base
from energistics.etp.v12.datatypes import (
    DataValue,
    ErrorCode,
    SupportedDataObject,
    SupportedProtocol,
)
from energistics.etp.v12.protocol.core import (
    Acknowledge,
    Authorize,
    AuthorizeResponse,
    CloseSession,
    OpenSession,
    Ping,
    Pong,
    ProtocolException,
    RequestSession,
)
from tests.test_etp_objects.conftest import avro_roundtrip


def test_request_session() -> None:
    rs = RequestSession(
        application_name="test",
        application_version="1.2.3",
        client_instance_id=uuid.uuid4(),
        requested_protocols=[
            SupportedProtocol(
                protocol=3,
                role="store",
            ),
            SupportedProtocol(
                protocol=9,
                role="store",
            ),
        ],
        supported_data_objects=[
            SupportedDataObject(
                qualified_type="eml20.*",
                data_object_capabilities={
                    "MaxDataObjectSize": DataValue(item=10),
                },
            ),
        ],
        supported_compression=["gzip"],
    )
    assert rs._protocol == energistics.base.Protocol.CORE
    assert rs._message_type == 1
    assert not rs._is_multipart

    ret_rs = avro_roundtrip(rs)
    assert rs == ret_rs


def test_open_session() -> None:
    os = OpenSession(
        application_name="test",
        application_version="1.2.3",
        server_instance_id=uuid.uuid4(),
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
        supported_data_objects=[
            SupportedDataObject(
                qualified_type="eml20.*",
                data_object_capabilities={
                    "MaxDataObjectSize": DataValue(item=10),
                    "SupportsGet": DataValue(item=True),
                    "SupportsPut": DataValue(item=True),
                    "SupportsDelete": DataValue(item=False),
                },
            ),
        ],
        supported_compression="gzip",
        earliest_retained_change_time=10,
        session_id=uuid.uuid1(),
        endpoint_capabilities={
            "MaxWebSocketMessagePayloadSize": DataValue(item=500000),
            "MaxMessagePayloadUncompressedSize": DataValue(item=50000),
        },
    )
    assert os._protocol == energistics.base.Protocol.CORE
    assert os._message_type == 2
    assert not os._is_multipart

    ret_os = avro_roundtrip(os)
    assert os == ret_os


def test_close_session() -> None:
    cs = CloseSession(reason="test")

    assert cs._protocol == energistics.base.Protocol.CORE
    assert cs._message_type == 5
    assert not cs._is_multipart

    ret_cs = avro_roundtrip(cs)
    assert cs == ret_cs


def test_authorize() -> None:
    token = "eadhb..."
    a = Authorize(authorization=f"Bearer {token}")

    assert a._protocol == energistics.base.Protocol.CORE
    assert a._message_type == 6
    assert not a._is_multipart

    ret_a = avro_roundtrip(a)
    assert a == ret_a


def test_authorize_reponse() -> None:
    ar = AuthorizeResponse(success=True)

    assert ar._protocol == energistics.base.Protocol.CORE
    assert ar._message_type == 7
    assert not ar._is_multipart

    ret_ar = avro_roundtrip(ar)
    assert ar == ret_ar

    with pytest.raises(pydantic.ValidationError):
        AuthorizeResponse(success=True, challenges=["bar"])


def test_ping() -> None:
    p = Ping()
    assert p._protocol == energistics.base.Protocol.CORE
    assert p._message_type == 8
    assert not p._is_multipart

    ret_p = avro_roundtrip(p)
    assert p == ret_p


def test_pong() -> None:
    p = Pong()
    assert p._protocol == energistics.base.Protocol.CORE
    assert p._message_type == 9
    assert not p._is_multipart

    ret_p = avro_roundtrip(p)
    assert p == ret_p


def test_protocol_exception() -> None:
    pe = ProtocolException(error=dict(message="foo", code=1))

    assert pe._protocol == energistics.base.Protocol.CORE
    assert pe._message_type == 1000
    assert pe._is_multipart

    ret_pe = avro_roundtrip(pe)
    assert pe == ret_pe

    assert pe.error is not None
    assert pe.error.code == ErrorCode.ENOROLE

    with pytest.raises(pydantic.ValidationError):
        ProtocolException(error=None)
    with pytest.raises(pydantic.ValidationError):
        ProtocolException(
            error=dict(message="foo", code=2),
            errors={"bar": dict(message="bar", code=3)},
        )


def test_acknowledge() -> None:
    a = Acknowledge()

    assert a._protocol == energistics.base.Protocol.CORE
    assert a._message_type == 1001
    assert not a._is_multipart

    ret_a = avro_roundtrip(a)
    assert a == ret_a
