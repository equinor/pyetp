import uuid

import pytest

import energistics.base
from energistics.etp.v12.protocol.transaction import (
    CommitTransaction,
    CommitTransactionResponse,
    RollbackTransaction,
    RollbackTransactionResponse,
    StartTransaction,
    StartTransactionResponse,
)
from tests.test_etp_objects.conftest import avro_roundtrip


def test_start_transaction() -> None:
    st = StartTransaction(
        read_only=True,
        message="foo",
        dataspace_uris=[
            "eml:///dataspace('foo/bar')",
            "eml:///dataspace('bar/foo')",
        ],
    )
    assert st._protocol == energistics.base.Protocol.TRANSACTION
    assert st._message_type == 1
    assert not st._is_multipart

    assert st == avro_roundtrip(st)

    with pytest.raises(ExceptionGroup):
        StartTransaction(
            read_only=False,
            dataspace_uris=[
                "not-valid",
            ],
        )


def test_start_transaction_response() -> None:
    _str = StartTransactionResponse(
        transaction_uuid=uuid.uuid1(),
    )

    assert _str.successful
    assert _str.failure_reason == ""

    assert _str._protocol == energistics.base.Protocol.TRANSACTION
    assert _str._message_type == 2
    assert not _str._is_multipart

    assert _str == avro_roundtrip(_str)


def test_commit_transaction() -> None:
    ct = CommitTransaction(
        transaction_uuid=uuid.uuid1(),
    )
    assert ct._protocol == energistics.base.Protocol.TRANSACTION
    assert ct._message_type == 3
    assert not ct._is_multipart

    assert ct == avro_roundtrip(ct)


def test_commit_transaction_response() -> None:
    ctr = CommitTransactionResponse(
        transaction_uuid=uuid.uuid1(),
    )

    assert ctr.successful
    assert ctr.failure_reason == ""

    assert ctr._protocol == energistics.base.Protocol.TRANSACTION
    assert ctr._message_type == 5
    assert not ctr._is_multipart

    assert ctr == avro_roundtrip(ctr)


def test_rollback_transaction() -> None:
    rt = RollbackTransaction(
        transaction_uuid=uuid.uuid1(),
    )
    assert rt._protocol == energistics.base.Protocol.TRANSACTION
    assert rt._message_type == 4
    assert not rt._is_multipart

    assert rt == avro_roundtrip(rt)


def test_rollback_transaction_response() -> None:
    rtr = RollbackTransactionResponse(
        transaction_uuid=uuid.uuid1(),
    )

    assert rtr.successful
    assert rtr.failure_reason == ""

    assert rtr._protocol == energistics.base.Protocol.TRANSACTION
    assert rtr._message_type == 6
    assert not rtr._is_multipart

    assert rtr == avro_roundtrip(rtr)
