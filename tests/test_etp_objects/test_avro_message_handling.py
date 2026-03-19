import gzip

import pytest

import energistics.etp.v12.protocol.core
import energistics.etp.v12.protocol.transaction
from energistics.avro_handler import (
    GzipCompression,
    decode_message,
    encode_message,
)
from energistics.etp.v12.datatypes import MessageHeader
from energistics.etp.v12.datatypes.message_header import MessageHeaderFlags


def test_encode_decode_roundtrip() -> None:
    body = energistics.etp.v12.protocol.core.CloseSession(reason="roundtrip-test")
    header = MessageHeader.from_etp_protocol_body(
        body, message_flags=MessageHeaderFlags.FIN, message_id=4
    )

    message = encode_message(header, body)

    ret_header, ret_body = decode_message(message)
    assert header == ret_header
    assert body == ret_body


def test_encode_decode_gzip_roundtrip() -> None:
    body = energistics.etp.v12.protocol.transaction.StartTransaction(
        dataspace_uris=["eml:///dataspace('foo/bar')"],
        read_only=False,
    )
    header = MessageHeader.from_etp_protocol_body(
        body,
        message_id=20,
        message_flags=MessageHeaderFlags.FIN | MessageHeaderFlags.COMPRESSED,
    )

    with pytest.raises(ValueError):
        encode_message(header, body)

    message = encode_message(header, body, compression_func=GzipCompression.compress)
    assert message == encode_message(header, body, compression_func=gzip.compress)

    with pytest.raises(ValueError):
        decode_message(message)

    ret_header, ret_body = decode_message(
        message, decompression_func=GzipCompression.decompress
    )
    assert ret_header == header
    assert ret_body == body
