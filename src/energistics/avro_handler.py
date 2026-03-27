import gzip
import io
import logging
import typing
from collections.abc import Callable

import fastavro

import energistics.base
import energistics.etp.v12.protocol.core
import energistics.etp.v12.protocol.data_array
import energistics.etp.v12.protocol.dataspace
import energistics.etp.v12.protocol.discovery
import energistics.etp.v12.protocol.store
import energistics.etp.v12.protocol.transaction
from energistics.etp.v12.datatypes.message_header import MessageHeader

logger = logging.getLogger(__name__)


CORE_PROTOCOLS = [
    getattr(energistics.etp.v12.protocol.core, o)
    for o in energistics.etp.v12.protocol.core.__all__
]
DISCOVERY_PROTOCOLS = [
    getattr(energistics.etp.v12.protocol.discovery, o)
    for o in energistics.etp.v12.protocol.discovery.__all__
]
STORE_PROTOCOLS = [
    getattr(energistics.etp.v12.protocol.store, o)
    for o in energistics.etp.v12.protocol.store.__all__
]
DATA_ARRAY_PROTOCOLS = [
    getattr(energistics.etp.v12.protocol.data_array, o)
    for o in energistics.etp.v12.protocol.data_array.__all__
]
TRANSACTION_PROTOCOLS = [
    getattr(energistics.etp.v12.protocol.transaction, o)
    for o in energistics.etp.v12.protocol.transaction.__all__
]
DATASPACE_PROTOCOLS = [
    getattr(energistics.etp.v12.protocol.dataspace, o)
    for o in energistics.etp.v12.protocol.dataspace.__all__
]

PROTOCOLS = [
    *CORE_PROTOCOLS,
    *DISCOVERY_PROTOCOLS,
    *STORE_PROTOCOLS,
    *DATA_ARRAY_PROTOCOLS,
    *TRANSACTION_PROTOCOLS,
    *DATASPACE_PROTOCOLS,
]

PROTOCOLS_MAP: dict[
    tuple[energistics.base.Protocol, int],
    typing.Type[energistics.base.ETPBaseProtocolModel],
] = {(p._protocol, p._message_type): p for p in PROTOCOLS}


def get_schema_class(
    protocol: energistics.base.Protocol, message_type: int
) -> typing.Type[energistics.base.ETPBaseProtocolModel]:

    if message_type == 1000:
        return energistics.etp.v12.protocol.core.ProtocolException

    if message_type == 1001:
        return energistics.etp.v12.protocol.core.Acknowledge

    schema = PROTOCOLS_MAP.get((protocol, message_type))

    if schema is None:
        raise NotImplementedError(
            f"The schema with {protocol = } and {message_type = } has not been "
            "implemented yet"
        )

    return schema


def encode_message(
    header: MessageHeader,
    body: energistics.base.ETPBaseProtocolModel,
    compression_func: Callable[[bytes], bytes] | None = None,
) -> bytes:
    logger.debug(f"Encoding message with header: {header}")

    if header.is_compressed() and compression_func is None:
        raise ValueError(
            "Message header indicates that compression is used, but no compression "
            "function specified"
        )

    with io.BytesIO() as fo:
        fastavro.write.schemaless_writer(
            fo=fo,
            schema=header.avro_schema,
            record=header.model_dump(by_alias=True),
        )
        header_bytes = fo.getvalue()

    with io.BytesIO() as fo:
        fastavro.write.schemaless_writer(
            fo=fo,
            schema=body.avro_schema,
            record=body.model_dump(by_alias=True),
        )
        body_bytes = fo.getvalue()

    if header.is_compressed():
        assert compression_func is not None
        body_bytes = compression_func(body_bytes)

    return header_bytes + body_bytes


def decode_message(
    message: bytes, decompression_func: Callable[[bytes], bytes] | None = None
) -> tuple[MessageHeader, energistics.base.ETPBaseProtocolModel]:
    with io.BytesIO(message) as fo:
        # TODO: Remove the `#type: ignore`-below once a new release of
        # `fastavro` is in place (greater than `1.12.1`).
        header_record = fastavro.read.schemaless_reader(
            fo=fo,
            writer_schema=MessageHeader.avro_schema,
            return_record_name=True,
            return_named_type_override=True,
        )  # type: ignore
        assert isinstance(header_record, dict)
        header = MessageHeader(**header_record)
        body_bytes = fo.read()

    logger.debug(f"Decoding message with header {header}")

    if header.is_compressed() and decompression_func is None:
        raise ValueError(
            "Message header indicates that compression is used, but no decompression "
            "function specified"
        )

    body_cls = get_schema_class(header.protocol, header.message_type)

    if header.is_compressed():
        assert decompression_func is not None
        body_bytes = decompression_func(body_bytes)

    with io.BytesIO(body_bytes) as fo:
        # TODO: Remove the `#type: ignore`-below once a new release of
        # `fastavro` is in place (greater than `1.12.1`).
        body_record = fastavro.read.schemaless_reader(
            fo=fo,
            writer_schema=body_cls.avro_schema,
            return_record_name=True,
            return_named_type_override=True,
        )  # type: ignore
        assert isinstance(body_record, dict)
        body = body_cls(**body_record)

    return header, body


class CompressionAlgorithm(typing.Protocol):
    name: str

    @staticmethod
    def compress(obj: bytes) -> bytes: ...

    @staticmethod
    def decompress(raw_data: bytes) -> bytes: ...


class GzipCompression:
    name: str = "gzip"

    @staticmethod
    def compress(obj: bytes) -> bytes:
        return gzip.compress(obj)

    @staticmethod
    def decompress(raw_data: bytes) -> bytes:
        with io.BytesIO(raw_data) as b, gzip.open(b, "rb") as f:
            return f.read()
