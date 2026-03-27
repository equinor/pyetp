import io

import fastavro

import energistics.base
from energistics.etp.v12.datatypes import Uuid


def _avro_roundtrip(
    obj: energistics.base.ETPBaseModel | Uuid,
) -> energistics.base.ETPBaseModel | Uuid:
    cls = type(obj)
    with io.BytesIO() as fb:
        fastavro.write.schemaless_writer(
            fo=fb,
            schema=obj.avro_schema,
            record=obj.model_dump(by_alias=True),
        )
        bs = fb.getvalue()

    with io.BytesIO(bs) as fb:
        # TODO: Remove the `#type: ignore`-below once a new release of
        # `fastavro` is in place (greater than `1.12.1`).
        record = fastavro.read.schemaless_reader(
            fo=fb,
            writer_schema=cls.avro_schema,
            return_record_name=True,
            return_named_type_override=True,
        )  # type: ignore

        assert isinstance(record, dict | bytes)
        if isinstance(record, dict):
            ret_obj = cls(**record)
        else:
            # Handle the case when we get bytes in return.
            # Explicitly verify that `cls` is `Uuid`.
            assert issubclass(cls, Uuid)
            ret_obj = cls(record)

    return ret_obj


def avro_roundtrip(obj: energistics.base.ETPBaseModel) -> energistics.base.ETPBaseModel:
    ret_obj = _avro_roundtrip(obj)
    assert not isinstance(ret_obj, Uuid)
    return ret_obj


def avro_roundtrip_uuid(obj: Uuid) -> Uuid:
    ret_obj = _avro_roundtrip(obj)
    assert isinstance(ret_obj, Uuid)
    return ret_obj
