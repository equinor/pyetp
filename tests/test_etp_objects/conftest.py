import io

import fastavro

import energistics.base


def avro_roundtrip(obj: energistics.base.ETPType) -> energistics.base.ETPType:
    cls = type(obj)
    with io.BytesIO() as fb:
        fastavro.write.schemaless_writer(
            fo=fb,
            schema=obj.avro_schema,
            record=obj.model_dump(by_alias=True),
        )
        bs = fb.getvalue()

    with io.BytesIO(bs) as fb:
        record = fastavro.read.schemaless_reader(
            fo=fb,
            writer_schema=cls.avro_schema,
            return_record_name=True,
            return_named_type_override=True,
        )

        assert isinstance(record, dict)
        ret_obj = cls(**record)

    return ret_obj
