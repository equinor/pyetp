import typing

import pydantic

import energistics.base

UUIDType: typing.TypeAlias = (
    pydantic.UUID1 | pydantic.UUID3 | pydantic.UUID4 | pydantic.UUID5
)


def serialize_uuid(value: UUIDType) -> bytes:
    return value.bytes


@energistics.base.add_avro_metadata
class Uuid(pydantic.RootModel[UUIDType], energistics.base.ETPMetaData):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "fixed",
        "namespace": "Energistics.Etp.v12.Datatypes",
        "name": "Uuid",
        "size": 16,
        "fullName": "Energistics.Etp.v12.Datatypes.Uuid",
        "depends": [],
    }

    root: typing.Annotated[
        UUIDType,
        pydantic.PlainSerializer(serialize_uuid),
    ]
