import typing
import uuid

import pydantic

import energistics.base

UUIDType: typing.TypeAlias = (
    pydantic.UUID1 | pydantic.UUID3 | pydantic.UUID4 | pydantic.UUID5
)


def serialize_uuid(value: UUIDType) -> bytes:
    if isinstance(value, bytes):
        return value
    return value.bytes


def validate_uuid(value: typing.Any) -> typing.Any:
    if isinstance(value, bytes):
        return uuid.UUID(bytes=value)
    if isinstance(value, str):
        return uuid.UUID(value)
    if isinstance(value, uuid.UUID):
        return uuid.UUID(str(value))
    if isinstance(value, int):
        return uuid.UUID(int=value)

    raise ValueError(
        f"Unable to coerce uuid '{value}' of type '{type(value)}' into a "
        "uuid.UUID-object"
    )


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
        pydantic.PlainValidator(validate_uuid),
    ]
