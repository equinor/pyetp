import typing

import pydantic

import energistics.base
from energistics.etp.v12.datatypes.any_sparse_array import AnySparseArray
from energistics.etp.v12.datatypes.array_of_boolean import ArrayOfBoolean
from energistics.etp.v12.datatypes.array_of_bytes import ArrayOfBytes
from energistics.etp.v12.datatypes.array_of_double import ArrayOfDouble
from energistics.etp.v12.datatypes.array_of_float import ArrayOfFloat
from energistics.etp.v12.datatypes.array_of_int import ArrayOfInt
from energistics.etp.v12.datatypes.array_of_long import ArrayOfLong
from energistics.etp.v12.datatypes.array_of_nullable_boolean import (
    ArrayOfNullableBoolean,
)
from energistics.etp.v12.datatypes.array_of_nullable_int import ArrayOfNullableInt
from energistics.etp.v12.datatypes.array_of_nullable_long import ArrayOfNullableLong
from energistics.etp.v12.datatypes.array_of_string import ArrayOfString
from energistics.types import ETPArrayType

ETPExtendedArrayType: typing.TypeAlias = ETPArrayType | AnySparseArray
ItemPrimitiveTypes: typing.TypeAlias = bool | int | float | str | bytes | None


ItemType: typing.TypeAlias = ETPExtendedArrayType | ItemPrimitiveTypes


def serialize_item(item: ItemType) -> typing.Any:
    if isinstance(item, ETPExtendedArrayType):
        return (
            type(item).avro_schema["fullName"],
            item.model_dump(by_alias=True),
        )

    return item


cls_list: list[typing.Type[ETPExtendedArrayType]] = [
    ArrayOfBoolean,
    ArrayOfNullableBoolean,
    ArrayOfInt,
    ArrayOfNullableInt,
    ArrayOfLong,
    ArrayOfNullableLong,
    ArrayOfFloat,
    ArrayOfDouble,
    ArrayOfString,
    ArrayOfBytes,
    AnySparseArray,
]


validator_map: dict[str, typing.Type[ETPExtendedArrayType]] = {
    c.full_name: c for c in cls_list
}


def validate_item(item: typing.Any) -> typing.Any:
    if isinstance(item, (tuple, list)):
        cls = validator_map.get(item[0])
        if cls is not None:
            return cls(**item[1])

    return item


@energistics.base.add_avro_metadata
class DataValue(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes",
        "name": "DataValue",
        "fields": [
            {
                "name": "item",
                "type": [
                    "null",
                    "boolean",
                    "int",
                    "long",
                    "float",
                    "double",
                    "string",
                    "Energistics.Etp.v12.Datatypes.ArrayOfBoolean",
                    "Energistics.Etp.v12.Datatypes.ArrayOfNullableBoolean",
                    "Energistics.Etp.v12.Datatypes.ArrayOfInt",
                    "Energistics.Etp.v12.Datatypes.ArrayOfNullableInt",
                    "Energistics.Etp.v12.Datatypes.ArrayOfLong",
                    "Energistics.Etp.v12.Datatypes.ArrayOfNullableLong",
                    "Energistics.Etp.v12.Datatypes.ArrayOfFloat",
                    "Energistics.Etp.v12.Datatypes.ArrayOfDouble",
                    "Energistics.Etp.v12.Datatypes.ArrayOfString",
                    "Energistics.Etp.v12.Datatypes.ArrayOfBytes",
                    "bytes",
                    "Energistics.Etp.v12.Datatypes.AnySparseArray",
                ],
            }
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.DataValue",
        "depends": [
            "Energistics.Etp.v12.Datatypes.ArrayOfBoolean",
            "Energistics.Etp.v12.Datatypes.ArrayOfNullableBoolean",
            "Energistics.Etp.v12.Datatypes.ArrayOfInt",
            "Energistics.Etp.v12.Datatypes.ArrayOfNullableInt",
            "Energistics.Etp.v12.Datatypes.ArrayOfLong",
            "Energistics.Etp.v12.Datatypes.ArrayOfNullableLong",
            "Energistics.Etp.v12.Datatypes.ArrayOfFloat",
            "Energistics.Etp.v12.Datatypes.ArrayOfDouble",
            "Energistics.Etp.v12.Datatypes.ArrayOfString",
            "Energistics.Etp.v12.Datatypes.ArrayOfBytes",
            "Energistics.Etp.v12.Datatypes.AnySparseArray",
        ],
    }

    item: typing.Annotated[
        ItemType,
        pydantic.BeforeValidator(validate_item),
        pydantic.PlainSerializer(serialize_item),
    ]
