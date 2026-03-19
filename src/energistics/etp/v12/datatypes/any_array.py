import typing

import pydantic

import energistics.base
from energistics.etp.v12.datatypes.array_of_boolean import ArrayOfBoolean
from energistics.etp.v12.datatypes.array_of_double import ArrayOfDouble
from energistics.etp.v12.datatypes.array_of_float import ArrayOfFloat
from energistics.etp.v12.datatypes.array_of_int import ArrayOfInt
from energistics.etp.v12.datatypes.array_of_long import ArrayOfLong
from energistics.etp.v12.datatypes.array_of_string import ArrayOfString
from energistics.types import ETPBasicArrayType


def serialize_item(item: ETPBasicArrayType) -> typing.Any:
    if isinstance(item, ETPBasicArrayType):
        return (
            type(item).avro_schema["fullName"],
            item.model_dump(by_alias=True),
        )

    return item


cls_list: list[typing.Type[ETPBasicArrayType]] = [
    ArrayOfBoolean,
    ArrayOfInt,
    ArrayOfLong,
    ArrayOfFloat,
    ArrayOfDouble,
    ArrayOfString,
]


validator_map: dict[str, typing.Type[ETPBasicArrayType]] = {
    c.full_name: c for c in cls_list
}


def validate_item(item: typing.Any) -> typing.Any:
    if isinstance(item, (tuple, list)):
        cls = validator_map[item[0]]
        return cls(**item[1])

    return item


@energistics.base.add_avro_metadata
class AnyArray(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes",
        "name": "AnyArray",
        "fields": [
            {
                "name": "item",
                "type": [
                    "Energistics.Etp.v12.Datatypes.ArrayOfBoolean",
                    "Energistics.Etp.v12.Datatypes.ArrayOfInt",
                    "Energistics.Etp.v12.Datatypes.ArrayOfLong",
                    "Energistics.Etp.v12.Datatypes.ArrayOfFloat",
                    "Energistics.Etp.v12.Datatypes.ArrayOfDouble",
                    "Energistics.Etp.v12.Datatypes.ArrayOfString",
                    "bytes",
                ],
            }
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.AnyArray",
        "depends": [
            "Energistics.Etp.v12.Datatypes.ArrayOfBoolean",
            "Energistics.Etp.v12.Datatypes.ArrayOfInt",
            "Energistics.Etp.v12.Datatypes.ArrayOfLong",
            "Energistics.Etp.v12.Datatypes.ArrayOfFloat",
            "Energistics.Etp.v12.Datatypes.ArrayOfDouble",
            "Energistics.Etp.v12.Datatypes.ArrayOfString",
        ],
    }

    item: typing.Annotated[
        ETPBasicArrayType | bytes,
        pydantic.PlainSerializer(serialize_item),
        pydantic.BeforeValidator(validate_item),
    ]
