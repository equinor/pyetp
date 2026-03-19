import enum

import energistics.base

_avro_schema: energistics.base.AvroSchemaType = {
    "type": "enum",
    "namespace": "Energistics.Etp.v12.Datatypes",
    "name": "AnyArrayType",
    "symbols": [
        "arrayOfBoolean",
        "arrayOfInt",
        "arrayOfLong",
        "arrayOfFloat",
        "arrayOfDouble",
        "arrayOfString",
        "bytes",
    ],
    "fullName": "Energistics.Etp.v12.Datatypes.AnyArrayType",
    "depends": [],
}


class AnyArrayType(enum.StrEnum):
    ARRAY_OF_BOOLEAN = "arrayOfBoolean"
    ARRAY_OF_INT = "arrayOfInt"
    ARRAY_OF_LONG = "arrayOfLong"
    ARRAY_OF_FLOAT = "arrayOfFloat"
    ARRAY_OF_DOUBLE = "arrayOfDouble"
    ARRAY_OF_STRING = "arrayOfString"
    BYTES = "bytes"
