import enum

import energistics.base

_avro_schema: energistics.base.AvroSchemaType = {
    "type": "enum",
    "namespace": "Energistics.Etp.v12.Datatypes",
    "name": "AnyLogicalArrayType",
    "symbols": [
        "arrayOfBoolean",
        "arrayOfInt8",
        "arrayOfUInt8",
        "arrayOfInt16LE",
        "arrayOfInt32LE",
        "arrayOfInt64LE",
        "arrayOfUInt16LE",
        "arrayOfUInt32LE",
        "arrayOfUInt64LE",
        "arrayOfFloat32LE",
        "arrayOfDouble64LE",
        "arrayOfInt16BE",
        "arrayOfInt32BE",
        "arrayOfInt64BE",
        "arrayOfUInt16BE",
        "arrayOfUInt32BE",
        "arrayOfUInt64BE",
        "arrayOfFloat32BE",
        "arrayOfDouble64BE",
        "arrayOfString",
        "arrayOfCustom",
    ],
    "fullName": "Energistics.Etp.v12.Datatypes.AnyLogicalArrayType",
    "depends": [],
}


class AnyLogicalArrayType(enum.StrEnum):
    ARRAY_OF_BOOLEAN = "arrayOfBoolean"
    ARRAY_OF_INT8 = "arrayOfInt8"
    ARRAY_OF_UINT8 = "arrayOfUInt8"
    ARRAY_OF_INT16_LE = "arrayOfInt16LE"
    ARRAY_OF_INT32_LE = "arrayOfInt32LE"
    ARRAY_OF_INT64_LE = "arrayOfInt64LE"
    ARRAY_OF_UINT16_LE = "arrayOfUInt16LE"
    ARRAY_OF_UINT32_LE = "arrayOfUInt32LE"
    ARRAY_OF_UINT64_LE = "arrayOfUInt64LE"
    ARRAY_OF_FLOAT32_LE = "arrayOfFloat32LE"
    ARRAY_OF_DOUBLE64_LE = "arrayOfDouble64LE"
    ARRAY_OF_INT16_BE = "arrayOfInt16BE"
    ARRAY_OF_INT32_BE = "arrayOfInt32BE"
    ARRAY_OF_INT64_BE = "arrayOfInt64BE"
    ARRAY_OF_UINT16_BE = "arrayOfUInt16BE"
    ARRAY_OF_UINT32_BE = "arrayOfUInt32BE"
    ARRAY_OF_UINT64_BE = "arrayOfUInt64BE"
    ARRAY_OF_FLOAT32_BE = "arrayOfFloat32BE"
    ARRAY_OF_DOUBLE64_BE = "arrayOfDouble64BE"
    ARRAY_OF_STRING = "arrayOfString"
    ARRAY_OF_CUSTOM = "arrayOfCustom"
