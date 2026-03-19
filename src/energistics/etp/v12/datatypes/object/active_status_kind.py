import enum

import energistics.base

_avro_schema: energistics.base.AvroSchemaType = {
    "type": "enum",
    "namespace": "Energistics.Etp.v12.Datatypes.Object",
    "name": "ActiveStatusKind",
    "symbols": ["Active", "Inactive"],
    "fullName": "Energistics.Etp.v12.Datatypes.Object.ActiveStatusKind",
    "depends": [],
}


class ActiveStatusKind(enum.StrEnum):
    ACTIVE = "Active"
    INACTIVE = "Inactive"
