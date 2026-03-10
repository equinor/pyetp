import enum

import energistics.base

_avro_schema: energistics.base.AvroSchemaType = {
    "type": "enum",
    "namespace": "Energistics.Etp.v12.Datatypes.Object",
    "name": "RelationshipKind",
    "symbols": ["Primary", "Secondary", "Both"],
    "fullName": "Energistics.Etp.v12.Datatypes.Object.RelationshipKind",
    "depends": [],
}


class RelationshipKind(enum.StrEnum):
    PRIMARY = "Primary"
    SECONDARY = "Secondary"
    BOTH = "Both"
