import enum

import energistics.base

_avro_schema: energistics.base.AvroSchemaType = {
    "type": "enum",
    "namespace": "Energistics.Etp.v12.Datatypes.Object",
    "name": "ContextScopeKind",
    "symbols": ["self", "sources", "targets", "sourcesOrSelf", "targetsOrSelf"],
    "fullName": "Energistics.Etp.v12.Datatypes.Object.ContextScopeKind",
    "depends": [],
}


class ContextScopeKind(enum.StrEnum):
    SELF = "self"
    SOURCES = "sources"
    TARGETS = "targets"
    SOURCES_OR_SELF = "sourcesOrSelf"
    TARGETS_OR_SELF = "targetsOrSelf"
