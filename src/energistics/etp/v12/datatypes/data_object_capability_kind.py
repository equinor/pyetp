import enum
import typing

import energistics.base

_avro_schema: energistics.base.AvroSchemaType = {
    "type": "enum",
    "namespace": "Energistics.Etp.v12.Datatypes",
    "name": "DataObjectCapabilityKind",
    "symbols": [
        "ActiveTimeoutPeriod",
        "MaxContainedDataObjectCount",
        "MaxDataObjectSize",
        "OrphanedChildrenPrunedOnDelete",
        "SupportsGet",
        "SupportsPut",
        "SupportsDelete",
        "MaxSecondaryIndexCount",
    ],
    "fullName": "Energistics.Etp.v12.Datatypes.DataObjectCapabilityKind",
    "depends": [],
}


class DataObjectCapabilityKind(enum.StrEnum):
    ACTIVE_TIMEOUT_PERIOD = "ActiveTimeoutPeriod"
    MAX_CONTAINED_DATA_OBJECT_COUNT = "MaxContainedDataObjectCount"
    MAX_DATA_OBJECT_SIZE = "MaxDataObjectSize"
    ORPHANED_CHILDREN_PRUNED_ON_DELETE = "OrphanedChildrenPrunedOnDelete"
    SUPPORTS_GET = "SupportsGet"
    SUPPORTS_PUT = "SupportsPut"
    SUPPORTS_DELETE = "SupportsDelete"
    MAX_SECONDARY_INDEX_COUNT = "MaxSecondaryIndexCount"

    def get_valid_type(self) -> typing.Type[bool | int]:
        if self in [
            DataObjectCapabilityKind.ORPHANED_CHILDREN_PRUNED_ON_DELETE,
            DataObjectCapabilityKind.SUPPORTS_GET,
            DataObjectCapabilityKind.SUPPORTS_PUT,
            DataObjectCapabilityKind.SUPPORTS_DELETE,
        ]:
            return bool
        return int
