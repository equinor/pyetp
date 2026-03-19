import typing

from pydantic import Field

import energistics.base
from energistics.etp.v12.datatypes.object.active_status_kind import ActiveStatusKind
from energistics.etp.v12.datatypes.object.context_info import ContextInfo
from energistics.etp.v12.datatypes.object.context_scope_kind import ContextScopeKind


@energistics.base.add_protocol_avro_metadata
class GetResources(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Discovery",
        "name": "GetResources",
        "protocol": "3",
        "messageType": "1",
        "senderRole": "customer",
        "protocolRoles": "store,customer",
        "multipartFlag": False,
        "fields": [
            {
                "name": "context",
                "type": "Energistics.Etp.v12.Datatypes.Object.ContextInfo",
            },
            {
                "name": "scope",
                "type": "Energistics.Etp.v12.Datatypes.Object.ContextScopeKind",
            },
            {"name": "countObjects", "type": "boolean", "default": False},
            {"name": "storeLastWriteFilter", "type": ["null", "long"]},
            {
                "name": "activeStatusFilter",
                "type": [
                    "null",
                    "Energistics.Etp.v12.Datatypes.Object.ActiveStatusKind",
                ],
            },
            {"name": "includeEdges", "type": "boolean", "default": False},
        ],
        "fullName": "Energistics.Etp.v12.Protocol.Discovery.GetResources",
        "depends": [
            "Energistics.Etp.v12.Datatypes.Object.ContextInfo",
            "Energistics.Etp.v12.Datatypes.Object.ContextScopeKind",
            "Energistics.Etp.v12.Datatypes.Object.ActiveStatusKind",
        ],
    }

    context: ContextInfo
    scope: ContextScopeKind
    count_objects: bool = Field(alias="countObjects", default=False)
    store_last_write_filter: int | None = Field(
        alias="storeLastWriteFilter", default=None
    )
    active_status_filter: ActiveStatusKind | None = Field(
        alias="activeStatusFilter", default=None
    )
    include_edges: bool = Field(alias="includeEdges", default=False)
