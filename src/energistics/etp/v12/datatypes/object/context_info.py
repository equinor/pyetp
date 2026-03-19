import typing

from pydantic import AfterValidator, Field, PositiveInt

import energistics.base
from energistics.etp.v12.datatypes.object.relationship_kind import RelationshipKind
from energistics.validators import (
    check_data_object_types,
    check_dataspace_or_data_object_uri,
)


def check_valid_navigable_edges(navigable_edges: RelationshipKind) -> RelationshipKind:
    if navigable_edges == RelationshipKind.BOTH:
        raise ValueError(
            f"Navigable edges can only be {RelationshipKind.PRIMARY} or "
            f"{RelationshipKind.SECONDARY} in ContextInfo"
        )

    return navigable_edges


@energistics.base.add_avro_metadata
class ContextInfo(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes.Object",
        "name": "ContextInfo",
        "fields": [
            {"name": "uri", "type": "string"},
            {"name": "depth", "type": "int"},
            {
                "name": "dataObjectTypes",
                "type": {"type": "array", "items": "string"},
                "default": [],
            },
            {
                "name": "navigableEdges",
                "type": "Energistics.Etp.v12.Datatypes.Object.RelationshipKind",
            },
            {"name": "includeSecondaryTargets", "type": "boolean", "default": False},
            {"name": "includeSecondarySources", "type": "boolean", "default": False},
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.Object.ContextInfo",
        "depends": ["Energistics.Etp.v12.Datatypes.Object.RelationshipKind"],
    }

    uri: typing.Annotated[str, AfterValidator(check_dataspace_or_data_object_uri)]
    depth: PositiveInt
    data_object_types: typing.Annotated[
        list[str],
        Field(alias="dataObjectTypes", default_factory=list),
        AfterValidator(check_data_object_types),
    ]
    navigable_edges: typing.Annotated[
        RelationshipKind,
        Field(alias="navigableEdges"),
        AfterValidator(check_valid_navigable_edges),
    ]
    include_secondary_targets: bool = Field(
        alias="includeSecondaryTargets", default=False
    )
    include_secondary_sources: bool = Field(
        alias="includeSecondarySources", default=False
    )
