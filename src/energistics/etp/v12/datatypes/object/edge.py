import typing

from pydantic import Field, model_validator

import energistics.base
import energistics.uris
from energistics.etp.v12.datatypes.data_value import DataValue
from energistics.etp.v12.datatypes.object.relationship_kind import RelationshipKind


@energistics.base.add_avro_metadata
class Edge(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes.Object",
        "name": "Edge",
        "fields": [
            {"name": "sourceUri", "type": "string"},
            {"name": "targetUri", "type": "string"},
            {
                "name": "relationshipKind",
                "type": "Energistics.Etp.v12.Datatypes.Object.RelationshipKind",
            },
            {
                "name": "customData",
                "type": {
                    "type": "map",
                    "values": "Energistics.Etp.v12.Datatypes.DataValue",
                },
                "default": {},
            },
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.Object.Edge",
        "depends": [
            "Energistics.Etp.v12.Datatypes.Object.RelationshipKind",
            "Energistics.Etp.v12.Datatypes.DataValue",
        ],
    }

    source_uri: str = Field(alias="sourceUri")
    target_uri: str = Field(alias="targetUri")
    relationship_kind: RelationshipKind = Field(alias="relationshipKind")
    custom_data: typing.Mapping[str, DataValue] = Field(
        alias="customData", default_factory=dict
    )

    @model_validator(mode="after")
    def check_uri(self) -> typing.Self:
        energistics.uris.DataObjectURI.validate_uri(self.source_uri)
        energistics.uris.DataObjectURI.validate_uri(self.target_uri)

        return self

    @model_validator(mode="after")
    def check_relationship_kind(self) -> typing.Self:
        if self.relationship_kind == RelationshipKind.BOTH:
            raise ValueError(
                f"Relationship kind can only be {RelationshipKind.PRIMARY} or "
                f"{RelationshipKind.SECONDARY} in Edge"
            )

        return self
