import typing

from pydantic import Field

import energistics.base


@energistics.base.add_avro_metadata
class PutResponse(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes.Object",
        "name": "PutResponse",
        "fields": [
            {
                "name": "createdContainedObjectUris",
                "type": {"type": "array", "items": "string"},
                "default": [],
            },
            {
                "name": "deletedContainedObjectUris",
                "type": {"type": "array", "items": "string"},
                "default": [],
            },
            {
                "name": "joinedContainedObjectUris",
                "type": {"type": "array", "items": "string"},
                "default": [],
            },
            {
                "name": "unjoinedContainedObjectUris",
                "type": {"type": "array", "items": "string"},
                "default": [],
            },
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.Object.PutResponse",
        "depends": [],
    }

    created_contained_object_uris: list[str] = Field(
        alias="createdContainedObjectUris", default_factory=list
    )
    deleted_contained_object_uris: list[str] = Field(
        alias="deletedContainedObjectUris", default_factory=list
    )
    joined_contained_object_uris: list[str] = Field(
        alias="joinedContainedObjectUris", default_factory=list
    )
    unjoined_contained_object_uris: list[str] = Field(
        alias="unjoinedContainedObjectUris", default_factory=list
    )
