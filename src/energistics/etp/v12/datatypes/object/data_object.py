import typing

from pydantic import Field, model_validator

import energistics.base
from energistics.etp.v12.datatypes.object.resource import Resource
from energistics.etp.v12.datatypes.uuid import Uuid


@energistics.base.add_avro_metadata
class DataObject(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes.Object",
        "name": "DataObject",
        "fields": [
            {
                "name": "resource",
                "type": "Energistics.Etp.v12.Datatypes.Object.Resource",
            },
            {"name": "format", "type": "string", "default": "xml"},
            {"name": "blobId", "type": ["null", "Energistics.Etp.v12.Datatypes.Uuid"]},
            {"name": "data", "type": "bytes", "default": ""},
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.Object.DataObject",
        "depends": [
            "Energistics.Etp.v12.Datatypes.Object.Resource",
            "Energistics.Etp.v12.Datatypes.Uuid",
        ],
    }

    resource: Resource
    format: str = "xml"
    blob_id: Uuid | None = Field(alias="blobId", default=None)
    data: bytes = b""

    @model_validator(mode="after")
    def check_blob_and_data(self) -> typing.Self:
        if len(self.data) == 0 and self.blob_id is None:
            raise ValueError("Either the data field or the blob-id must be populated")

        if self.blob_id is not None and len(self.data) > 0:
            raise ValueError("If the blob-id is set, then data must be empty")

        return self
