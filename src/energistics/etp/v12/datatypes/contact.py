import typing

from pydantic import Field

import energistics.base


@energistics.base.add_avro_metadata
class Contact(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes",
        "name": "Contact",
        "fields": [
            {"name": "organizationName", "type": "string", "default": ""},
            {"name": "contactName", "type": "string", "default": ""},
            {"name": "contactPhone", "type": "string", "default": ""},
            {"name": "contactEmail", "type": "string", "default": ""},
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.Contact",
        "depends": [],
    }

    organization_name: str = Field(alias="organizationName", default="")
    contact_name: str = Field(alias="contactName", default="")
    contact_phone: str = Field(alias="contactPhone", default="")
    contact_email: str = Field(alias="contactEmail", default="")
