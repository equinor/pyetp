import typing

from pydantic import AfterValidator, Field

import energistics.base
from energistics.validators import check_data_object_types, check_dataspace_uri


@energistics.base.add_protocol_avro_metadata
class GetDeletedResources(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Discovery",
        "name": "GetDeletedResources",
        "protocol": "3",
        "messageType": "5",
        "senderRole": "customer",
        "protocolRoles": "store,customer",
        "multipartFlag": False,
        "fields": [
            {"name": "dataspaceUri", "type": "string"},
            {"name": "deleteTimeFilter", "type": ["null", "long"]},
            {
                "name": "dataObjectTypes",
                "type": {"type": "array", "items": "string"},
                "default": [],
            },
        ],
        "fullName": "Energistics.Etp.v12.Protocol.Discovery.GetDeletedResources",
        "depends": [],
    }

    dataspace_uri: typing.Annotated[
        str, Field(alias="dataspaceUri"), AfterValidator(check_dataspace_uri)
    ]
    delete_time_filter: int | None = Field(alias="deleteTimeFilter")
    data_object_types: typing.Annotated[
        list[str],
        Field(alias="dataObjectTypes", default_factory=list),
        AfterValidator(check_data_object_types),
    ]
