import typing

from pydantic import AfterValidator, Field

import energistics.base
import energistics.uris
from energistics.etp.v12.datatypes.array_of_string import ArrayOfString


def check_data_object_uris(
    uris: typing.Mapping[str, ArrayOfString],
) -> typing.Mapping[str, ArrayOfString]:
    errors = []
    for k, v in uris.items():
        for uri in v.values:
            if not energistics.uris.DataObjectURI.is_valid_uri(uri):
                errors.append(
                    ValueError(
                        f"Uri '{uri}' (from element '{k}') is not a valid data object "
                        "uri"
                    )
                )

    if len(errors) > 0:
        raise ExceptionGroup(
            f"There were {len(errors)} invalid data object uris",
            errors,
        )

    return uris


@energistics.base.add_protocol_avro_metadata
class DeleteDataObjectsResponse(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Store",
        "name": "DeleteDataObjectsResponse",
        "protocol": "4",
        "messageType": "10",
        "senderRole": "store",
        "protocolRoles": "store,customer",
        "multipartFlag": True,
        "fields": [
            {
                "name": "deletedUris",
                "type": {
                    "type": "map",
                    "values": "Energistics.Etp.v12.Datatypes.ArrayOfString",
                },
            }
        ],
        "fullName": "Energistics.Etp.v12.Protocol.Store.DeleteDataObjectsResponse",
        "depends": ["Energistics.Etp.v12.Datatypes.ArrayOfString"],
    }

    deleted_uris: typing.Annotated[
        typing.Mapping[str, ArrayOfString],
        Field(alias="deletedUris"),
        AfterValidator(check_data_object_uris),
    ]
