import typing

from pydantic import AfterValidator, Field

import energistics.base
import energistics.uris


def check_dataspace_uris(dataspace_uris: list[str]) -> list[str]:
    errors = []
    for du in dataspace_uris:
        if not energistics.uris.DataspaceURI.is_valid_uri(du):
            errors.append(ValueError(f"Uri '{du}' is not a valid dataspace uri"))
    if len(errors) > 0:
        raise ExceptionGroup(
            f"There were {len(errors)} in the dataspace uris",
            errors,
        )

    return dataspace_uris


@energistics.base.add_protocol_avro_metadata
class StartTransaction(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Transaction",
        "name": "StartTransaction",
        "protocol": "18",
        "messageType": "1",
        "senderRole": "customer",
        "protocolRoles": "store,customer",
        "multipartFlag": False,
        "fields": [
            {"name": "readOnly", "type": "boolean"},
            {"name": "message", "type": "string", "default": ""},
            {
                "name": "dataspaceUris",
                "type": {"type": "array", "items": "string"},
                "default": [""],
            },
        ],
        "fullName": "Energistics.Etp.v12.Protocol.Transaction.StartTransaction",
        "depends": [],
    }

    read_only: bool = Field(alias="readOnly")
    message: str = ""
    dataspace_uris: typing.Annotated[
        list[str],
        Field(alias="dataspaceUris", default_factory=[]),
        AfterValidator(check_dataspace_uris),
    ]
