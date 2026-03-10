import typing

from pydantic import AfterValidator

import energistics.base
import energistics.uris


def check_data_object_uris(uris: typing.Mapping[str, str]) -> typing.Mapping[str, str]:
    errors = []
    for k, v in uris.items():
        if not energistics.uris.DataObjectURI.is_valid_uri(v):
            errors.append(
                ValueError(
                    f"Uri '{v}' (from element '{k}') is not a valid data object uri"
                )
            )

    if len(errors) > 0:
        raise ExceptionGroup(
            f"There were {len(errors)} invalid data object uris",
            errors,
        )

    return uris


@energistics.base.add_protocol_avro_metadata
class GetDataObjects(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Store",
        "name": "GetDataObjects",
        "protocol": "4",
        "messageType": "1",
        "senderRole": "customer",
        "protocolRoles": "store,customer",
        "multipartFlag": False,
        "fields": [
            {"name": "uris", "type": {"type": "map", "values": "string"}},
            {"name": "format", "type": "string", "default": "xml"},
        ],
        "fullName": "Energistics.Etp.v12.Protocol.Store.GetDataObjects",
        "depends": [],
    }

    uris: typing.Annotated[
        typing.Mapping[str, str], AfterValidator(check_data_object_uris)
    ]
    format: str = "xml"
