import re
import typing

import energistics.base
import energistics.uris


@energistics.base.add_protocol_avro_metadata
class DeleteDataspaces(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Dataspace",
        "name": "DeleteDataspaces",
        "protocol": "24",
        "messageType": "4",
        "senderRole": "customer",
        "protocolRoles": "store,customer",
        "multipartFlag": False,
        "fields": [{"name": "uris", "type": {"type": "map", "values": "string"}}],
        "fullName": "Energistics.Etp.v12.Protocol.Dataspace.DeleteDataspaces",
        "depends": [],
    }

    uris: typing.Mapping[str, str]

    def model_post_init(self, __context: typing.Any) -> None:
        errors = []
        for k, uri in self.uris.items():
            m = re.match(energistics.uris.DATASPACE_URI_PATTERN, uri)

            if m is None:
                error = ValueError(f"Uri({k}) '{uri}' is not a valid ETP dataspace uri")
                errors.append(error)

        if len(errors) > 0:
            # NOTE: This does not raise pydantic.ValidationError, but instead an
            # ExceptionGroup of ValueError's.
            raise ExceptionGroup(
                f"There were {len(errors)} invalid ETP dataspace uris",
                errors,
            )
