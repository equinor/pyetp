import typing

from pydantic import Field

import energistics.base


@energistics.base.add_protocol_avro_metadata
class GetDataspaces(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Dataspace",
        "name": "GetDataspaces",
        "protocol": "24",
        "messageType": "1",
        "senderRole": "customer",
        "protocolRoles": "store,customer",
        "multipartFlag": False,
        "fields": [{"name": "storeLastWriteFilter", "type": ["null", "long"]}],
        "fullName": "Energistics.Etp.v12.Protocol.Dataspace.GetDataspaces",
        "depends": [],
    }

    store_last_write_filter: int | None = Field(
        alias="storeLastWriteFilter", default=None
    )
