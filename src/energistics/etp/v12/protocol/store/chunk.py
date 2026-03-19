import typing

from pydantic import Field

import energistics.base
from energistics.etp.v12.datatypes.uuid import Uuid


@energistics.base.add_protocol_avro_metadata
class Chunk(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Store",
        "name": "Chunk",
        "protocol": "4",
        "messageType": "8",
        "senderRole": "store,customer",
        "protocolRoles": "store,customer",
        "multipartFlag": True,
        "fields": [
            {"name": "blobId", "type": "Energistics.Etp.v12.Datatypes.Uuid"},
            {"name": "data", "type": "bytes"},
            {"name": "final", "type": "boolean"},
        ],
        "fullName": "Energistics.Etp.v12.Protocol.Store.Chunk",
        "depends": ["Energistics.Etp.v12.Datatypes.Uuid"],
    }

    blob_id: Uuid = Field(alias="blobId")
    data: bytes
    final: bool
