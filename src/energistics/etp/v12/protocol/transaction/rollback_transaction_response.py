import typing

from pydantic import Field

import energistics.base
from energistics.etp.v12.datatypes.uuid import Uuid


@energistics.base.add_protocol_avro_metadata
class RollbackTransactionResponse(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Transaction",
        "name": "RollbackTransactionResponse",
        "protocol": "18",
        "messageType": "6",
        "senderRole": "store",
        "protocolRoles": "store,customer",
        "multipartFlag": False,
        "fields": [
            {"name": "transactionUuid", "type": "Energistics.Etp.v12.Datatypes.Uuid"},
            {"name": "successful", "type": "boolean", "default": True},
            {"name": "failureReason", "type": "string", "default": ""},
        ],
        "fullName": "Energistics.Etp.v12.Protocol.Transaction.RollbackTransactionResponse",
        "depends": ["Energistics.Etp.v12.Datatypes.Uuid"],
    }

    transaction_uuid: Uuid = Field(alias="transactionUuid")
    successful: bool = True
    failure_reason: str = Field(alias="failureReason", default="")
