import typing

from pydantic import Field

import energistics.base
from energistics.etp.v12.datatypes.uuid import Uuid


@energistics.base.add_protocol_avro_metadata
class RollbackTransaction(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Transaction",
        "name": "RollbackTransaction",
        "protocol": "18",
        "messageType": "4",
        "senderRole": "customer",
        "protocolRoles": "store,customer",
        "multipartFlag": False,
        "fields": [
            {"name": "transactionUuid", "type": "Energistics.Etp.v12.Datatypes.Uuid"}
        ],
        "fullName": "Energistics.Etp.v12.Protocol.Transaction.RollbackTransaction",
        "depends": ["Energistics.Etp.v12.Datatypes.Uuid"],
    }

    transaction_uuid: Uuid = Field(alias="transactionUuid")
