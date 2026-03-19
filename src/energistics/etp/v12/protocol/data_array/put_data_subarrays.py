import typing

from pydantic import Field

import energistics.base
from energistics.etp.v12.datatypes.data_array_types.put_data_subarrays_type import (
    PutDataSubarraysType,
)


@energistics.base.add_protocol_avro_metadata
class PutDataSubarrays(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.DataArray",
        "name": "PutDataSubarrays",
        "protocol": "9",
        "messageType": "5",
        "senderRole": "customer",
        "protocolRoles": "store,customer",
        "multipartFlag": False,
        "fields": [
            {
                "name": "dataSubarrays",
                "type": {
                    "type": "map",
                    "values": "Energistics.Etp.v12.Datatypes.DataArrayTypes.PutDataSubarraysType",
                },
            }
        ],
        "fullName": "Energistics.Etp.v12.Protocol.DataArray.PutDataSubarrays",
        "depends": [
            "Energistics.Etp.v12.Datatypes.DataArrayTypes.PutDataSubarraysType"
        ],
    }

    data_subarrays: typing.Mapping[str, PutDataSubarraysType] = Field(
        alias="dataSubarrays"
    )
