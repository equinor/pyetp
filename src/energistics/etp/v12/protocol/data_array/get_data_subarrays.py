import typing

from pydantic import Field

import energistics.base
from energistics.etp.v12.datatypes.data_array_types.get_data_subarrays_type import (
    GetDataSubarraysType,
)


@energistics.base.add_protocol_avro_metadata
class GetDataSubarrays(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.DataArray",
        "name": "GetDataSubarrays",
        "protocol": "9",
        "messageType": "3",
        "senderRole": "customer",
        "protocolRoles": "store,customer",
        "multipartFlag": False,
        "fields": [
            {
                "name": "dataSubarrays",
                "type": {
                    "type": "map",
                    "values": "Energistics.Etp.v12.Datatypes.DataArrayTypes.GetDataSubarraysType",
                },
            }
        ],
        "fullName": "Energistics.Etp.v12.Protocol.DataArray.GetDataSubarrays",
        "depends": [
            "Energistics.Etp.v12.Datatypes.DataArrayTypes.GetDataSubarraysType"
        ],
    }

    data_subarrays: typing.Mapping[str, GetDataSubarraysType] = Field(
        alias="dataSubarrays"
    )
