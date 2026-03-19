import typing

from pydantic import Field

import energistics.base
from energistics.etp.v12.datatypes.data_array_types.data_array import DataArray


@energistics.base.add_protocol_avro_metadata
class GetDataSubarraysResponse(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.DataArray",
        "name": "GetDataSubarraysResponse",
        "protocol": "9",
        "messageType": "8",
        "senderRole": "store",
        "protocolRoles": "store,customer",
        "multipartFlag": True,
        "fields": [
            {
                "name": "dataSubarrays",
                "type": {
                    "type": "map",
                    "values": "Energistics.Etp.v12.Datatypes.DataArrayTypes.DataArray",
                },
                "default": {},
            }
        ],
        "fullName": "Energistics.Etp.v12.Protocol.DataArray.GetDataSubarraysResponse",
        "depends": ["Energistics.Etp.v12.Datatypes.DataArrayTypes.DataArray"],
    }

    data_subarrays: typing.Mapping[str, DataArray] = Field(
        alias="dataSubarrays", default_factory=dict
    )
