import typing

from pydantic import Field

import energistics.base
from energistics.etp.v12.datatypes.data_array_types.data_array import DataArray


@energistics.base.add_protocol_avro_metadata
class GetDataArraysResponse(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.DataArray",
        "name": "GetDataArraysResponse",
        "protocol": "9",
        "messageType": "1",
        "senderRole": "store",
        "protocolRoles": "store,customer",
        "multipartFlag": True,
        "fields": [
            {
                "name": "dataArrays",
                "type": {
                    "type": "map",
                    "values": "Energistics.Etp.v12.Datatypes.DataArrayTypes.DataArray",
                },
                "default": {},
            }
        ],
        "fullName": "Energistics.Etp.v12.Protocol.DataArray.GetDataArraysResponse",
        "depends": ["Energistics.Etp.v12.Datatypes.DataArrayTypes.DataArray"],
    }

    data_arrays: typing.Mapping[str, DataArray] = Field(
        alias="dataArrays", default_factory=dict
    )
