import typing

from pydantic import Field

import energistics.base
from energistics.etp.v12.datatypes.data_array_types.put_data_arrays_type import (
    PutDataArraysType,
)


@energistics.base.add_protocol_avro_metadata
class PutDataArrays(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.DataArray",
        "name": "PutDataArrays",
        "protocol": "9",
        "messageType": "4",
        "senderRole": "customer",
        "protocolRoles": "store,customer",
        "multipartFlag": False,
        "fields": [
            {
                "name": "dataArrays",
                "type": {
                    "type": "map",
                    "values": "Energistics.Etp.v12.Datatypes.DataArrayTypes.PutDataArraysType",
                },
            }
        ],
        "fullName": "Energistics.Etp.v12.Protocol.DataArray.PutDataArrays",
        "depends": ["Energistics.Etp.v12.Datatypes.DataArrayTypes.PutDataArraysType"],
    }

    data_arrays: typing.Mapping[str, PutDataArraysType] = Field(alias="dataArrays")
