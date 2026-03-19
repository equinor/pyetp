import typing

from pydantic import Field

import energistics.base
from energistics.etp.v12.datatypes.data_array_types.put_uninitialized_data_array_type import (
    PutUninitializedDataArrayType,
)


@energistics.base.add_protocol_avro_metadata
class PutUninitializedDataArrays(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.DataArray",
        "name": "PutUninitializedDataArrays",
        "protocol": "9",
        "messageType": "9",
        "senderRole": "customer",
        "protocolRoles": "store,customer",
        "multipartFlag": False,
        "fields": [
            {
                "name": "dataArrays",
                "type": {
                    "type": "map",
                    "values": "Energistics.Etp.v12.Datatypes.DataArrayTypes.PutUninitializedDataArrayType",
                },
            }
        ],
        "fullName": "Energistics.Etp.v12.Protocol.DataArray.PutUninitializedDataArrays",
        "depends": [
            "Energistics.Etp.v12.Datatypes.DataArrayTypes.PutUninitializedDataArrayType"
        ],
    }

    data_arrays: typing.Mapping[str, PutUninitializedDataArrayType] = Field(
        alias="dataArrays"
    )
