import typing

from pydantic import Field

import energistics.base
from energistics.etp.v12.datatypes.data_array_types.data_array_identifier import (
    DataArrayIdentifier,
)


@energistics.base.add_protocol_avro_metadata
class GetDataArrayMetadata(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.DataArray",
        "name": "GetDataArrayMetadata",
        "protocol": "9",
        "messageType": "6",
        "senderRole": "customer",
        "protocolRoles": "store,customer",
        "multipartFlag": False,
        "fields": [
            {
                "name": "dataArrays",
                "type": {
                    "type": "map",
                    "values": "Energistics.Etp.v12.Datatypes.DataArrayTypes.DataArrayIdentifier",
                },
            }
        ],
        "fullName": "Energistics.Etp.v12.Protocol.DataArray.GetDataArrayMetadata",
        "depends": ["Energistics.Etp.v12.Datatypes.DataArrayTypes.DataArrayIdentifier"],
    }

    data_arrays: typing.Mapping[str, DataArrayIdentifier] = Field(alias="dataArrays")
