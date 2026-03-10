import typing

from pydantic import Field

import energistics.base
from energistics.etp.v12.datatypes.data_array_types.data_array_metadata import (
    DataArrayMetadata,
)


@energistics.base.add_protocol_avro_metadata
class GetDataArrayMetadataResponse(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.DataArray",
        "name": "GetDataArrayMetadataResponse",
        "protocol": "9",
        "messageType": "7",
        "senderRole": "store",
        "protocolRoles": "store,customer",
        "multipartFlag": True,
        "fields": [
            {
                "name": "arrayMetadata",
                "type": {
                    "type": "map",
                    "values": "Energistics.Etp.v12.Datatypes.DataArrayTypes.DataArrayMetadata",
                },
                "default": {},
            }
        ],
        "fullName": "Energistics.Etp.v12.Protocol.DataArray.GetDataArrayMetadataResponse",
        "depends": ["Energistics.Etp.v12.Datatypes.DataArrayTypes.DataArrayMetadata"],
    }

    array_metadata: typing.Mapping[str, DataArrayMetadata] = Field(
        alias="arrayMetadata", default_factory=dict
    )
