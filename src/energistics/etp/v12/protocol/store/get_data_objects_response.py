import typing

from pydantic import Field

import energistics.base
from energistics.etp.v12.datatypes.object.data_object import DataObject


@energistics.base.add_protocol_avro_metadata
class GetDataObjectsResponse(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Store",
        "name": "GetDataObjectsResponse",
        "protocol": "4",
        "messageType": "4",
        "senderRole": "store",
        "protocolRoles": "store,customer",
        "multipartFlag": True,
        "fields": [
            {
                "name": "dataObjects",
                "type": {
                    "type": "map",
                    "values": "Energistics.Etp.v12.Datatypes.Object.DataObject",
                },
                "default": {},
            }
        ],
        "fullName": "Energistics.Etp.v12.Protocol.Store.GetDataObjectsResponse",
        "depends": ["Energistics.Etp.v12.Datatypes.Object.DataObject"],
    }

    data_objects: typing.Mapping[str, DataObject] = Field(
        alias="dataObjects", default_factory=dict
    )
