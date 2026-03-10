import typing

from pydantic import Field

import energistics.base
from energistics.etp.v12.datatypes.object.data_object import DataObject


@energistics.base.add_protocol_avro_metadata
class PutDataObjects(energistics.base.ETPBaseProtocolModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Protocol.Store",
        "name": "PutDataObjects",
        "protocol": "4",
        "messageType": "2",
        "senderRole": "customer",
        "protocolRoles": "store,customer",
        "multipartFlag": True,
        "fields": [
            {
                "name": "dataObjects",
                "type": {
                    "type": "map",
                    "values": "Energistics.Etp.v12.Datatypes.Object.DataObject",
                },
            },
            {"name": "pruneContainedObjects", "type": "boolean", "default": False},
        ],
        "fullName": "Energistics.Etp.v12.Protocol.Store.PutDataObjects",
        "depends": ["Energistics.Etp.v12.Datatypes.Object.DataObject"],
    }

    data_objects: typing.Mapping[str, DataObject] = Field(alias="dataObjects")
    prune_contained_objects: bool = Field(alias="pruneContainedObjects", default=False)
