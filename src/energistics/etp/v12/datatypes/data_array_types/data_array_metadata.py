import typing

import numpy as np
from pydantic import Field

import energistics.base
from energistics.array_mapping import TransportArrayTypeMapping
from energistics.etp.v12.datatypes.any_array_type import AnyArrayType
from energistics.etp.v12.datatypes.any_logical_array_type import AnyLogicalArrayType
from energistics.etp.v12.datatypes.data_value import DataValue


@energistics.base.add_avro_metadata
class DataArrayMetadata(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes.DataArrayTypes",
        "name": "DataArrayMetadata",
        "fields": [
            {"name": "dimensions", "type": {"type": "array", "items": "long"}},
            {
                "name": "preferredSubarrayDimensions",
                "type": {"type": "array", "items": "long"},
                "default": [],
            },
            {
                "name": "transportArrayType",
                "type": "Energistics.Etp.v12.Datatypes.AnyArrayType",
            },
            {
                "name": "logicalArrayType",
                "type": "Energistics.Etp.v12.Datatypes.AnyLogicalArrayType",
            },
            {"name": "storeLastWrite", "type": "long"},
            {"name": "storeCreated", "type": "long"},
            {
                "name": "customData",
                "type": {
                    "type": "map",
                    "values": "Energistics.Etp.v12.Datatypes.DataValue",
                },
                "default": {},
            },
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.DataArrayTypes.DataArrayMetadata",
        "depends": [
            "Energistics.Etp.v12.Datatypes.AnyArrayType",
            "Energistics.Etp.v12.Datatypes.AnyLogicalArrayType",
            "Energistics.Etp.v12.Datatypes.DataValue",
        ],
    }

    dimensions: list[int]
    preferred_subarray_dimensions: list[int] = Field(
        alias="preferredSubarrayDimensions", default_factory=list
    )
    transport_array_type: AnyArrayType = Field(alias="transportArrayType")
    logical_array_type: AnyLogicalArrayType = Field(alias="logicalArrayType")
    store_last_write: int = Field(alias="storeLastWrite")
    store_created: int = Field(alias="storeCreated")
    custom_data: typing.Mapping[str, DataValue] = Field(
        alias="customData", default_factory=dict
    )

    # TODO: Add validation of transport and logical array types once this is
    # implemented from the server.

    def get_transport_array_size(self) -> int:
        dtype = TransportArrayTypeMapping.get_dtype(self.transport_array_type)
        return int(np.prod(self.dimensions) * dtype.itemsize)
