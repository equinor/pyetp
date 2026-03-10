import typing

import energistics.base
from energistics.etp.v12.datatypes.data_array_types.data_array_identifier import (
    DataArrayIdentifier,
)
from energistics.etp.v12.datatypes.data_array_types.data_array_metadata import (
    DataArrayMetadata,
)


@energistics.base.add_avro_metadata
class PutUninitializedDataArrayType(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes.DataArrayTypes",
        "name": "PutUninitializedDataArrayType",
        "fields": [
            {
                "name": "uid",
                "type": "Energistics.Etp.v12.Datatypes.DataArrayTypes.DataArrayIdentifier",
            },
            {
                "name": "metadata",
                "type": "Energistics.Etp.v12.Datatypes.DataArrayTypes.DataArrayMetadata",
            },
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.DataArrayTypes.PutUninitializedDataArrayType",
        "depends": [
            "Energistics.Etp.v12.Datatypes.DataArrayTypes.DataArrayIdentifier",
            "Energistics.Etp.v12.Datatypes.DataArrayTypes.DataArrayMetadata",
        ],
    }

    uid: DataArrayIdentifier
    metadata: DataArrayMetadata
