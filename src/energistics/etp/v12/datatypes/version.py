import typing

import energistics.base


@energistics.base.add_avro_metadata
class Version(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes",
        "name": "Version",
        "fields": [
            {"name": "major", "type": "int", "default": 0},
            {"name": "minor", "type": "int", "default": 0},
            {"name": "revision", "type": "int", "default": 0},
            {"name": "patch", "type": "int", "default": 0},
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.Version",
        "depends": [],
    }

    major: int = 0
    minor: int = 0
    revision: int = 0
    patch: int = 0
