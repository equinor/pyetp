import datetime
import typing
from dataclasses import dataclass

from energistics.etp.v12.datatypes import DataValue
from energistics.etp.v12.datatypes.object import Dataspace


@dataclass
class ACL:
    legal_tags: list[str]
    other_relevant_data_countries: list[str]
    owners: list[str]
    viewers: list[str]

    @classmethod
    def from_custom_data(cls, custom_data: dict[str, DataValue]) -> typing.Self:
        legal_tags = (
            custom_data.pop("legaltags").item.values
            if "legaltags" in custom_data
            else []
        )
        other_relevant_data_countries = (
            custom_data.pop("otherRelevantDataCountries").item.values
            if "otherRelevantDataCountries" in custom_data
            else []
        )
        owners = (
            custom_data.pop("owners").item.values if "owners" in custom_data else []
        )
        viewers = (
            custom_data.pop("viewers").item.values if "viewers" in custom_data else []
        )

        return cls(
            legal_tags=legal_tags,
            other_relevant_data_countries=other_relevant_data_countries,
            owners=owners,
            viewers=viewers,
        )


@dataclass
class RDDMSDataspace:
    uri: str
    path: str
    store_created: datetime.datetime
    store_last_write: datetime.datetime
    acl: ACL
    other_custom_data: dict[str, DataValue]

    @classmethod
    def from_etp_dataspace(cls, dataspace: Dataspace) -> typing.Self:
        custom_data = dataspace.custom_data.copy()
        acl = ACL.from_custom_data(custom_data)
        store_created = (
            dataspace.store_created
            if isinstance(dataspace.store_created, datetime.datetime)
            else datetime.datetime.fromtimestamp(
                dataspace.store_created / 1e6,
                datetime.timezone.utc,
            )
        )
        store_last_write = (
            dataspace.store_last_write
            if isinstance(dataspace.store_last_write, datetime.datetime)
            else datetime.datetime.fromtimestamp(
                dataspace.store_last_write / 1e6,
                datetime.timezone.utc,
            )
        )

        return cls(
            uri=dataspace.uri,
            path=dataspace.path,
            store_created=store_created,
            store_last_write=store_last_write,
            acl=acl,
            other_custom_data=custom_data,
        )
