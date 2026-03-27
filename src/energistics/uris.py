from __future__ import annotations

import dataclasses
import re
import typing

# This pattern is modified from the original ECMAScript syntax used in the ETP
# v1.2. spec to a valid format for Python. Note that some ETP-server
# implementations do not allow all formats specified in the ETP v1.2 spec.
# Notably the open-etp-server does not support the default dataspace with the
# URI 'eml:///', which will be accepted by the pattern below.
DATASPACE_URI_PATTERN = re.compile(
    r"^eml:\/\/\/(?:dataspace\('(?P<dataspace>[^']*?(?:''[^']*?)*)'\))?$"
)

# This pattern is modified from the original ECMAScript syntax used in the ETP
# v1.2. spec to a valid format for Python. Note that some ETP-server
# implementations do not allow all formats specified in the ETP v1.2 spec.
# Notably the open-etp-server does not support the default dataspace with the
# URI 'eml:///', which will be accepted by the pattern below.
DATA_OBJECT_URI_PATTERN = re.compile(
    r"^eml:\/\/\/"
    + r"(?:dataspace\('(?P<dataspace>[^']*?(?:''[^']*?)*)'\)\/)?"
    + r"(?P<domain>witsml|resqml|prodml|eml)"
    + r"(?P<domain_version>[1-9]\d)\.(?P<object_type>\w+)"
    + r"\((?:(?P<uuid>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    + r"[0-9a-fA-F]{12})"
    + r"|uuid=(?P<uuid2>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    + r"[0-9a-fA-F]{12}),version='(?P<version>[^']*?(?:''[^']*?)*)')\)$"
)


@dataclasses.dataclass
class DataspaceURI:
    uri: str
    dataspace: str | None

    def __post_init__(self) -> None:
        m = re.match(DATASPACE_URI_PATTERN, self.uri)

        if m is None:
            raise ValueError(f"Uri '{self.uri}' is an invalid ETP dataspace uri.")

        dataspace = m.group("dataspace")
        if self.dataspace != dataspace:
            raise ValueError(
                f"Dataspace path '{self.dataspace}' is not equal to the path in "
                f"the full uri '{dataspace}' (uri: '{self.uri}')."
            )

    @staticmethod
    def is_valid_uri(uri: str | DataspaceURI) -> bool:
        return (
            isinstance(uri, DataspaceURI)
            or re.match(DATASPACE_URI_PATTERN, uri) is not None
        )

    @staticmethod
    def validate_uri(uri: str | DataspaceURI) -> None:
        if not DataspaceURI.is_valid_uri(uri):
            raise ValueError(f"Uri '{uri}' is not a valid ETP dataspace uri.")

    @classmethod
    def from_dataspace_path(cls, dataspace: str | None) -> typing.Self:
        uri = f"eml:///dataspace('{dataspace}')" if dataspace is not None else "eml:///"

        return cls(uri=uri, dataspace=dataspace)

    @classmethod
    def from_uri(cls, uri: str | typing.Self) -> typing.Self:
        if isinstance(uri, DataspaceURI):
            return cls(**dataclasses.asdict(uri))

        m = re.match(DATASPACE_URI_PATTERN, uri)

        if m is None:
            raise ValueError(f"Uri '{uri}' is an invalid ETP dataspace uri.")

        dataspace = m.group("dataspace")

        return cls(uri=uri, dataspace=dataspace)

    @classmethod
    def from_any_etp_uri(cls, uri: str | typing.Self | DataObjectURI) -> DataspaceURI:
        if isinstance(uri, DataspaceURI):
            return cls(**dataclasses.asdict(uri))

        if isinstance(uri, DataObjectURI):
            return uri.get_dataspace_uri()

        if not uri.startswith("eml:/"):
            return cls.from_dataspace_path(uri)

        if uri.endswith("')"):
            return cls.from_uri(uri)

        return DataObjectURI.from_uri(uri).get_dataspace_uri()

    def __str__(self) -> str:
        return self.uri


@dataclasses.dataclass
class DataObjectURI:
    uri: str
    dataspace: str | None
    domain: str
    domain_version: str
    object_type: str
    uuid: str
    version: str | None

    def __post_init__(self) -> None:
        m = re.match(DATA_OBJECT_URI_PATTERN, self.uri)

        if m is None:
            raise ValueError(f"Uri '{self.uri}' is an invalid ETP data object uri.")

        assert m.group("dataspace") == self.dataspace
        assert m.group("domain") == self.domain
        assert m.group("domain_version") == self.domain_version

        if m.group("uuid") is not None:
            assert m.group("uuid") == self.uuid
        else:
            assert m.group("uuid2") == self.uuid
            assert self.version is not None and m.group("version") == self.version

    def get_dataspace_uri(self) -> DataspaceURI:
        return DataspaceURI.from_dataspace_path(self.dataspace)

    @staticmethod
    def is_valid_uri(uri: str | DataObjectURI) -> bool:
        return (
            isinstance(uri, DataObjectURI)
            or re.match(DATA_OBJECT_URI_PATTERN, uri) is not None
        )

    @staticmethod
    def validate_uri(uri: str | DataObjectURI) -> None:
        if not DataObjectURI.is_valid_uri(uri):
            raise ValueError(f"Uri '{uri}' is not a valid ETP data object uri.")

    @classmethod
    def from_uri(cls, uri: str | typing.Self) -> typing.Self:
        if isinstance(uri, DataObjectURI):
            return cls(**dataclasses.asdict(uri))

        m = re.match(DATA_OBJECT_URI_PATTERN, uri)

        if m is None:
            raise ValueError(f"Uri '{uri}' is an invalid ETP data object uri.")

        dataspace = m.group("dataspace")
        domain = m.group("domain")
        domain_version = m.group("domain_version")
        object_type = m.group("object_type")
        uuid = m.group("uuid") or m.group("uuid2")
        version = m.group("version")

        return cls(
            uri=uri,
            dataspace=dataspace,
            domain=domain,
            domain_version=domain_version,
            object_type=object_type,
            uuid=uuid,
            version=version,
        )

    @classmethod
    def from_parts(
        cls,
        dataspace: str | None,
        domain: str,
        domain_version: str,
        object_type: str,
        uuid: str,
        version: str | None,
    ) -> typing.Self:
        uri = (
            (
                f"eml:///dataspace('{dataspace}')/"
                if dataspace is not None
                else "eml:///"
            )
            + f"{domain}{domain_version}.{object_type}"
            + (f"({uuid})" if version is None else f"(uuid={uuid},version='{version}')")
        )

        return cls(
            uri=uri,
            dataspace=dataspace,
            domain=domain,
            domain_version=domain_version,
            object_type=object_type,
            uuid=uuid,
            version=version,
        )

    def __str__(self) -> str:
        return self.uri
