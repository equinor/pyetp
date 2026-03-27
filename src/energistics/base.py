import abc
import enum
import importlib
import typing

from pydantic import BaseModel, ConfigDict

AvroSchemaValues: typing.TypeAlias = (
    list["AvroSchemaValues"]
    | dict[str, "AvroSchemaValues"]
    | str
    | int
    | float
    | bool
    | None
)
AvroSchemaType: typing.TypeAlias = dict[str, AvroSchemaValues]


class ETPMetaData(abc.ABC):
    avro_schema: typing.ClassVar[AvroSchemaType]
    full_name: typing.ClassVar[str]


class ETPBaseModel(BaseModel, ETPMetaData):
    model_config = ConfigDict(populate_by_name=True)


class Protocol(enum.IntEnum):
    CORE = 0
    CHANNEL_STREAMING = 1
    CHANNEL_DATA_FRAME = 2
    DISCOVERY = 3
    STORE = 4
    STORE_NOTIFICATION = 5
    GROWING_OBJECT = 6
    GROWING_OBJECT_NOTIFICATION = 7
    DATA_ARRAY = 9
    DISCOVERY_QUERY = 13
    STORE_QUERY = 14
    GROWING_OBJECT_QUERY = 16
    TRANSACTION = 18
    CHANNEL_SUBSCRIBE = 21
    CHANNEL_DATA_LOAD = 22
    DATASPACE = 24
    SUPPORTED_TYPES = 25


class Role(enum.StrEnum):
    CLIENT = "client"
    SERVER = "server"
    PRODUCER = "producer"
    CONSUMER = "consumer"
    STORE = "store"
    CUSTOMER = "customer"


class ETPBaseProtocolModel(ETPBaseModel):
    _protocol: typing.ClassVar[Protocol]
    _message_type: typing.ClassVar[int]
    _is_multipart: typing.ClassVar[bool]


def get_avro_schema_from_class(cls_full_name: str) -> AvroSchemaType:
    namespace_map = {
        "Energistics": "energistics",
        "Etp": "etp",
        "v12": "v12",
        "Datatypes": "datatypes",
        "Protocol": "protocol",
        "DataArrayTypes": "data_array_types",
        "ChannelData": "channel_data",
        "Object": "object",
    }
    *namespaces, cls_name = cls_full_name.split(".")
    cls_file_name = "".join(f"_{c.lower()}" if c.isupper() else c for c in cls_name)[1:]
    path = ".".join([*[namespace_map[n] for n in namespaces], cls_file_name])
    cls = getattr(importlib.import_module(path), cls_name)

    if issubclass(cls, enum.Enum):
        # We store the avro schemas in the module, not the enums themselves.
        return typing.cast(
            AvroSchemaType,
            getattr(importlib.import_module(path), "_avro_schema"),
        )
    return typing.cast(AvroSchemaType, cls._avro_schema)


def serialize_avro_schema(
    prev_serialized: list[str], schema: AvroSchemaValues | AvroSchemaType
) -> tuple[AvroSchemaValues | AvroSchemaType, list[str]]:
    if isinstance(schema, dict):
        if "fullName" in schema and schema["fullName"] in prev_serialized:
            return schema["fullName"], prev_serialized

    if isinstance(schema, bool | int | float | None):
        return schema, prev_serialized

    if isinstance(schema, str):
        if schema.startswith("Energistics.Etp.v12") and schema not in prev_serialized:
            # return serialize_avro_schema(prev_serialized, avro_map[schema])
            return serialize_avro_schema(
                prev_serialized, get_avro_schema_from_class(schema)
            )
        return schema, prev_serialized

    if not isinstance(schema, dict):
        raise TypeError(
            f"Expected schema to be of type {AvroSchemaType}, got {type(schema)}"
        )

    new_schema = {}
    for k, v in schema.items():
        if k in [
            "name",
            "namespace",
            "protocol",
            "messageType",
            "senderRole",
            "protocolRoles",
            "multipartFlag",
            "fullName",
            "depends",
        ]:
            new_schema[k] = v

        elif isinstance(v, list):
            new_schema[k] = list()
            for i in range(len(v)):
                ret_schema, prev_serialized = serialize_avro_schema(
                    prev_serialized,
                    v[i],
                )

                typing.cast(list[AvroSchemaValues], new_schema[k]).append(ret_schema)

        else:
            new_schema[k], prev_serialized = serialize_avro_schema(
                prev_serialized,
                v,
            )

    if "fullName" in new_schema:
        assert isinstance(new_schema["fullName"], str)
        prev_serialized = [*prev_serialized, new_schema["fullName"]]

    return new_schema, prev_serialized


def add_avro_metadata(cls: typing.Any) -> typing.Any:
    cls.avro_schema = typing.cast(
        AvroSchemaType, serialize_avro_schema([], cls._avro_schema)[0]
    )
    cls.full_name = cls._avro_schema["fullName"]
    return cls


def add_protocol_avro_metadata(cls: typing.Any) -> typing.Any:
    cls = add_avro_metadata(cls)
    cls._protocol = Protocol(int(cls._avro_schema["protocol"]))
    cls._message_type = int(cls._avro_schema["messageType"])
    cls._is_multipart = cls._avro_schema["multipartFlag"]
    return cls
