import typing

from xsdata.formats.dataclass.models.generics import DerivedElement
from xsdata.formats.dataclass.serializers import XmlSerializer
from xsdata.formats.dataclass.serializers.config import SerializerConfig


class MetaSub(typing.Protocol):
    @property
    def namespace(self) -> str: ...


class MetaObj(typing.Protocol):
    @property
    def target_namespace(self) -> str: ...


class RO201SubObj(typing.Protocol):
    @property
    def Meta(self) -> typing.Type[MetaSub]: ...


class RO201Obj(typing.Protocol):
    @property
    def Meta(self) -> typing.Type[MetaObj]: ...


def serialize_resqml_v201_object(
    obj: RO201Obj | RO201SubObj | DerivedElement[RO201Obj | RO201SubObj],
) -> bytes:
    serializer = XmlSerializer(config=SerializerConfig())

    if isinstance(obj, DerivedElement):
        namespace = getattr(obj.value.Meta, "namespace", None) or getattr(
            obj.value.Meta, "target_namespace", None
        )
        name = obj.value.__class__.__name__
    else:
        namespace = getattr(obj.Meta, "namespace", None) or getattr(
            obj.Meta, "target_namespace", None
        )
        name = obj.__class__.__name__

    if namespace is None:
        raise AttributeError(f"No XML namespace found for object {obj}")

    # This is a solution to enforce the inclusion of the `xsi:type`-attribute
    # on the generated XML-elements.
    if not isinstance(obj, DerivedElement) and name.startswith("obj_"):
        obj = DerivedElement(
            qname=f"{{{namespace}}}{name[4:]}",
            value=obj,
            type=f"{{{namespace}}}{name}",
        )

    return str.encode(
        serializer.render(
            obj,
            ns_map={
                "eml": "http://www.energistics.org/energyml/data/commonv2",
                "resqml2": "http://www.energistics.org/energyml/data/resqmlv2",
            },
        )
    )
