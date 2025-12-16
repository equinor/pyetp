from xsdata.formats.dataclass.models.generics import DerivedElement
from xsdata.formats.dataclass.serializers import XmlSerializer
from xsdata.formats.dataclass.serializers.config import SerializerConfig

import resqml_objects.v201 as ro_201


def serialize_resqml_v201_object(
    obj: ro_201.AbstractObject | DerivedElement[ro_201.AbstractObject],
) -> bytes:
    serializer = XmlSerializer(config=SerializerConfig())

    namespace = getattr(obj.Meta, "namespace", None) or obj.Meta.target_namespace
    name = obj.__class__.__name__

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
