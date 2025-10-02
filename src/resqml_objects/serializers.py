from xsdata.formats.dataclass.serializers import XmlSerializer
from xsdata.formats.dataclass.serializers.config import SerializerConfig

import resqml_objects.v201 as ro_201


def serialize_resqml_v201_object(obj: ro_201.AbstractObject) -> bytes:
    serializer = XmlSerializer(config=SerializerConfig())

    return str.encode(serializer.render(obj))
