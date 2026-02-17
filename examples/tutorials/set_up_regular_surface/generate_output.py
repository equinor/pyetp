import pathlib

import rich
from lxml import etree

from resqml_objects.serializers import (
    serialize_resqml_v201_object,
)

from set_up_regular_surface import epc, crs, gri


p = pathlib.Path("examples") / "tutorials" / "set_up_regular_surface"
width = 80


with (
    open(p / "epc_obj.txt", "w") as f1,
    open(p / "crs_obj.txt", "w") as f2,
    open(p / "gri_obj.txt", "w") as f3,
):
    rich.console.Console(width=width, file=f1).print(epc)
    rich.console.Console(width=width, file=f2).print(crs)
    rich.console.Console(width=width, file=f3).print(gri)


with (
    open(p / "epc_xml.txt", "w") as f1,
    open(p / "crs_xml.txt", "w") as f2,
    open(p / "gri_xml.txt", "w") as f3,
):
    f1.write(
        etree.tostring(
            etree.fromstring(serialize_resqml_v201_object(epc)),
            pretty_print=True,
        ).decode(),
    )
    f2.write(
        etree.tostring(
            etree.fromstring(serialize_resqml_v201_object(crs)),
            pretty_print=True,
        ).decode()
    )
    f3.write(
        etree.tostring(
            etree.fromstring(serialize_resqml_v201_object(gri)),
            pretty_print=True,
        ).decode()
    )
