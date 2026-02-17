import importlib
import pathlib

import numpy as np
from lxml import etree

from resqml_objects.parsers import parse_resqml_v201_object

mod = importlib.import_module(
    "examples.tutorials.set_up_regular_surface.set_up_regular_surface"
)

p = pathlib.Path("examples") / "tutorials" / "set_up_regular_surface"


def test_set_up_regular_surface() -> None:
    epc_xml = etree.tostring(etree.parse(p / "epc_xml.txt"))
    crs_xml = etree.tostring(etree.parse(p / "crs_xml.txt"))
    gri_xml = etree.tostring(etree.parse(p / "gri_xml.txt"))

    ret_epc = parse_resqml_v201_object(epc_xml)
    ret_crs = parse_resqml_v201_object(crs_xml)
    ret_gri = parse_resqml_v201_object(gri_xml)

    assert ret_epc == mod.epc
    assert ret_crs == mod.crs
    assert ret_gri == mod.gri

    assert (
        mod.X.shape
        == mod.Y.shape
        == (
            ret_gri.grid2d_patch.slowest_axis_count,
            ret_gri.grid2d_patch.fastest_axis_count,
        )
    )

    origin = np.array([mod.X[0, 0], mod.Y[0, 0]])
    go = ret_gri.grid2d_patch.geometry.points.supporting_geometry.origin
    np.testing.assert_equal(origin, np.array([go.coordinate1, go.coordinate2]))
