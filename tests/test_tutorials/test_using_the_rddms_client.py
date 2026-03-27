import resqml_objects.v201 as ro
from energistics.etp.v12.datatypes.object import Dataspace, Resource
from examples.tutorials.using_the_rddms_client.using_the_rddms_client import (
    crs,
    dataspaces,
    gri,
    gri_lo,
    gri_resources,
)
from rddms_io.data_types import LinkedObjects


def test_using_the_rddms_client() -> None:
    assert isinstance(dataspaces, list)
    for d in dataspaces:
        assert isinstance(d, Dataspace)

    assert isinstance(gri_resources, list)
    for g in gri_resources:
        assert isinstance(g, Resource)

    assert isinstance(gri_lo, LinkedObjects)

    assert isinstance(gri, ro.obj_Grid2dRepresentation)
    assert isinstance(crs, ro.obj_LocalDepth3dCrs)

    assert len(dataspaces) == 1
    assert dataspaces[0].path == "rddms_io/demo"
    assert len(gri_resources) == 1

    gri_uri = gri.get_etp_data_object_uri(dataspaces[0].path)
    crs_uri = crs.get_etp_data_object_uri(dataspaces[0].path)
    assert gri_uri == gri_resources[0].uri
    assert gri_uri == gri_lo.start_uri
    assert (
        len(gri_lo.target_resources) == 1 and crs_uri == gri_lo.target_resources[0].uri
    )
