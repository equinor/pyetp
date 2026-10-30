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
    assert "rddms_io/demo" in [d.path for d in dataspaces]
    dataspace = next(filter(lambda d: d.path == "rddms_io/demo", dataspaces))

    assert isinstance(gri_resources, list)
    for g in gri_resources:
        assert isinstance(g, Resource)

    assert isinstance(gri_lo, LinkedObjects)

    assert isinstance(gri, ro.obj_Grid2dRepresentation)
    assert isinstance(crs, ro.obj_LocalDepth3dCrs)

    assert len(gri_resources) == 1

    gri_uri = gri.get_etp_data_object_uri(dataspace.path)
    crs_uri = crs.get_etp_data_object_uri(dataspace.path)
    assert gri_uri == gri_resources[0].uri
    assert gri_uri == gri_lo.start_uri
    assert (
        len(gri_lo.target_resources) == 1 and crs_uri == gri_lo.target_resources[0].uri
    )
