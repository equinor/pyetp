import datetime
import uuid

import numpy as np
import numpy.typing as npt
import pytest
import pytest_asyncio
from xsdata.models.datatype import XmlDateTime

import resqml_objects.v201 as resqml_objects
from pyetp.client import ETPClient, connect
from pyetp.config import SETTINGS

SETTINGS.application_name = "geomin_testing"
SETTINGS.etp_url = "ws://localhost:9100"
SETTINGS.etp_timeout = 30
# The max size comes from the websockets library on received messages!
SETTINGS.MaxWebSocketMessagePayloadSize = 2**20
dataspace = "test/test"


async def get_app_token(rc=None):
    return None


@pytest_asyncio.fixture
async def eclient():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ws_open = sock.connect_ex(("127.0.0.1", 9100)) == 0

    if not ws_open:
        pytest.skip(
            reason="websocket for test server not open", allow_module_level=True
        )

    async with connect() as client:
        yield client


@pytest_asyncio.fixture
async def duri(eclient: ETPClient):
    uri = eclient.dataspace_uri("test/test")
    try:
        resp = await eclient.put_dataspaces_no_raise([""], [""], [""], [""], uri)
        # assert len(resp) == 1, "created one dataspace"
        yield uri
    finally:
        resp = await eclient.delete_dataspaces(uri)
        assert len(resp) == 1, "should cleanup our test dataspace"


@pytest.fixture
def random_2d_resqml_grid() -> tuple[
    resqml_objects.EpcExternalPartReference,
    resqml_objects.LocalDepth3dCrs,
    resqml_objects.Grid2dRepresentation,
    npt.NDArray[np.float32],
]:
    shape = (201, 203)
    scaling = 100.0
    z_values = 2 * (np.random.random(shape).astype(np.float32) - 0.5) * scaling

    return construct_2d_resqml_grid_from_array(z_values)


def construct_2d_resqml_grid_from_array(
    z_values: npt.NDArray[np.number],
) -> tuple[
    resqml_objects.EpcExternalPartReference,
    resqml_objects.LocalDepth3dCrs,
    resqml_objects.Grid2dRepresentation,
    npt.NDArray[np.float32],
]:
    x0 = 2 * (np.random.random() - 0.5) * 1e6
    y0 = 2 * (np.random.random() - 0.5) * 1e6
    dx = x0 / z_values.shape[0]
    dy = y0 / z_values.shape[1]

    xoffset = 2 * (np.random.random() - 0.5) * 1e3
    yoffset = 2 * (np.random.random() - 0.5) * 1e3
    zoffset = 2 * (np.random.random() - 0.5) * 1e3
    rotation = 360 * np.random.random()

    # Valid EPSG codes are between 1024 (inclusive) and 32767 (exclusive).
    vertical_epsg = np.random.randint(low=1024, high=32767)
    horizontal_epsg = np.random.randint(low=1024, high=32767)

    common_citation_fields = dict(
        creation=XmlDateTime.from_string(
            datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
        ),
        originator="pyetp",
        format="equinor:pyetp-testing",
    )
    schema_version = "2.0"

    epc = resqml_objects.EpcExternalPartReference(
        citation=resqml_objects.Citation(
            title="Hdf Proxy",
            **common_citation_fields,
        ),
        schema_version=schema_version,
        uuid=str(uuid.uuid4()),
        mime_type="application/x-hdf5",
    )

    crs = resqml_objects.LocalDepth3dCrs(
        citation=resqml_objects.Citation(
            title="Random CRS",
            **common_citation_fields,
        ),
        schema_version=schema_version,
        uuid=str(uuid.uuid4()),
        xoffset=xoffset,
        yoffset=yoffset,
        zoffset=zoffset,
        areal_rotation=resqml_objects.PlaneAngleMeasure(
            value=rotation,
            uom=resqml_objects.PlaneAngleUom.DEGA,
        ),
        projected_axis_order=resqml_objects.AxisOrder2d.EASTING_NORTHING,
        projected_uom=resqml_objects.LengthUom.M,
        vertical_uom=resqml_objects.LengthUom.M,
        zincreasing_downward=True,
        vertical_crs=resqml_objects.VerticalCrsEpsgCode(
            epsg_code=vertical_epsg,
        ),
        projected_crs=resqml_objects.ProjectedCrsEpsgCode(
            epsg_code=horizontal_epsg,
        ),
    )

    gri = resqml_objects.Grid2dRepresentation(
        uuid=(grid_uuid := str(uuid.uuid4())),
        schema_version=schema_version,
        surface_role=resqml_objects.SurfaceRole.MAP,
        citation=resqml_objects.Citation(
            title="Random z-values",
            **common_citation_fields,
        ),
        grid2d_patch=resqml_objects.Grid2dPatch(
            patch_index=0,
            # NumPy-arrays are by default C-ordered, meaning that the last
            # index is the index that changes most rapidly. In this case this
            # means that the columns are the fastest changing axis.
            fastest_axis_count=z_values.shape[1],
            slowest_axis_count=z_values.shape[0],
            geometry=resqml_objects.PointGeometry(
                local_crs=resqml_objects.DataObjectReference(
                    # NOTE: See Energistics Identifier Specification 4.0
                    # (it is downloaded alongside the RESQML v2.0.1
                    # standard) section 4.1 for an explanation on the
                    # format of content_type.
                    content_type=(
                        f"application/x-resqml+xml;version={schema_version};"
                        f"type={crs.__class__.__name__}"
                    ),
                    title=crs.citation.title,
                    uuid=crs.uuid,
                ),
                points=resqml_objects.Point3dZValueArray(
                    supporting_geometry=resqml_objects.Point3dLatticeArray(
                        origin=resqml_objects.Point3d(
                            coordinate1=x0,
                            coordinate2=y0,
                            coordinate3=0.0,
                        ),
                        offset=[
                            # Offset for the y-direction, i.e., the fastest axis
                            resqml_objects.Point3dOffset(
                                offset=resqml_objects.Point3d(
                                    coordinate1=0.0,
                                    coordinate2=1.0,
                                    coordinate3=0.0,
                                ),
                                spacing=resqml_objects.DoubleConstantArray(
                                    value=dy,
                                    count=z_values.shape[1] - 1,
                                ),
                            ),
                            # Offset for the x-direction, i.e., the slowest axis
                            resqml_objects.Point3dOffset(
                                offset=resqml_objects.Point3d(
                                    coordinate1=1.0,
                                    coordinate2=0.0,
                                    coordinate3=0.0,
                                ),
                                spacing=resqml_objects.DoubleConstantArray(
                                    value=dx,
                                    count=z_values.shape[0] - 1,
                                ),
                            ),
                        ],
                    ),
                    zvalues=resqml_objects.DoubleHdf5Array(
                        values=resqml_objects.Hdf5Dataset(
                            path_in_hdf_file=f"/RESQML/{grid_uuid}/zvalues",
                            hdf_proxy=resqml_objects.DataObjectReference(
                                content_type=(
                                    f"application/x-eml+xml;version={schema_version};"
                                    f"type={epc.__class__.__name__}"
                                ),
                                title=epc.citation.title,
                                uuid=epc.uuid,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    return epc, crs, gri, z_values
