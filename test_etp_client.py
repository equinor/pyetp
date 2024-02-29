import tempfile
from pathlib import Path

import numpy as np
import pytest
import xtgeo
from fastapi.testclient import TestClient

import map_api.resqml_objects as resqml_objects
from map_api.etp_client.client import ETPClient
from map_api.etp_client.uri import DataspaceURI
from map_api.main import DeleteMapBody, MapPayload, NewMapInterface, app, projectCRSData


def create_surface(ncol: int, nrow: int):
    surface = xtgeo.RegularSurface(
        name="testsurface",
        ncol=ncol,
        nrow=nrow,
        xori=np.random.rand() * 1000,
        yori=np.random.rand() * 1000000,
        xinc=23.0,
        yinc=47.3,
        rotation=33.0,
        values=np.random.random((nrow, ncol)).astype(np.float32),
    )
    surface.unrotate()
    return surface


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', [create_surface(3, 4), create_surface(100, 40)])
async def test_rddms_roundtrip(eclient: ETPClient, surface: xtgeo.RegularSurface, duri: DataspaceURI):
    # NOTE: xtgeo calls the first axis (axis 0) of the values-array
    # columns, and the second axis by rows.

    epsg_code = 23031
    epc_uri, crs_uri, gri_uri = await eclient.put_xtgeo_surface(surface, epsg_code, dataspace=duri)

    epc, crs, gri = await eclient.get_resqml_objects(epc_uri, crs_uri, gri_uri)
    newsurf = await eclient.get_xtgeo_surface(epc_uri, gri_uri)
    array = np.array(newsurf.values.filled(np.nan))

    np.testing.assert_allclose(array, np.array(surface.values.filled(np.nan)))

    print(epc, crs, gri)

    assert isinstance(epc, resqml_objects.EpcExternalPartReference)
    assert isinstance(crs, resqml_objects.LocalDepth3dCrs)
    assert isinstance(gri, resqml_objects.Grid2dRepresentation)

    assert crs.projected_crs.epsg_code == epsg_code
    assert surface.rotation == crs.areal_rotation.value
    assert array.shape[0] == gri.grid2d_patch.slowest_axis_count == surface.values.shape[0] == surface.ncol
    assert array.shape[1] == gri.grid2d_patch.fastest_axis_count == surface.values.shape[1] == surface.nrow

    supporting_geometry = gri.grid2d_patch.geometry.points.supporting_geometry
    assert isinstance(supporting_geometry, resqml_objects.Point3dLatticeArray)

    assert surface.xori == supporting_geometry.origin.coordinate1
    assert surface.yori == supporting_geometry.origin.coordinate2
    assert surface.xinc == supporting_geometry.offset[0].spacing.value
    assert surface.yinc == supporting_geometry.offset[1].spacing.value


@pytest.fixture
def surface_path():
    surface = create_surface(3, 4)
    with tempfile.NamedTemporaryFile(mode='+bw') as fp:
        fp.close()
        surface.to_file(Path(fp.name), fformat='irap_binary')
        yield Path(fp.name)


def test_etp_api(client: TestClient, surface_path: Path):

    # test success upload
    crs = projectCRSData(epsg=12345, name='test')
    payload = MapPayload(
        projectId='test', projectCRS=crs, filePath=str(surface_path), transformPipeline='',
        metadata=NewMapInterface(name='test', description='test', crsName='test', format='irap_binary', zUnit='m', mapType='value')
    )

    response = client.post(
        app.url_path_for('parse_map_binary_data'),
        content=payload.json()
    )
    assert response.status_code == 200, "endpoint should be OK"

    # TODO: test fail upload delete
    # rerror = client.post(
    #     app.url_path_for('parse_map_binary_data'),
    #     content=payload.model_dump_json()
    # )
    # assert rerror.status_code == 409, "should return conflic"

    # test success delete
    payload = DeleteMapBody(**response.json())
    response = client.request(
        'DELETE',
        app.url_path_for('delete_map'),
        content=payload.json()
    )
    assert response.status_code == 200, "endpoint should be OK"

    # test fail delete
    response = client.request(
        'DELETE',
        app.url_path_for('delete_map'),
        content=payload.json()
    )
    assert response.status_code == 404, "should be conflict"


def test_etp_api_validation_error(client: TestClient):

    # test fail delete
    response = client.request(
        'DELETE',
        app.url_path_for('delete_map'),
        json=dict(rddmsURLs=['http://localhost/test'])
    )
    print(response.text)
    assert response.status_code == 422, "should be invalid"
