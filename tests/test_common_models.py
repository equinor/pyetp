import uuid

import numpy as np

import resqml_objects.v201 as ro
from resqml_objects.surface_helpers import RegularGridParameters
from resqml_objects.v201.common_models import RegularSurfaceModels


def test_regular_surface_depth_model() -> None:
    originator = "resqml-objects-tester"
    title = "Test surface"
    shape = tuple(np.random.randint(10, 123, size=2).tolist())

    x = np.linspace(
        -20 * (np.random.random() + 0.1), 20 * (np.random.random() + 0.1), shape[0]
    )
    y = np.linspace(
        -20 * (np.random.random() + 0.1), 20 * (np.random.random() + 0.1), shape[1]
    )
    surf = -np.exp(
        -(np.linspace(-1, 1, shape[0])[:, None] ** 2)
        - np.linspace(-1, 1, shape[1]) ** 2
    )

    origin = np.array([x[0], y[0]])
    spacing = np.array([x[1] - x[0], y[1] - y[0]])

    grid_angle = 2 * np.pi * (np.random.random() - 0.5)

    vertical_epsg_code = 1234
    projected_epsg_code = 23456

    ml_objects, data_arrays = RegularSurfaceModels.get_depth_model(
        originator=originator,
        title=title,
        origin=origin,
        spacing=spacing,
        angle_in_rad=grid_angle,
        surf=surf,
        vertical_epsg_code=vertical_epsg_code,
        projected_epsg_code=projected_epsg_code,
        zincreasing_downward=False,
        projected_uom=ro.LengthUom.FT_BN_B,
        vertical_uom=ro.LengthUom.FT_BN_A,
        uuid_epc=(uuid_epc := str(uuid.uuid4())),
        uuid_crs=(uuid_crs := str(uuid.uuid4())),
        uuid_gri=(uuid_gri := str(uuid.uuid4())),
    )

    epc, crs, gri = ml_objects

    assert epc.citation.originator == originator
    assert epc.uuid == uuid_epc
    assert crs.citation.originator == originator
    assert crs.uuid == uuid_crs
    assert gri.citation.originator == originator
    assert gri.uuid == uuid_gri

    assert crs.projected_uom == ro.LengthUom.FT_BN_B
    assert crs.vertical_uom == ro.LengthUom.FT_BN_A
    assert not crs.zincreasing_downward

    rsp = gri.get_regular_surface_parameters(crs=crs)

    assert shape == rsp.shape
    np.testing.assert_allclose(origin, rsp.origin)
    np.testing.assert_allclose(spacing, rsp.spacing)
    assert abs(grid_angle - rsp.angle) < 1e-12

    X, Y = gri.get_xy_grid(crs=crs)
    rgp = RegularGridParameters.from_xy_grid(X, Y)
    assert rgp.shape == shape
    np.testing.assert_allclose(rgp.origin, origin)
    np.testing.assert_allclose(rgp.spacing, spacing)
