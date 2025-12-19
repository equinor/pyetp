import numpy as np

from resqml_objects.surface_helpers import RegularGridParameters


def test_2d_rotation() -> None:
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    c = np.array([1.0, 1.0]) / np.sqrt(2)

    r = np.column_stack([a, b, c])
    angle = np.pi / 4.0

    r_r = RegularGridParameters.rotate_2d_vector(r, angle=angle)
    # Here we fetch each rotated unit vector, and we therefore index each
    # column.
    a_r, b_r, c_r = r_r[:, 0], r_r[:, 1], r_r[:, 2]

    np.testing.assert_allclose(a_r, np.array([1.0, 1.0]) / np.sqrt(2.0), atol=1e-14)
    np.testing.assert_allclose(b_r, np.array([-1.0, 1.0]) / np.sqrt(2.0), atol=1e-14)
    np.testing.assert_allclose(c_r, np.array([0.0, 1.0]), atol=1e-14)

    r_rev = RegularGridParameters.rotate_2d_vector(r_r, angle=-angle)
    np.testing.assert_allclose(r_rev, r, atol=1e-14)


def test_angle_and_unit_vectors() -> None:
    r = RegularGridParameters.angle_to_unit_vectors(0.0)
    np.testing.assert_equal(r[:, 0], np.array([1.0, 0.0]))
    np.testing.assert_equal(r[:, 1], np.array([0.0, 1.0]))

    r = RegularGridParameters.angle_to_unit_vectors(-np.pi / 4.0)
    np.testing.assert_allclose(r[:, 0], np.array([1.0, -1.0]) / np.sqrt(2.0))
    np.testing.assert_allclose(r[:, 1], np.array([1.0, 1.0]) / np.sqrt(2.0))

    angles = 8 * np.pi * (np.random.random(100) - 0.5)

    for angle in angles:
        r_r = RegularGridParameters.angle_to_unit_vectors(angle)
        ret_angle = RegularGridParameters.unit_vectors_to_angle(r_r)

        while angle > np.pi:
            angle -= 2.0 * np.pi

        while angle < -np.pi:
            angle += 2.0 * np.pi

        np.testing.assert_allclose(angle, np.angle(r_r[0, 0] + 1j * r_r[1, 0]))
        np.testing.assert_allclose(angle, ret_angle)


def test_regular_grid_parameters() -> None:
    x = np.linspace(0, 1, 101)
    y = np.linspace(1, 2, 103)

    X, Y = np.meshgrid(x, y, indexing="ij")
    crs_angle = 2 * np.pi * (np.random.random() - 0.5)

    rgp_1 = RegularGridParameters.from_xy_grid(X, Y, crs_angle=crs_angle)
    rgp_2 = RegularGridParameters.from_xy_grid_vectors(x, y, crs_angle=crs_angle)

    np.testing.assert_equal(rgp_1.shape, rgp_2.shape)
    np.testing.assert_equal(rgp_1.origin, rgp_2.origin)
    np.testing.assert_equal(rgp_1.spacing, rgp_2.spacing)
    np.testing.assert_equal(rgp_1.unit_vectors, rgp_2.unit_vectors)
    np.testing.assert_equal(rgp_1.crs_angle, rgp_2.crs_angle)

    X_u, Y_u = rgp_1.to_xy_grid(with_crs_angle=False)
    np.testing.assert_allclose(X_u, X)
    np.testing.assert_allclose(Y_u, Y)

    X_1, Y_1 = rgp_1.to_xy_grid()
    X_2, Y_2 = rgp_2.to_xy_grid()
    np.testing.assert_equal(X_1, X_2)
    np.testing.assert_equal(Y_1, Y_2)

    r_r = RegularGridParameters.rotate_2d_vector(
        np.vstack([np.ravel(X) - X[0, 0], np.ravel(Y) - Y[0, 0]]),
        angle=crs_angle,
    )

    # In this case fetch the `x` and `y` components separately, instead of
    # fetching each point. Hence, we index the rows.
    X_r = r_r[0].reshape(X.shape) + X[0, 0]
    Y_r = r_r[1].reshape(Y.shape) + Y[0, 0]
    np.testing.assert_allclose(X_r, X_1, atol=1e-14)
    np.testing.assert_allclose(Y_r, Y_1, atol=1e-14)


def test_regular_grid_parameters_rotated() -> None:
    x = np.linspace(0, 1, 101)
    y = np.linspace(1, 2, 103)

    X, Y = np.meshgrid(x, y, indexing="ij")
    crs_angle = 2 * np.pi * (np.random.random() - 0.5)

    r_r = RegularGridParameters.rotate_2d_vector(
        np.vstack([np.ravel(X) - X[0, 0], np.ravel(Y) - Y[0, 0]]),
        angle=crs_angle,
    )

    X_r = r_r[0].reshape(X.shape) + X[0, 0]
    Y_r = r_r[1].reshape(Y.shape) + Y[0, 0]

    rgp_1 = RegularGridParameters.from_xy_grid(X_r, Y_r, crs_angle=0.0)
    rgp_2 = RegularGridParameters.from_xy_grid_vectors(x, y, crs_angle=crs_angle)

    np.testing.assert_allclose(rgp_1.shape, rgp_2.shape)
    np.testing.assert_allclose(rgp_1.origin, rgp_2.origin)
    np.testing.assert_allclose(rgp_1.spacing, rgp_2.spacing)

    X_1, Y_1 = rgp_1.to_xy_grid()
    X_2, Y_2 = rgp_2.to_xy_grid()
    np.testing.assert_allclose(X_1, X_2)
    np.testing.assert_allclose(Y_1, Y_2)

    np.testing.assert_allclose(X_r, X_1, atol=1e-14)
    np.testing.assert_allclose(Y_r, Y_1, atol=1e-14)


def test_regular_grid_parameters_double_rotated() -> None:
    x = np.linspace(0, 1, 101)
    y = np.linspace(1, 2, 103)

    X, Y = np.meshgrid(x, y, indexing="ij")
    grid_angle = 2 * np.pi * (np.random.random() - 0.5)
    crs_angle = 2 * np.pi * (np.random.random() - 0.5)

    r = RegularGridParameters.rotate_2d_vector(
        np.vstack([np.ravel(X) - X[0, 0], np.ravel(Y) - Y[0, 0]]),
        angle=grid_angle,
    )

    X_r = r[0].reshape(X.shape) + X[0, 0]
    Y_r = r[1].reshape(Y.shape) + Y[0, 0]

    r_r = RegularGridParameters.rotate_2d_vector(
        np.vstack([np.ravel(X_r) - X_r[0, 0], np.ravel(Y_r) - Y_r[0, 0]]),
        angle=crs_angle,
    )
    X_rr = r_r[0].reshape(X.shape) + X[0, 0]
    Y_rr = r_r[1].reshape(Y.shape) + Y[0, 0]

    rgp_1 = RegularGridParameters.from_xy_grid(X_r, Y_r, crs_angle=crs_angle)
    rgp_2 = RegularGridParameters.from_xy_grid(X_rr, Y_rr, crs_angle=0.0)

    X_1, Y_1 = rgp_1.to_xy_grid()
    X_2, Y_2 = rgp_2.to_xy_grid()
    np.testing.assert_allclose(X_1, X_2)
    np.testing.assert_allclose(Y_1, Y_2)

    r_u = RegularGridParameters.rotate_2d_vector(
        np.vstack([np.ravel(X_1) - X_1[0, 0], np.ravel(Y_1) - Y_1[0, 0]]),
        angle=-crs_angle - grid_angle,
    )
    X_u = r_u[0].reshape(X_1.shape) + X_1[0, 0]
    Y_u = r_u[1].reshape(Y_1.shape) + Y_1[0, 0]

    np.testing.assert_allclose(X_u, X, atol=1e-14)
    np.testing.assert_allclose(Y_u, Y, atol=1e-14)
