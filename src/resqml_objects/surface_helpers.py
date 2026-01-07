import typing
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

DType = typing.TypeVar("DType", bound=np.float32 | np.float64)


def rotate_2d_vector(r: npt.NDArray[DType], angle: float) -> npt.NDArray[DType]:
    """
    Function used to rotate a set of `N` 2d-vectors stacked in an array of
    shape `(2, N)` by an angle `angle` (in radians) counter-clockwise. This
    function does not remove the origin of the vectors, so that is up to the
    user if this should be done.

    Returns
    -------
    npt.NDArray[DType]
        The rotated 2d vector with the same shape as the input vector.
    """
    if not np.shape(r)[0] == 2:
        raise TypeError("Array 'r' must have shape '(2,)' or '(2, None)'")

    if len(np.shape(r)) == 1:
        r = r.reshape(-1, 1)

    c = np.cos(angle / 2.0)
    s = np.sin(angle / 2.0)

    return (c**2 - s**2) * r + 2 * c * s * np.vstack([-r[1], r[0]])


def angle_to_unit_vectors(
    angle: float,
) -> typing.Annotated[npt.NDArray[np.float64], dict(shape=(2, 2))]:
    """
    Function that constructs a pair of orthonormal unit vectors from an angle
    (in radians), where the first unit vector (the `x`-direction) is rotated
    counter-clockwise to the `[1.0, 0.0]`-axis, and the second unit vector is
    rotated `pi/2.0` compared to the `x`-unit vector.
    """

    return rotate_2d_vector(np.eye(2).astype(np.float64), angle)


def unit_vectors_to_angle(
    unit_vectors: typing.Annotated[npt.NDArray[DType], dict(shape=(2, 2))],
) -> float:
    """
    Function returning the angle (in radians) from the `x`-axis (i.e., the
    `[1.0, 0.0]`-vector) to the `unit_vectors[:, 0]`-vector. This function
    takes in a pair of unit vectors, where the second vector (`unit_vectors[:,
    1]`) is only used to check that the vectors are orthonormal, and that it is
    rotated `np.pi / 2.0` counter-clockwise relative to the first vector.

    This function uses `np.atan2` to choose the correct quadrant.

    Note this entire function can be replaced by `np.atan2` (or `np.angle`
    making the `y`-component of the `x`-vector complex) for the `x`-vector of
    the unit vectors, but is kept to give an interface where you can pass in
    both unit vectors and verify that they are orthonormal and rotated the
    expected way relative to each other.
    """
    x_vec = unit_vectors[:, 0]
    y_vec = unit_vectors[:, 1]

    tol = np.finfo(x_vec.dtype).eps * 100
    if not np.allclose(x_vec @ y_vec, 0.0, atol=tol):
        raise ValueError("Unit vectors are not orthonormal")

    angle = np.atan2(x_vec[1], x_vec[0])

    # Check to ensure that the `y`-vector is rotated `pi / 2.0`
    # counter-clockwise from the `x`-vector.
    y_angle = np.atan2(y_vec[1], y_vec[0])

    if angle == np.pi / 2.0:
        if y_angle >= 0.0:
            np.testing.assert_allclose(
                angle,
                y_angle - np.pi / 2.0,
                atol=tol,
            )
        else:
            np.testing.assert_allclose(
                angle,
                y_angle + np.pi,
                atol=tol,
            )
    elif np.pi / 2.0 < angle <= np.pi:
        np.testing.assert_allclose(angle, y_angle + 3 * np.pi / 2.0, atol=tol)
    else:
        np.testing.assert_allclose(
            angle,
            y_angle - np.pi / 2.0,
            atol=tol,
        )

    return angle


@dataclass
class RegularGridParameters(typing.Generic[DType]):
    """
    Dataclass acting as a set of helper functions and container for regular
    two-dimensional `X` and `Y` grids that follow a surface `Z`. The purpose of
    this dataclass is to either take in uniformly spaced, regular, grids `X`
    and `Y` (can be rotated), or grid vectors `x` and `y`, and store the
    `shape`, `origin`, `spacing` and `unit_vectors` that are needed to
    reconstruct the full grids, and provide methods that returns the full
    grids. Furthermore, the class has optional `crs_angle` (set to `0.0` by
    default, and measured in radians) which makes it convenient to work with
    rotated grids inside rotated local coordinate reference systems.

    Attributes
    ----------
    shape: tuple[int, int]
        The shape of the full `X` and `Y` grids and the corresponding `Z`-array
        of surface values.
    origin: typing.Annotated[npt.NDArray[DType], dict(shape=(2,))]
        A two-dimensional array of origin values. Corresponds to `X[0, 0]` and
        `Y[0, 0]` in the full grids.
    spacing: typing.Annotated[npt.NDArray[DType], dict(shape=(2,))]
        A two-dimensional array with the grid spacing in each direction.
    unit_vectors: typing.Annotated[npt.NDArray[DType], dict(shape=(2, 2))]
        Two two-dimensional unit vectors of the `X` and `Y` grids. The vectors
        lie in the columns of `unit_vectors`, i.e., `vec_1 = unit_vectors[:,
        0]` and `vec_2 = unit_vectors[:, 1]`.
    crs_angle: float
        The rotation of the local coordinate reference system, if applicable.
        The default is `0.0`, i.e., an urotated coordinate system is used.
    crs_offset: typing.Annotated[npt.NDArray[DType], dict(shape=(2,))] | None
        The offset of the origin of the local coordinate reference system, if
        applicable. The default is `None` which gets defaulted to `array([0.0,
        0.0])` of type `DType`.

    Note that we construct the edge of `X` from `shape[0]`, `origin[0]`,
    `spacing[0]`, and `unit_vectors[:, 0]`, and the edge of `Y` from the second
    element/column.
    """

    shape: tuple[int, int]
    origin: typing.Annotated[npt.NDArray[DType], dict(shape=(2,))]
    spacing: typing.Annotated[npt.NDArray[DType], dict(shape=(2,))]
    unit_vectors: typing.Annotated[npt.NDArray[DType], dict(shape=(2, 2))]
    crs_angle: float = 0.0
    crs_offset: typing.Annotated[npt.NDArray[DType], dict(shape=(2,))] | None = None

    # Attaching the helper functions to this class to avoid doing extra
    # imports.
    rotate_2d_vector = staticmethod(rotate_2d_vector)
    angle_to_unit_vectors = staticmethod(angle_to_unit_vectors)
    unit_vectors_to_angle = staticmethod(unit_vectors_to_angle)

    def __post_init__(self) -> None:
        if self.crs_offset is None:
            self.crs_offset = np.zeros_like(self.origin)

    def to_xy_grid(
        self, to_global_crs: bool = True
    ) -> tuple[
        npt.NDArray[DType],
        npt.NDArray[DType],
    ]:
        vec_1 = self.unit_vectors[:, 0]
        vec_2 = self.unit_vectors[:, 1]
        origin = self.origin

        if to_global_crs:
            # Here we construct the unit vectors of the global CRS as seen from
            # the local CRS (global-in-local -> ginl). Hence, the negative sign
            # on the rotation. The global unit vectors are in the columns (axis
            # 1) of the 2 x 2 matrix.
            ginl_unit_vectors = self.angle_to_unit_vectors(-self.crs_angle)
            # Next, we find the elements of the grid unit vectors in the global
            # coordinate system.
            #
            #   (new_vec_i)_k = (vec_i)_k @ e_k,
            #
            # where `e_k` is the `k`'th column of `ginl_unit_vectors` and
            # `(vec_i)_k` is the `k`'th element of `vec_i` (with `i` being
            # either `1` or `2`). Below we do the full transformation in a
            # single call.
            new_unit_vectors = np.einsum(
                "ij, kj -> ik", self.unit_vectors, ginl_unit_vectors
            )

            vec_1 = new_unit_vectors[:, 0]
            vec_2 = new_unit_vectors[:, 1]

            # Computing the origin of the surface in the global CRS by adding
            # in the offset of the local CRS.
            origin = origin + self.crs_offset

        edge_1 = vec_1 * self.spacing[0] * np.arange(self.shape[0]).reshape(-1, 1)
        edge_2 = vec_2 * self.spacing[1] * np.arange(self.shape[1]).reshape(-1, 1)

        X = edge_1[:, 0].reshape(-1, 1) + edge_2[:, 0]
        Y = edge_1[:, 1].reshape(-1, 1) + edge_2[:, 1]

        return X + origin[0], Y + origin[1]

    @classmethod
    def from_xy_grid(
        cls,
        X: npt.NDArray[DType],
        Y: npt.NDArray[DType],
        crs_angle: float = 0.0,
        crs_offset: npt.NDArray[DType] | None = None,
    ) -> typing.Self:
        if len(np.shape(np.squeeze(X))) == 1 and len(np.shape(np.squeeze(Y))) == 1:
            return cls.from_xy_grid_vectors(
                X, Y, crs_angle=crs_angle, crs_offset=crs_offset
            )

        assert len(np.shape(X)) == 2
        assert np.shape(X) == np.shape(Y)
        assert X.dtype == Y.dtype

        x_col_diffs = np.diff(X, axis=1)
        y_col_diffs = np.diff(Y, axis=1)
        x_row_diffs = np.diff(X, axis=0)
        y_row_diffs = np.diff(Y, axis=0)

        tol = np.finfo(X.dtype).eps * 10
        # Check that the spacing is uniform in all directions.
        np.testing.assert_allclose(x_col_diffs, x_col_diffs[0, 0], atol=tol)
        np.testing.assert_allclose(y_col_diffs, y_col_diffs[0, 0], atol=tol)
        np.testing.assert_allclose(x_row_diffs, x_row_diffs[0, 0], atol=tol)
        np.testing.assert_allclose(y_row_diffs, y_row_diffs[0, 0], atol=tol)

        xvec = np.array([x_row_diffs[0, 0], y_row_diffs[0, 0]])
        yvec = np.array([x_col_diffs[0, 0], y_col_diffs[0, 0]])

        spacing = np.array([np.linalg.norm(xvec), np.linalg.norm(yvec)])
        xu = xvec / spacing[0]
        yu = yvec / spacing[1]

        # Verify that the unit vectors are orthonormal.
        np.testing.assert_allclose(abs(xu @ yu) ** 2, 0, atol=tol)

        unit_vectors = np.column_stack([xu, yu])
        # TODO: Remove these
        np.testing.assert_equal(unit_vectors[:, 0], xu)
        np.testing.assert_equal(unit_vectors[:, 1], yu)

        return cls(
            shape=np.shape(X),
            origin=np.array([X[0, 0], Y[0, 0]]),
            spacing=spacing,
            unit_vectors=unit_vectors,
            crs_angle=crs_angle,
            crs_offset=crs_offset,
        )

    @classmethod
    def from_xy_grid_vectors(
        cls,
        x: npt.NDArray[DType],
        y: npt.NDArray[DType],
        crs_angle: float = 0.0,
        crs_offset: npt.NDArray[DType] | None = None,
    ) -> typing.Self:
        x = np.squeeze(x)
        y = np.squeeze(y)

        if len(np.shape(x)) != 1 or len(np.shape(y)) != 1:
            raise ValueError(
                "The 'x' and 'y' grid vectors must be 1-d arrays (after using "
                "`np.squeeze`)"
            )

        x_spacing = np.diff(x)
        y_spacing = np.diff(y)

        tol = np.finfo(x.dtype).eps * 10
        if not np.allclose(x_spacing, x_spacing[0], atol=tol) or not np.allclose(
            y_spacing, y_spacing[0], atol=tol
        ):
            raise ValueError("The 'x' and 'y' grid vectors must have a uniform spacing")

        assert x.dtype == y.dtype

        shape = (len(x), len(y))
        unit_vectors = np.eye(2).astype(x.dtype)

        return cls(
            shape=shape,
            origin=np.array([x[0], y[0]]),
            spacing=np.array([x_spacing[0], y_spacing[0]]),
            unit_vectors=unit_vectors,
            crs_angle=crs_angle,
            crs_offset=crs_offset,
        )
