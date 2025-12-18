import typing
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class RegularGridParameters:
    shape: tuple[int, int]
    origin: tuple[float, float]
    dr: tuple[float, float]
    unit_vectors: tuple[tuple[float, float], tuple[float, float]]

    @classmethod
    def from_xy_grid(
        cls,
        X: npt.NDArray[np.float32 | np.float64],
        Y: npt.NDArray[np.float32 | np.float64],
    ) -> typing.Self:
        if len(np.shape(np.squeeze(X))) == 1 and len(np.shape(np.squeeze(Y))) == 1:
            return cls.from_xy_grid_vectors(X, Y)

        assert len(np.shape(X)) == 2
        assert np.shape(X) == np.shape(Y)

        x_col_diffs = np.diff(X, axis=1)
        y_col_diffs = np.diff(Y, axis=1)
        x_row_diffs = np.diff(X, axis=0)
        y_row_diffs = np.diff(Y, axis=0)

        # Check that the spacing is uniform in all directions
        np.testing.assert_allclose(x_col_diffs, x_col_diffs[0, 0])
        np.testing.assert_allclose(y_col_diffs, y_col_diffs[0, 0])
        np.testing.assert_allclose(x_row_diffs, x_row_diffs[0, 0])
        np.testing.assert_allclose(y_row_diffs, y_row_diffs[0, 0])

        xvec = np.array([x_col_diffs[0], y_col_diffs[0]])
        yvec = np.array([x_row_diffs[0], y_row_diffs[0]])

        dr = np.array([xvec / np.linalg.norm(xvec), yvec / np.linalg.norm(yvec)])
        xu = xvec / dr[0]
        yu = yvec / dr[1]

        return cls(
            shape=np.shape(X),
            origin=(X[0, 0], Y[0, 0]),
            dr=tuple(dr.tolist()),
            unit_vectors=tuple(tuple(xu.tolist()), tuple(yu.tolist())),
        )

    @classmethod
    def from_xy_grid_vectors(
        cls,
        x: npt.NDArray[np.float32 | np.float64],
        y: npt.NDArray[np.float32 | np.float64],
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

        if not np.allclose(x_spacing, x_spacing[0]) or not np.allclose(
            y_spacing, y_spacing[0]
        ):
            raise ValueError(
                "The 'x' and 'y' grid vectors must have a uniform spacing"
            )

        shape = (len(x), len(y))
        origin = (float(x[0]), float(y[0]))
        dr = (float(x_spacing[0]), float(y_spacing[0]))
        unit_vectors = (
            (1.0, 0.0),
            (0.0, 1.0),
        )

        return cls(
            shape=shape,
            origin=origin,
            dr=dr,
            unit_vectors=unit_vectors,
        )
