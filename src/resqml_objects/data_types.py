import typing

import numpy as np
import numpy.typing as npt


class RegularSurfaceParameters(typing.NamedTuple):
    shape: tuple[int, int]
    origin: typing.Annotated[npt.NDArray[np.float64], dict(shape=(2,))]
    spacing: typing.Annotated[npt.NDArray[np.float64], dict(shape=(2,))]
    angle: float
