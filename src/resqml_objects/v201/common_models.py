import typing
import uuid

import numpy as np
import numpy.typing as npt

import resqml_objects.v201.generated as ro
from resqml_objects.surface_helpers import angle_to_unit_vectors

Vec2D: typing.TypeAlias = typing.Annotated[
    npt.NDArray[np.float64],
    dict(shape=(2,)),
]
SurfaceArrayType: typing.TypeAlias = typing.Annotated[
    npt.NDArray[np.float64],
    dict(shape=(None, None)),
]


class RegularSurfaceModel:
    # class Rotation:
    #     pass

    @staticmethod
    def get_depth_model(
        originator: str,
        title: str,
        origin: Vec2D,
        spacing: Vec2D,
        surf: SurfaceArrayType,
        vertical_epsg_code: int,
        projected_epsg_code: int,
        # rotation: Rotation,
        unit_vec_1: Vec2D | None = None,
        unit_vec_2: Vec2D | None = None,
        angle_in_rad: float | np.float64 | None = None,
        zincreasing_downward: bool = True,
        uuid_epc: str | uuid.UUID | None = None,
        uuid_crs: str | uuid.UUID | None = None,
        uuid_gri: str | uuid.UUID | None = None,
    ) -> tuple[
        tuple[
            ro.obj_EpcExternalPartReference,
            ro.obj_LocalDepth3dCrs,
            ro.obj_Grid2dRepresentation,
        ],
        dict[str, SurfaceArrayType],
    ]:
        if angle_in_rad is None and (unit_vec_1 is None or unit_vec_2 is None):
            raise ValueError(
                "Specify either an angle (in radians) or a pair of unit vectors"
            )

        if angle_in_rad is not None:
            unit_vectors = angle_to_unit_vectors(angle=angle_in_rad)
            # The `x`- and `y`-coordinates lie in the columns of
            # `unit_vectors`. Hence, the first unit vector corresponds to the
            # first row (`unit_vectors[0]`), and the second unit vector the
            # second row (`unit_vectors[1]`).
            unit_vec_1 = unit_vectors[0]
            unit_vec_2 = unit_vectors[1]

        uuid_epc = uuid_epc or str(uuid.uuid4())
        uuid_crs = uuid_crs or str(uuid.uuid4())
        uuid_gri = uuid_gri or str(uuid.uuid4())

        epc = ro.obj_EpcExternalPartReference(
            citation=ro.Citation(
                title=f"Hdf proxy for {title}",
                originator=originator,
            ),
            uuid=uuid_epc,
        )
        crs = ro.obj_LocalDepth3dCrs(
            citation=ro.Citation(
                title=f"Local crs for {title}",
                originator=originator,
            ),
            uuid=uuid_crs,
            vertical_crs=ro.VerticalCrsEpsgCode(epsg_code=vertical_epsg_code),
            projected_crs=ro.ProjectedCrsEpsgCode(epsg_code=projected_epsg_code),
            zincreasing_downward=zincreasing_downward,
        )

        gri = ro.obj_Grid2dRepresentation.from_regular_surface(
            citation=ro.Citation(title=title, originator=originator),
            crs=crs,
            epc_external_part_reference=epc,
            shape=surf.shape,
            origin=origin,
            spacing=spacing,
            unit_vec_1=unit_vec_1,
            unit_vec_2=unit_vec_2,
            uuid=uuid_gri,
        )

        path_in_hdf_file = (
            gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file
        )
        data_arrays = {path_in_hdf_file: surf}

        return (epc, crs, gri), data_arrays
