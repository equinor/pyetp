import typing
import uuid

import numpy as np
import numpy.typing as npt

import resqml_objects.v201.generated as ro
from resqml_objects.surface_helpers import rotate_2d_vector

Vec2D: typing.TypeAlias = typing.Annotated[
    npt.NDArray[np.float64],
    dict(shape=(2,)),
]
SurfaceArrayType: typing.TypeAlias = typing.Annotated[
    npt.NDArray[np.float64],
    dict(shape=(None, None)),
]


class RegularSurfaceModels:
    @staticmethod
    def get_depth_model(
        originator: str,
        title: str,
        origin: Vec2D,
        spacing: Vec2D,
        angle_in_rad: float | np.float64,
        surf: SurfaceArrayType,
        vertical_epsg_code: int,
        projected_epsg_code: int,
        zincreasing_downward: bool = True,
        projected_uom: ro.LengthUom = ro.LengthUom.M,
        vertical_uom: ro.LengthUom = ro.LengthUom.M,
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
        unit_vec_1 = np.squeeze(
            rotate_2d_vector(np.array([1.0, 0.0]), angle=angle_in_rad)
        )
        unit_vec_2 = np.squeeze(
            rotate_2d_vector(np.array([0.0, 1.0]), angle=angle_in_rad)
        )

        uuid_epc = uuid_epc or uuid.uuid4()
        uuid_crs = uuid_crs or uuid.uuid4()
        uuid_gri = uuid_gri or uuid.uuid4()

        epc = ro.obj_EpcExternalPartReference(
            citation=ro.Citation(
                title=f"Hdf proxy for {title}",
                originator=originator,
            ),
            uuid=str(uuid_epc),
        )
        crs = ro.obj_LocalDepth3dCrs(
            citation=ro.Citation(
                title=f"Local crs for {title}",
                originator=originator,
            ),
            uuid=str(uuid_crs),
            vertical_crs=ro.VerticalCrsEpsgCode(epsg_code=vertical_epsg_code),
            projected_crs=ro.ProjectedCrsEpsgCode(epsg_code=projected_epsg_code),
            zincreasing_downward=zincreasing_downward,
            projected_uom=projected_uom,
            vertical_uom=vertical_uom,
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
            uuid=str(uuid_gri),
        )

        points = gri.grid2d_patch.geometry.points
        assert isinstance(points, ro.Point3dZValueArray)
        zvalues = points.zvalues
        assert isinstance(zvalues, ro.DoubleHdf5Array)
        path_in_hdf_file = zvalues.values.path_in_hdf_file
        data_arrays = {path_in_hdf_file: surf}

        return (epc, crs, gri), data_arrays
