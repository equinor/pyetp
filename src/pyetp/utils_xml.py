import datetime
import logging
import typing as T
from uuid import uuid4

import numpy as np
from xsdata.models.datatype import XmlDateTime

import resqml_objects.v201 as ro
from resqml_objects.v201.utils import (
    get_content_type_string,
    get_data_object_reference,
    resqml_schema_version,
    common_schema_version,
)
from pyetp.config import SETTINGS

if T.TYPE_CHECKING:
    from xtgeo import RegularSurface


def create_common_citation(title: str):
    return ro.Citation(
        title=title,
        creation=XmlDateTime.from_string(
            datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
        ),
        originator=SETTINGS.application_name,
        format=f"{SETTINGS.application_name}:v{SETTINGS.application_version}",
    )


def create_common_crs(
    title: str,
    projected_epsg,
    rotation: float = 0.0,
    resqml_schema_version: str = resqml_schema_version,
):
    return ro.LocalDepth3dCrs(
        citation=create_common_citation(f"CRS for {title}"),
        schema_version=resqml_schema_version,
        uuid=str(uuid4()),
        # NOTE: I assume that we let the CRS have no offset, and add any offset
        # in the grid instead.
        xoffset=0.0,
        yoffset=0.0,
        zoffset=0.0,
        areal_rotation=ro.PlaneAngleMeasure(
            # Here rotation should be zero!
            value=rotation,
            uom=ro.PlaneAngleUom.DEGA,
        ),
        # NOTE: Verify that this is the projected axis order
        projected_axis_order=ro.AxisOrder2d.EASTING_NORTHING,
        projected_uom=ro.LengthUom.M,
        vertical_uom=ro.LengthUom.M,
        zincreasing_downward=True,
        vertical_crs=ro.VerticalCrsEpsgCode(epsg_code=projected_epsg),
        projected_crs=ro.ProjectedCrsEpsgCode(
            epsg_code=projected_epsg,
        ),
    )


def create_epc(common_schema_version: str = common_schema_version):
    return ro.EpcExternalPartReference(
        citation=create_common_citation("Hdf Proxy"),
        schema_version=common_schema_version,
        uuid=str(uuid4()),
        mime_type="application/x-hdf5",
    )


def parse_xtgeo_surface_to_resqml_grid(surf: "RegularSurface", projected_epsg: int):
    # Build the RESQML-objects "manually" from the generated dataclasses.
    # Their content is described also in the RESQML v2.0.1 standard that is
    # available for download here:
    # https://publications.opengroup.org/standards/energistics-standards/v231a

    title = surf.name or "regularsurface"

    # NOTE: xtgeo uses nrow for axis 1 in the array, and ncol for axis 0.  This
    # means that surf.nrow is the fastest changing axis, and surf.ncol the
    # slowest changing axis, and we have surf.values.shape == (surf.ncol,
    # surf.nrow). The author of this note finds that confusing, but such is
    # life.
    epc, crs, gri = instantiate_resqml_grid(
        title,
        surf.get_rotation(),
        surf.xori,
        surf.yori,
        surf.xinc,
        surf.yinc,
        surf.ncol,
        surf.nrow,
        projected_epsg,
    )
    return epc, crs, gri


def instantiate_resqml_grid(
    name: str,
    rotation: float,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    nx: int,
    ny: int,
    epsg: int,
    resqml_schema_version: str = resqml_schema_version,
):
    epc = create_epc()
    crs = create_common_crs(name, epsg, rotation)

    gri = ro.Grid2dRepresentation(
        uuid=(grid_uuid := str(uuid4())),
        schema_version=resqml_schema_version,
        surface_role=ro.SurfaceRole.MAP,
        citation=create_common_citation(name),
        grid2d_patch=ro.Grid2dPatch(
            # TODO: Perhaps we can use this for tiling?
            patch_index=0,
            # NumPy-arrays are C-ordered, meaning that the last index is
            # the index that changes most rapidly. However, xtgeo uses nrow for
            # axis 1 in the array, and ncol for axis 0. This means that
            # surf.nrow is the fastest changing axis, and surf.ncol the slowest
            # changing axis (as surf.values.shape == (surf.ncol, surf.nrow))
            fastest_axis_count=ny,
            slowest_axis_count=nx,
            geometry=ro.PointGeometry(
                local_crs=get_data_object_reference(crs),
                points=ro.Point3dZValueArray(
                    supporting_geometry=ro.Point3dLatticeArray(
                        origin=ro.Point3d(
                            coordinate1=x0,
                            coordinate2=y0,
                            coordinate3=0.0,
                        ),
                        # NOTE: The ordering in the offset-list should be
                        # preserved when the data is passed back and forth.
                        # However, _we_ need to ensure a consistent ordering
                        # for ourselves. In this setup I have set the slowest
                        # axis to come first, i.e., the x-axis or axis 0 in
                        # NumPy. The reason is so that it corresponds with the
                        # origin above where "coordinate1" is set to be the
                        # x0-coordinate, and "coordinate2" the y0-coordinate.
                        # However, we can change this as we see fit.
                        offset=[
                            # Offset for x-direction, i.e., the slowest axis
                            ro.Point3dOffset(
                                offset=ro.Point3d(
                                    coordinate1=1.0,
                                    coordinate2=0.0,
                                    coordinate3=0.0,
                                ),
                                spacing=ro.DoubleConstantArray(
                                    value=dx,
                                    count=nx - 1,
                                ),
                            ),
                            # Offset for y-direction, i.e., the fastest axis
                            ro.Point3dOffset(
                                offset=ro.Point3d(
                                    coordinate1=0.0,
                                    coordinate2=1.0,
                                    coordinate3=0.0,
                                ),
                                spacing=ro.DoubleConstantArray(
                                    value=dy,
                                    count=ny - 1,
                                ),
                            ),
                        ],
                    ),
                    zvalues=ro.DoubleHdf5Array(
                        values=ro.Hdf5Dataset(
                            path_in_hdf_file=f"/RESQML/{grid_uuid}/zvalues",
                            hdf_proxy=get_data_object_reference(epc),
                        ),
                    ),
                ),
            ),
        ),
    )
    return epc, crs, gri
