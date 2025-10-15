import logging
import typing

import xtgeo
from etptypes.energistics.etp.v12.datatypes.data_array_types.data_array_identifier import (
    DataArrayIdentifier,
)

import resqml_objects.v201 as ro
from pyetp.client import ETPClient
from pyetp.uri import DataObjectURI


def check_inside(x: float, y: float, patch: ro.Grid2dPatch):
    xori = patch.geometry.points.supporting_geometry.origin.coordinate1
    yori = patch.geometry.points.supporting_geometry.origin.coordinate2
    xmax = xori + (
        patch.geometry.points.supporting_geometry.offset[0].spacing.value
        * patch.geometry.points.supporting_geometry.offset[0].spacing.count
    )
    ymax = yori + (
        patch.geometry.points.supporting_geometry.offset[1].spacing.value
        * patch.geometry.points.supporting_geometry.offset[1].spacing.count
    )
    if x < xori:
        return False
    if y < yori:
        return False
    if x > xmax:
        return False
    if y > ymax:
        return False
    return True


def find_closest_index(x, y, patch: ro.Grid2dPatch):
    x_ind = (
        x - patch.geometry.points.supporting_geometry.origin.coordinate1
    ) / patch.geometry.points.supporting_geometry.offset[0].spacing.value
    y_ind = (
        y - patch.geometry.points.supporting_geometry.origin.coordinate2
    ) / patch.geometry.points.supporting_geometry.offset[1].spacing.value
    return round(x_ind), round(y_ind)


async def get_surface_value_x_y(
    etp_client: ETPClient,
    epc_uri: typing.Union[DataObjectURI, str],
    gri_uri: typing.Union[DataObjectURI, str],
    crs_uri: typing.Union[DataObjectURI, str],
    x: typing.Union[int, float],
    y: typing.Union[int, float],
    method: typing.Literal["bilinear", "nearest"],
):
    # parallelized using subarray
    (gri,) = await etp_client.get_resqml_objects(gri_uri)
    xori = gri.grid2d_patch.geometry.points.supporting_geometry.origin.coordinate1
    yori = gri.grid2d_patch.geometry.points.supporting_geometry.origin.coordinate2
    xinc = gri.grid2d_patch.geometry.points.supporting_geometry.offset[0].spacing.value
    yinc = gri.grid2d_patch.geometry.points.supporting_geometry.offset[1].spacing.value
    max_x_index_in_gri = gri.grid2d_patch.geometry.points.supporting_geometry.offset[
        0
    ].spacing.count
    max_y_index_in_gri = gri.grid2d_patch.geometry.points.supporting_geometry.offset[
        1
    ].spacing.count
    buffer = 4
    if not check_inside(x, y, gri.grid2d_patch):
        logging.info(f"Points not inside {x}:{y} {gri}")
        return
    uid = DataArrayIdentifier(
        uri=str(epc_uri),
        pathInResource=gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file,
    )
    if max_x_index_in_gri <= 10 or max_y_index_in_gri <= 10:
        surf = await etp_client.get_xtgeo_surface(epc_uri, gri_uri, crs_uri)
        return surf.get_value_from_xy((x, y), sampling=method)

    x_ind, y_ind = find_closest_index(x, y, gri.grid2d_patch)
    if method == "nearest":
        arr = await etp_client.get_subarray(uid, [x_ind, y_ind], [1, 1])
        return arr[0][0]
    min_x_ind = max(x_ind - (buffer / 2), 0)
    min_y_ind = max(y_ind - (buffer / 2), 0)
    count_x = min(max_x_index_in_gri - min_x_ind, buffer)
    count_y = min(max_y_index_in_gri - min_y_ind, buffer)
    # shift start index to left if not enough buffer on right
    if count_x < buffer:
        x_index_to_add = 3 - count_x
        min_x_ind_new = max(0, min_x_ind - x_index_to_add)
        count_x = count_x + min_x_ind - min_x_ind_new + 1
        min_x_ind = min_x_ind_new
    if count_y < buffer:
        y_index_to_add = 3 - count_y
        min_y_ind_new = max(0, min_y_ind - y_index_to_add)
        count_y = count_y + min_y_ind - min_y_ind_new + 1
        min_y_ind = min_y_ind_new
    arr = await etp_client.get_subarray(uid, [min_x_ind, min_y_ind], [count_x, count_y])
    new_x_ori = xori + (min_x_ind * xinc)
    new_y_ori = yori + (min_y_ind * yinc)
    regridded = xtgeo.RegularSurface(
        ncol=arr.shape[0],
        nrow=arr.shape[1],
        xori=new_x_ori,
        yori=new_y_ori,
        xinc=xinc,
        yinc=yinc,
        rotation=0.0,
        values=arr.flatten(),
    )
    return regridded.get_value_from_xy((x, y))
