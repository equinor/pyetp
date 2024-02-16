import datetime
import typing as T
from uuid import uuid4

import lxml.etree as ET
import numpy as np
import xtgeo
from xsdata.formats.dataclass.context import XmlContext
from xsdata.formats.dataclass.parsers import XmlParser
from xsdata.formats.dataclass.serializers import XmlSerializer
from xsdata.formats.dataclass.serializers.config import SerializerConfig
from xsdata.models.datatype import XmlDateTime

import map_api.resqml_objects as ro

from .types import (ArrayOfBoolean, ArrayOfDouble, ArrayOfFloat, ArrayOfInt,
                    ArrayOfLong, DataArray, DataObject)

APPLICATION_NAME = "geomint"
APPLICATION_VERSION = "0.0.1"


SUPPORTED_ARRAY_TYPES = {
    ArrayOfFloat: np.float32,
    ArrayOfBoolean: np.bool_,
    ArrayOfInt: np.int32,
    ArrayOfLong: np.int64,
    ArrayOfDouble: np.float64,
}


def get_data_object_type(obj: ro.AbstractObject):
    return obj.__class__.__name__


def parse_resqml_objects(data_objects: T.List[DataObject]):
    # This function creates a list of resqml-objects from the returned xml from
    # the ETP-server. It dynamically finds the relevant resqml dataclass using
    # the object name found in the xml. Its intention is to be used after
    # calling the get_data_objects-protocol.

    # Set up an XML-parser from xsdata.
    parser = XmlParser(context=XmlContext())

    return [
        parser.from_bytes(
            data_object.data,
            getattr(ro, ET.QName(ET.fromstring(data_object.data).tag).localname),
        )
        for data_object in data_objects
    ]


def resqml_to_xml(obj: ro.AbstractObject):
    serializer = XmlSerializer(config=SerializerConfig())
    return str.encode(serializer.render(obj))


def etp_data_array_to_numpy(data_array: DataArray):

    itemtype = type(data_array.data.item)
    if itemtype not in SUPPORTED_ARRAY_TYPES:
        raise TypeError(f"Not {type(data_array.data.item)} supported")

    dims: T.Tuple[int, ...] = tuple(map(int, data_array.dimensions))
    dtype = SUPPORTED_ARRAY_TYPES.get(itemtype)

    return np.asarray(data_array.data.item.values, dtype=dtype).reshape(dims)


def numpy_to_etp_data_array(data: np.ndarray):
    from etptypes.energistics.etp.v12.datatypes.any_array import AnyArray

    dtype = data.dtype

    arraytype = [item[0] for item in SUPPORTED_ARRAY_TYPES.items() if item[1] == data.dtype]
    if not len(arraytype):
        raise TypeError(f"Not {type(dtype)} supported")

    cls = arraytype[0]
    return DataArray(
        dimensions=data.shape,  # type: ignore
        data=AnyArray(item=cls(values=data.flatten().tolist()))
    )


def create_common_citation(title: str):

    return ro.Citation(
        title=title,
        creation=XmlDateTime.from_string(
            datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
        ),
        originator=APPLICATION_NAME,
        format=f"equinor:{APPLICATION_NAME}:v{APPLICATION_VERSION}",
    )


def parse_xtgeo_surface_to_resqml_grid(surf: 'xtgeo.RegularSurface', projected_epsg: int):
    # Build the RESQML-objects "manually" from the generated dataclasses.
    # Their content is described also in the RESQML v2.0.1 standard that is
    # available for download here:
    # https://publications.opengroup.org/standards/energistics-standards/v231a

    schema_version = "2.0"
    title = surf.name or "regularsurface"

    epc = ro.EpcExternalPartReference(
        citation=create_common_citation("Hdf Proxy"),
        schema_version=schema_version,
        uuid=str(uuid4()),
        mime_type="application/x-hdf5",
    )

    assert np.abs(surf.get_rotation()) < 1e-7, "Maps should have no rotation!"

    crs = ro.LocalDepth3dCrs(
        citation=create_common_citation(f"CRS for {title}"),
        schema_version=schema_version,
        uuid=str(uuid4()),
        # NOTE: I assume that we let the CRS have no offset, and add any offset
        # in the grid instead.
        xoffset=0.0,
        yoffset=0.0,
        zoffset=0.0,
        areal_rotation=ro.PlaneAngleMeasure(
            # Here rotation should be zero!
            value=surf.get_rotation(),
            uom=ro.PlaneAngleUom.DEGA,
        ),
        # NOTE: Verify that this is the projected axis order
        projected_axis_order=ro.AxisOrder2d.EASTING_NORTHING,
        projected_uom=ro.LengthUom.M,
        vertical_uom=ro.LengthUom.M,
        zincreasing_downward=True,
        vertical_crs=ro.VerticalUnknownCrs(
            unknown="unknown",
        ),
        projected_crs=ro.ProjectedCrsEpsgCode(
            epsg_code=projected_epsg,
        ),
    )

    x0 = surf.xori
    y0 = surf.yori
    dx = surf.xinc
    dy = surf.yinc
    # NOTE: xtgeo uses nrow for axis 1 in the array, and ncol for axis 0.  This
    # means that surf.nrow is the fastest changing axis, and surf.ncol the
    # slowest changing axis, and we have surf.values.shape == (surf.ncol,
    # surf.nrow). The author of this note finds that confusing, but such is
    # life.
    nx = surf.ncol
    ny = surf.nrow

    gri = ro.Grid2dRepresentation(
        uuid=(grid_uuid := str(uuid4())),
        schema_version=schema_version,
        surface_role=ro.SurfaceRole.MAP,
        citation=create_common_citation(title),
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
                local_crs=ro.DataObjectReference(
                    # NOTE: See Energistics Identifier Specification 4.0
                    # (it is downloaded alongside the RESQML v2.0.1
                    # standard) section 4.1 for an explanation on the
                    # format of content_type.
                    content_type=f"application/x-resqml+xml;version={schema_version};type={get_data_object_type(crs)}",
                    title=crs.citation.title,
                    uuid=crs.uuid,
                ),
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
                            hdf_proxy=ro.DataObjectReference(
                                content_type=f"application/x-eml+xml;version={schema_version};type={get_data_object_type(epc)}",
                                title=epc.citation.title,
                                uuid=epc.uuid,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    return epc, crs, gri
