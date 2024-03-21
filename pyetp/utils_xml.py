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

import pyetp.resqml_objects as ro
from pyetp.config import SETTINGS

from .types import DataObject

schema_version = "2.0"


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


def create_common_citation(title: str):

    return ro.Citation(
        title=title,
        creation=XmlDateTime.from_string(
            datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
        ),
        originator=SETTINGS.application_name,
        format=f"{SETTINGS.application_name}:v{SETTINGS.application_version}",
    )


def create_common_crs(title: str, projected_epsg, rotation: float = 0.0):
    return ro.LocalDepth3dCrs(
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
            value=rotation,
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


def create_epc(schema_version="2.0"):
    return ro.EpcExternalPartReference(
        citation=create_common_citation("Hdf Proxy"),
        schema_version=schema_version,
        uuid=str(uuid4()),
        mime_type="application/x-hdf5",
    )


def parse_xtgeo_surface_to_resqml_grid(surf: 'xtgeo.RegularSurface', projected_epsg: int):
    # Build the RESQML-objects "manually" from the generated dataclasses.
    # Their content is described also in the RESQML v2.0.1 standard that is
    # available for download here:
    # https://publications.opengroup.org/standards/energistics-standards/v231a

    title = surf.name or "regularsurface"

    assert np.abs(surf.get_rotation()) < 1e-7, "Maps should have no rotation!"

    epc = create_epc()
    crs = create_common_crs(title, projected_epsg, surf.get_rotation())   # Here rotation should be zero!

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


def convert_epc_mesh_to_resqml_mesh(epc_filename, title_in, projected_epsg):
    import numpy as np
    import resqpy.model as rq
    import resqpy.unstructured as rug

    title = title_in or "hexamesh"

    model = rq.Model(epc_filename)
    assert model is not None

    #
    # read mesh:  vertex positions and cell definitions
    #
    hexa_uuid = model.uuid(obj_type='UnstructuredGridRepresentation', title=title_in)
    assert hexa_uuid is not None
    hexa = rug.HexaGrid(model, uuid=hexa_uuid)
    assert hexa is not None
    assert hexa.cell_shape == 'hexahedral'
    hexa.check_hexahedral()

    crs = create_common_crs(title, projected_epsg)

    epc = ro.EpcExternalPartReference(
        citation=create_common_citation("Hdf Proxy"),
        schema_version=schema_version,
        uuid=str(uuid4()),
        mime_type="application/x-hdf5",
    )

    cellshape = ro.CellShape.HEXAHEDRAL if (hexa.cell_shape == "hexahedral") else ro.CellShape.TETRAHEDRAL

    geom = ro.UnstructuredGridGeometry(
        local_crs=ro.DataObjectReference(
            content_type=f"application/x-resqml+xml;version={schema_version};type={get_data_object_type(crs)}",
            title=crs.citation.title,
            uuid=crs.uuid,
        ),
        node_count=hexa.node_count or -1,
        face_count=hexa.face_count or -1,
        cell_shape=cellshape,
        points=ro.Point3dHdf5Array(
            coordinates=ro.Hdf5Dataset(
                path_in_hdf_file=f"/RESQML/{str(hexa_uuid)}/points",
                hdf_proxy=ro.DataObjectReference(
                    content_type=f"application/x-eml+xml;version={schema_version};type={get_data_object_type(epc)}",
                    title=epc.citation.title,
                    uuid=str(epc.uuid),
                ),
            )
        ),
        nodes_per_face=ro.ResqmlJaggedArray(
            elements=ro.IntegerHdf5Array(
                null_value=-1,
                values=ro.Hdf5Dataset(
                    path_in_hdf_file=f"/RESQML/{str(hexa_uuid)}/nodes_per_face",
                    hdf_proxy=ro.DataObjectReference(
                        content_type=f"application/x-eml+xml;version={schema_version};type={get_data_object_type(epc)}",
                        title=epc.citation.title,
                        uuid=str(epc.uuid),
                    ),
                )
            ),
            cumulative_length=ro.IntegerHdf5Array(
                null_value=-1,
                values=ro.Hdf5Dataset(
                    path_in_hdf_file=f"/RESQML/{str(hexa_uuid)}/nodes_per_face_cl",
                    hdf_proxy=ro.DataObjectReference(
                        content_type=f"application/x-eml+xml;version={schema_version};type={get_data_object_type(epc)}",
                        title=epc.citation.title,
                        uuid=str(epc.uuid),
                    ),
                )
            ),
        ),
        faces_per_cell=ro.ResqmlJaggedArray(
            elements=ro.IntegerHdf5Array(
                null_value=-1,
                values=ro.Hdf5Dataset(
                    path_in_hdf_file=f"/RESQML/{str(hexa_uuid)}/faces_per_cell",
                    hdf_proxy=ro.DataObjectReference(
                        content_type=f"application/x-eml+xml;version={schema_version};type={get_data_object_type(epc)}",
                        title=epc.citation.title,
                        uuid=str(epc.uuid),
                    ),
                )
            ),
            cumulative_length=ro.IntegerHdf5Array(
                null_value=-1,
                values=ro.Hdf5Dataset(
                    path_in_hdf_file=f"/RESQML/{str(hexa_uuid)}/faces_per_cell_cl",
                    hdf_proxy=ro.DataObjectReference(
                        content_type=f"application/x-eml+xml;version={schema_version};type={get_data_object_type(epc)}",
                        title=epc.citation.title,
                        uuid=str(epc.uuid),
                    ),
                )
            ),
        ),
        cell_face_is_right_handed=ro.BooleanHdf5Array(
            values=ro.Hdf5Dataset(
                path_in_hdf_file=f"/RESQML/{str(hexa_uuid)}/cell_face_is_right_handed",
                hdf_proxy=ro.DataObjectReference(
                    content_type=f"application/x-eml+xml;version={schema_version};type={get_data_object_type(epc)}",
                    title=epc.citation.title,
                    uuid=str(epc.uuid),
                ),
            )
        )
    )

    #
    uns = ro.UnstructuredGridRepresentation(
        uuid=str(hexa.uuid),
        schema_version=schema_version,
        # surface_role=resqml_objects.SurfaceRole.MAP,
        citation=create_common_citation(hexa.title),
        cell_count=hexa.cell_count or -1,
        geometry=geom,
    )

    return uns, crs, epc, hexa


def convert_epc_mesh_property_to_resqml_mesh(epc_filename, hexa, prop_title, uns, epc):
    import resqpy.model as rq
    import resqpy.property as rqp

    def uom_for_prop_title(pt):
        if (pt == "Age"):
            return ro.ResqmlUom.A_1
        if (pt == "Temperature"):
            return ro.ResqmlUom.DEG_C
        if (pt == "LayerID"):
            return ro.ResqmlUom.EUC
        if (pt == "Porosity_initial"):
            return ro.ResqmlUom.M3_M3
        if (pt == "Porosity_decay"):
            return ro.ResqmlUom.VALUE_1_M
        if (pt == "Density_solid"):
            return ro.ResqmlUom.KG_M3
        if (pt == "insulance_thermal"):
            return ro.ThermalInsulanceUom.DELTA_K_M2_W
        if (pt == "Radiogenic_heat_production"):
            return ro.ResqmlUom.U_W_M3
        return ro.ResqmlUom.EUC

    model = rq.Model(epc_filename)
    assert model is not None
    prop_uuid = model.uuid(title=prop_title)
    prop = rqp.Property(model, uuid=prop_uuid)

    continuous = prop.is_continuous()

    if (prop.local_property_kind_uuid() is None):
        propertykind0 = None
    else:
        pk = rqp.PropertyKind(model, uuid=prop.local_property_kind_uuid())
        propertykind0 = ro.PropertyKind(
            schema_version=schema_version,
            citation=create_common_citation(f"{prop_title}"),
            naming_system="urn:resqml:bp.com:resqpy",
            is_abstract=False,
            representative_uom=uom_for_prop_title(prop_title),
            parent_property_kind=ro.StandardPropertyKind(
                kind=ro.ResqmlPropertyKind.CONTINUOUS if continuous else ro.ResqmlPropertyKind.DISCRETE
            ),
            uuid=str(pk.uuid),
        )

    pov = ro.PatchOfValues(
        values=ro.DoubleHdf5Array(
            values=ro.Hdf5Dataset(
                path_in_hdf_file=f"/RESQML/{str(prop_uuid)}/values",
                hdf_proxy=ro.DataObjectReference(
                    content_type=f"application/x-eml+xml;version={schema_version};type={get_data_object_type(epc)}",
                    title=epc.citation.title,
                    uuid=str(epc.uuid),
                ),
            )
        ) if continuous else
        ro.IntegerHdf5Array(
            values=ro.Hdf5Dataset(
                path_in_hdf_file=f"/RESQML/{str(prop_uuid)}/values",
                hdf_proxy=ro.DataObjectReference(
                    content_type=f"application/x-eml+xml;version={schema_version};type={get_data_object_type(epc)}",
                    title=epc.citation.title,
                    uuid=str(epc.uuid),
                ),
            ),
            null_value=int(1e30),
        )
    )

    if (continuous):
        cprop0 = ro.ContinuousProperty(
            schema_version=schema_version,
            citation=create_common_citation(f"{prop_title}"),
            uuid=str(prop.uuid),
            uom=prop.uom(),
            count=1,
            indexable_element=prop.indexable_element(),
            supporting_representation=ro.DataObjectReference(
                content_type=f"application/x-resqml+xml;version={schema_version};type={get_data_object_type(uns)}",
                title=uns.citation.title,
                uuid=uns.uuid,
            ),
            property_kind=ro.LocalPropertyKind(
                local_property_kind=ro.DataObjectReference(
                    content_type=f"application/x-resqml+xml;version={schema_version};type={get_data_object_type(propertykind0)}",
                    title=propertykind0.citation.title,
                    uuid=propertykind0.uuid,
                )
            ) if (propertykind0 is not None) else ro.StandardPropertyKind(kind=prop.property_kind()),
            minimum_value=[prop.minimum_value() or 0.0],
            maximum_value=[prop.maximum_value() or 1.0],
            facet=[ro.PropertyKindFacet(
                facet=ro.Facet.WHAT,
                value=prop_title,  # prop.facet(),
            )],
            patch_of_values=[pov],
        )
    else:
        cprop0 = ro.DiscreteProperty(
            schema_version=schema_version,
            citation=create_common_citation(f"{prop_title}"),
            uuid=str(prop.uuid),
            # uom = prop.uom(),
            count=1,
            indexable_element=prop.indexable_element(),
            supporting_representation=ro.DataObjectReference(
                content_type=f"application/x-resqml+xml;version={schema_version};type={get_data_object_type(uns)}",
                title=uns.citation.title,
                uuid=uns.uuid,
            ),
            property_kind=ro.LocalPropertyKind(
                local_property_kind=ro.DataObjectReference(
                    content_type=f"application/x-resqml+xml;version={schema_version};type={get_data_object_type(propertykind0)}",
                    title=propertykind0.citation.title,
                    uuid=propertykind0.uuid,
                )
            ) if (propertykind0 is not None) else ro.StandardPropertyKind(kind=prop.property_kind()),
            minimum_value=[int(prop.minimum_value() or 0)],
            maximum_value=[int(prop.maximum_value() or 1)],
            facet=[ro.PropertyKindFacet(
                facet=ro.Facet.WHAT,
                value=prop_title,  # prop.facet(),
            )],
            patch_of_values=[pov],
        )

    return cprop0, prop, propertykind0
