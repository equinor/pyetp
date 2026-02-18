import datetime

import numpy as np

import resqml_objects.v201 as ro


originator = "<name/username/email>"

creation = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)

epc_uuid = "d53c9c04-e83a-4ad7-87fc-567a0dd5e660"
crs_uuid = "dbe0e6ba-1ea6-4dd7-b541-9c4c14c16f62"
gri_uuid = "be3dc02d-ed9d-45b1-b291-6edced323411"

epc = ro.obj_EpcExternalPartReference(
    citation=ro.Citation(
        title="Demo epc",
        originator=originator,
        creation=creation,
    ),
    uuid=epc_uuid,
)

crs = ro.obj_LocalDepth3dCrs(
    citation=ro.Citation(
        title="Demo crs",
        originator=originator,
        creation=creation,
    ),
    uuid=crs_uuid,
    vertical_crs=ro.VerticalUnknownCrs(unknown="Mean sea level"),
    projected_crs=ro.ProjectedCrsEpsgCode(epsg_code=23031),
)

shape = (101, 103)
origin = np.array([10.0, 11.0])
spacing = np.array([1.0, 0.9])
u1 = np.array([np.sqrt(3.0) / 2.0, 0.5])
u2 = np.array([-0.5, np.sqrt(3.0) / 2.0])

gri = ro.obj_Grid2dRepresentation.from_regular_surface(
    citation=ro.Citation(
        title="Demo grid",
        originator=originator,
        creation=creation,
    ),
    uuid=gri_uuid,
    crs=crs,
    epc_external_part_reference=epc,
    shape=shape,
    origin=origin,
    spacing=spacing,
    unit_vec_1=u1,
    unit_vec_2=u2,
)

X, Y = gri.get_xy_grid(crs=crs)
