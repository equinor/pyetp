import asyncio

import numpy as np

from rddms_io import rddms_connect
import resqml_objects.v201 as ro

from rddms_io.data_types import LinkedObjects
from energistics.etp.v12.datatypes.object import Dataspace, Resource

z = np.random.random((101, 103))
origin = np.array([10.0, 11.0])
spacing = np.array([1.0, 0.9])
u1 = np.array([np.sqrt(3.0) / 2.0, 0.5])
u2 = np.array([-0.5, np.sqrt(3.0) / 2.0])

originator = "<name/username/email>"
epc = ro.obj_EpcExternalPartReference(
    citation=ro.Citation(title="Demo epc", originator=originator)
)
crs = ro.obj_LocalDepth3dCrs(
    citation=ro.Citation(
        title="Demo crs",
        originator=originator,
    ),
    vertical_crs=ro.VerticalCrsEpsgCode(epsg_code=6230),
    projected_crs=ro.ProjectedCrsEpsgCode(epsg_code=23031),
)
gri = ro.obj_Grid2dRepresentation.from_regular_surface(
    citation=ro.Citation(
        title="Demo grid",
        originator=originator,
    ),
    crs=crs,
    epc_external_part_reference=epc,
    shape=z.shape,
    origin=origin,
    spacing=spacing,
    unit_vec_1=u1,
    unit_vec_2=u2,
)


async def main() -> tuple[
    list[Dataspace],
    list[Resource],
    LinkedObjects,
]:
    uri = "ws://localhost:9100"
    data_partition_id = ""
    access_token = ""
    dataspace_path = "rddms_io/demo"

    async with rddms_connect(
        uri=uri,
        data_partition_id=data_partition_id,
        authorization=access_token,
    ) as rddms_client:
        await rddms_client.create_dataspace(
            dataspace_path,
            legal_tags=["legal-tag-1", "legal-tag-2"],
            other_relevant_data_countries=["country-code"],
            owners=["owners"],
            viewers=["viewers-1", "viewers-2"],
            ignore_if_exists=True,
        )

        assert isinstance(
            gri.grid2d_patch.geometry.points, ro.Point3dZValueArray
        )
        assert isinstance(
            gri.grid2d_patch.geometry.points.zvalues, ro.DoubleHdf5Array
        )

        await rddms_client.upload_model(
            dataspace_path,
            ml_objects=[epc, crs, gri],
            data_arrays={
                gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file: z,
            },
        )

        dataspaces = await rddms_client.list_dataspaces()

        gri_resources = await rddms_client.list_objects_under_dataspace(
            dataspace_path,
            data_object_types=[ro.obj_Grid2dRepresentation],
        )

        gri_lo = await rddms_client.list_linked_objects(
            start_uri=gri_resources[0].uri,
        )

        ret_models = await rddms_client.download_models(
            ml_uris=[gri_lo.start_uri],
            download_arrays=True,
            download_linked_objects=True,
        )

        assert len(ret_models) == 1

        ret_model = ret_models[0]
        ret_gri = ret_model.obj

        assert isinstance(ret_gri, ro.obj_Grid2dRepresentation)
        assert isinstance(
            ret_gri.grid2d_patch.geometry.points, ro.Point3dZValueArray
        )
        assert isinstance(
            ret_gri.grid2d_patch.geometry.points.zvalues, ro.DoubleHdf5Array
        )

        assert len(ret_model.linked_models) == 1

        ret_crs = ret_model.linked_models[0].obj
        ret_z = ret_model.arrays[
            ret_gri.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file
        ]

        assert ret_gri == gri
        assert ret_crs == crs
        np.testing.assert_equal(z, ret_z)

        all_resources = await rddms_client.list_objects_under_dataspace(
            dataspace_path,
        )
        await rddms_client.delete_model(ml_uris=[a.uri for a in all_resources])
        await rddms_client.delete_dataspace(dataspace_path)

    return dataspaces, gri_resources, gri_lo


dataspaces, gri_resources, gri_lo = asyncio.run(main())
