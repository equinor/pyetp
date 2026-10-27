import asyncio
import logging
import time
import typing

import numpy as np
import numpy.typing as npt

import resqml_objects.v201 as ro
from pyetp import ETPError
from rddms_io import rddms_connect
from resqml_objects.surface_helpers import (
    angle_to_unit_vectors,
)

logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s", level=logging.WARNING
)

logger = logging.getLogger("rddms_io.client")
logger.setLevel(logging.DEBUG)

logger = logging.getLogger("pyetp.client")
logger.setLevel(logging.DEBUG)

logger = logging.getLogger("websockets.client")
logger.setLevel(logging.DEBUG)

etp_url = "ws://localhost:9100"
dataspace_path = "foo/bar"

crs = ro.obj_LocalDepth3dCrs(
    citation=ro.Citation(title="Demo CRS", originator="demo-user"),
    vertical_crs=ro.VerticalUnknownCrs(unknown="Mean sea level"),
    projected_crs=ro.ProjectedCrsEpsgCode(epsg_code=23031),
)
epc = ro.obj_EpcExternalPartReference(
    citation=ro.Citation(title="Demo epc", originator="demo-user"),
)

rng = np.random.default_rng(21092026)


class Surface(typing.NamedTuple):
    grid: ro.obj_Grid2dRepresentation
    array: np.ndarray[tuple[int, int], np.dtype[np.float64]]


def get_random_grid(
    shape: tuple[int, int],
    crs: ro.obj_LocalDepth3dCrs,
    epc: ro.obj_EpcExternalPartReference,
    rng: np.random.Generator = rng,
) -> Surface:
    x_0 = rng.uniform(low=532_000, high=537_000)
    y_0 = rng.uniform(low=6_730_000, high=6_735_000)

    origin = np.array([x_0, y_0])
    spacing = rng.uniform(low=5, high=25, size=2)

    angle = rng.uniform(low=0.0, high=2 * np.pi)
    unit_vectors = angle_to_unit_vectors(angle=angle)

    grid = ro.obj_Grid2dRepresentation.from_regular_surface(
        citation=ro.Citation(title="Demo grid", originator="demo-user"),
        crs=crs,
        epc_external_part_reference=epc,
        shape=shape,
        origin=origin,
        spacing=spacing,
        unit_vec_1=unit_vectors[:, 0],
        unit_vec_2=unit_vectors[:, 1],
    )

    z_values = rng.uniform(low=0, high=2500, size=shape)

    return Surface(grid=grid, array=z_values)


surface = get_random_grid(shape=(20_000, 10_000), crs=crs, epc=epc)
print(f"{np.size(surface.array) * surface.array.dtype.itemsize / 1e9} GB")

pir = surface.grid.grid2d_patch.geometry.points.zvalues.values.path_in_hdf_file
data_arrays = {pir: surface.array}
ml_objects = [surface.grid, crs, epc]


async def upload_surface(
    ml_objects: list[ro.AbstractCitedDataObject],
    data_arrays: dict[str, npt.NDArray[np.float64]],
) -> None:
    async with rddms_connect(uri=etp_url) as rddms_client:
        try:
            await rddms_client.create_dataspace(dataspace_uri=dataspace_path)
            print(f"Created dataspace: {dataspace_path}")
        except ETPError as e:
            assert e.code == 5
            print(f"Dataspace: {dataspace_path} already exists")

        return await rddms_client.upload_model(
            dataspace_uri=dataspace_path,
            ml_objects=ml_objects,
            data_arrays=data_arrays,
        )


t0 = time.time()
gri_uri, crs_uri, epc_uri = asyncio.run(
    upload_surface(ml_objects, data_arrays), debug=True
)
t1 = time.time()
print(f"Time to upload surface: {t1 - t0} sec")


async def download_surface(ml_uris: list[str]) -> None:
    async with rddms_connect(uri=etp_url) as rddms_client:
        return await rddms_client.download_models(ml_uris=ml_uris, download_arrays=True)


t0 = time.time()
gri_model, crs_model, epc_model = asyncio.run(
    download_surface([gri_uri, crs_uri, epc_uri]), debug=True
)
t1 = time.time()
print(f"Time to download surface: {t1 - t0} sec")
assert gri_model.obj == surface.grid
assert crs_model.obj == crs
assert epc_model.obj == epc

np.testing.assert_equal(gri_model.arrays[pir], surface.array)


async def clean_up() -> None:
    async with rddms_connect(uri=etp_url) as rddms_client:
        resources = await rddms_client.list_objects_under_dataspace(
            dataspace_uri=dataspace_path
        )

        # Delete all objects.
        await rddms_client.delete_model(ml_uris=[r.uri for r in resources])

        # Verify that the dataspace is now empty.
        resources = await rddms_client.list_objects_under_dataspace(
            dataspace_uri=dataspace_path
        )
        assert len(resources) == 0


asyncio.run(clean_up())


# client = RDDMSClientSync(uri=etp_url)
# client.create_dataspace(dataspace_uri=dataspace_path, ignore_if_exists=True)
#
# t0 = time.time()
# client.upload_model(
#     dataspace_uri=dataspace_path,
#     ml_objects=[*ml_objects, crs, epc],
#     data_arrays=data_arrays,
# )
# t1 = time.time()
# print(f"Time to upload surface using the synchronous client: {t1 - t0} sec")
#
# resources = client.list_objects_under_dataspace(dataspace_uri=dataspace_path)
# client.delete_model(ml_uris=[r.uri for r in resources])
