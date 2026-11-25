import asyncio

from rddms_io import RDDMSClient, rddms_connect

etp_server_url = "ws://localhost:9100"


async def delete_dataspace_and_content(
    rddms_client: RDDMSClient, dataspace_uri: str
) -> None:
    await rddms_client.delete_dataspace_content(dataspace_uri)
    await rddms_client.delete_dataspace(dataspace_uri)


async def main() -> None:
    async with rddms_connect(uri=etp_server_url) as rddms_client:
        dataspaces = await rddms_client.list_dataspaces()

        for dataspace in dataspaces:
            await delete_dataspace_and_content(rddms_client, dataspace.uri)


asyncio.run(main())
