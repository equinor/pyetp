from fastapi import Header

from map_api.config import SETTINGS

from .client import connect

# TODO: Parse auth to get user and retain/cache etp connection for each user


async def get_eclient(
    authorization: str = Header(default=None),
):
    async with connect(SETTINGS.etp_url, SETTINGS.duri, authorization=authorization) as client:
        yield client
