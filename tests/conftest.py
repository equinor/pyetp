import datetime
import socket
import urllib.parse
import uuid

import numpy as np
import numpy.typing as npt
import pytest
import pytest_asyncio
from xsdata.models.datatype import XmlDateTime

import resqml_objects.v201 as resqml_objects
from pyetp.client import ETPClient, etp_connect
from energistics.uris import DataspaceURI

etp_server_url = "ws://localhost:9100"

dataspace = "test/test"


async def get_app_token(rc=None):
    return None


def check_if_server_is_accesible() -> bool:
    parsed_url = urllib.parse.urlparse(etp_server_url)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    return sock.connect_ex((parsed_url.hostname, parsed_url.port)) == 0


skip_decorator = pytest.mark.skipif(
    not check_if_server_is_accesible(),
    reason="websocket for test server not open",
)
