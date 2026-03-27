import socket
import urllib.parse

import pytest

etp_server_url = "ws://localhost:9100"

dataspace = "test/test"


def check_if_server_is_accesible() -> bool:
    parsed_url = urllib.parse.urlparse(etp_server_url)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    return sock.connect_ex((parsed_url.hostname, parsed_url.port)) == 0


skip_decorator = pytest.mark.skipif(
    not check_if_server_is_accesible(),
    reason="websocket for test server not open",
)
