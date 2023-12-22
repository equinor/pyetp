import json


ETP_SERVER_URL = "wss://interop-rddms.azure-api.net"
PSS_DATASPACE = "demo/pss-data-gateway"
MAX_WEBSOCKET_MESSAGE_SIZE = int(1.6e7)  # From the published ETP server

# TODO: Check pathing when the api is called
with open("package.json", "r") as f:
    jschema = json.load(f)
    APPLICATION_NAME = jschema["name"]
    APPLICATION_VERSION = jschema["version"]


async def upload_resqml_surface(resqml_objects, surface_values):
    return "RDDMSURL, but which one?"
