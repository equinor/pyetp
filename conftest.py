
import fakeredis
import pytest
from fastapi.testclient import TestClient

from map_api.db import get_cache
from map_api.main import app

ETP_SERVER_URL = "ws://localhost:9002"
DATASPACE = "test/pss-data-gateway"


def get_fake_cache():
    return fakeredis.aioredis.FakeRedis()


app.dependency_overrides[get_cache] = get_fake_cache


@pytest.fixture
def client():
    return TestClient(app)
