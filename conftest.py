
import fakeredis
import pytest
from fastapi.testclient import TestClient

from map_api.db import get_cache
from map_api.main import app


def get_fake_cache():
    return fakeredis.aioredis.FakeRedis()


app.dependency_overrides[get_cache] = get_fake_cache


@pytest.fixture
def client():
    return TestClient(app)
