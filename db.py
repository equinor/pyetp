
import redis.asyncio as redis

from map_api.config import SETTINGS

pool = redis.ConnectionPool.from_url(SETTINGS.redis_dns)


def get_cache():
    return redis.Redis.from_pool(pool)
