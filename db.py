
import redis.asyncio as redis

from map_api.config import SETTINGS

if SETTINGS.redis_password is not None:
    SETTINGS.redis_dns.password = SETTINGS.redis_password

pool = redis.ConnectionPool.from_url(SETTINGS.redis_dns)


def get_cache():
    return redis.Redis.from_pool(pool)
