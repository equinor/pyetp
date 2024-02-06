from os import environ

import redis.asyncio as redis

pool = redis.ConnectionPool.from_url(environ.get('REDISHOST', 'redis://localhost:6379'))

def get_cache():
    return redis.Redis.from_pool(pool)
