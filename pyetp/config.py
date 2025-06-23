from typing import Optional

from pydantic import AnyUrl, BaseSettings, Field


class WebSocketUrl(AnyUrl):
    allowed_schemes = {'wss', 'ws'}


class Settings(BaseSettings):

    class Config:
        env_prefix = ''  # defaults to no prefix, i.e. ""
        fields = {
            'redis_dns': {'env': ['redishost', 'redis_host', 'redis_dns']},
            'redis_password': {'env': ['redis_password', 'redispass']}
        }

    application_name: str = Field(default='pyetp')
    application_version: str = Field(default='0.0.33')

    #dataspace: str = Field(default='demo/pss-data-gateway')
    etp_url: WebSocketUrl = Field(default='wss://host.com')
    etp_timeout: float = Field(default=15., description="Timeout in seconds")
    data_partition: Optional[str] = None
    MaxWebSocketMessagePayloadSize: int = Field(default=500000000)


SETTINGS = Settings()
