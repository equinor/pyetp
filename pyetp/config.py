from typing import Optional

from pydantic import AnyUrl, BaseSettings, Field, RedisDsn, SecretStr

from pyetp.uri import DataspaceURI
from py_etp_client.etpconfig import ETPConfig

class WebSocketUrl(AnyUrl):
    allowed_schemes = {'wss', 'ws'}


class Settings(BaseSettings):

    class Config:
        env_prefix = ''  # defaults to no prefix, i.e. ""
        fields = {
            'redis_dns': {'env': ['redishost', 'redis_host', 'redis_dns']},
            'redis_password': {'env': ['redis_password', 'redispass']}
        }

    application_name: str = Field(default='etpClient')
    application_version: str = Field(default='0.0.1')

    dataspace: str = Field(default='demo/pss-data-gateway')
    port: int = 443
    etp_url: WebSocketUrl = Field(default='wss://host.com')
    etp_timeout: float = Field(default=60., description="Timeout in seconds")
    data_partition: Optional[str] = None

    @property
    def duri(self):
        return DataspaceURI.from_name(self.dataspace)


SETTINGS = Settings()

