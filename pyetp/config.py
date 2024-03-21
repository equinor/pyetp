from pydantic import AnyUrl, BaseSettings, Field, RedisDsn, SecretStr
from pyetp.uri import DataspaceURI

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
    etp_url: WebSocketUrl = Field(default='wss://host.com')
    etp_timeout: float = Field(default=60., description="Timeout in seconds")

    @property
    def duri(self):
        return DataspaceURI.from_name(self.dataspace)




SETTINGS = Settings()