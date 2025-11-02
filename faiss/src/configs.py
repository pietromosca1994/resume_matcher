from pydantic.dataclasses import dataclass
from typing import Union

from custom_types.config_types import BaseConfig

@dataclass
class PostgresClientConfig(BaseConfig):
    host: str
    port: int
    database: str
    user: str = "user"
    password: str = "password"

    @classmethod
    def auto(cls):
        return cls(
            host="localhost",
            port=5432,
            database="mydatabase"
        )

    def __post_init__(self):
        pass 

@dataclass
class FAISSDBConfig(BaseConfig):
    postgres_config: Union[PostgresClientConfig, None]=None

    @classmethod
    def auto(cls):
        return cls()

    def __post_init__(self):
        pass 

# %%
