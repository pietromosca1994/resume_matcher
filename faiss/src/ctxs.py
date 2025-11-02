from dataclasses import dataclass 
from sqlalchemy.orm import Session
from typing import Union

from custom_types.ctx_types import BaseCtx

# @dataclass
# class ParserCtx(BaseCtx):
#     postgres_client: 

# @dataclass
class FAISSDBCtx(BaseCtx):
    postgres_client: Union[Session, None]=None

    def __post_init__(self):
        pass