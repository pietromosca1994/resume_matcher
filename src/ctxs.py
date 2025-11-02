from dataclasses import dataclass 
from sqlalchemy.orm import Session
from typing import Union

from .custom_types.ctx_types import BaseCtx
from .faiss_client import FAISSClient
# @dataclass
# class ParserCtx(BaseCtx):
#     postgres_client: 

# @dataclass
class FAISSClientCtx(BaseCtx):
    postgres_client: Union[Session, None]=None

    def __post_init__(self):
        pass

class FAISSDBCtx(BaseCtx):
    postgres_client: Union[Session, None]=None

    def __post_init__(self):
        pass

class ResumeProcessorCtx(BaseCtx):
    postgres_client: Union[Session, None]=None
    faiss_client: Union[FAISSClient, None]=None 

    def __post_init__(self):
        pass

class MatchingEngineCtx(BaseCtx):
    postgres_client: Union[Session, None]=None
    faiss_client: Union[FAISSClient, None]=None 

    def __post_init__(self):
        pass