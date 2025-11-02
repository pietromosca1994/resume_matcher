from dataclasses import dataclass
from abc import ABC, abstractmethod
# from pydantic import BaseModel

from .serializablemodel import SerializableModel

@dataclass
class BaseCtx(ABC, SerializableModel):
    
    @abstractmethod
    def __post_init__(self):
        ''' Method to check that the provided parameters are valid '''
        pass 