# from dataclasses import dataclass
from pydantic.dataclasses import dataclass
from abc import ABC, abstractmethod

from .serializablemodel import SerializableModel

@dataclass
class BaseConfig(ABC, SerializableModel):

    @classmethod
    @abstractmethod
    def auto(cls, *args, **kwargs):
        '''Method to provide an automatic configuration'''
        return cls() 
    
    @abstractmethod
    def __post_init__(self):
        ''' Method to check that the provided parameters are valid '''
        pass
