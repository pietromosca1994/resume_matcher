from pydantic.dataclasses import dataclass
from dataclasses import field
from typing import Union, Literal   

from .custom_types.config_types import BaseConfig

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
class FAISSClientConfig(BaseConfig):
    base_url: str = "http://localhost:8000"
    timeout: int = 10
    
    @classmethod
    def auto(cls):
        return FAISSClientConfig()

    def __post_init__(self):
        pass 
    
@dataclass
class PostgresClientConfig(BaseConfig):
    host: str="localhost"
    port: int="5432"
    database: str="postgres"
    user: str = "user"
    password: str = "password"

    @classmethod
    def auto(cls):
        return cls(
            host="localhost",
            port=5432,
            database="postgres"
        )

    def __post_init__(self):
        pass 

@dataclass
class ResumeProcessorConfig(BaseConfig):
    faiss_client_config: FAISSClientConfig = field(default_factory=FAISSClientConfig)
    postgres_client_config: Union[PostgresClientConfig, None]= field(default_factory=PostgresClientConfig)
    resumes_table_id: str = 'resumes'
    skills_index_id: str = 'skills_index'
    titles_index_id: str = 'titles_index'
    resume_skills_index_id: str = 'resume_skills_index'
    resume_experiences_index_id: str = 'resume_experiences_index'
    resume_titles_index_id: str = 'resume_titles_index'
    embedding_method: Literal["bert_l6", "bert_l12", "roberta", "distilbert"]="bert_l6"
    index_type: Literal['flatl2', 'flat1d', 'flatip']='flatip'

    @classmethod
    def auto(cls):
        return cls()

    def __post_init__(self):
        pass 

@dataclass
class MatchingEngineConfig(BaseConfig):
    faiss_client_config: FAISSClientConfig = field(default_factory=FAISSClientConfig)
    postgres_client_config: Union[PostgresClientConfig, None]= field(default_factory=PostgresClientConfig)
    resumes_table_id: str = 'resumes'
    skills_index_id: str = 'skills_index'
    titles_index_id: str = 'titles_index'
    resume_skills_index_id: str = 'resume_skills_index'
    resume_experiences_index_id: str = 'resume_experiences_index'
    resume_titles_index_id: str = 'resume_titles_index'
    embedding_method: Literal["bert_l6", "bert_l12", "roberta", "distilbert"]="bert_l6"
    index_type: Literal['flatl2', 'flat1d', 'flatip']='flatip'

    @classmethod
    def auto(cls):
        return cls()

    def __post_init__(self):
        pass 

# %%
@dataclass
class FAISSDBConfig(BaseConfig):
    postgres_config: Union[PostgresClientConfig, None]=None

    @classmethod
    def auto(cls):
        return cls()

    def __post_init__(self):
        pass 