#%%
import faiss
from typing import Dict, Literal, Union, List
import numpy as np
import logging 
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, UniqueConstraint, MetaData, Table
from sqlalchemy.dialects.postgresql import JSONB
import hashlib
import json 

from .custom_types.class_types import BaseClass
from .configs import ResumeProcessorConfig
from .ctxs import ResumeProcessorCtx
from .faiss_client import FAISSClient
from .nlp_processor import NLProcessor

class ResumeProcessor(BaseClass):
    def __init__(self, 
                 config: ResumeProcessorConfig,
                 ctx: ResumeProcessorCtx=ResumeProcessorCtx(),
                 verbose: int=logging.INFO):
        super().__init__(config, ctx, verbose)
        self.config: ResumeProcessorConfig
        self.ctx: ResumeProcessorCtx

        self.init_postgres_client()
        self.init_faiss_client()
        self.nlp_processor = NLProcessor()
    
    def init_postgres_client(self):
        # PostgreSQL URI
        database_url = (
            f"postgresql://{self.config.postgres_client_config.user}:"
            f"{self.config.postgres_client_config.password}@"
            f"{self.config.postgres_client_config.host}:"
            f"{self.config.postgres_client_config.port}/"
            f"{self.config.postgres_client_config.database}"
        )
        engine = create_engine(database_url, echo=False)
        Session = sessionmaker(bind=engine)
        self.ctx.postgres_client = Session()

        try:
            with engine.connect() as connection:
                result=connection.execute(text("SELECT 1"))
                print(f"Connection to {self.config.postgres_client_config.database} successful, result:", result.scalar())
        except Exception as e:
            print("Connection failed:", e)

    def init_faiss_client(self):
        self.ctx.faiss_client = FAISSClient(self.config.faiss_client_config)

    def run(self, resume: Dict):
        self.meta=self.get_resume_meta(resume)
        self.logger.info(f'Processing resume from {self.meta}')

        self.process_skills(resume)
        self.process_experience(resume)
        self.process_title(resume)

        self.add_resume_to_postgres(resume)
    
    @staticmethod
    def get_resume_meta(resume: Dict):
        meta={
            "first_name": resume['first_name'],
            "last_name": resume['last_name'],
            "email": resume['email'],
        }
        return meta
    
    def add_resume_to_postgres(self, resume: Dict):
        pass 

    def process_skills(self, resume: Dict):
        # fetch skills 
        skills=resume['skills']

        # normalize
        skills=[self.nlp_processor.normalize(skill) for skill in skills]

        # search semantic 
        self.logger.debug(f'Base skills {skills}')
        skills=self._expand_semantic(skills, self.config.skills_index_id, 4)
        self.logger.debug(f'Semantic skills {skills}')

        # normalize
        skills=' '.join(skills)
        
        
        # generate embeddings
        skills_emb=self.nlp_processor.get_embeddings(skills,
                                                     method=self.config.embedding_method)
        
        # save to index
        self.ctx.faiss_client.build_index(self.config.resume_skills_index_id,
                                          self.config.index_type,
                                          size=skills_emb.shape[1])
        self.ctx.faiss_client.add_vector(self.config.resume_skills_index_id, 
                                         skills_emb, 
                                         normalize=True,
                                         meta=self.meta)
                
    def process_experience(self, resume: Dict):
        experiences=[]
        for experience in resume['experiences']:
            experiences.append(experience['description'])
        experiences=" ".join(experiences)
        
        # normalize
        experiences=self.nlp_processor.remove_noise(experiences)
        experiences=self.nlp_processor.normalize(experiences)
        experiences=self.nlp_processor.remove_punctuation(experiences)

        # generate embeddings 
        experiences_emb=self.nlp_processor.get_embeddings(experiences,
                                                          method=self.config.embedding_method)

        # save to index 
        self.ctx.faiss_client.build_index(self.config.resume_experiences_index_id,
                                          self.config.index_type,
                                          size=experiences_emb.shape[1])
        self.ctx.faiss_client.add_vector(self.config.resume_experiences_index_id, 
                                         experiences_emb, 
                                         normalize=True,
                                         meta=self.meta)
        pass
    
    def process_title(self, resume: Dict):
        # fetch titles
        titles=[]
        for experience in resume['experiences']:

            titles.append(self.nlp_processor.normalize(experience['role']))

        # semantic search
        self.logger.debug(f'Base titles {titles}')
        titles=self._expand_semantic(titles, self.config.titles_index_id, 2)
        self.logger.debug(f'Semantic titles {titles}')

        # normalize
        titles=' '.join(titles)
        titles=self.nlp_processor.remove_punctuation(titles)

        # generate embeddings
        titles_emb=self.nlp_processor.get_embeddings(titles,
                                                     method=self.config.embedding_method)
        
        # save to index
        self.ctx.faiss_client.build_index(self.config.resume_titles_index_id,
                                          self.config.index_type,
                                          size=titles_emb.shape[1])
        self.ctx.faiss_client.add_vector(self.config.resume_titles_index_id, 
                                         titles_emb, 
                                         normalize=True,
                                         meta=self.meta)
        pass

    def _expand_semantic(self, elements: List[str], index_id: str, k: int=4):
        
        semantic_elements=[]
        for element in elements:
            vector=self.nlp_processor.get_embeddings(element, method=self.config.embedding_method)
            response=self.ctx.faiss_client.query(vector, index_id, k, True)
            for element in response:
                semantic_elements.append(element['meta']['key'])
        elements.extend(semantic_elements)
        
        # remove duplicates
        elements=list(set(elements))
        
        return elements
    
    def add_resume_to_postgres(self, resume: Dict):
        """
        Adds a resume to the PostgreSQL database using the email as the unique ID.
        If the record already exists, it updates it.
        """
        if not hasattr(self.ctx, 'postgres_client') or self.ctx.postgres_client is None:
            self.logger.error("Postgres client not initialized.")
            return

        table_name = self.config.resumes_table_id

        # Ensure table exists
        metadata = MetaData()
        resumes_table = Table(
            table_name,
            metadata,
            Column("email", String, primary_key=True),
            Column("first_name", String),
            Column("last_name", String),
            Column("telephone_number", String),
            Column("resume", JSONB),
        )
        engine = self.ctx.postgres_client.get_bind()
        metadata.create_all(bind=engine, checkfirst=True)

        # Serialize the resume
        resume_json = json.dumps(resume)

        # Upsert statement: insert or update all relevant columns
        insert_stmt = text(f"""
            INSERT INTO {table_name} (email, first_name, last_name, telephone_number, resume)
            VALUES (:email, :first_name, :last_name, :telephone_number, :resume)
            ON CONFLICT (email)
            DO UPDATE SET
                first_name = EXCLUDED.first_name,
                last_name = EXCLUDED.last_name,
                telephone_number = EXCLUDED.telephone_number,
                resume = EXCLUDED.resume
        """)

        params = {
            "email": resume.get('email'),
            "first_name": resume.get('first_name'),
            "last_name": resume.get('last_name'),
            "telephone_number": resume.get('phone', None),
            "resume": resume_json
        }

        # Execute in a session
        with self.ctx.postgres_client as session:
            session.execute(insert_stmt, params)
            session.commit()

        self.logger.info(f"Resume for {resume['email']} added/updated in Postgres.")
# %%
