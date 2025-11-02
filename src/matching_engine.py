from typing import Dict, Literal, Union, List
import numpy as np
import logging 
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, UniqueConstraint, MetaData, Table
from sqlalchemy.dialects.postgresql import JSONB
import json 
from collections import defaultdict

from .custom_types.class_types import BaseClass
from .configs import MatchingEngineConfig
from .ctxs import MatchingEngineCtx
from .faiss_client import FAISSClient
from .nlp_processor import NLProcessor

class MatchingEngine(BaseClass):
    def __init__(self, 
                 config: MatchingEngineConfig,
                 ctx: MatchingEngineCtx=MatchingEngineCtx(),
                 verbose: int=logging.INFO):
        super().__init__(config, ctx, verbose)
        self.config: MatchingEngineConfig
        self.ctx: MatchingEngineCtx

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

    def run(self, job_description: Dict,
            skill_w: float=0.5,
            experience_w: float=0.2,
            title_w: float=0.3,
            top_k: int=10):

        if sum([skill_w, experience_w, title_w])!=1.0:
            raise ValueError('The sum of the weights should be 1')
        
        skill_matches=self.match_skills(job_description, 50)
        title_matches=self.match_titles(job_description, 50)
        experience_matches=self.match_experiences(job_description, 50)
        
        # compute scores
        scores = defaultdict(lambda: {'skills_score': 0, 'experience_score': 0, 'title_score': 0})

        for match in skill_matches:
            email = match['meta']['email']
            scores[email]['skills_score'] = float(match['distance'])

        for match in experience_matches:
            email = match['meta']['email']
            scores[email]['experience_score'] = float(match['distance'])

        for match in title_matches:
            email = match['meta']['email']
            scores[email]['title_score'] = float(match['distance'])

        # Compute weighted total score
        for email, score_dict in scores.items():
            total_score = (
                score_dict['skills_score'] * skill_w +
                score_dict['experience_score'] * experience_w +
                score_dict['title_score'] * title_w
            )
            scores[email]['total_score'] = total_score

        # Sort candidates by total score descending
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
        
        # return top k
        top_candidates=sorted_candidates[:top_k]

        enriched_results = []
        for email, metrics in top_candidates:
            candidate_data = self.fetch_resume_by_email(email)
            if candidate_data:
                enriched_results.append({
                    "email": email,
                    "first_name": candidate_data.get("first_name"),
                    "last_name": candidate_data.get("last_name"),
                    "resume": candidate_data.get("resume"),
                    "skills_score": metrics["skills_score"],
                    "experience_score": metrics["experience_score"],
                    "title_score": metrics["title_score"],
                    "total_score": metrics["total_score"]
                })
            else:
                # Fallback in case no record is found
                enriched_results.append({
                    "email": email,
                    "first_name": None,
                    "last_name": None,
                    "resume": None,
                    "skills_score": metrics["skills_score"],
                    "experience_score": metrics["experience_score"],
                    "title_score": metrics["title_score"],
                    "total_score": metrics["total_score"]
                })

        return enriched_results

    def match_skills(self, job_description: Dict, k: int=10):
        # fetch skills 
        skills=job_description['required_skills']

        # normalize
        skills=[self.nlp_processor.normalize(skill) for skill in skills]

        # search semantic 
        self.logger.debug(f'Base skills {skills}')
        skills=self._expand_semantic(skills, self.config.skills_index_id, 4)
        self.logger.debug(f'Semantic skills {skills}')

        # normalize
        skills=' '.join(skills)

        skill_matches=self._match(skills, self.config.resume_skills_index_id, k)

        return skill_matches
    
    def match_titles(self, job_description: Dict, k: int=10): 
        # fetch titles
        titles=self.nlp_processor.normalize(job_description['job_title'])

        # semantic search
        self.logger.debug(f'Base titles {titles}')
        titles=self._expand_semantic([titles], self.config.titles_index_id, 2)
        self.logger.debug(f'Semantic titles {titles}')

        # normalize
        titles=' '.join(titles)
        titles=self.nlp_processor.remove_punctuation(titles)

        title_matches=self._match(titles, self.config.resume_titles_index_id, k)

        return title_matches
    
    def match_experiences(self, job_description: Dict, k: int =10):
        
        # normalize
        experiences=self.nlp_processor.remove_noise(job_description['job_description'])
        experiences=self.nlp_processor.normalize(experiences)
        experiences=self.nlp_processor.remove_punctuation(experiences)

        experience_matches=self._match(experiences, self.config.resume_experiences_index_id, k)

        return experience_matches

    def _match(self, element: str, index_id: str, k: int):
        vector=self.nlp_processor.get_embeddings(element,
                                                      method=self.config.embedding_method)
        response=self.ctx.faiss_client.query(vector, index_id, k, True)

        return response 
        
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
    
    def fetch_resume_by_email(self, email: str):
        """
        Fetch first_name, last_name, and resume JSON from Postgres for a given email.
        Returns a dict or None if not found.
        """
        if not hasattr(self.ctx, 'postgres_client') or self.ctx.postgres_client is None:
            self.logger.error("Postgres client not initialized. Call init_postgres_client() first.")
            return None

        query = text("""
            SELECT first_name, last_name, resume
            FROM resumes
            WHERE email = :email
            LIMIT 1
        """)

        try:
            with self.ctx.postgres_client as session:
                result = session.execute(query, {"email": email}).fetchone()

            if result:
                first_name, last_name, resume_json = result
                self.logger.info(f"Fetched resume for {email}: {first_name} {last_name}")
                return {
                    "email": email,
                    "first_name": first_name,
                    "last_name": last_name,
                    "resume": resume_json
                }
            else:
                self.logger.warning(f"No resume found for {email}")
                return None

        except Exception as e:
            self.logger.error(f"Error fetching resume for {email}: {e}")
            return None