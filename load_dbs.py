#%%
# import modules
import json
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Table, Column, String, Float, Integer, MetaData, create_engine
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Dict, Literal
from glob import glob

from nlp_processor import NLProcessor
from src.faiss_client import FAISSClient   
from src.configs import FAISSClientConfig, ResumeProcessorConfig 
from src.resume_processor import ResumeProcessor

# %%
# Initialize Modules
# Inilitialize NLP Processor
nlp_processor = NLProcessor()

# PostgreSQL URI
DATABASE_USER = 'user'
DATABASE_PASSWORD = 'password'
DATABASE_URL = f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@localhost:5432/postgres"

engine = create_engine(DATABASE_URL, echo=False)
Session = sessionmaker(bind=engine)
session = Session()

# initialize FAISS client
faiss_client_config = FAISSClientConfig()
faiss_client = FAISSClient(faiss_client_config)

# initialize Resume Processor
resume_processor_config=ResumeProcessorConfig()
resume_processor=ResumeProcessor(resume_processor_config)

#%% 
def load_elements(ontology: List[Dict], 
                  index_id: str, 
                  index_type: Literal['flatl2', 'flatip']='flatl2',
                  embedding_method: Literal['bert_l6']='bert_l6'):
    elements = []
    for element in ontology:
        elements.append(nlp_processor.normalize(element["base"]))
        elements.extend([nlp_processor.normalize(skill) for skill in element["related"]])
    elements = list(set(elements))
    print(f"Total unique elements loaded: {len(elements)}")

    faiss_client.build_index(index_id, index_type=index_type, size=384)
    for element in elements: 
        # add to vector db
        vector=nlp_processor.get_embeddings(element, method=embedding_method)
        faiss_client.add_vector(
            index_id=index_id,
            vector=vector,
            normalize=True,
            meta={"key": element}
        )
        print(f'✅ Added {element} to index {index_id}')

def load_ontology(ontology: List[Dict], table: str): 
    relations=[]
    for element in ontology:
        for related in element["related"]:
            relations.append((nlp_processor.normalize(element["base"]), 
                              nlp_processor.normalize(related)))
    print(f"Total relations loaded: {len(relations)}")
    
    for base, related in relations:
        add_relation_to_db(base, 
                           related, 
                           table,
                           similarity=1, 
                           source='ontology')
        # also mirror
        add_relation_to_db(related,
                           base,
                           table,
                           similarity=1, 
                           source='ontology')
    pass 

def add_relation_to_db(base: str, 
                       related: str, 
                       table: str,
                       similarity: float = 1.0, source: str = "ontology"):
    """
    Adds a base-related relation entry to the specified table.
    If the table doesn't exist, it will be created automatically.
    """
    try:
        # Access the engine from the session
        engine = session.get_bind()
        metadata = MetaData()

        # Reflect existing tables
        metadata.reflect(bind=engine)

        # If table doesn't exist, create it dynamically
        if table not in metadata.tables:
            relation_table = Table(
                table,
                metadata,
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("base", String, nullable=False),
                Column("related", String, nullable=False),
                Column("similarity", Float, default=1.0),
                Column("source", String, default="ontology")
            )
            metadata.create_all(engine)
            print(f"Created table '{table}' in the database.")
        else:
            relation_table = metadata.tables[table]

        # Insert the new relation
        insert_stmt = relation_table.insert().values(
            base=base,
            related=related,
            similarity=similarity,
            source=source
        )
        session.execute(insert_stmt)
        session.commit()

        print(f"✅ Added relation: {base} → {related} (Table: {table})")

    except SQLAlchemyError as e:
        session.rollback()
        print(f"❌ Failed to add relation {base} → {related}: {e}")


#%%
if __name__=='__main__':
    # Load skill ontology
    with open("./postgres/data/skill_ontology.json", "r") as f:
        skill_ontology = json.load(f)

    # Load titles ontology
    with open("./postgres/data/title_ontology.json", "r") as f:
        title_ontology = json.load(f)

    # load resumes
    resume_files=glob('./data/*.json')
    resumes=[]
    for resume_file in resume_files: 
        with open(resume_file, 'r') as f: 
            resumes.append(json.load(f))

    # load skills 
    load_elements(skill_ontology, 'skills_index', index_type='flatip')
    load_ontology(skill_ontology, 'skills_ontology')

    # load titles 
    load_elements(title_ontology, 'titles_index', index_type='flatip')
    load_ontology(title_ontology, 'titles_ontology')

    # load resumes
    for resume in resumes: 
        resume_processor.run(resume)
