#%%
# import modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.')))

from src.faiss_client import FAISSClient   
from src.configs import FAISSClientConfig, PostgresClientConfig
from nlp_processor import NLProcessor

# %%
# initialize FAISS client
faiss_client_config = FAISSClientConfig()
faiss_client = FAISSClient(faiss_client_config)

# Inilitialize NLP Processor
nlp_processor = NLProcessor()

#%%
vector=nlp_processor.get_embeddings('python', method="bert_l6")
faiss_client.query(vector, 'skills_index', 10, True)

# %%
