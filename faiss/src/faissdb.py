#%%
import faiss
from typing import Dict, Literal, Union
import numpy as np
import logging 
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, UniqueConstraint, MetaData, Table
from sqlalchemy.dialects.postgresql import JSONB
import json 

from custom_types.class_types import BaseClass
from configs import FAISSDBConfig
from ctxs import FAISSDBCtx

Base = declarative_base()
# class FaissMeta(Base): 
#     __tablename__ = "faiss_meta"
#     id = Column(Integer, primary_key=True)
#     index = Column(String, primary_key=True)
#     meta = Column(JSONB, index=True) 

class FAISSDB(BaseClass):
    def __init__(self,
                 config: FAISSDBConfig=FAISSDBConfig.auto(),
                 ctx: FAISSDBCtx=FAISSDBCtx(),
                 verbose: int=logging.INFO): 
        super().__init__(config, ctx, verbose)
        self.indexes: Dict[str, faiss.Index] = {}
        self.config: FAISSDBConfig
        self.ctx: FAISSDBCtx

        if config.postgres_config is not None:
            self.init_postgres_client()
            self.meta=True

    def init_postgres_client(self):
        # PostgreSQL URI
        database_url = (
            f"postgresql://{self.config.postgres_config.user}:"
            f"{self.config.postgres_config.password}@"
            f"{self.config.postgres_config.host}:"
            f"{self.config.postgres_config.port}/"
            f"{self.config.postgres_config.database}"
        )
        engine = create_engine(database_url, echo=False)
        Session = sessionmaker(bind=engine)
        self.ctx.postgres_client = Session()

        try:
            with engine.connect() as connection:
                result=connection.execute(text("SELECT 1"))
                print(f"Connection to {self.config.postgres_config.database} successful, result:", result.scalar())
        except Exception as e:
            print("Connection failed:", e)

        Base.metadata.create_all(engine)

    def _create_index_table(self, index_id: str):
        """
        Dynamically create a table to store metadata for a specific FAISS index.
        """
        if not self.meta:
            return

        metadata = MetaData()
        table_name = f"faiss_meta_{index_id}"

        table = Table(
            table_name,
            metadata,
            Column("idx", Integer, primary_key=True),
            Column("meta", JSONB, index=True),     
        )

        # Use checkfirst=True to avoid errors if table already exists
        engine = self.ctx.postgres_client.get_bind()  # gets the underlying engine
        metadata.create_all(bind=engine, checkfirst=True)
    
    def build_index(self, index_id: str, 
                    index_type: Literal['flatl2', 'flat1d', 'flatip'],
                    size: int):
        if index_id not in self.indexes.keys():
            if index_type == 'flatl2':
                index=faiss.IndexFlatL2(size)
            elif index_type == 'flat1d':
                index=faiss.IndexFlat1D(size)
            elif index_type == 'flatip':
                index=faiss.IndexFlatIP(size)
            
            self.indexes[index_id] = index
            self._create_index_table(index_id)
            
            self.logger.info(f"Built FAISS index {index_id} of type {index_type} with size {size}.")
        else:
            self.logger.warning(f'Index {index_id} already existing. Skipping the creation of index {index_id}')
        pass 

    def add_vector(self, index_id: str, vector: np.ndarray, normalize: bool=False, meta: Dict=None):
        # add vector
        if index_id not in self.indexes.keys():
            self.logger.error(f"Index {index_id} not found.")
            return  
        
        if type(vector) is not np.ndarray:
            vector = np.array(vector)
            self.logger.warning(f"Converted input vectors to numpy array.")

        if vector.dtype != np.float32:
            vector = vector.astype("float32")
        
        if normalize:
            faiss.normalize_L2(vector)
        
        index = self.indexes[index_id]
        index.add(vector)
        self.logger.info(f"Added {len(vector)} vectors to index {index_id}.")

        # add meta
        idx=self.get_index_size(index_id)-1
        self._add_meta(index_id, idx, meta)

    # def _add_meta(self, meta: Union[Dict, None]):
    #     if self.meta:
    #         self.ctx.postgres_client.add(FaissMeta(meta=meta)) if meta is not None else {'id': self.get_index_size()}
    #         self.ctx.postgres_client.commit()


    def _add_meta(self, index_id: str, idx: int, meta: Union[Dict, None]):
        """
        Add metadata for a vector, using SHA256 of the vector as its ID.
        """
        if not self.meta:
            return

        table_name = f"faiss_meta_{index_id}"

        # Serialize meta dict to JSON string
        meta_json = json.dumps(meta)

        insert_stmt = text(f"""
            INSERT INTO {table_name} (idx, meta)
            VALUES (:idx, :meta)
            ON CONFLICT (idx) DO UPDATE SET meta = EXCLUDED.meta
        """)
        params = {"idx": idx, "meta": meta_json}

        with self.ctx.postgres_client as session:
            session.execute(insert_stmt, params)
            session.commit()

    def _get_meta(self, index_id: str, idx: int) -> Union[Dict, None]:
        """
        Retrieve metadata for a vector using SHA256 hash as ID.
        """
        idx=int(idx)
        if self.meta and idx>=0:
            table_name = f"faiss_meta_{index_id}"

            query = text(f"SELECT meta FROM {table_name} WHERE idx = :idx")

            # Execute the query using the vector_hash parameter
            result = self.ctx.postgres_client.execute(query, {"idx": idx}).fetchone()

            if result is None:
                return None

            # Deserialize JSON string back into dict if necessary
            import json
            meta = result[0]
            if isinstance(meta, str):
                return json.loads(meta)
            return meta
        
        else:
            return None
    
    def query(self, vector: np.ndarray, index_id: str, k: int, normalize: bool=False):
        if index_id not in self.indexes.keys():
            self.logger.error(f"Index {index_id} not found.")
            return  
        
        if vector.dtype != np.float32:
            vector = vector.astype("float32")

        vector=vector.reshape(1, -1)
        
        if normalize:
            faiss.normalize_L2(vector)

        index = self.indexes[index_id]
        D, I = index.search(vector, k)

        results=[]
        for dist, idx in zip(D[0], I[0]):
            results.append({
                "index": int(idx),
                "distance": float(dist),
                "meta": self._get_meta(index_id, idx)
            })

        return results
    
    def get_index_size(self, index_id: str) -> int:
        if index_id not in self.indexes.keys():
            self.logger.error(f"Index {index_id} not found.")
            return 0  
        
        index = self.indexes[index_id]
        return index.ntotal
    
    def get_vector(self, index_id: str, idx: int) -> Dict:
        if index_id not in self.indexes.keys():
            self.logger.error(f"Index {index_id} not found.")
            return np.array([])  
        
        index = self.indexes[index_id]
        vector = index.reconstruct(int(idx))

        result={
            "index": int(idx),
            "vector": vector.tolist(),
            "meta": self._get_meta(index_id, idx)
        }
        return result
    
# %%
