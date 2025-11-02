# app/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Literal, Optional
import numpy as np
import logging
import dotenv 
import os 

from faissdb import FAISSDB
from configs import FAISSDBConfig, PostgresClientConfig
from ctxs import FAISSDBCtx

dotenv.load_dotenv()

# ------------------------------------------------------------------------
# Pydantic request models

class IndexCreateRequest(BaseModel):
    index_name: str
    index_type: Literal["flatl2", "flat1d", "flatip"]
    size: int

class VectorAddRequest(BaseModel):
    index_id: str
    vector: List[List[float]]
    normalize: Optional[bool] = False
    meta: Optional[Dict] = None

class QueryRequest(BaseModel):
    index_id: str
    vector: List[float]
    k: int = 5
    normalize: Optional[bool] = False

# ------------------------------------------------------------------------
# Initialize API and FAISS client

app = FastAPI(title="FAISS Vector Search API", version="1.0")

config = FAISSDBConfig(
    postgres_config=PostgresClientConfig(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        database=os.getenv("POSTGRES_DATABASE", 'faissdb'),
        user=os.getenv("POSTGRES_USER", 'user'),
        password=os.getenv("POSTGRES_PASSWORD", 'password')
    )
)

ctx = FAISSDBCtx()
faiss_db = FAISSDB(config=config, ctx=ctx, verbose=logging.INFO)

# ------------------------------------------------------------------------
# Routes

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "FAISS Vector Search API"}

@app.post("/index/create")
def create_index(request: IndexCreateRequest):
    try:
        faiss_db.build_index(request.index_name, request.index_type, request.size)
        return {"status": "success", "message": f"Index '{request.index_name}' created."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector/add")
def add_vector(request: VectorAddRequest):
    try:
        vector = np.array(request.vector, dtype=np.float32)
        faiss_db.add_vector(
            index_id=request.index_id,
            vector=vector,
            normalize=request.normalize,
            meta=request.meta,
        )
        return {"status": "success", "count": len(vector)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector/query")
def query_vector(request: QueryRequest):
    try:
        vector = np.array(request.vector, dtype=np.float32)
        results = faiss_db.query(
            vector=vector,
            index_id=request.index_id,
            k=request.k,
            normalize=request.normalize,
        )
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/{index_id}/size")
def get_index_size(index_id: str):
    try:
        size = faiss_db.get_index_size(index_id)
        return {"index_id": index_id, "size": size}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/{index_id}/{idx}/vector")
def get_vector(index_id: str, idx: int):
    try:
        vector = faiss_db.get_vector(index_id, idx)
        if vector is None:
            raise HTTPException(status_code=404, detail=f"Vector {idx} not found in {index_id}")
        meta=faiss_db._get_meta(index_id, vector)
        return {"index_id": index_id, "idx": idx, "vector": vector.tolist(), "meta": meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    log_level = "debug"  # enables debug logging

    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )