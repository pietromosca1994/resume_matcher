import requests
from typing import Dict, Literal, List, Optional, Any, Union
import numpy as np
import logging

from .custom_types.class_types import BaseClass
from .configs import FAISSClientConfig

class FAISSClient(BaseClass):
    """
    API client for FAISS vector database operations.
    Communicates with the FastAPI server instead of directly managing FAISS indexes.
    """
    
    def __init__(
        self, 
        config: FAISSClientConfig,
        ctx = None,
        verbose: int = logging.INFO,
    ):
        """
        Initialize the FAISS API client.
        
        Args:
            base_url: Base URL of the FastAPI server
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            verbose: Logging level
        """
        super().__init__(config, ctx, verbose)
        self.config: FAISSClientConfig
        self.session = requests.Session()
        
        # Setup logging
        logging.basicConfig(level=verbose)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make a GET request to the API.
        
        Args:
            endpoint: API endpoint (e.g., '/index/size')
            params: Optional query parameters
            
        Returns:
            JSON response as dictionary
        """
        url = f"{self.config.base_url}{endpoint}"
        try:
            self.logger.debug(f"GET {url} with params: {params}")
            response = self.session.get(url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"GET request to {url} failed: {e}")
            raise
    
    def _post(
        self, 
        endpoint: str, 
        data: Optional[Dict] = None, 
        json: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make a POST request to the API.
        
        Args:
            endpoint: API endpoint (e.g., '/vector/add')
            data: Optional form data
            json: Optional JSON payload
            
        Returns:
            JSON response as dictionary
        """
        url = f"{self.config.base_url}{endpoint}"
        try:
            self.logger.debug(f"POST {url} with json: {json}")
            response = self.session.post(
                url, 
                data=data, 
                json=json, 
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"POST request to {url} failed: {e}")
            raise
    
    def build_index(
        self, 
        index_name: str, 
        index_type: Literal['flatl2', 'flat1d', 'flatip'], 
        size: int
    ) -> Dict[str, Any]:
        """
        Create a new FAISS index.
        
        Args:
            index_name: Name of the index
            index_type: Type of FAISS index
            size: Dimensionality of vectors
            
        Returns:
            API response
        """
        payload = {
            'index_name': index_name,
            'index_type': index_type,
            'size': size
        }
        
        response = self._post('/index/create', json=payload)
        self.logger.info(f"Built index '{index_name}' of type '{index_type}' with size {size}")
        return response
    
    def add_vector(
        self, 
        index_id: str, 
        vector: np.ndarray, 
        normalize: bool = False, 
        meta: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Add a vector or batch of vectors to an index.

        Args:
            index_id: Name of the index
            vector: Numpy array of shape [d] or [n, d]
            normalize: Whether to normalize vectors
            meta: Optional metadata dictionary

        Returns:
            API response
        """
        # Ensure vector is 2D (list of lists) for the API
        if isinstance(vector, np.ndarray):
            if len(vector.shape) == 1:
                vector = vector.reshape(1, -1)
            vector_list = vector.tolist()
        else:
            vector_list = vector  # assume already list of lists

        payload = {
            "index_id": index_id,
            "vector": vector_list,  # matches Pydantic model
            "normalize": normalize,
            "meta": meta
        }

        response = self._post("/vector/add", json=payload)
        self.logger.info(f"Added {response.get('count', 0)} vector to index '{index_id}'")
        return response
    
    def query(
        self, 
        vector: np.ndarray, 
        index_id: str, 
        k: int = 5, 
        normalize: bool = False
    ) -> Union[List[Dict[str, Any]], None]:
        """
        Query the index for nearest neighbors.
        
        Args:
            vector: Query vector (1D numpy array)
            index_id: Name of the index
            k: Number of nearest neighbors to return
            normalize: Whether to normalize the query vector
            
        Returns:
            List of results with 'index', 'distance', and optional 'meta' fields
        """
        # Convert numpy array to list for JSON serialization
        if isinstance(vector, np.ndarray):
            if len(vector.shape) > 1:
                vector = vector.flatten()
            vector_list = vector.tolist()
        else:
            vector_list = vector
        
        payload = {
            'index_id': index_id,
            'vector': vector_list,
            'k': k,
            'normalize': normalize
        }
        
        response = self._post('/vector/query', json=payload)
        results = response.get('results', [])
        if results:
            self.logger.debug(f"Query returned {len(results)} results from index '{index_id}'")
        else:
            self.logger.warning('Query returned no results')
        return results

    def get_index_size(self, index_id: str) -> int:
        """
        Get the number of vectors in an index.
        
        Args:
            index_id: Name of the index
            
        Returns:
            Number of vectors in the index
        """
        response = self._get(f'/{index_id}/size')
        size = response.get('size', 0)
        self.logger.debug(f"Index '{index_id}' contains {size} vectors")
        return size
    
    def get_vector(self, index_id: str, idx: int): 
        response = self._get(f'/{index_id}/{idx}/vector')
        vector = np.array(response.get('vector', []), dtype=np.float32)
        self.logger.debug(f"Vector {idx}: {vector}")
        return vector
    
    def close(self):
        """Close the session."""
        self.session.close()
        self.logger.info("API client session closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()