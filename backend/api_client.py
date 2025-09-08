"""
Client library for the Face Recognition REST API.
Provides easy-to-use methods for interacting with the secure API.
"""

import requests
import numpy as np
import base64
import json
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class FaceRecognitionClient:
    """
    Client for the Face Recognition REST API.
    Handles authentication, request formatting, and response parsing.
    """
    
    def __init__(self, base_url: str = "http://localhost:5000", api_key: str = None):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API server
            api_key: API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({'X-API-Key': api_key})
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """
        Make a request to the API with error handling.
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            data: Request data for POST requests
            
        Returns:
            Response data as dictionary
            
        Raises:
            requests.RequestException: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                response = self.session.get(url)
            elif method == 'POST':
                response = self.session.post(url, json=data)
            elif method == 'DELETE':
                response = self.session.delete(url)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        return self._make_request('GET', '/api/health')
    
    def store_face(self, name: str, face_vector: np.ndarray) -> Dict[str, Any]:
        """
        Store a face vector in the database.
        
        Args:
            name: Name associated with the face
            face_vector: Face vector as numpy array
            
        Returns:
            Response with face_id and status
        """
        if not isinstance(face_vector, np.ndarray):
            raise ValueError("face_vector must be a numpy array")
        
        # Encode vector as base64
        vector_b64 = base64.b64encode(face_vector.tobytes()).decode('utf-8')
        
        data = {
            'name': name,
            'face_vector': vector_b64
        }
        
        return self._make_request('POST', '/api/faces', data)
    
    def match_face(self, face_vector: np.ndarray, threshold: float = 0.8) -> Dict[str, Any]:
        """
        Find matching face in the database.
        
        Args:
            face_vector: Query face vector as numpy array
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            Response with match results
        """
        if not isinstance(face_vector, np.ndarray):
            raise ValueError("face_vector must be a numpy array")
        
        # Encode vector as base64
        vector_b64 = base64.b64encode(face_vector.tobytes()).decode('utf-8')
        
        data = {
            'face_vector': vector_b64,
            'threshold': threshold
        }
        
        return self._make_request('POST', '/api/faces/match', data)
    
    def get_face(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get face vector by name.
        
        Args:
            name: Name of the person
            
        Returns:
            Face data or None if not found
        """
        try:
            response = self._make_request('GET', f'/api/faces/{name}')
            return response
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    def list_faces(self) -> Dict[str, Any]:
        """
        List all stored faces.
        
        Returns:
            Response with list of faces
        """
        return self._make_request('GET', '/api/faces')
    
    def delete_face(self, name: str) -> bool:
        """
        Delete a face from the database (requires admin API key).
        
        Args:
            name: Name of the person to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            response = self._make_request('DELETE', f'/api/faces/{name}')
            return response.get('success', False)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return False
            raise

# Convenience functions for common operations
def create_client(api_key: str = None, base_url: str = "http://localhost:5000") -> FaceRecognitionClient:
    """Create a new API client instance."""
    return FaceRecognitionClient(base_url, api_key)

def store_face_vector(name: str, face_vector: np.ndarray, api_key: str, base_url: str = "http://localhost:5000") -> int:
    """
    Store a face vector (convenience function).
    
    Args:
        name: Name associated with the face
        face_vector: Face vector as numpy array
        api_key: API key for authentication
        base_url: Base URL of the API server
        
    Returns:
        face_id: ID of the stored face
    """
    client = create_client(api_key, base_url)
    response = client.store_face(name, face_vector)
    return response['face_id']

def find_matching_face(face_vector: np.ndarray, threshold: float = 0.8, api_key: str = None, base_url: str = "http://localhost:5000") -> Optional[Tuple[str, float]]:
    """
    Find matching face (convenience function).
    
    Args:
        face_vector: Query face vector as numpy array
        threshold: Minimum similarity threshold (0-1)
        api_key: API key for authentication
        base_url: Base URL of the API server
        
    Returns:
        Tuple of (name, similarity) or None if no match
    """
    client = create_client(api_key, base_url)
    response = client.match_face(face_vector, threshold)
    
    if response.get('match_found', False):
        return (response['name'], response['similarity'])
    return None
