"""
Database connection module for face recognition system.
Follows cursor rules for secure PostgreSQL integration.
"""

import psycopg2
import psycopg2.pool
import logging
from typing import Optional, List, Tuple, Dict, Any
from contextlib import contextmanager
import numpy as np
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConnectionError(Exception):
    """Custom exception for database connection errors."""
    pass

class FaceDatabase:
    """
    Database handler for face recognition system.
    Handles face vector storage, retrieval, and matching with proper security.
    """
    
    def __init__(self, min_connections: int = 1, max_connections: int = 10):
        """
        Initialize database connection pool.
        
        Args:
            min_connections: Minimum number of connections in pool
            max_connections: Maximum number of connections in pool
        """
        self.connection_pool = None
        self._initialize_pool(min_connections, max_connections)
        self._create_tables()
    
    def _initialize_pool(self, min_connections: int, max_connections: int) -> None:
        """Initialize connection pool with error handling."""
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                min_connections,
                max_connections,
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            logger.info("Database connection pool initialized successfully")
        except psycopg2.Error as e:
            logger.error(f"Failed to initialize database connection pool: {e}")
            raise DatabaseConnectionError(f"Database connection failed: {e}")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with automatic cleanup."""
        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
        except psycopg2.Error as e:
            logger.error(f"Database operation failed: {e}")
            if conn:
                conn.rollback()
            raise DatabaseConnectionError(f"Database operation failed: {e}")
        finally:
            if conn:
                self.connection_pool.putconn(conn)
    
    def _create_tables(self) -> None:
        """Create necessary tables if they don't exist."""
        create_faces_table = """
        CREATE TABLE IF NOT EXISTS faces (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            face_vector BYTEA NOT NULL,
            vector_dimension INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name)
        );
        """
        
        create_face_encodings_table = """
        CREATE TABLE IF NOT EXISTS face_encodings (
            id SERIAL PRIMARY KEY,
            face_id INTEGER REFERENCES faces(id) ON DELETE CASCADE,
            encoding_data BYTEA NOT NULL,
            encoding_type VARCHAR(50) DEFAULT 'numpy_array',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_faces_table)
                    cur.execute(create_face_encodings_table)
                    conn.commit()
                    logger.info("Database tables created/verified successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise DatabaseConnectionError(f"Table creation failed: {e}")
    
    def store_face_vector(self, name: str, face_vector: np.ndarray) -> int:
        """
        Store a face vector in the database.
        
        Args:
            name: Name associated with the face
            face_vector: Normalized face vector as numpy array
            
        Returns:
            face_id: ID of the stored face
            
        Raises:
            DatabaseConnectionError: If storage fails
        """
        if not isinstance(face_vector, np.ndarray):
            raise ValueError("face_vector must be a numpy array")
        
        # Normalize vector before storage (as per cursor rules)
        normalized_vector = face_vector / np.linalg.norm(face_vector)
        
        insert_query = """
        INSERT INTO faces (name, face_vector, vector_dimension)
        VALUES (%s, %s, %s)
        ON CONFLICT (name) 
        DO UPDATE SET 
            face_vector = EXCLUDED.face_vector,
            vector_dimension = EXCLUDED.vector_dimension,
            updated_at = CURRENT_TIMESTAMP
        RETURNING id;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_query, (
                        name,
                        normalized_vector.tobytes(),
                        len(normalized_vector)
                    ))
                    face_id = cur.fetchone()[0]
                    conn.commit()
                    logger.info(f"Face vector stored for '{name}' with ID {face_id}")
                    return face_id
        except Exception as e:
            logger.error(f"Failed to store face vector for '{name}': {e}")
            raise DatabaseConnectionError(f"Face vector storage failed: {e}")
    
    def get_face_vector(self, name: str) -> Optional[np.ndarray]:
        """
        Retrieve a face vector by name.
        
        Args:
            name: Name of the person
            
        Returns:
            face_vector: Normalized face vector or None if not found
        """
        select_query = """
        SELECT face_vector, vector_dimension 
        FROM faces 
        WHERE name = %s;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(select_query, (name,))
                    result = cur.fetchone()
                    
                    if result:
                        vector_bytes, dimension = result
                        face_vector = np.frombuffer(vector_bytes, dtype=np.float32)
                        return face_vector.reshape(dimension)
                    return None
        except Exception as e:
            logger.error(f"Failed to retrieve face vector for '{name}': {e}")
            return None
    
    def find_matching_face(self, query_vector: np.ndarray, threshold: float = 0.8) -> Optional[Tuple[str, float]]:
        """
        Find the best matching face using cosine similarity.
        
        Args:
            query_vector: Normalized query face vector
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            Tuple of (name, similarity_score) or None if no match found
        """
        if not isinstance(query_vector, np.ndarray):
            raise ValueError("query_vector must be a numpy array")
        
        # Normalize query vector
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        select_query = """
        SELECT name, face_vector, vector_dimension 
        FROM faces;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(select_query)
                    results = cur.fetchall()
                    
                    best_match = None
                    best_score = 0
                    
                    for name, vector_bytes, dimension in results:
                        stored_vector = np.frombuffer(vector_bytes, dtype=np.float32).reshape(dimension)
                        
                        # Calculate cosine similarity
                        similarity = np.dot(query_vector, stored_vector)
                        
                        if similarity > best_score and similarity >= threshold:
                            best_score = similarity
                            best_match = name
                    
                    if best_match:
                        logger.info(f"Best match found: '{best_match}' with similarity {best_score:.3f}")
                        return (best_match, best_score)
                    else:
                        logger.info("No matching face found above threshold")
                        return None
                        
        except Exception as e:
            logger.error(f"Failed to find matching face: {e}")
            return None
    
    def get_all_faces(self) -> List[Dict[str, Any]]:
        """
        Retrieve all stored faces.
        
        Returns:
            List of dictionaries containing face information
        """
        select_query = """
        SELECT id, name, vector_dimension, created_at, updated_at 
        FROM faces 
        ORDER BY name;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(select_query)
                    results = cur.fetchall()
                    
                    faces = []
                    for row in results:
                        faces.append({
                            'id': row[0],
                            'name': row[1],
                            'vector_dimension': row[2],
                            'created_at': row[3],
                            'updated_at': row[4]
                        })
                    
                    return faces
        except Exception as e:
            logger.error(f"Failed to retrieve all faces: {e}")
            return []
    
    def delete_face(self, name: str) -> bool:
        """
        Delete a face from the database.
        
        Args:
            name: Name of the person to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        delete_query = """
        DELETE FROM faces 
        WHERE name = %s;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(delete_query, (name,))
                    deleted_count = cur.rowcount
                    conn.commit()
                    
                    if deleted_count > 0:
                        logger.info(f"Face '{name}' deleted successfully")
                        return True
                    else:
                        logger.warning(f"Face '{name}' not found for deletion")
                        return False
        except Exception as e:
            logger.error(f"Failed to delete face '{name}': {e}")
            return False
    
    def clear_all_faces(self) -> bool:
        """
        Clear all faces from the database.
        
        Returns:
            True if clearing successful, False otherwise
        """
        delete_query = "DELETE FROM faces;"
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(delete_query)
                    conn.commit()
                    logger.info("All faces cleared from database")
                    return True
        except Exception as e:
            logger.error(f"Failed to clear all faces: {e}")
            return False
    
    def close(self) -> None:
        """Close the database connection pool."""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Database connection pool closed")


# Global database instance
db = None

def get_database() -> FaceDatabase:
    """Get or create the global database instance."""
    global db
    if db is None:
        db = FaceDatabase()
    return db

def close_database() -> None:
    """Close the global database instance."""
    global db
    if db:
        db.close()
        db = None
