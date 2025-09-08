"""
Test script for database functionality.
Run this to verify your PostgreSQL setup is working correctly.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import get_database, close_database
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER

def test_database_connection():
    """Test basic database connectivity."""
    print("Testing database connection...")
    
    try:
        db = get_database()
        print("✅ Database connection successful!")
        return db
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

def test_face_storage(db):
    """Test storing and retrieving face vectors."""
    print("\nTesting face vector storage...")
    
    # Create a dummy face vector (128-dimensional, typical for face recognition)
    dummy_vector = np.random.rand(128).astype(np.float32)
    test_name = "test_person"
    
    try:
        # Store the face vector
        face_id = db.store_face_vector(test_name, dummy_vector)
        print(f"✅ Face vector stored with ID: {face_id}")
        
        # Retrieve the face vector
        retrieved_vector = db.get_face_vector(test_name)
        if retrieved_vector is not None:
            # Check if vectors are similar (they should be normalized)
            similarity = np.dot(dummy_vector / np.linalg.norm(dummy_vector), 
                              retrieved_vector / np.linalg.norm(retrieved_vector))
            print(f"✅ Face vector retrieved successfully. Similarity: {similarity:.6f}")
        else:
            print("❌ Failed to retrieve face vector")
            
    except Exception as e:
        print(f"❌ Face storage test failed: {e}")

def test_face_matching(db):
    """Test face matching functionality."""
    print("\nTesting face matching...")
    
    # Create a query vector similar to the stored one
    query_vector = np.random.rand(128).astype(np.float32)
    
    try:
        # Try to find a match
        match_result = db.find_matching_face(query_vector, threshold=0.5)
        
        if match_result:
            name, similarity = match_result
            print(f"✅ Match found: {name} with similarity {similarity:.3f}")
        else:
            print("ℹ️ No match found (expected for random vector)")
            
    except Exception as e:
        print(f"❌ Face matching test failed: {e}")

def test_get_all_faces(db):
    """Test retrieving all faces."""
    print("\nTesting get all faces...")
    
    try:
        faces = db.get_all_faces()
        print(f"✅ Retrieved {len(faces)} faces from database")
        
        for face in faces:
            print(f"  - {face['name']} (ID: {face['id']}, Dimension: {face['vector_dimension']})")
            
    except Exception as e:
        print(f"❌ Get all faces test failed: {e}")

def test_cleanup(db):
    """Clean up test data."""
    print("\nCleaning up test data...")
    
    try:
        success = db.delete_face("test_person")
        if success:
            print("✅ Test data cleaned up successfully")
        else:
            print("ℹ️ Test data not found (already cleaned up)")
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")

def main():
    """Run all database tests."""
    print("=" * 50)
    print("PostgreSQL Database Test Suite")
    print("=" * 50)
    
    print(f"Database Configuration:")
    print(f"  Host: {DB_HOST}")
    print(f"  Port: {DB_PORT}")
    print(f"  Database: {DB_NAME}")
    print(f"  User: {DB_USER}")
    print()
    
    # Test connection
    db = test_database_connection()
    if db is None:
        print("\n❌ Cannot proceed without database connection.")
        print("Please check your PostgreSQL setup and environment variables.")
        return
    
    # Run tests
    test_face_storage(db)
    test_face_matching(db)
    test_get_all_faces(db)
    test_cleanup(db)
    
    # Close database
    close_database()
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()
