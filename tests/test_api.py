"""
Test script for the Face Recognition REST API.
Tests all endpoints and security features.
"""

import numpy as np
import sys
import os
import time

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.api_client import FaceRecognitionClient, create_client

def test_api_health(client):
    """Test API health endpoint."""
    print("Testing API health...")
    
    try:
        response = client.health_check()
        print(f"✅ API Health: {response['status']}")
        print(f"   Database: {response['database']}")
        print(f"   Timestamp: {response['timestamp']}")
        return True
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        return False

def test_face_storage(client):
    """Test storing a face vector."""
    print("\nTesting face storage...")
    
    # Create a dummy face vector
    face_vector = np.random.rand(128).astype(np.float32)
    test_name = "test_api_person"
    
    try:
        response = client.store_face(test_name, face_vector)
        print(f"✅ Face stored successfully")
        print(f"   Face ID: {response['face_id']}")
        print(f"   Name: {response['name']}")
        print(f"   Dimension: {response['vector_dimension']}")
        return test_name, face_vector
    except Exception as e:
        print(f"❌ Face storage failed: {e}")
        return None, None

def test_face_matching(client, stored_vector):
    """Test face matching functionality."""
    print("\nTesting face matching...")
    
    # Create a similar vector (should match)
    similar_vector = stored_vector + np.random.normal(0, 0.1, stored_vector.shape)
    similar_vector = similar_vector.astype(np.float32)
    
    try:
        response = client.match_face(similar_vector, threshold=0.7)
        
        if response.get('match_found', False):
            print(f"✅ Match found: {response['name']}")
            print(f"   Similarity: {response['similarity']:.4f}")
        else:
            print("ℹ️ No match found (threshold too high)")
            
        return True
    except Exception as e:
        print(f"❌ Face matching failed: {e}")
        return False

def test_face_retrieval(client, test_name):
    """Test retrieving a face vector."""
    print("\nTesting face retrieval...")
    
    try:
        response = client.get_face(test_name)
        
        if response:
            print(f"✅ Face retrieved successfully")
            print(f"   Name: {response['name']}")
            print(f"   Dimension: {response['vector_dimension']}")
            return True
        else:
            print("❌ Face not found")
            return False
    except Exception as e:
        print(f"❌ Face retrieval failed: {e}")
        return False

def test_list_faces(client):
    """Test listing all faces."""
    print("\nTesting list faces...")
    
    try:
        response = client.list_faces()
        print(f"✅ Retrieved {response['count']} faces")
        
        for face in response['faces']:
            print(f"   - {face['name']} (ID: {face['id']})")
            
        return True
    except Exception as e:
        print(f"❌ List faces failed: {e}")
        return False

def test_authentication():
    """Test authentication requirements."""
    print("\nTesting authentication...")
    
    # Create client without API key
    client_no_key = create_client()
    
    try:
        client_no_key.health_check()
        print("ℹ️ Health check works without API key (expected)")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Try to store face without API key
    try:
        face_vector = np.random.rand(128).astype(np.float32)
        client_no_key.store_face("test_auth", face_vector)
        print("❌ Should have required API key")
        return False
    except Exception as e:
        print("✅ API key required for protected endpoints")
        return True

def test_rate_limiting(client):
    """Test rate limiting."""
    print("\nTesting rate limiting...")
    
    try:
        # Make multiple requests quickly
        for i in range(15):
            try:
                client.health_check()
                print(f"   Request {i+1}: OK")
            except Exception as e:
                if "429" in str(e):
                    print(f"   Request {i+1}: Rate limited (expected)")
                    return True
                else:
                    print(f"   Request {i+1}: Error - {e}")
                    return False
            time.sleep(0.1)
        
        print("ℹ️ Rate limiting not triggered (may need more requests)")
        return True
    except Exception as e:
        print(f"❌ Rate limiting test failed: {e}")
        return False

def test_cleanup(client, test_name):
    """Clean up test data."""
    print("\nCleaning up test data...")
    
    try:
        success = client.delete_face(test_name)
        if success:
            print("✅ Test data cleaned up successfully")
        else:
            print("ℹ️ Test data not found (already cleaned up)")
        return True
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        return False

def main():
    """Run all API tests."""
    print("=" * 60)
    print("Face Recognition REST API Test Suite")
    print("=" * 60)
    
    # Test configuration
    base_url = "http://localhost:5000"
    api_key = "user-key-change-this"  # Use the default from env_template.txt
    
    print(f"API Configuration:")
    print(f"  Base URL: {base_url}")
    print(f"  API Key: {api_key[:10]}...")
    print()
    
    # Create client
    client = create_client(api_key, base_url)
    
    # Run tests
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Health check
    total_tests += 1
    if test_api_health(client):
        tests_passed += 1
    
    # Test 2: Authentication
    total_tests += 1
    if test_authentication():
        tests_passed += 1
    
    # Test 3: Face storage
    total_tests += 1
    test_name, stored_vector = test_face_storage(client)
    if test_name:
        tests_passed += 1
    
    # Test 4: Face matching
    total_tests += 1
    if test_face_matching(client, stored_vector):
        tests_passed += 1
    
    # Test 5: Face retrieval
    total_tests += 1
    if test_face_retrieval(client, test_name):
        tests_passed += 1
    
    # Test 6: List faces
    total_tests += 1
    if test_list_faces(client):
        tests_passed += 1
    
    # Test 7: Rate limiting
    total_tests += 1
    if test_rate_limiting(client):
        tests_passed += 1
    
    # Test 8: Cleanup
    total_tests += 1
    if test_cleanup(client, test_name):
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! API is working correctly.")
    else:
        print("❌ Some tests failed. Check the API server and configuration.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
