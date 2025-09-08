"""
Secure REST API for face recognition system.
Follows cursor rules for secure database integration and API design.
"""

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
import os
import jwt
import datetime
from functools import wraps
import numpy as np
from typing import Optional, Dict, Any
import base64
import sys

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import get_database, close_database
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('API_SECRET_KEY', 'your-secret-key-change-this')
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-jwt-secret-change-this')

# Enable CORS for frontend integration
CORS(app, origins=['http://localhost:3000', 'http://localhost:5000'])

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# API Keys for authentication (in production, use proper user management)
API_KEYS = {
    'admin': os.getenv('ADMIN_API_KEY', 'admin-key-change-this'),
    'user': os.getenv('USER_API_KEY', 'user-key-change-this')
}

def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        if api_key not in API_KEYS.values():
            return jsonify({'error': 'Invalid API key'}), 401
        
        return f(*args, **kwargs)
    return decorated_function

def require_admin_key(f):
    """Decorator to require admin API key."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        if api_key != API_KEYS['admin']:
            return jsonify({'error': 'Admin access required'}), 403
        
        return f(*args, **kwargs)
    return decorated_function

def validate_face_vector(vector_data: str) -> Optional[np.ndarray]:
    """Validate and decode face vector from base64 string."""
    try:
        # Decode base64 string to bytes
        vector_bytes = base64.b64decode(vector_data)
        # Convert to numpy array
        vector = np.frombuffer(vector_bytes, dtype=np.float32)
        
        # Validate vector properties
        if len(vector) == 0:
            return None
        if len(vector) > 10000:  # Reasonable limit for face vectors
            return None
            
        return vector
    except Exception as e:
        logger.error(f"Invalid face vector format: {e}")
        return None

def log_api_request(endpoint: str, user_ip: str, success: bool, details: str = ""):
    """Log API requests for audit purposes."""
    logger.info(f"API Request: {endpoint} | IP: {user_ip} | Success: {success} | Details: {details}")

@app.before_request
def before_request():
    """Set up database connection for each request."""
    g.db = get_database()

@app.teardown_request
def teardown_request(exception):
    """Clean up database connection after each request."""
    if hasattr(g, 'db'):
        close_database()

@app.route('/api/health', methods=['GET'])
@limiter.limit("100 per minute")
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'database': 'connected',
        'timestamp': datetime.datetime.utcnow().isoformat()
    })

@app.route('/api/faces', methods=['POST'])
@require_api_key
@limiter.limit("10 per minute")
def store_face():
    """Store a face vector in the database."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        name = data.get('name')
        face_vector_b64 = data.get('face_vector')
        
        # Validate inputs
        if not name or not isinstance(name, str):
            return jsonify({'error': 'Valid name required'}), 400
        
        if not face_vector_b64 or not isinstance(face_vector_b64, str):
            return jsonify({'error': 'Valid face_vector required'}), 400
        
        # Clean and validate name
        name = name.strip()
        if len(name) > 255 or len(name) < 1:
            return jsonify({'error': 'Name must be 1-255 characters'}), 400
        
        # Validate and decode face vector
        face_vector = validate_face_vector(face_vector_b64)
        if face_vector is None:
            return jsonify({'error': 'Invalid face vector format'}), 400
        
        # Store in database
        face_id = g.db.store_face_vector(name, face_vector)
        
        log_api_request('/api/faces', request.remote_addr, True, f"Stored face for {name}")
        
        return jsonify({
            'success': True,
            'face_id': face_id,
            'name': name,
            'vector_dimension': len(face_vector)
        }), 201
        
    except Exception as e:
        logger.error(f"Error storing face: {e}")
        log_api_request('/api/faces', request.remote_addr, False, str(e))
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/faces/match', methods=['POST'])
@require_api_key
@limiter.limit("20 per minute")
def match_face():
    """Find matching face in the database."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        face_vector_b64 = data.get('face_vector')
        threshold = data.get('threshold', 0.8)
        
        # Validate inputs
        if not face_vector_b64 or not isinstance(face_vector_b64, str):
            return jsonify({'error': 'Valid face_vector required'}), 400
        
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            return jsonify({'error': 'Threshold must be between 0 and 1'}), 400
        
        # Validate and decode face vector
        face_vector = validate_face_vector(face_vector_b64)
        if face_vector is None:
            return jsonify({'error': 'Invalid face vector format'}), 400
        
        # Find match
        match_result = g.db.find_matching_face(face_vector, threshold)
        
        if match_result:
            name, similarity = match_result
            log_api_request('/api/faces/match', request.remote_addr, True, f"Match found: {name}")
            return jsonify({
                'success': True,
                'match_found': True,
                'name': name,
                'similarity': round(similarity, 4)
            })
        else:
            log_api_request('/api/faces/match', request.remote_addr, True, "No match found")
            return jsonify({
                'success': True,
                'match_found': False,
                'message': 'No matching face found'
            })
            
    except Exception as e:
        logger.error(f"Error matching face: {e}")
        log_api_request('/api/faces/match', request.remote_addr, False, str(e))
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/faces/<name>', methods=['GET'])
@require_api_key
@limiter.limit("30 per minute")
def get_face(name):
    """Get face vector by name."""
    try:
        # Validate name
        if not name or not isinstance(name, str):
            return jsonify({'error': 'Valid name required'}), 400
        
        name = name.strip()
        if len(name) > 255:
            return jsonify({'error': 'Name too long'}), 400
        
        # Get face vector
        face_vector = g.db.get_face_vector(name)
        
        if face_vector is not None:
            # Encode vector as base64 for transmission
            vector_b64 = base64.b64encode(face_vector.tobytes()).decode('utf-8')
            
            log_api_request(f'/api/faces/{name}', request.remote_addr, True)
            return jsonify({
                'success': True,
                'name': name,
                'face_vector': vector_b64,
                'vector_dimension': len(face_vector)
            })
        else:
            log_api_request(f'/api/faces/{name}', request.remote_addr, True, "Face not found")
            return jsonify({
                'success': True,
                'found': False,
                'message': 'Face not found'
            }), 404
            
    except Exception as e:
        logger.error(f"Error retrieving face: {e}")
        log_api_request(f'/api/faces/{name}', request.remote_addr, False, str(e))
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/faces', methods=['GET'])
@require_api_key
@limiter.limit("10 per minute")
def list_faces():
    """List all stored faces."""
    try:
        faces = g.db.get_all_faces()
        
        # Convert timestamps to ISO format
        for face in faces:
            if face['created_at']:
                face['created_at'] = face['created_at'].isoformat()
            if face['updated_at']:
                face['updated_at'] = face['updated_at'].isoformat()
        
        log_api_request('/api/faces', request.remote_addr, True, f"Retrieved {len(faces)} faces")
        return jsonify({
            'success': True,
            'faces': faces,
            'count': len(faces)
        })
        
    except Exception as e:
        logger.error(f"Error listing faces: {e}")
        log_api_request('/api/faces', request.remote_addr, False, str(e))
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/faces/<name>', methods=['DELETE'])
@require_admin_key
@limiter.limit("5 per minute")
def delete_face(name):
    """Delete a face from the database (admin only)."""
    try:
        # Validate name
        if not name or not isinstance(name, str):
            return jsonify({'error': 'Valid name required'}), 400
        
        name = name.strip()
        if len(name) > 255:
            return jsonify({'error': 'Name too long'}), 400
        
        # Delete face
        success = g.db.delete_face(name)
        
        if success:
            log_api_request(f'/api/faces/{name}', request.remote_addr, True, "Face deleted")
            return jsonify({
                'success': True,
                'message': f'Face "{name}" deleted successfully'
            })
        else:
            log_api_request(f'/api/faces/{name}', request.remote_addr, True, "Face not found for deletion")
            return jsonify({
                'success': False,
                'message': f'Face "{name}" not found'
            }), 404
            
    except Exception as e:
        logger.error(f"Error deleting face: {e}")
        log_api_request(f'/api/faces/{name}', request.remote_addr, False, str(e))
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit exceeded."""
    return jsonify({'error': 'Rate limit exceeded'}), 429

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.getenv('API_PORT', 5000))
    
    # Run in production mode
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,  # Set to False for production
        threaded=True
    )
