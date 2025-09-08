# Secure REST API Documentation

This document describes the secure REST API for the face recognition system, which follows cursor rules for secure database integration and API design.

## 🔒 Security Features

The API implements multiple layers of security following cursor rules:

### 1. **Authentication**
- **API Key Authentication**: All endpoints (except health check) require a valid API key
- **Role-Based Access**: Different API keys for admin and user roles
- **Header-Based**: API keys sent via `X-API-Key` header

### 2. **Input Validation**
- **Parameterized Queries**: All database operations use parameterized queries
- **Input Sanitization**: All inputs are validated and sanitized
- **Type Checking**: Strict type validation for all parameters
- **Size Limits**: Reasonable limits on input sizes

### 3. **Rate Limiting**
- **Per-Endpoint Limits**: Different limits for different operations
- **IP-Based Tracking**: Rate limiting based on client IP address
- **Graceful Handling**: Proper error responses for rate limit exceeded

### 4. **Audit Logging**
- **Request Logging**: All API requests are logged with IP and success status
- **Error Tracking**: Detailed error logging for debugging
- **Security Events**: Authentication failures and suspicious activities logged

### 5. **Data Protection**
- **Base64 Encoding**: Face vectors transmitted as base64 strings
- **Vector Validation**: Face vectors validated before processing
- **No Raw Images**: Only face vectors stored, never raw images

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy the secure template
cp env_template_secure.txt .env

# Edit .env with your actual values
# Make sure to change all the default keys!
```

### 3. Start the API Server
```bash
python backend/api_server.py
```

### 4. Test the API
```bash
python tests/test_api.py
```

## 📡 API Endpoints

### Health Check
```http
GET /api/health
```
**No authentication required**

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### Store Face Vector
```http
POST /api/faces
X-API-Key: your-api-key
Content-Type: application/json

{
  "name": "John Doe",
  "face_vector": "base64-encoded-vector-string"
}
```

**Rate Limit:** 10 per minute

**Response:**
```json
{
  "success": true,
  "face_id": 1,
  "name": "John Doe",
  "vector_dimension": 128
}
```

### Match Face
```http
POST /api/faces/match
X-API-Key: your-api-key
Content-Type: application/json

{
  "face_vector": "base64-encoded-vector-string",
  "threshold": 0.8
}
```

**Rate Limit:** 20 per minute

**Response (Match Found):**
```json
{
  "success": true,
  "match_found": true,
  "name": "John Doe",
  "similarity": 0.9234
}
```

**Response (No Match):**
```json
{
  "success": true,
  "match_found": false,
  "message": "No matching face found"
}
```

### Get Face by Name
```http
GET /api/faces/{name}
X-API-Key: your-api-key
```

**Rate Limit:** 30 per minute

**Response:**
```json
{
  "success": true,
  "name": "John Doe",
  "face_vector": "base64-encoded-vector-string",
  "vector_dimension": 128
}
```

### List All Faces
```http
GET /api/faces
X-API-Key: your-api-key
```

**Rate Limit:** 10 per minute

**Response:**
```json
{
  "success": true,
  "faces": [
    {
      "id": 1,
      "name": "John Doe",
      "vector_dimension": 128,
      "created_at": "2024-01-15T10:30:00.000Z",
      "updated_at": "2024-01-15T10:30:00.000Z"
    }
  ],
  "count": 1
}
```

### Delete Face (Admin Only)
```http
DELETE /api/faces/{name}
X-API-Key: admin-api-key
```

**Rate Limit:** 5 per minute

**Response:**
```json
{
  "success": true,
  "message": "Face \"John Doe\" deleted successfully"
}
```

## 🔑 Authentication

### API Keys
The API uses two types of API keys:

1. **User API Key**: For regular operations (store, match, retrieve, list)
2. **Admin API Key**: For administrative operations (delete)

### Setting API Keys
Add these to your `.env` file:
```env
ADMIN_API_KEY=your-super-secret-admin-key
USER_API_KEY=your-user-api-key
```

### Using API Keys
Include the API key in the `X-API-Key` header:
```bash
curl -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"name":"John","face_vector":"..."}' \
     http://localhost:5000/api/faces
```

## 🛡️ Security Best Practices

### 1. **API Key Management**
- ✅ Use strong, unique API keys
- ✅ Rotate keys regularly
- ✅ Never commit API keys to version control
- ✅ Use different keys for different environments

### 2. **Network Security**
- ✅ Use HTTPS in production
- ✅ Configure firewall rules
- ✅ Use VPN for remote access
- ✅ Monitor network traffic

### 3. **Input Validation**
- ✅ Validate all inputs server-side
- ✅ Sanitize user-provided data
- ✅ Use parameterized queries
- ✅ Implement size limits

### 4. **Rate Limiting**
- ✅ Monitor rate limit usage
- ✅ Adjust limits based on usage patterns
- ✅ Implement progressive rate limiting
- ✅ Log rate limit violations

### 5. **Audit and Monitoring**
- ✅ Monitor API access logs
- ✅ Set up alerts for suspicious activity
- ✅ Regular security reviews
- ✅ Keep dependencies updated

## 🧪 Testing

### Run API Tests
```bash
# Start the API server first
python backend/api_server.py

# In another terminal, run tests
python tests/test_api.py
```

### Manual Testing with curl
```bash
# Health check
curl http://localhost:5000/api/health

# Store face (replace with your API key)
curl -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"name":"Test","face_vector":"base64-vector"}' \
     http://localhost:5000/api/faces

# List faces
curl -H "X-API-Key: your-api-key" \
     http://localhost:5000/api/faces
```

## 🔧 Configuration

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `API_PORT` | API server port | 5000 |
| `API_SECRET_KEY` | Flask secret key | (generated) |
| `JWT_SECRET_KEY` | JWT secret key | (generated) |
| `ADMIN_API_KEY` | Admin API key | (generated) |
| `USER_API_KEY` | User API key | (generated) |

### Rate Limits
| Endpoint | Limit | Description |
|----------|-------|-------------|
| Health Check | 100/min | Basic health monitoring |
| Store Face | 10/min | Prevent abuse of storage |
| Match Face | 20/min | Allow reasonable matching |
| Get Face | 30/min | Allow frequent lookups |
| List Faces | 10/min | Prevent enumeration |
| Delete Face | 5/min | Prevent mass deletion |

## 🚨 Error Handling

### HTTP Status Codes
- `200` - Success
- `201` - Created (face stored)
- `400` - Bad Request (invalid input)
- `401` - Unauthorized (missing/invalid API key)
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found (face not found)
- `429` - Too Many Requests (rate limit exceeded)
- `500` - Internal Server Error

### Error Response Format
```json
{
  "error": "Error message description"
}
```

## 🔄 Integration Examples

### Python Client Usage
```python
from backend.api_client import create_client
import numpy as np

# Create client
client = create_client(api_key="your-api-key")

# Store face
face_vector = np.random.rand(128).astype(np.float32)
response = client.store_face("John Doe", face_vector)
print(f"Stored face with ID: {response['face_id']}")

# Match face
match_result = client.match_face(face_vector, threshold=0.8)
if match_result.get('match_found'):
    print(f"Found match: {match_result['name']}")
```

### JavaScript/Node.js Usage
```javascript
const axios = require('axios');

const client = axios.create({
  baseURL: 'http://localhost:5000',
  headers: {
    'X-API-Key': 'your-api-key',
    'Content-Type': 'application/json'
  }
});

// Store face
const response = await client.post('/api/faces', {
  name: 'John Doe',
  face_vector: 'base64-encoded-vector'
});
```

## 🔍 Monitoring and Logging

### Log Format
```
API Request: /api/faces | IP: 192.168.1.100 | Success: True | Details: Stored face for John Doe
```

### Monitoring Checklist
- [ ] API response times
- [ ] Error rates
- [ ] Rate limit usage
- [ ] Authentication failures
- [ ] Database connection health
- [ ] Memory and CPU usage

## 🚀 Production Deployment

### 1. **Environment Setup**
```bash
# Use production-grade secrets
export API_SECRET_KEY="your-super-secret-production-key"
export ADMIN_API_KEY="your-admin-production-key"
export USER_API_KEY="your-user-production-key"
```

### 2. **Web Server Configuration**
```bash
# Use Gunicorn for production
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 backend.api_server:app
```

### 3. **SSL/TLS Setup**
```bash
# Configure HTTPS with Let's Encrypt or similar
# Update CORS origins to your domain
```

### 4. **Monitoring Setup**
```bash
# Set up monitoring with tools like:
# - Prometheus + Grafana
# - ELK Stack
# - AWS CloudWatch
```

The API is now ready for secure production use! 🎉
