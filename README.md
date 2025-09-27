# Face Recognition System

A PostgreSQL-based face recognition system with real-time webcam detection and face matching capabilities.

## 🚀 Features

- **Real-time Face Detection**: Uses OpenCV Haar Cascade for face detection
- **PostgreSQL Database**: Stores face embeddings with proper normalization and similarity matching
- **Multiple Face Recognition Approaches**:
  - PyTorch-based embeddings (`face_detection_pytorch_integrated.py`)
  - Histogram-based features (`face_detection_pretrained.py`)
- **RetinaFace Support**: Optional RetinaFace model integration
- **API Server**: RESTful API for face storage and matching
- **SQLite Fallback**: Alternative SQLite-based face embeddings system

## 📋 Prerequisites

- Python 3.8+
- PostgreSQL 12+
- Webcam/Camera

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd raspberrypi-server
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up PostgreSQL database**:
   - Install PostgreSQL
   - Create database: `face_recognition_db`
   - See `DATABASE_SETUP.md` for detailed instructions

4. **Configure environment**:
   ```bash
   cp env_template_secure.txt .env
   # Edit .env with your database credentials
   ```

## 🎯 Quick Start

### Option 1: PyTorch-based Face Recognition
```bash
python face_detection_pytorch_integrated.py
```

### Option 2: Histogram-based Face Recognition
```bash
python face_detection_pretrained.py
```

### Controls:
- `q` - quit
- `a` - add unknown face to database
- `s` - show database statistics
- `t` - test embedding extraction

## 🗄️ Database Schema

The system uses PostgreSQL with the following tables:

### `faces` table:
- `id`: Primary key (auto-increment)
- `name`: Person's name (unique)
- `face_vector`: Binary storage of normalized face vector
- `vector_dimension`: Dimension of the face vector
- `created_at`: Timestamp when record was created
- `updated_at`: Timestamp when record was last updated

### `face_encodings` table:
- `id`: Primary key (auto-increment)
- `face_id`: Foreign key to faces table
- `encoding_data`: Additional encoding data
- `encoding_type`: Type of encoding
- `created_at`: Timestamp when record was created

## 🔧 Configuration

Edit `.env` file with your settings:

```env
# Database Settings
DB_HOST=localhost
DB_PORT=5432
DB_NAME=face_recognition_db
DB_USER=postgres
DB_PASSWORD=your_password

# API Security (optional)
API_SECRET_KEY=your-secret-key
JWT_SECRET_KEY=your-jwt-key
ADMIN_API_KEY=admin-key
USER_API_KEY=user-key
API_PORT=5000
```

## 📊 API Usage

Start the API server:
```bash
python backend/api_server.py
```

### Endpoints:

- `GET /api/health` - Health check
- `POST /api/faces` - Store face vector
- `POST /api/faces/match` - Find matching face
- `GET /api/faces/<name>` - Get face by name
- `GET /api/faces` - Get all faces

## 🧪 Testing

Run database tests:
```bash
python tests/test_database.py
```

Run API tests:
```bash
python tests/test_api.py
```

## 📁 Project Structure

```
├── backend/
│   ├── database.py          # PostgreSQL database handler
│   ├── face_embeddings.py   # SQLite fallback embeddings
│   ├── api_server.py        # Flask API server
│   └── api_client.py        # API client
├── models/
│   ├── detector.py          # RetinaFace detector
│   └── model_info.txt       # Model information
├── tests/
│   ├── test_database.py     # Database tests
│   └── test_api.py          # API tests
├── face_detection_pytorch_integrated.py  # Main PyTorch app
├── face_detection_pretrained.py          # Histogram-based app
├── retinaface_detector.py   # RetinaFace integration
├── config.py                # Configuration
├── requirements.txt         # Dependencies
└── DATABASE_SETUP.md       # Database setup guide
```

## 🔍 SQL Queries for pgAdmin

View all faces with embeddings:
```sql
SELECT 
    id,
    name,
    vector_dimension,
    created_at,
    updated_at,
    LENGTH(face_vector) as embedding_size_bytes
FROM faces 
ORDER BY name;
```

Get database statistics:
```sql
SELECT 
    COUNT(*) as total_faces,
    AVG(vector_dimension) as avg_vector_dimension,
    MIN(created_at) as first_face_added,
    MAX(created_at) as last_face_added
FROM faces;
```

## ⚠️ Troubleshooting

### Common Issues:

1. **PostgreSQL Connection Error**:
   - Check if PostgreSQL is running
   - Verify credentials in `.env` file
   - Ensure database exists

2. **Face Recognition Issues**:
   - Make sure you have at least 2 people in database
   - Check lighting conditions
   - Ensure face is clearly visible

3. **Import Errors**:
   - Install missing dependencies: `pip install -r requirements.txt`
   - Check Python path

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- OpenCV for face detection
- PostgreSQL for database storage
- PyTorch for neural network embeddings
- RetinaFace for advanced face detection