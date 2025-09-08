# PostgreSQL Database Setup Guide
port 5432
This guide will help you set up PostgreSQL for the face recognition system following the cursor rules for secure database integration.

## Prerequisites

1. **PostgreSQL Installation**
   - Windows: Download from https://www.postgresql.org/download/windows/
   - macOS: `brew install postgresql`
   - Ubuntu/Debian: `sudo apt-get install postgresql postgresql-contrib`

2. **Python Dependencies**
   - The required `psycopg2-binary` package has been added to `requirements.txt`
   - Run: `pip install -r requirements.txt`

## Step-by-Step Setup

### 1. Install PostgreSQL

**Windows:**
1. Download PostgreSQL installer from the official website
2. Run the installer and follow the setup wizard
3. Remember the password you set for the `postgres` user
4. Keep the default port (5432)

**macOS:**
```bash
brew install postgresql
brew services start postgresql
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### 2. Create Database and User

Connect to PostgreSQL as the superuser:

**Windows (using pgAdmin or psql):**
```sql
-- Create database
CREATE DATABASE face_recognition_db;

-- Create user (optional, you can use postgres user)
CREATE USER face_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE face_recognition_db TO face_user;
```

**macOS/Linux:**
```bash
sudo -u postgres psql
```

Then run the SQL commands above.

### 3. Configure Environment Variables

1. Copy the template file:
   ```bash
   cp env_template.txt .env
   ```

2. Edit `.env` with your actual database credentials:
   ```env
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=face_recognition_db
   DB_USER=postgres  # or your custom user
   DB_PASSWORD=your_actual_password
   ```

### 4. Test the Database Connection

Run the test script to verify everything is working:

```bash
python tests/test_database.py
```

You should see output like:
```
==================================================
PostgreSQL Database Test Suite
==================================================
Database Configuration:
  Host: localhost
  Port: 5432
  Database: face_recognition_db
  User: postgres

Testing database connection...
✅ Database connection successful!

Testing face vector storage...
✅ Face vector stored with ID: 1
✅ Face vector retrieved successfully. Similarity: 1.000000

Testing face matching...
ℹ️ No match found (expected for random vector)

Testing get all faces...
✅ Retrieved 1 faces from database
  - test_person (ID: 1, Dimension: 128)

Cleaning up test data...
✅ Test data cleaned up successfully

✅ All tests completed!
```

## Security Features Implemented

The database module follows cursor rules for secure PostgreSQL integration:

1. **Parameterized Queries**: All SQL queries use parameterized statements to prevent SQL injection
2. **Connection Pooling**: Efficient connection management with automatic cleanup
3. **Error Handling**: Graceful handling of database connection errors
4. **Environment Variables**: No hardcoded credentials in the code
5. **Vector Normalization**: Face vectors are normalized before storage for consistent matching
6. **Input Validation**: All inputs are validated before database operations

## Database Schema

The system creates two main tables:

### `faces` table
- `id`: Primary key (auto-increment)
- `name`: Person's name (unique)
- `face_vector`: Binary storage of normalized face vector
- `vector_dimension`: Dimension of the face vector
- `created_at`: Timestamp when record was created
- `updated_at`: Timestamp when record was last updated

### `face_encodings` table
- `id`: Primary key (auto-increment)
- `face_id`: Foreign key to faces table
- `encoding_data`: Additional encoding data
- `encoding_type`: Type of encoding (default: 'numpy_array')
- `created_at`: Timestamp when record was created

## Usage Examples

```python
from backend.database import get_database, close_database
import numpy as np

# Get database instance
db = get_database()

# Store a face vector
face_vector = np.random.rand(128).astype(np.float32)
face_id = db.store_face_vector("John Doe", face_vector)

# Find matching face
query_vector = np.random.rand(128).astype(np.float32)
match = db.find_matching_face(query_vector, threshold=0.8)
if match:
    name, similarity = match
    print(f"Found match: {name} with similarity {similarity}")

# Get all stored faces
faces = db.get_all_faces()
for face in faces:
    print(f"Stored face: {face['name']}")

# Clean up
close_database()
```

## Troubleshooting

### Connection Errors
- Verify PostgreSQL is running: `sudo systemctl status postgresql`
- Check credentials in `.env` file
- Ensure database exists: `psql -U postgres -d face_recognition_db`

### Permission Errors
- Grant proper permissions to your database user
- Check PostgreSQL logs: `sudo tail -f /var/log/postgresql/postgresql-*.log`

### Import Errors
- Ensure `psycopg2-binary` is installed: `pip install psycopg2-binary`
- Check Python path includes the project root

## Next Steps

Once the database is set up and tested:

1. Integrate the database with your face detection model
2. Implement real-time face recognition using the database
3. Add face vector extraction from your PyTorch model
4. Set up webcam integration for live face recognition

The database is now ready to store and match face vectors securely!
