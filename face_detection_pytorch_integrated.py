#!/usr/bin/env python3
"""
Face Detection Demo with PyTorch Model Integration + PostgreSQL Database
Uses your existing PyTorch model for face embeddings instead of face_recognition library
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import sys
from typing import List, Tuple, Optional
from facenet_pytorch import InceptionResnetV1

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import PostgreSQL database functions
from backend.database import get_database, close_database

# =============================================================================
# PYTORCH MODEL FOR FACE EMBEDDINGS (SIMPLIFIED)
# =============================================================================

class FaceNetEmbeddingModel:
    """
    FaceNet-based face embedding extraction using pretrained model
    This provides much better discriminative embeddings than untrained models
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load pretrained FaceNet model (VGG-Face2 trained)
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.embedding_dim = 512  # FaceNet produces 512-dimensional embeddings
        print(f"FaceNet model loaded on {self.device}")
    
    def forward(self, x):
        """
        Extract embeddings from face images
        Args:
            x: Tensor of shape (batch_size, 3, 160, 160) with face images
        Returns:
            Normalized embeddings of shape (batch_size, 512)
        """
        with torch.no_grad():
            embeddings = self.model(x)
            # L2 normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

# =============================================================================
# FACE DETECTOR CLASS
# =============================================================================

class OpenCVFaceDetector:
    """
    Face detector using OpenCV Haar Cascade
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade classifier")
        
        print("OpenCV face detector initialized successfully")
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces in frame"""
        faces = []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_rects = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in face_rects:
                confidence = 0.8  # Default confidence for Haar cascade
                faces.append((x, y, x + w, y + h, confidence))
            
        except Exception as e:
            print(f"Error in face detection: {e}")
        
        return faces

# =============================================================================
# PYTORCH + POSTGRESQL FACE RECOGNITION
# =============================================================================

class PyTorchPostgreSQLFaceRecognition:
    """
    Face recognition system using PyTorch model + PostgreSQL database
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize FaceNet model and PostgreSQL connection"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize FaceNet model (pretrained)
        self.model = FaceNetEmbeddingModel()
        
        # Initialize PostgreSQL database
        self.db = get_database()
        print("FaceNet + PostgreSQL face recognition system initialized")
    
    def get_embedding_from_face(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding using FaceNet model
        
        Args:
            face_crop: Cropped face image as numpy array (BGR)
            
        Returns:
            Face embedding vector or None if extraction fails
        """
        try:
            # Resize to FaceNet standard size (160x160)
            face_resized = cv2.resize(face_crop, (160, 160))
            
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1] and convert to tensor
            face_normalized = face_rgb.astype(np.float32) / 255.0
            face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1).unsqueeze(0)
            face_tensor = face_tensor.to(self.device)
            
            # Extract embedding using FaceNet
            embedding = self.model.forward(face_tensor)
            embedding = embedding.cpu().numpy().flatten()
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"Failed to extract embedding: {e}")
            return None
    
    def find_matching_face(self, embedding: np.ndarray, threshold: float = 0.8, confidence_threshold: float = 0.95) -> Optional[Tuple[str, float]]:
        """
        Find matching face in PostgreSQL database with confidence check
        
        Args:
            embedding: Face embedding to match
            threshold: Minimum similarity threshold for database search
            confidence_threshold: High threshold for confident matches (above this = known person, below = stranger)
        
        Returns:
            Tuple of (name, similarity) if confident match found, None otherwise
        """
        try:
            match = self.db.find_matching_face(embedding, threshold)
            if match is not None:
                name, similarity = match
                # Only return match if similarity is above confidence threshold
                if similarity >= confidence_threshold:
                    return match
                else:
                    # Similarity too low - treat as stranger
                    return None
            return None
        except Exception as e:
            print(f"Failed to find matching face: {e}")
            return None
    
    def store_new_face(self, name: str, embedding: np.ndarray) -> Optional[int]:
        """Store new face in PostgreSQL database"""
        try:
            face_id = self.db.store_face_vector(name, embedding)
            print(f"Stored face '{name}' with ID: {face_id}")
            return face_id
        except Exception as e:
            print(f"Failed to store face: {e}")
            return None
    
    def prompt_and_store_unknown_face(self, face_crop: np.ndarray) -> Tuple[str, bool]:
        """Prompt user to label unknown face and store it"""
        try:
            embedding = self.get_embedding_from_face(face_crop)
            if embedding is None:
                return "", False
            
            name = input("No match found. Enter name to add (or leave blank to skip): ").strip()
            
            if name:
                face_id = self.store_new_face(name, embedding)
                return name, face_id is not None
            else:
                return "", False
                
        except Exception as e:
            print(f"Error in prompt and store: {e}")
            return "", False
    
    def close(self):
        """Close database connection"""
        close_database()

# =============================================================================
# MAIN WEBCAM LOOP
# =============================================================================

def main():
    """Main function to run PyTorch + PostgreSQL face detection"""
    print("Initializing PyTorch + PostgreSQL Face Recognition...")
    
    try:
        # Initialize face detector
        detector = OpenCVFaceDetector(confidence_threshold=0.5)
        
        # Initialize PyTorch + PostgreSQL face recognition
        face_recognition = PyTorchPostgreSQLFaceRecognition()
        
        # Initialize webcam
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Webcam opened successfully.")
        print("Controls:")
        print("  'q' - quit")
        print("  'a' - add unknown face to database")
        print("  's' - show all faces in database")
        print("  't' - test embedding extraction")
        
        # Set webcam properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break
            
            # Detect faces
            faces = detector.detect_faces(frame)
            result_frame = frame.copy()
            
            # Process each detected face
            for (x1, y1, x2, y2, confidence) in faces:
                # Ensure coordinates are within frame bounds
                x1c = max(0, x1)
                y1c = max(0, y1)
                x2c = min(frame.shape[1], x2)
                y2c = min(frame.shape[0], y2)
                
                if x2c <= x1c or y2c <= y1c:
                    continue
                
                # Crop face
                face_crop = frame[y1c:y2c, x1c:x2c]
                
                # Initialize label
                label_text = f"Face: {confidence:.2f}"
                label_color = (0, 255, 0)  # Green for unknown
                
                try:
                    # Extract embedding using PyTorch model
                    embedding = face_recognition.get_embedding_from_face(face_crop)
                    
                    if embedding is not None:
                        # Try to find match in PostgreSQL database with confidence threshold
                        match = face_recognition.find_matching_face(embedding, threshold=0.8, confidence_threshold=0.6)
                        
                        if match is not None:
                            name, score = match
                            label_text = f"{name} ({score:.3f})"
                            label_color = (0, 255, 0)  # Green for confident match
                        else:
                            # Check if there's a low-confidence match to show "Stranger"
                            low_match = face_recognition.db.find_matching_face(embedding, threshold=0.8)
                            if low_match is not None:
                                _, low_score = low_match
                                if low_score < 0.6:
                                    label_text = f"Stranger ({low_score:.3f})"
                                    label_color = (0, 165, 255)  # Orange for stranger
                                else:
                                    label_text = "Unknown (press 'a' to add)"
                                    label_color = (0, 255, 255)  # Yellow for unknown
                            else:
                                label_text = "Unknown (press 'a' to add)"
                                label_color = (0, 255, 255)  # Yellow for unknown
                    else:
                        label_text = "Embedding failed"
                        label_color = (0, 0, 255)  # Red for error
                        
                except Exception as e:
                    print(f"Error processing face: {e}")
                    label_text = "Processing error"
                    label_color = (0, 0, 255)
                
                # Draw bounding box and label
                cv2.rectangle(result_frame, (x1c, y1c), (x2c, y2c), label_color, 2)
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw label background
                cv2.rectangle(result_frame, (x1c, y1c - label_size[1] - 10), 
                             (x1c + label_size[0], y1c), label_color, -1)
                
                # Draw label text
                cv2.putText(result_frame, label_text, (x1c, y1c - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Add info text
            info_text = f"Faces: {len(faces)} | q: quit | a: add | s: show | t: test"
            cv2.putText(result_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('PyTorch + PostgreSQL Face Recognition', result_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('a') and len(faces) > 0:
                # Add the first detected face to database
                x1, y1, x2, y2, _ = faces[0]
                x1c = max(0, x1)
                y1c = max(0, y1)
                x2c = min(frame.shape[1], x2)
                y2c = min(frame.shape[0], y2)
                
                if x2c > x1c and y2c > y1c:
                    face_crop = frame[y1c:y2c, x1c:x2c]
                    name, success = face_recognition.prompt_and_store_unknown_face(face_crop)
                    if success:
                        print(f"✅ Successfully added '{name}' to PostgreSQL database!")
                    else:
                        print(f"❌ Failed to add '{name}' to database")
            elif key == ord('s'):
                # Show all faces in database
                try:
                    all_faces = face_recognition.db.get_all_faces()
                    print(f"\n📊 Faces in PostgreSQL database ({len(all_faces)} total):")
                    for face in all_faces:
                        print(f"  - {face['name']} (ID: {face['id']}, Dimension: {face['vector_dimension']})")
                    print()
                except Exception as e:
                    print(f"Error retrieving faces: {e}")
            elif key == ord('t'):
                # Test embedding extraction
                if len(faces) > 0:
                    x1, y1, x2, y2, _ = faces[0]
                    x1c = max(0, x1)
                    y1c = max(0, y1)
                    x2c = min(frame.shape[1], x2)
                    y2c = min(frame.shape[0], y2)
                    
                    if x2c > x1c and y2c > y1c:
                        face_crop = frame[y1c:y2c, x1c:x2c]
                        embedding = face_recognition.get_embedding_from_face(face_crop)
                        if embedding is not None:
                            print(f"✅ Embedding extracted: shape={embedding.shape}, norm={np.linalg.norm(embedding):.3f}")
                        else:
                            print("❌ Failed to extract embedding")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        face_recognition.close()
        print("Webcam closed. PostgreSQL connection closed. Demo finished.")

if __name__ == "__main__":
    main()
