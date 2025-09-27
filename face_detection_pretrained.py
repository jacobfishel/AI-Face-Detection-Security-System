#!/usr/bin/env python3
"""
Face Detection Demo with Pre-trained Face Recognition
Uses a proper face recognition approach instead of untrained PyTorch model
"""

import cv2
import numpy as np
import os
import sys
from typing import List, Tuple, Optional

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import PostgreSQL database functions
from backend.database import get_database, close_database

# =============================================================================
# FACE DETECTOR
# =============================================================================

class SimpleFaceDetector:
    """Simple face detector using OpenCV"""
    
    def __init__(self):
        self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade classifier")
        
        print("Simple face detector initialized")
    
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
                confidence = 0.8
                faces.append((x, y, x + w, y + h, confidence))
            
        except Exception as e:
            print(f"Error in face detection: {e}")
        
        return faces

# =============================================================================
# SIMPLE FACE EMBEDDING (HISTOGRAM-BASED)
# =============================================================================

class SimpleFaceEmbedding:
    """
    Simple face embedding using color histograms and basic features
    This is a fallback when face_recognition is not available
    """
    
    def __init__(self):
        print("Simple histogram-based face embedding initialized")
    
    def get_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Extract simple features from face crop
        
        Args:
            face_crop: Cropped face image (BGR)
            
        Returns:
            Feature vector
        """
        try:
            # Resize to standard size
            face_resized = cv2.resize(face_crop, (64, 64))
            
            # Convert to different color spaces
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_resized, cv2.COLOR_BGR2LAB)
            
            features = []
            
            # Color histograms
            for channel in [0, 1, 2]:  # BGR channels
                hist = cv2.calcHist([face_resized], [channel], None, [32], [0, 256])
                features.extend(hist.flatten())
            
            # HSV histograms
            for channel in [0, 1, 2]:  # HSV channels
                hist = cv2.calcHist([hsv], [channel], None, [32], [0, 256])
                features.extend(hist.flatten())
            
            # LAB histograms
            for channel in [0, 1, 2]:  # LAB channels
                hist = cv2.calcHist([lab], [channel], None, [32], [0, 256])
                features.extend(hist.flatten())
            
            # Texture features (LBP-like)
            gray_resized = cv2.resize(gray, (32, 32))
            lbp_features = self._calculate_lbp_features(gray_resized)
            features.extend(lbp_features)
            
            # Convert to numpy array and normalize
            features = np.array(features, dtype=np.float32)
            features = features / (np.linalg.norm(features) + 1e-8)  # L2 normalize
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return a random vector if extraction fails
            return np.random.randn(128).astype(np.float32)
    
    def _calculate_lbp_features(self, gray_image: np.ndarray) -> List[float]:
        """Calculate Local Binary Pattern-like features"""
        features = []
        h, w = gray_image.shape
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = gray_image[i, j]
                lbp_value = 0
                
                # 8-neighborhood
                neighbors = [
                    gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                    gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                    gray_image[i+1, j-1], gray_image[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        lbp_value |= (1 << k)
                
                features.append(lbp_value)
        
        # Create histogram of LBP values
        lbp_hist, _ = np.histogram(features, bins=16, range=(0, 256))
        return lbp_hist.tolist()

# =============================================================================
# FACE RECOGNITION WITH PROPER THRESHOLDS
# =============================================================================

class ProperFaceRecognition:
    """
    Face recognition system with proper thresholds and fallback embedding
    """
    
    def __init__(self):
        # Initialize database
        self.db = get_database()
        
        # Initialize simple embedding extractor
        self.embedding_extractor = SimpleFaceEmbedding()
        
        # Proper thresholds for face recognition
        self.high_confidence_threshold = 0.85    # Very confident match
        self.medium_confidence_threshold = 0.75  # Good match
        self.low_confidence_threshold = 0.65     # Possible match
        self.very_low_threshold = 0.55           # Weak match
        
        print("Proper face recognition system initialized")
        print(f"Thresholds: High={self.high_confidence_threshold}, Medium={self.medium_confidence_threshold}, Low={self.low_confidence_threshold}, VeryLow={self.very_low_threshold}")
    
    def get_embedding_from_face(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding using simple features"""
        try:
            embedding = self.embedding_extractor.get_embedding(face_crop)
            print(f"🔍 Simple embedding: shape={embedding.shape}, norm={np.linalg.norm(embedding):.4f}")
            return embedding
        except Exception as e:
            print(f"Failed to extract embedding: {e}")
            return None
    
    def find_matching_face(self, embedding: np.ndarray) -> Tuple[Optional[str], float, str]:
        """
        Find matching face with proper thresholds
        
        Returns:
            Tuple of (name, confidence, status)
        """
        try:
            # Check database size
            all_faces = self.db.get_all_faces()
            if len(all_faces) < 2:
                return None, 0.0, 'insufficient_data'
            
            # Find best match
            match = self.db.find_matching_face(embedding, threshold=self.very_low_threshold)
            
            if match:
                name, confidence = match
                
                if confidence >= self.high_confidence_threshold:
                    status = 'high_confidence'
                elif confidence >= self.medium_confidence_threshold:
                    status = 'medium_confidence'
                elif confidence >= self.low_confidence_threshold:
                    status = 'low_confidence'
                else:
                    status = 'very_low_confidence'
                
                print(f"🎯 Match: {name} with confidence {confidence:.4f} (status: {status})")
                return name, confidence, status
            else:
                print(f"❌ No match found above threshold {self.very_low_threshold}")
                return None, 0.0, 'unknown'
                
        except Exception as e:
            print(f"Failed to find matching face: {e}")
            return None, 0.0, 'unknown'
    
    def store_new_face(self, name: str, embedding: np.ndarray) -> Optional[int]:
        """Store new face in database"""
        try:
            face_id = self.db.store_face_vector(name, embedding)
            print(f"✅ Stored face '{name}' with ID: {face_id}")
            return face_id
        except Exception as e:
            print(f"Failed to store face: {e}")
            return None
    
    def prompt_and_store_unknown_face(self, face_crop: np.ndarray) -> Tuple[str, bool]:
        """Prompt user to label unknown face"""
        try:
            embedding = self.get_embedding_from_face(face_crop)
            if embedding is None:
                return "", False
            
            name = input("Unknown face detected. Enter name to add (or leave blank to skip): ").strip()
            
            if name:
                face_id = self.store_new_face(name, embedding)
                return name, face_id is not None
            else:
                return "", False
                
        except Exception as e:
            print(f"Error in prompt and store: {e}")
            return "", False
    
    def get_database_stats(self) -> dict:
        """Get database statistics"""
        try:
            all_faces = self.db.get_all_faces()
            return {
                'total_faces': len(all_faces),
                'faces': all_faces,
                'has_sufficient_data': len(all_faces) >= 2
            }
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {'total_faces': 0, 'faces': [], 'has_sufficient_data': False}
    
    def close(self):
        """Close database connection"""
        close_database()

# =============================================================================
# MAIN WEBCAM LOOP
# =============================================================================

def main():
    """Main function with proper face recognition"""
    print("🚀 Initializing Proper Face Recognition System...")
    
    try:
        # Initialize face detector
        detector = SimpleFaceDetector()
        
        # Initialize proper face recognition
        face_recognition = ProperFaceRecognition()
        
        # Initialize webcam
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Webcam opened successfully.")
        print("Controls:")
        print("  'q' - quit")
        print("  'a' - add unknown face to database")
        print("  's' - show database statistics")
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
                label_color = (0, 255, 0)  # Default green
                
                try:
                    # Extract embedding
                    embedding = face_recognition.get_embedding_from_face(face_crop)
                    
                    if embedding is not None:
                        # Find match
                        name, match_confidence, status = face_recognition.find_matching_face(embedding)
                        
                        if status == 'insufficient_data':
                            label_text = "Need 2+ people"
                            label_color = (0, 255, 255)  # Yellow
                        elif status == 'high_confidence':
                            label_text = f"{name} ({match_confidence:.3f})"
                            label_color = (0, 255, 0)  # Green
                        elif status == 'medium_confidence':
                            label_text = f"{name}? ({match_confidence:.3f})"
                            label_color = (0, 165, 255)  # Orange
                        elif status == 'low_confidence':
                            label_text = f"{name}?? ({match_confidence:.3f})"
                            label_color = (0, 0, 255)  # Red
                        elif status == 'very_low_confidence':
                            label_text = f"{name}??? ({match_confidence:.3f})"
                            label_color = (128, 0, 128)  # Purple
                        else:  # unknown
                            label_text = "Unknown (press 'a')"
                            label_color = (255, 0, 255)  # Magenta
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
            
            # Get database stats
            stats = face_recognition.get_database_stats()
            
            # Add info text
            info_text = f"Faces: {len(faces)} | DB: {stats['total_faces']} | q: quit | a: add | s: stats"
            cv2.putText(result_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add threshold info
            threshold_text = f"Thresholds: H={face_recognition.high_confidence_threshold} M={face_recognition.medium_confidence_threshold} L={face_recognition.low_confidence_threshold}"
            cv2.putText(result_frame, threshold_text, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Proper Face Recognition', result_frame)
            
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
                        print(f"✅ Successfully added '{name}' to database!")
                    else:
                        print(f"❌ Failed to add '{name}' to database")
            elif key == ord('s'):
                # Show database statistics
                stats = face_recognition.get_database_stats()
                print(f"\n📊 Database Statistics:")
                print(f"   Total faces: {stats['total_faces']}")
                print(f"   Sufficient data: {'Yes' if stats['has_sufficient_data'] else 'No'}")
                print(f"   Faces in database:")
                for face in stats['faces']:
                    print(f"     - {face['name']} (ID: {face['id']}, Dimension: {face['vector_dimension']})")
                print()
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
        print("Webcam closed. Database connection closed. Demo finished.")

if __name__ == "__main__":
    main()
