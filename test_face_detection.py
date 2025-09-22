"""
Test script for face detection functionality
"""

import cv2
import numpy as np
from retinaface_detector import RetinaFaceDetector

def test_face_detection():
    """
    Test the face detection functionality
    """
    print("Testing face detection...")
    
    # Initialize detector
    detector = RetinaFaceDetector(confidence_threshold=0.5)
    
    # Create a test image (you can replace this with loading an actual image)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some text to the test image
    cv2.putText(test_image, "Test Image - No Faces", (50, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Test face detection
    faces = detector.detect_faces(test_image)
    print(f"Detected {len(faces)} faces in test image")
    
    # Test drawing boxes
    result_image = detector.draw_boxes(test_image, faces)
    
    # Save test result
    cv2.imwrite("test_result.jpg", result_image)
    print("Test result saved as 'test_result.jpg'")
    
    print("Face detection test completed successfully!")

if __name__ == "__main__":
    test_face_detection()
