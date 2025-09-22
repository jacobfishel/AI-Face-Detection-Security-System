# Face Detection Demo

This demo combines your existing PyTorch model with real-time face detection using OpenCV and RetinaFace.

## Files Created

- `face_detection_demo.py` - Main demo file containing:
  - Your existing PyTorch `WiderFaceNN` model (unchanged)
  - `RetinaFaceDetector` class for face detection
  - Webcam loop for real-time face detection
- `retinaface_detector.py` - Standalone RetinaFace detector class
- `download_retinaface_model.py` - Script to set up model directory
- `test_face_detection.py` - Test script for face detection

## How to Run

### Basic Usage (OpenCV Haar Cascade)
```bash
python face_detection_demo.py
```

This will:
1. Open your webcam
2. Detect faces using OpenCV's built-in Haar Cascade classifier
3. Draw bounding boxes around detected faces
4. Display real-time video with face detection overlay
5. Press 'q' to quit

### With RetinaFace ONNX Model (Optional)

To use the more accurate RetinaFace model:

1. Install ONNX Runtime:
   ```bash
   pip install onnxruntime
   ```

2. Download a RetinaFace ONNX model and place it in `models/retinaface.onnx`

3. Run the demo:
   ```bash
   python face_detection_demo.py
   ```

## Features

- **Modular Design**: The `RetinaFaceDetector` class is completely separate from your PyTorch model
- **Fallback Detection**: If RetinaFace model is not available, automatically falls back to OpenCV Haar Cascade
- **Real-time Performance**: Optimized for webcam feed with configurable confidence thresholds
- **Easy Integration**: Your existing PyTorch model code remains completely unchanged

## RetinaFaceDetector Class Methods

- `__init__(model_path=None, confidence_threshold=0.5)` - Initialize detector
- `detect_faces(frame)` - Detect faces in a frame, returns list of (x1, y1, x2, y2, confidence)
- `draw_boxes(frame, faces)` - Draw bounding boxes on frame

## Model Sources

For RetinaFace ONNX models, check:
- [InsightFace](https://github.com/deepinsight/insightface)
- [PyTorch RetinaFace](https://github.com/biubug6/Pytorch_Retinaface)
- [ONNX Model Zoo](https://github.com/onnx/models)

## Requirements

- OpenCV (already installed)
- NumPy (already installed)
- PyTorch (already installed)
- ONNX Runtime (optional, for RetinaFace models)

## Your PyTorch Model

Your existing `WiderFaceNN` model is preserved exactly as-is in the demo file. It's completely separate from the face detection functionality and won't interfere with your existing code.
