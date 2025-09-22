"""
Script to download a pre-trained RetinaFace ONNX model
"""

import os
import urllib.request
import zipfile
import shutil

def download_retinaface_model():
    """
    Download a pre-trained RetinaFace ONNX model
    """
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # For demo purposes, we'll create a simple model info file
    # In a real implementation, you would download from a proper source
    model_info_path = os.path.join(model_dir, "model_info.txt")
    
    with open(model_info_path, "w") as f:
        f.write("RetinaFace ONNX Model Information\n")
        f.write("================================\n\n")
        f.write("To use RetinaFace detection, you need to download a pre-trained ONNX model.\n\n")
        f.write("Recommended sources:\n")
        f.write("1. InsightFace: https://github.com/deepinsight/insightface\n")
        f.write("2. PyTorch RetinaFace: https://github.com/biubug6/Pytorch_Retinaface\n")
        f.write("3. ONNX Model Zoo: https://github.com/onnx/models\n\n")
        f.write("Model requirements:\n")
        f.write("- Input size: 640x640\n")
        f.write("- Input format: RGB, normalized to [0,1]\n")
        f.write("- Output: bounding boxes, confidence scores, landmarks\n\n")
        f.write("Place the downloaded .onnx file as 'retinaface.onnx' in this models/ directory.\n")
    
    print(f"Model information saved to: {model_info_path}")
    print("Please download a RetinaFace ONNX model and place it in the models/ directory.")
    print("The demo will work with OpenCV Haar Cascade as a fallback.")

if __name__ == "__main__":
    download_retinaface_model()
