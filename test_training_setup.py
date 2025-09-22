#!/usr/bin/env python3
"""
Test script to verify training setup
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import kagglehub
        print("✓ KaggleHub imported successfully")
    except ImportError as e:
        print(f"✗ KaggleHub import failed: {e}")
        return False
    
    try:
        from data.kaggle_dataset import KaggleWiderFaceDataset
        print("✓ KaggleWiderFaceDataset imported successfully")
    except ImportError as e:
        print(f"✗ KaggleWiderFaceDataset import failed: {e}")
        return False
    
    try:
        from backend.model import WiderFaceNN
        print("✓ WiderFaceNN model imported successfully")
    except ImportError as e:
        print(f"✗ WiderFaceNN import failed: {e}")
        return False
    
    try:
        from config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, DEVICE
        print(f"✓ Config loaded - Device: {DEVICE}, Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")
    except ImportError as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test that the model can be created"""
    print("\nTesting model creation...")
    
    try:
        from backend.model import WiderFaceNN
        import torch
        
        model = WiderFaceNN()
        print(f"✓ Model created successfully")
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            pred_boxes, conf_scores = model(dummy_input)
            print(f"✓ Forward pass successful - Boxes shape: {pred_boxes.shape}, Conf shape: {conf_scores.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_dataset_creation():
    """Test that the dataset can be created (without downloading)"""
    print("\nTesting dataset creation...")
    
    try:
        from data.kaggle_dataset import KaggleWiderFaceDataset
        
        # Try to create dataset without downloading first
        print("Attempting to create dataset without download...")
        dataset = KaggleWiderFaceDataset('train', download=False)
        print(f"✓ Dataset created successfully, length: {len(dataset)}")
        return True
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        print("This might be expected if the dataset hasn't been downloaded yet.")
        return False

def test_kaggle_credentials():
    """Test Kaggle API credentials"""
    print("\nTesting Kaggle API credentials...")
    
    try:
        import kagglehub
        
        # Try to list datasets (this requires authentication)
        print("Testing Kaggle API access...")
        # This will fail if credentials aren't set up
        kagglehub.dataset_list()
        print("✓ Kaggle API credentials are working")
        return True
    except Exception as e:
        print(f"✗ Kaggle API test failed: {e}")
        print("You may need to set up Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Create API token and download kaggle.json")
        print("3. Place kaggle.json in ~/.kaggle/ directory")
        return False

if __name__ == '__main__':
    print("WIDERFace Training Setup Test")
    print("=" * 40)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test model creation
    if not test_model_creation():
        all_tests_passed = False
    
    # Test dataset creation
    if not test_dataset_creation():
        all_tests_passed = False
    
    # Test Kaggle credentials
    if not test_kaggle_credentials():
        all_tests_passed = False
    
    print("\n" + "=" * 40)
    if all_tests_passed:
        print("✅ All tests passed! Training setup is ready.")
    else:
        print("❌ Some tests failed. Please fix the issues above before training.")

