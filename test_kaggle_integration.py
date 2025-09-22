#!/usr/bin/env python3
"""
Test script to verify KaggleHub WIDERFace integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.kaggle_dataset import KaggleWiderFaceDataset
import torch
from torch.utils.data import DataLoader

def test_kaggle_integration():
    """Test the KaggleHub WIDERFace dataset integration"""
    print("Testing KaggleHub WIDERFace integration...")
    
    try:
        # Test dataset initialization (without downloading first)
        print("1. Testing dataset initialization...")
        dataset = KaggleWiderFaceDataset('train', download=False)
        print(f"   ✓ Dataset initialized successfully")
        print(f"   ✓ Dataset path: {dataset.dataset_path}")
        
        # Test dataset length
        print("2. Testing dataset length...")
        dataset_len = len(dataset)
        print(f"   ✓ Dataset length: {dataset_len}")
        
        # Test getting a single item
        print("3. Testing single item retrieval...")
        if dataset_len > 0:
            image, target = dataset[0]
            print(f"   ✓ Image shape: {image.shape}")
            print(f"   ✓ Target keys: {list(target.keys())}")
            print(f"   ✓ Number of boxes: {len(target['boxes'])}")
        else:
            print("   ⚠ Dataset is empty")
        
        # Test DataLoader
        print("4. Testing DataLoader...")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn)
        print(f"   ✓ DataLoader created successfully")
        
        # Test getting a batch
        print("5. Testing batch retrieval...")
        for i, (images, targets) in enumerate(dataloader):
            print(f"   ✓ Batch {i+1}: images shape {images.shape}, {len(targets)} targets")
            if i >= 2:  # Only test first few batches
                break
        
        print("\n✅ All tests passed! KaggleHub integration is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have Kaggle API credentials set up")
        print("2. Place your kaggle.json file in ~/.kaggle/ directory")
        print("3. Or set KAGGLE_CONFIG_DIR environment variable")
        print("4. Make sure you have internet connection to download the dataset")
        return False

def test_download():
    """Test downloading the dataset"""
    print("\nTesting dataset download...")
    try:
        dataset = KaggleWiderFaceDataset('train', download=True)
        print("✅ Dataset download test passed!")
        return True
    except Exception as e:
        print(f"❌ Dataset download test failed: {e}")
        return False

if __name__ == '__main__':
    print("KaggleHub WIDERFace Integration Test")
    print("=" * 50)
    
    # Test download first
    download_success = test_download()
    
    if download_success:
        # Test integration
        integration_success = test_kaggle_integration()
        
        if integration_success:
            print("\n🎉 All tests completed successfully!")
            print("Your KaggleHub WIDERFace integration is ready to use.")
        else:
            print("\n⚠️  Integration tests failed. Please check the error messages above.")
    else:
        print("\n⚠️  Download test failed. Please check your Kaggle API setup.")
