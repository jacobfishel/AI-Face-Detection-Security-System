# KaggleHub WIDERFace Integration

This document describes the integration of KaggleHub for downloading and using the WIDERFace dataset instead of manually downloaded files.

## Overview

The project has been updated to use KaggleHub API to automatically download and manage the WIDERFace dataset. This eliminates the need for manual dataset downloads and provides a more streamlined workflow.

## Changes Made

### 1. New Files
- `data/kaggle_dataset.py` - New dataset loader using KaggleHub
- `test_kaggle_integration.py` - Test script to verify integration
- `KAGGLEHUB_INTEGRATION.md` - This documentation file

### 2. Modified Files
- `requirements.txt` - Added kagglehub dependency
- `config.py` - Removed local file paths, added KaggleHub configuration
- `backend/train.py` - Updated to use new KaggleHub dataset loader
- `backend/parse_wider.py` - Updated to work with KaggleHub downloaded data, made output_path optional
- `env_template_secure.txt` - Updated with KaggleHub configuration

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Kaggle API Credentials

1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
2. Click "Create New Token" to download your `kaggle.json` file
3. Place the `kaggle.json` file in one of these locations:
   - **Windows**: `C:\Users\<Your Username>\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`
4. Alternatively, set the `KAGGLE_CONFIG_DIR` environment variable to point to the directory containing `kaggle.json`

### 3. Optional Configuration

Create a `.env` file based on `env_template_secure.txt` and optionally set:
```bash
KAGGLE_DOWNLOAD_PATH=/path/to/custom/download/location
```

If not set, the dataset will be downloaded to the default cache location.

## Usage

### Basic Usage

The new `KaggleWiderFaceDataset` class automatically handles dataset downloading and parsing:

```python
from data.kaggle_dataset import KaggleWiderFaceDataset
from torch.utils.data import DataLoader

# Initialize dataset (will download automatically if not present)
train_dataset = KaggleWiderFaceDataset('train', download=True)
val_dataset = KaggleWiderFaceDataset('val', download=True)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=train_dataset.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=val_dataset.collate_fn)
```

### Training

The training script has been updated to use the new dataset loader:

```bash
python backend/train.py
```

### Testing Integration

Run the test script to verify everything is working:

```bash
python test_kaggle_integration.py
```

### Manual Dataset Parsing

If you need to manually parse annotations from the downloaded dataset:

```python
from backend.parse_wider import download_widerface_dataset, parse_annotation_from_kaggle

# Download dataset
dataset_path = download_widerface_dataset()

# Parse annotations
parse_annotation_from_kaggle(dataset_path, 'train', 'parsed_train.json')
parse_annotation_from_kaggle(dataset_path, 'val', 'parsed_val.json')
```

## Dataset Structure

The KaggleHub downloaded dataset follows this structure:
```
~/.cache/kagglehub/datasets/xiaofeng/wider-face/
├── WIDER_train/
│   └── images/
├── WIDER_val/
│   └── images/
├── WIDER_test/
│   └── images/
└── wider_face_split/
    ├── wider_face_train_bbx_gt.txt
    ├── wider_face_val_bbx_gt.txt
    └── wider_face_test_filelist.txt
```

## Key Features

1. **Automatic Download**: Dataset is downloaded automatically on first use
2. **Caching**: Downloaded dataset is cached locally for future use
3. **Error Handling**: Comprehensive error handling for network issues and missing credentials
4. **Backward Compatibility**: The new dataset class maintains the same interface as the original
5. **Flexible Configuration**: Support for custom download paths via environment variables
6. **Code Reuse**: Uses your existing parsing logic from `parse_wider.py` instead of duplicating code

## Troubleshooting

### Common Issues

1. **"Error downloading dataset"**
   - Check your internet connection
   - Verify Kaggle API credentials are set up correctly
   - Ensure you have accepted the dataset terms on Kaggle

2. **"Import kagglehub could not be resolved"**
   - Run `pip install kagglehub` to install the package

3. **"Authentication failed"**
   - Check that your `kaggle.json` file is in the correct location
   - Verify the API token is valid and not expired

4. **"Dataset not found"**
   - Make sure you have access to the WIDERFace dataset on Kaggle
   - Check that the dataset identifier is correct

### Getting Help

- Check the [KaggleHub documentation](https://github.com/Kaggle/kagglehub)
- Verify your [Kaggle API setup](https://www.kaggle.com/docs/api)
- Check the [WIDERFace dataset page](https://www.kaggle.com/datasets/xiaofeng/wider-face)

## Migration from Local Dataset

If you were previously using a locally downloaded WIDERFace dataset:

1. The new system will automatically download the dataset from Kaggle
2. Your existing training code should work without changes
3. You can remove the old local dataset files if desired
4. Update your environment variables to remove the old dataset paths

## Performance Notes

- First-time dataset download may take some time depending on your internet connection
- Subsequent runs will use the cached dataset for faster loading
- The dataset is automatically validated during download
