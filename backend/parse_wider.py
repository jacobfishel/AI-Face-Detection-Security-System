# File to parse the wider dataset into the proper format to train data
# Updated to work with KaggleHub downloaded WIDERFace dataset
import json
import tests.parse_wider_test as test
import os
import kagglehub
from pathlib import Path
from config import KAGGLE_DATASET_ID, KAGGLE_DOWNLOAD_PATH


# ---------------- CODE -------------- #
# Functions to work with KaggleHub downloaded WIDERFace dataset

def download_widerface_dataset():
    """Download WIDERFace dataset using KaggleHub"""
    print("Downloading WIDERFace dataset from Kaggle...")
    try:
        if KAGGLE_DOWNLOAD_PATH:
            dataset_path = kagglehub.dataset_download(KAGGLE_DATASET_ID, path=KAGGLE_DOWNLOAD_PATH)
        else:
            dataset_path = kagglehub.dataset_download(KAGGLE_DATASET_ID)
        print(f"Dataset downloaded to: {dataset_path}")
        return Path(dataset_path)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Make sure you have Kaggle API credentials set up.")
        print("Place your kaggle.json file in ~/.kaggle/ or set KAGGLE_CONFIG_DIR environment variable.")
        raise

def parse_annotation_from_kaggle(dataset_path, split, output_path=None):
    """
    Parse annotations from KaggleHub downloaded dataset
    
    Args:
        dataset_path: Path to the downloaded dataset
        split: Dataset split ('train', 'val', 'test')
        output_path: Path to save the parsed annotations JSON file (optional)
    """
    final_data_arr = []
    
    # Set up paths for KaggleHub downloaded dataset
    annotations_file = dataset_path / f"wider_face_split" / f"wider_face_{split}_bbx_gt.txt"
    
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotations_file}")
    
    with open(annotations_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        # Each image starts with a line containing the image path
        img_path = lines[i].strip()
        i += 1
        
        # Next line contains the number of faces
        if i >= len(lines):
            break
        num_faces = int(lines[i].strip())
        i += 1
        
        bboxes = []
        for _ in range(num_faces):
            if i >= len(lines):
                break
            # Parse bbox: x, y, w, h, blur, expression, illumination, invalid, occlusion, pose
            bbox_line = lines[i].strip().split()
            if len(bbox_line) >= 4:
                x, y, w, h = map(int, bbox_line[:4])
                # Only include valid bounding boxes (w > 0, h > 0)
                if w > 0 and h > 0:
                    bboxes.append([x, y, w, h])
            i += 1
        
        if bboxes:  # Only include images with valid bounding boxes
            final_data_arr.append({
                "filename": img_path,
                "bboxes": bboxes
            })
    
    # Only save to file if output_path is provided
    if output_path:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(final_data_arr, f, indent=2)
        
        print(f"Parsed {len(final_data_arr)} images for {split} split and saved to {output_path}")
    else:
        print(f"Parsed {len(final_data_arr)} images for {split} split")
    
    return final_data_arr

def parse_test_annotations_from_kaggle(dataset_path, output_path):
    """
    Parse test annotations from KaggleHub downloaded dataset
    
    Args:
        dataset_path: Path to the downloaded dataset
        output_path: Path to save the parsed test annotations JSON file
    """
    final_data_arr = []
    
    # Set up paths for KaggleHub downloaded dataset
    annotations_file = dataset_path / f"wider_face_split" / f"wider_face_test_filelist.txt"
    
    if not annotations_file.exists():
        raise FileNotFoundError(f"Test annotation file not found: {annotations_file}")
    
    with open(annotations_file, 'r') as f:
        data = f.readlines()

    for line in data:
        # Line contains relative path to test image
        img_path = line.strip()
        final_data_arr.append(img_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(final_data_arr, f, indent=2)
    
    print(f"Parsed {len(final_data_arr)} test images and saved to {output_path}")
    return final_data_arr



if __name__ == '__main__':
    # Example usage with KaggleHub
    try:
        # Download the dataset
        dataset_path = download_widerface_dataset()
        
        # Parse annotations for different splits
        output_dir = Path("parsed_annotations")
        output_dir.mkdir(exist_ok=True)
        
        # Parse train annotations
        train_output = output_dir / "parsed_wider_face_train_bbx_gt.json"
        parse_annotation_from_kaggle(dataset_path, 'train', train_output)
        
        # Parse validation annotations
        val_output = output_dir / "parsed_wider_face_val_bbx_gt.json"
        parse_annotation_from_kaggle(dataset_path, 'val', val_output)
        
        # Parse test annotations
        test_output = output_dir / "parsed_wider_face_test_filelist.json"
        parse_test_annotations_from_kaggle(dataset_path, test_output)
        
        print("All annotations parsed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have Kaggle API credentials set up.")

    

