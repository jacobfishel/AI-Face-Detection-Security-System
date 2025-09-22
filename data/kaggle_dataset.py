# Custom PyTorch Dataset class that loads WIDERFace dataset from KaggleHub

import torch
import os
import json
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import kagglehub
from pathlib import Path
import sys

# Add backend directory to path to import parse_wider functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from parse_wider import parse_annotation_from_kaggle


class KaggleWiderFaceDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', transform=None, download=True):
        """
        Initialize the Kaggle WIDERFace dataset
        
        Args:
            split (str): Dataset split ('train', 'val', 'test')
            transform: Albumentations transform pipeline
            download (bool): Whether to download the dataset if not present
        """
        self.split = split
        self.download = download
        
        # Download dataset using KaggleHub if needed
        if download:
            self.dataset_path = self._download_dataset()
        else:
            # Assume dataset is already downloaded
            self.dataset_path = Path.home() / '.cache' / 'kagglehub' / 'datasets' / 'xiaofeng' / 'wider-face'
        
        # Set up paths
        self.image_dir = self.dataset_path / f"WIDER_{split}" / "images"
        self.annotations_file = self.dataset_path / f"wider_face_split" / f"wider_face_{split}_bbx_gt.txt"
        
        # Parse annotations
        self.data = self._parse_annotations()
        
        # Set up transforms
        if transform is None:
            if split == 'train':
                self.transform = A.Compose([
                    A.Resize(512, 512),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.3))
            else:
                self.transform = A.Compose([
                    A.Resize(512, 512),
                    A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.3))
        else:
            self.transform = transform

    def _download_dataset(self):
        """Download the WIDERFace dataset using KaggleHub"""
        print("Downloading WIDERFace dataset from Kaggle...")
        try:
            dataset_path = kagglehub.dataset_download('xiaofeng/wider-face')
            print(f"Dataset downloaded to: {dataset_path}")
            return Path(dataset_path)
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Make sure you have Kaggle API credentials set up.")
            print("Place your kaggle.json file in ~/.kaggle/ or set KAGGLE_CONFIG_DIR environment variable.")
            raise

    def _parse_annotations(self):
        """Parse the WIDERFace annotation file using the existing parse_wider function"""
        # Use the existing parsing function from parse_wider.py
        # Pass None as output_path to avoid creating a file
        data = parse_annotation_from_kaggle(self.dataset_path, self.split, output_path=None)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        img_path = self.image_dir / entry["filename"]
        bboxes_raw = entry["bboxes"]

        # Convert to Pascal VOC format (x_min, y_min, x_max, y_max)
        boxes = []
        for box in bboxes_raw:
            x, y, w, h = box[:4]
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])

        boxes = torch.tensor(boxes, dtype=torch.float32)

        if boxes.numel() == 0:
            print(f"No valid bboxes for image {idx}, skipping")
            # Return a dummy sample instead of recursive call to avoid infinite recursion
            dummy_image = torch.zeros((3, 512, 512))
            dummy_target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([idx]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64)
            }
            return dummy_image, dummy_target

        # Filter out invalid bboxes
        boxes = boxes[boxes[:, 2] > boxes[:, 0]]    # x_max > x_min
        boxes = boxes[boxes[:, 3] > boxes[:, 1]]    # y_max > y_min
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        # If all boxes got filtered out, return dummy sample
        if boxes.shape[0] == 0:
            print(f"All boxes filtered out for image {idx}, returning dummy sample")
            dummy_image = torch.zeros((3, 512, 512))
            dummy_target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([idx]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64)
            }
            return dummy_image, dummy_target

        # Load and process image
        try:
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return dummy sample
            dummy_image = torch.zeros((3, 512, 512))
            dummy_target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([idx]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64)
            }
            return dummy_image, dummy_target

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes.tolist(), labels=labels.tolist())
            image = transformed["image"]
            boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.tensor(transformed["labels"], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }

        return image, target


def collate_fn(batch):
    """Collate function for DataLoader"""
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


# For backward compatibility, create an alias
WiderFaceDataset = KaggleWiderFaceDataset
