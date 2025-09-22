# Custom PyTorch Dataset class that loads images and boxes

import torch
import os
import json
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class WiderFaceDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, split, transform=None):
        self.root_dir = filepath
        self.image_dir = os.path.join(filepath, f"WIDER_{split}")

        with open(os.path.join(filepath, f"parsed_annotations/parsed_wider_face_{split}_bbx_gt.json")) as f:
            self.data = json.load(f)

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        img_path = os.path.join(self.image_dir, entry["filename"])
        bboxes_raw = entry["bboxes"]

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

        # I had an error where the boxes collapsed to 0 width, so i filter out invalid bboxes
        boxes = boxes[boxes[:, 2] > boxes[:, 0]]    #x_max > x_min
        boxes = boxes[boxes[:, 3] > boxes[:, 1]]    # y_max > y_min
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        #If all boxes got filtered out, return dummy sample
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

        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes.tolist(), labels=labels.tolist())
            image = transformed["image"]
            boxes = torch.tensor(ftransformed["bboxes"], dtype=torch.float32)
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
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
