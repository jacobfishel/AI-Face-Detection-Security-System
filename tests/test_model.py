# import torch
# import torch.nn.functional as F
# from torchvision.ops import box_iou
# from torch.utils.data import DataLoader, Subset
# import numpy as np
# import cv2
# from data.dataset import WiderFaceDataset, collate_fn
# from backend.model import WiderFaceNN
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from config import IOU_POS_THRESH, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, DEVICE
# from PIL import Image
# import json
# from torchvision.ops import nms


# PARSED_TEST_PATH = 'data/WIDER/parsed_annotations/parsed_wider_face_test_filelist.json'

# # Initialize model
# model = WiderFaceNN().to(DEVICE)

# transform = A.Compose([
#             A.Resize(512, 512),
#             A.Normalize(mean=[0.485, 0.456, 0.406],
#                         std=[0.229, 0.224, 0.225]),
#             ToTensorV2()
#         ])


# '''
# make a transformation
# load an image. 
# transform it
# do a forward pass of the image
# display the image with the bounding box drawn on it using opencv
# '''

# with open(PARSED_TEST_PATH, 'r') as f:
#     image_paths = json.load(f)

# for image in image_paths:

#     # Converts image to PIL format
#     PIL_image = Image.open(image).convert("RGB")
#     image = np.array(PIL_image) #to NP array

#     transformed = transform(image=image)['image'].to(DEVICE)
#     transformed = transformed.unsqueeze(0)


#     pred_boxes, pred_confs = model(transformed)

#     pred_boxes = pred_boxes.view(-1, 4)
#     pred_confs = torch.sigmoid(pred_confs).view(-1)

#     conf_thresh = 0.5
#     keep = pred_confs > conf_thresh
#     boxes_to_keep = pred_boxes[keep]
#     scores_to_keep = pred_confs[keep]

#     iou_thresh = 0.4
#     keep_idx = nms(boxes_to_keep, scores_to_keep, iou_thresh)

#     final_boxes = boxes_to_keep[keep_idx]
#     final_scores = scores_to_keep[keep_idx]

#     #draw boxes and display
#     img = np.array(PIL_image.copy())
#     print(img.shape)

#     height = img.shape[0]
#     width = img.shape[1]

#     h_ratio = height // 512
#     w_ratio = width //512

#     x1 = pred_boxes[:, 0]
#     x2 = pred_boxes[:, 2]
#     y1 = pred_boxes[:, 1]
#     y2 = pred_boxes[:, 3]

#     x1 = x1.detach().cpu().numpy()
#     x2 = x2.detach().cpu().numpy()
#     y1 = y1.detach().cpu().numpy()
#     y2 = y2.detach().cpu().numpy()



#     final_coords = np.stack([x1, y1, x2, y2], axis=1)


#     new_x1 = x1 // w_ratio
#     new_x2 = x2 // w_ratio
#     new_y1 = y1 // h_ratio
#     new_y2 = y2 // h_ratio

#     final_coords = np.stack([new_x1, new_y1, new_x2, new_y2], axis=1)
#     print(final_coords)


#     break


import torch
import torch.nn.functional as F
from torchvision.ops import box_iou, nms
from torch.utils.data import DataLoader, Subset
import numpy as np
import cv2
from data.dataset import WiderFaceDataset, collate_fn
from backend.model import WiderFaceNN
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import IOU_POS_THRESH, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, DEVICE
from PIL import Image
import json

PARSED_TEST_PATH = 'data/WIDER/parsed_annotations/parsed_wider_face_test_filelist.json'

# Initialize model
# Initialize model
model = WiderFaceNN().to(DEVICE)

# Warm-up with a dummy input to initialize fc1
dummy_input = torch.zeros(1, 3, 512, 512).to(DEVICE)
_ = model(dummy_input)

# Now load the weights
model.load_state_dict(torch.load('models/best_model.pth', map_location=DEVICE))

model.eval()

# Define transform for test images
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Load test image paths
with open(PARSED_TEST_PATH, 'r') as f:
    image_paths = json.load(f)

# Loop through test images
for image_path in image_paths:

    # Load and keep original image
    PIL_image = Image.open(image_path).convert("RGB")
    original_img = np.array(PIL_image.copy())  # For drawing

    orig_h, orig_w = original_img.shape[:2]

    # Transform image
    image = np.array(PIL_image)
    transformed = transform(image=image)['image'].to(DEVICE).unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        pred_boxes, pred_confs = model(transformed)

    pred_boxes = pred_boxes.view(-1, 4)
    pred_confs = torch.sigmoid(pred_confs).view(-1)

    # Filter by confidence threshold
    conf_thresh = 0.5
    keep = pred_confs > conf_thresh
    boxes_to_keep = pred_boxes[keep]
    scores_to_keep = pred_confs[keep]

    if boxes_to_keep.numel() == 0:
        print(f"No boxes passed threshold for {image_path}")
        continue

    # Apply NMS
    iou_thresh = 0.4
    keep_idx = nms(boxes_to_keep, scores_to_keep, iou_thresh)

    final_boxes = boxes_to_keep[keep_idx]

    # Rescale boxes back to original image size
    scale_x = orig_w / 512
    scale_y = orig_h / 512

    x1 = final_boxes[:, 0].detach().cpu().numpy() * scale_x
    y1 = final_boxes[:, 1].detach().cpu().numpy() * scale_y
    x2 = final_boxes[:, 2].detach().cpu().numpy() * scale_x
    y2 = final_boxes[:, 3].detach().cpu().numpy() * scale_y

    final_coords = np.stack([x1, y1, x2, y2], axis=1)

    # Draw boxes
    for box in final_coords:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display
    cv2.imshow("Predicted", original_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    break  # Remove this if you want to process all images
