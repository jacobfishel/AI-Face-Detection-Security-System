import torch
import torch.nn.functional as F
from torchvision.ops import box_iou
from torch.utils.data import DataLoader, Subset
import numpy as np
import cv2
from data.dataset import WiderFaceDataset, collate_fn
from backend.model import WiderFaceNN
from config import IOU_POS_THRESH, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, DEVICE

# Initialize model
model = WiderFaceNN().to(DEVICE)

# Load datasets (testers and real)
train_dataset = WiderFaceDataset('data/WIDER', 'train')
val_dataset = WiderFaceDataset('data/WIDER', 'val')

train_subset = Subset(train_dataset, list(range(200)))
val_subset = Subset(val_dataset, list(range(200)))




# Create DataLoaders
train_loader = DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

train_subset_loader = DataLoader(train_subset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_subset_loader = DataLoader(val_subset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# def train_model(train_loader, val_loader, model, n_epochs):
#     print(f"Training starting on {DEVICE}")
#     print(f"Training with a batch size: {BATCH_SIZE}, Epochs = {NUM_EPOCHS}")
#     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#     best_val_loss = float('inf')

#     losses, epochs = [], []

#     for epoch in range(n_epochs):
#         print(f"Epoch {epoch + 1}")
#         model.train()
#         running_loss = 0.0

#         for i, (images, targets) in enumerate(train_loader):
#             images = images.to(DEVICE)
#             print(f"First image = {images[0]}")
#             targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
#             print(f"first target = {targets[0]}")

#             optimizer.zero_grad()
#             pred_boxes, pred_confs = model(images)
#             print(f"Predicted boxes = {pred_boxes}")
#             print(f"Predicted confs = {pred_confs}")

#             pred_boxes = pred_boxes.view(-1, 4)
#             pred_confs = torch.sigmoid(pred_confs).view(-1)
#             target_boxes = torch.cat([t['boxes'] for t in targets], dim=0)

#             print(f"pred_boxes after view = {pred_boxes}")
#             print(f"pred_confs after sigmoid and view = {pred_confs}")
#             print(f"Target boxes = {target_boxes}")

#             if target_boxes.numel() == 0:
#                 continue

#             target_confs = torch.zeros(pred_boxes.size(0), device=DEVICE)
#             print(f"target_confs = {target_confs}")
#             ious = box_iou(pred_boxes, target_boxes)
#             print(f"ious = {ious}")
#             max_ious, gt_idx = ious.max(dim=1)
#             print(f"max_ious, gt_idx = {max_ious}, {gt_idx}")
#             positive_mask = max_ious > IOU_POS_THRESH
#             print(f"positive_mask = {positive_mask}")
#             target_confs[positive_mask] = 1
#             print(f"target_confs after mask = {target_confs}")

#             matched_gt_boxes = target_boxes[gt_idx]
#             print(f"matched_gt_boxes = {matched_gt_boxes}")

#             if positive_mask.sum() > 0:
#                 box_loss = F.smooth_l1_loss(pred_boxes[positive_mask], matched_gt_boxes[positive_mask])
#                 print(f"box_loss = {box_loss}")
#             else:
#                 box_loss = torch.tensor(0.0, device=DEVICE)
#                 print(f"box_loss since positive_mask.sum() <= 0 = {box_loss}")

#             conf_loss = F.binary_cross_entropy(pred_confs, target_confs)
#             print(f"conf_loss = {conf_loss}")
#             loss = box_loss + conf_loss
#             print(f"loss = {loss}")

#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             print(f"running loss = {running_loss}")
#             N = len(train_loader)
#             epochs.append(epoch + i / N)
#             losses.append(loss.item())

#         avg_train_loss = running_loss / len(train_loader)
#         print(f"Epoch {epoch + 1} Train Loss: {avg_train_loss:.4f}")

#         # Validation
#         model.eval()
#         val_loss_total = 0.0
#         with torch.no_grad():
#             for val_images, val_targets in val_loader:
#                 val_images = val_images.to(DEVICE)
#                 val_targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in val_targets]

#                 val_pred_boxes, val_pred_confs = model(val_images)
#                 val_pred_boxes = val_pred_boxes.view(-1, 4)
#                 val_pred_confs = torch.sigmoid(val_pred_confs).view(-1)

#                 val_target_boxes = torch.cat([t['boxes'] for t in val_targets], dim=0)
#                 if val_target_boxes.numel() == 0:
#                     continue

#                 val_target_confs = torch.zeros(val_pred_boxes.size(0), device=DEVICE)
#                 ious = box_iou(val_pred_boxes, val_target_boxes)
#                 max_ious, gt_idx = ious.max(dim=1)
#                 positive_mask = max_ious > IOU_POS_THRESH
#                 val_target_confs[positive_mask] = 1

#                 matched_gt_boxes = val_target_boxes[gt_idx]

#                 if positive_mask.sum() > 0:
#                     val_box_loss = F.smooth_l1_loss(val_pred_boxes[positive_mask], matched_gt_boxes[positive_mask])
#                 else:
#                     val_box_loss = torch.tensor(0.0, device=DEVICE)

#                 val_conf_loss = F.binary_cross_entropy(val_pred_confs, val_target_confs)
#                 val_loss = val_box_loss + val_conf_loss
#                 val_loss_total += val_loss.item()

#         avg_val_loss = val_loss_total / len(val_loader)
#         print(f"Epoch {epoch + 1} Val Loss: {avg_val_loss:.4f}")

#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             torch.save(model.state_dict(), "models/best_model.pth")
#             print("Best model saved!")

#     return np.array(epochs), np.array(losses)

def train_model(train_loader, val_loader, model, n_epochs):
    print(f"Training starting on {DEVICE}")
    print(f"Training with batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float('inf')

    losses, epochs = [], []

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        model.train()
        running_loss = 0.0

        for i, (images, targets) in enumerate(train_loader):
            images = images.to(DEVICE)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            if i == 0:  # Only debug-print the first batch of each epoch
                print(f"[TRAIN] images.shape: {images.shape}")
                print(f"[TRAIN] first target keys: {list(targets[0].keys())}")
                print(f"[TRAIN] first target boxes shape: {targets[0]['boxes'].shape}")

            optimizer.zero_grad()
            pred_boxes, pred_confs = model(images)

            pred_boxes = pred_boxes.view(-1, 4)
            pred_confs = torch.sigmoid(pred_confs).view(-1)
            target_boxes = torch.cat([t['boxes'] for t in targets], dim=0)

            if target_boxes.numel() == 0:
                continue

            target_confs = torch.zeros(pred_boxes.size(0), device=DEVICE)
            ious = box_iou(pred_boxes, target_boxes)
            max_ious, gt_idx = ious.max(dim=1)
            positive_mask = max_ious > IOU_POS_THRESH
            target_confs[positive_mask] = 1
            matched_gt_boxes = target_boxes[gt_idx]

            if positive_mask.sum() > 0:
                box_loss = F.smooth_l1_loss(pred_boxes[positive_mask], matched_gt_boxes[positive_mask])
            else:
                box_loss = torch.tensor(0.0, device=DEVICE)

            conf_loss = F.binary_cross_entropy(pred_confs, target_confs)
            loss = box_loss + conf_loss

            if i == 0:
                print(f"[TRAIN] box_loss: {box_loss.item():.4f}, conf_loss: {conf_loss.item():.4f}, total_loss: {loss.item():.4f}")

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            N = len(train_loader)
            epochs.append(epoch + i / N)
            losses.append(loss.item())

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Train Loss: {avg_train_loss:.4f}")

        # # Validation
        # model.eval()
        # val_loss_total = 0.0
        # with torch.no_grad():
        #     for j, (val_images, val_targets) in enumerate(val_loader):
        #         val_images = val_images.to(DEVICE)
        #         val_targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in val_targets]

        #         val_pred_boxes, val_pred_confs = model(val_images)
        #         val_pred_boxes = val_pred_boxes.view(-1, 4)
        #         val_pred_confs = torch.sigmoid(val_pred_confs).view(-1)

        #         val_target_boxes = torch.cat([t['boxes'] for t in val_targets], dim=0)
        #         if val_target_boxes.numel() == 0:
        #             continue

        #         val_target_confs = torch.zeros(val_pred_boxes.size(0), device=DEVICE)
        #         ious = box_iou(val_pred_boxes, val_target_boxes)
        #         max_ious, gt_idx = ious.max(dim=1)
        #         positive_mask = max_ious > IOU_POS_THRESH
        #         val_target_confs[positive_mask] = 1
        #         matched_gt_boxes = val_target_boxes[gt_idx]

        #         if positive_mask.sum() > 0:
        #             val_box_loss = F.smooth_l1_loss(val_pred_boxes[positive_mask], matched_gt_boxes[positive_mask])
        #         else:
        #             val_box_loss = torch.tensor(0.0, device=DEVICE)

        #         val_conf_loss = F.binary_cross_entropy(val_pred_confs, val_target_confs)
        #         val_loss = val_box_loss + val_conf_loss
        #         val_loss_total += val_loss.item()

        #         if j == 0:
        #             print(f"[VAL] box_loss: {val_box_loss.item():.4f}, conf_loss: {val_conf_loss.item():.4f}, total_loss: {val_loss.item():.4f}")

        # avg_val_loss = val_loss_total / len(val_loader)
        # print(f"Epoch {epoch + 1} Val Loss: {avg_val_loss:.4f}")

        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
    torch.save(model.state_dict(), "models/best_model.pth")
        #     print("Best model saved!")

    return np.array(epochs), np.array(losses)





if __name__ == '__main__':
    # epochs, losses = train_model(train_loader, val_loader, model, NUM_EPOCHS)
    # epochs, losses = train_model(train_subset_loader, val_subset_loader, model, NUM_EPOCHS)


    img = val_dataset.__getitem__(20)
    print(img)
