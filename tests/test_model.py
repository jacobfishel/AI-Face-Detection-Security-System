from backend.model import WiderFaceNN
import torch

model = WiderFaceNN()

model = model.load_state_dict(torch.load("models/best_model.pth"))
img = 'data/WIDER/WIDER_test/images/0--Parade/0_Parade_marchingband_1_9.jpg'

pred_boxes, conf_scored = model(img)