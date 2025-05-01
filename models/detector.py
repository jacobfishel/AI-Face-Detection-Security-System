# Neural network architecture (CNN with output for bounding boxes)
# 1. Unzip all the WIDER zip files
# 2. Parse the annotations from .txt files
# 3. Link each image with its bounding boxes
# 4. (Optional) Visualize a few to verify
# 5. Load this into a PyTorch Dataset
# 6. Train your model 🎯
from  data.dataset import WiderFaceDataset


model = WiderFaceDataset("data/WIDER/WIDER_test", )