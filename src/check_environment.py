import sys
import platform

import torch
import torchvision
import cv2
import numpy as np
import pandas as pd
import matplotlib
import sklearn

print("========== SYSTEM INFORMATION ==========")

print("Python:", sys.version)
print("Platform:", platform.platform())

print("\n========== LIBRARY VERSIONS ==========")

print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("OpenCV:", cv2.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Scikit-learn:", sklearn.__version__)

print("\n========== DEVICE INFORMATION ==========")

print("CUDA available:", torch.cuda.is_available())

device = torch.device("cpu")

print("Selected device:", device)

# Limit CPU threads for a more controlled experiment
torch.set_num_threads(4)

print("PyTorch CPU threads:", torch.get_num_threads())

print("\n========== TENSOR TEST ==========")

x = torch.rand(3, 3)
y = torch.rand(3, 3)

result = torch.matmul(x, y)

print("Matrix multiplication result:")
print(result)

print("\n========== OBJECT DETECTION TEST ==========")

boxes = torch.tensor([
    [10, 10, 100, 100],
    [20, 20, 110, 110],
    [200, 200, 300, 300]
], dtype=torch.float32)

scores = torch.tensor([0.9, 0.8, 0.7])

keep = torchvision.ops.nms(boxes, scores, 0.5)

print("NMS selected boxes:", keep.tolist())

print("\nEnvironment verification completed!")