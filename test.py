import torch
import cv2
import torch.nn as nn
import torch.optim as optim
from torchvision import models


model = models.densenet121(pretrained=True)
model.eval()