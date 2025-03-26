import torch.nn as nn
import torchvision.models as models
from config import DEVICE

def get_resnet50_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Binary classification
    return model.to(DEVICE)
