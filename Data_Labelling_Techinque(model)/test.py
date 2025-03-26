import torch
from config import DEVICE
from dataset import get_dataloaders
from model import get_resnet50_model
from utils import calculate_accuracy

if __name__ == "__main__":
    model = get_resnet50_model()
    _, test_loader = get_dataloaders()
    model.load_state_dict(torch.load("C:/Users/user/Desktop/Pytorch_onnx/Data_labelling_model/resnet50_binary_classification.pth"))
    model.to(DEVICE)

    calculate_accuracy(model, test_loader)
