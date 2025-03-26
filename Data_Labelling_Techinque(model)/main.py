import torch
from model import get_resnet50_model
from dataset import get_dataloaders
from train import train_model
from utils import calculate_accuracy
from config import MODEL_SAVE_PATH
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    model = get_resnet50_model()
    train_loader, test_loader = get_dataloaders()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_model(model, train_loader, criterion, optimizer)
    
    # Save the trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # Evaluate model
    calculate_accuracy(model, test_loader)
