import torch
from config import DEVICE, EPOCHS
from utils import calculate_accuracy
from model import get_resnet50_model
from dataset import get_dataloaders
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, criterion, optimizer, epochs=EPOCHS):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.2f}%")

if __name__ == "__main__":
    model = get_resnet50_model()
    train_loader, _ = get_dataloaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_model(model, train_loader, criterion, optimizer)
