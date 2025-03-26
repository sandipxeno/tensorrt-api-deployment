import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
from collections import OrderedDict

# Model Path & Uploads Folder
MODEL_PATH = "C:/Users/user/Desktop/Pytorch_onnx/Data_Labelling_Techinque(model)/model/resnet50_binary_classification.pth"
UPLOAD_FOLDER = "C:/Users/user/Desktop/Pytorch_onnx/Data_Labelling_Techinque(model)/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure uploads folder exists

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define Image Transform
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define Model Class
class Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(Classifier, self).__init__()
        self.encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.encoder(x)

# Load Model
def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found!")
        return None
    
    model = Classifier().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)

    # Fix if saved using DataParallel (multi-GPU)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # Remove "module." if present
        new_state_dict[new_key] = v

    try:
        model.load_state_dict(new_state_dict, strict=False)
    except RuntimeError as e:
        print("❌ Model loading error! Possible size mismatch.")
        print(e)
        return None

    model.eval()
    print("✅ Model loaded successfully.\n")
    return model

model = load_model()

# Prediction Function
def predict_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform_test(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()

        return "Dog" if prediction == 1 else "Cat"
    except Exception as e:
        return f"Error processing {image_path}: {str(e)}"

# 🔹 **Predict Single Image**
def predict_single(filename):
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(image_path):
        print(f"❌ Error: File '{filename}' not found in uploads/ folder!\n")
        return
    result = predict_image(image_path)
    print(f"📸 Image: {filename} → 🏷️ Prediction: {result}\n")

# 🔹 **Predict All Images in `uploads/`**
def predict_all():
    image_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not image_files:
        print("❌ No images found in uploads/ folder!\n")
        return

    print("🔍 Predicting all images in uploads/ folder...\n")
    for image_file in image_files:
        result = predict_image(os.path.join(UPLOAD_FOLDER, image_file))
        print(f"📸 {image_file} → 🏷️ {result}")
    print("\n✅ Batch prediction completed!\n")

# 🔥 **Main Execution**
if __name__ == "__main__":
    print("\n📌 Options:")
    print("1️⃣ Predict a single image → Enter filename (e.g., cat.jpg)")
    print("2️⃣ Predict all images in uploads/ → Press Enter\n")

    filename = input("🔹 Enter filename (or press Enter to predict all): ").strip()

    if filename:
        predict_single(filename)
    else:
        predict_all()



