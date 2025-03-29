import torch
from openvino.tools.mo import convert_model
import torchvision.models as models

# Load the model architecture (Make sure it matches the saved model)
model = models.resnet18()  # Change this to your model architecture

# Load the state dictionary
model.load_state_dict(torch.load("/Users/swedha/Documents/tensorrt-api-deployment/Data_Labelling_Techinque(model)/model/resnet50_binary_classification.pth"))

# Set the model to evaluation mode
model.eval()

# Convert it to OpenVINO IR format
ov_model = convert_model(model)
ov_model.save("/Users/swedha/Documents/tensorrt-api-deployment/openvino_model/model.xml")

print("Model successfully converted to OpenVINO IR format!")

