import torch
from openvino.tools.mo import convert_model
import torchvision.models as models

import openvino.tools.mo as mo

# Define paths
onnx_model_path = "C:/Users/user/Desktop/tensorrt-api-deployment/Data_Labelling_Conversion/onnx/resnet50_dog_cat.onnx"
ir_output_dir = "C:/Users/user/Desktop/tensorrt-api-deployment/FastAPI_Intergrating_Intel_OpenVINO_for_optimization/openvino/openvino_model"

# Convert ONNX to OpenVINO IR
mo.convert_model(onnx_model_path, output_dir=ir_output_dir, model_name="model")

print("Model successfully converted to OpenVINO IR format!")

