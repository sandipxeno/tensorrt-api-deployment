from openvino.runtime import Core
import numpy as np

# Initialize OpenVINO Core
ie = Core()

# Load the OpenVINO model
model_path = "/Users/swedha/Documents/tensorrt-api-deployment/openvino_model/resnet50_dog_cat.xml"
compiled_model = ie.compile_model(model_path, "CPU")

# Get input and output layers
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Create random input data (modify shape as per model)
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
result = compiled_model([input_data])

print("Inference Result:", result)
