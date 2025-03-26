import torch
import torchvision.models as models
import onnx
import onnxruntime as ort

# Ensure the model runs on CPU
device = torch.device("cpu")

# Load the trained ResNet50 model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(2048, 2)  # Modify last layer to match training

# Load model weights and move to CPU
model.load_state_dict(torch.load(
    r"C:/Users/user/Desktop/Pytorch_onnx/Data_Labelling_Techinque(model)/model/resnet50_binary_classification.pth", 
    map_location=device
))
model = model.to(device)
model.eval()

# Create a dummy input tensor (batch_size=1, channels=3, height=224, width=224)
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# Define ONNX save path
onnx_path = r"C:\Users\user\Desktop\Pytorch_onnx\Data_Labelling_Conversion\onnx\resnet50_dog_cat.onnx"

# Export to ONNX
torch.onnx.export(model, dummy_input, onnx_path,
                  export_params=True,        # Export trained weights
                  opset_version=11,          # ONNX version
                  input_names=['input'],     # Input layer name
                  output_names=['output'],   # Output layer name
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})  # Dynamic batch size

print(f"Model successfully converted and saved at: {onnx_path}")

# Verify the ONNX model
onnx_model = onnx.load(onnx_path)  
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")

# Run inference using ONNX Runtime (on CPU)
ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])  # Ensures it runs on CPU

# Convert dummy input to numpy (ONNX requires NumPy arrays)
onnx_output = ort_session.run(None, {'input': dummy_input.cpu().numpy()})

print("ONNX Model Output:", onnx_output)

