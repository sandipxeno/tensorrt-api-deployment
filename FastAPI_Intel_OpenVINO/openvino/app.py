from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from openvino.runtime import Core
import uvicorn

app = FastAPI()

# Load OpenVINO model
ie = Core()
model_path = "/Users/swedha/Documents/tensorrt-api-deployment/openvino_model/resnet50_dog_cat.xml"
compiled_model = ie.compile_model(model_path, "CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

CLASS_NAMES = ["Cat", "Dog"]  # Adjust based on model classes

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return {"error": "Invalid image format"}
    
    # Preprocess image
    image = cv2.resize(image, (224, 224))  # Adjust based on model input size
    image = np.transpose(image, (2, 0, 1))[None, :]  # Format for model
    image = image.astype(np.float32)
    image /= 255.0  # Normalize if needed
    
    # Run inference
    result = compiled_model([image])[output_layer]
    predicted_class = np.argmax(result)
    confidence = float(result[0][predicted_class])
    
    return {"prediction": CLASS_NAMES[predicted_class], "confidence": confidence}

@app.get("/")
def read_root():
    return {"message": "FastAPI with OpenVINO is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
