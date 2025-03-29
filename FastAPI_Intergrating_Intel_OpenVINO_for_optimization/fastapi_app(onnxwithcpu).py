from fastapi import FastAPI, UploadFile, File
import numpy as np
import onnxruntime as ort
import cv2


app = FastAPI()

# Load ONNX model for CPU inference
onnx_session = ort.InferenceSession("D:/Prodigal-3/tensorrt-api-deployment/Data_Labelling_Conversion/onnx/resnet50_dog_cat.onnx", providers=["CPUExecutionProvider"])

# Define class labels (update these with your actual class names)
class_labels = ["cat", "dog","somthing"]  # Modify based on your dataset

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # For numerical stability
    return exp_logits / np.sum(exp_logits)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess the image
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))  # Resize as per model input
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # Change to (C, H, W)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Run inference
    input_name = onnx_session.get_inputs()[0].name
    logits = onnx_session.run(None, {input_name: image})[0]

    # Convert logits to probabilities
    probabilities = softmax(logits[0])

    # Get predicted class
    predicted_class_index = np.argmax(probabilities)
    predicted_label = class_labels[predicted_class_index]

    return {"prediction": predicted_label, "confidence": float(probabilities[predicted_class_index])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

