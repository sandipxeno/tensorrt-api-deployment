import onnxruntime as ort
import numpy as np
import time
import psutil
import os
from PIL import Image
from openvino.runtime import Core

# Paths
onnx_model_path = "/Users/swedha/Documents/tensorrt-api-deployment/Data_Labelling_Conversion/onnx/resnet50_dog_cat.onnx"
openvino_model_path = "/Users/swedha/Documents/tensorrt-api-deployment/openvino_model/resnet50_dog_cat.xml"
results_file = "/Users/swedha/Documents/tensorrt-api-deployment/results.txt"
image_folder = "/Users/swedha/Documents/tensorrt-api-deployment/test_images"

# Load ONNX model
try:
    onnx_session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    onnx_input_name = onnx_session.get_inputs()[0].name
    print(f"✅ Loaded ONNX model: {onnx_model_path}")
except Exception as e:
    print(f"❌ Error loading ONNX model: {e}")
    exit()

# Load OpenVINO model
try:
    ie = Core()
    openvino_model = ie.compile_model(model=openvino_model_path, device_name="CPU")
    print(f"✅ Loaded OpenVINO model: {openvino_model_path}")
except Exception as e:
    print(f"❌ Error loading OpenVINO model: {e}")
    exit()

# Define classes
CLASSES = ["Cat", "Dog"]

def preprocess_image(image_path):
    """Preprocess image to match model input format."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))  # Resize to match model input
    image_array = np.asarray(image).astype(np.float32) / 255.0  # Normalize
    image_array = np.transpose(image_array, (2, 0, 1))  # Convert HWC to CHW
    return image_array

def load_batch_images(image_folder, batch_size=8):
    """Load multiple images and create a batch."""
    if not os.path.exists(image_folder):
        print(f"❌ Error: Image folder '{image_folder}' not found.")
        exit()

    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(("jpg", "png"))]
    if not image_files:
        print(f"❌ Error: No images found in folder '{image_folder}'.")
        exit()

    images = [preprocess_image(img) for img in image_files[:batch_size]]
    images = np.stack(images, axis=0)  # Convert to batch format
    return images, image_files[:batch_size]

def run_inference(session, images, input_name):
    """Run inference on ONNX model."""
    return session.run(None, {input_name: images})[0]

def run_openvino_inference(compiled_model, images):
    """Run inference on OpenVINO model (Batch processing)."""
    infer_request = compiled_model.create_infer_request()
    
    # Ensure batch size matches model expectation
    if images.shape[0] == 1:
        infer_request.infer({compiled_model.input(0): images})
        return infer_request.get_output_tensor(0).data
    else:
        results = []
        for image in images:  # Process images one by one
            input_tensor = np.expand_dims(image, axis=0)  # Shape: (1, 3, 224, 224)
            infer_request.infer({compiled_model.input(0): input_tensor})
            results.append(infer_request.get_output_tensor(0).data)
        return np.vstack(results)

def benchmark_model(session, images, input_name, model_name, num_runs=50):
    """Run batch inference and measure performance metrics."""
    times, cpu_usages, memory_usages = [], [], []

    for _ in range(num_runs):
        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=None)
        start_memory = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MB

        try:
            if model_name == "ONNX":
                result = run_inference(session, images, input_name)
            else:
                result = run_openvino_inference(session, images)
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            return None

        end_time = time.time()
        end_cpu = psutil.cpu_percent(interval=None)
        end_memory = psutil.virtual_memory().used / (1024 * 1024)

        times.append((end_time - start_time) * 1000)  # Convert to ms
        cpu_usages.append(end_cpu)
        memory_usages.append(end_memory - start_memory)

    avg_latency = np.mean(times)
    avg_throughput = (len(images) * num_runs) / (np.sum(times) / 1000)  # Images per second
    avg_cpu = np.mean(cpu_usages)
    avg_memory = np.mean(memory_usages)

    return avg_latency, avg_throughput, avg_cpu, avg_memory

if __name__ == "__main__":
    batch_size = 8
    num_runs = 50

    images, image_names = load_batch_images(image_folder, batch_size)
    
    # Benchmark ONNX
    print("\n🔹 Running ONNX model benchmark...")
    onnx_metrics = benchmark_model(onnx_session, images, onnx_input_name, "ONNX", num_runs)

    # Benchmark OpenVINO
    print("\n🔹 Running OpenVINO model benchmark...")
    openvino_metrics = benchmark_model(openvino_model, images, None, "OpenVINO", num_runs)

    if onnx_metrics and openvino_metrics:
        # Save results
        with open(results_file, "w", encoding="utf-8") as f:
            f.write("🔹 **Model Inference Performance Comparison** 🔹\n\n")
            f.write(f"{'Metric':<25}{'ONNX Model':<20}{'OpenVINO Model'}\n")
            f.write("="*70 + "\n")
            f.write(f"{'Batch Size':<25}{batch_size:<20}{batch_size}\n")
            f.write(f"{'Average Latency (ms)':<25}{onnx_metrics[0]:<20.2f}{openvino_metrics[0]:.2f}\n")
            f.write(f"{'Throughput (images/sec)':<25}{onnx_metrics[1]:<20.2f}{openvino_metrics[1]:.2f}\n")
            f.write(f"{'Avg CPU Usage (%)':<25}{onnx_metrics[2]:<20.2f}{openvino_metrics[2]:.2f}\n")
            f.write(f"{'Avg Memory Usage (MB)':<25}{onnx_metrics[3]:<20.2f}{openvino_metrics[3]:.2f}\n")

        print(f"\n✅ Benchmark results saved to: {results_file}")
    else:
        print("\n❌ Benchmark failed due to errors.")