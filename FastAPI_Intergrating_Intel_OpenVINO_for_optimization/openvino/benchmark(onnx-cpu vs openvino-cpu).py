import time
import os
import psutil
import numpy as np
import onnxruntime as ort
from PIL import Image
from openvino.runtime import Core

# Paths
onnx_model_path = "C:/Users/user/Desktop/tensorrt-api-deployment/Data_Labelling_Conversion/onnx/resnet50_dog_cat.onnx"
openvino_model_path = "C:/Users/user/Desktop/tensorrt-api-deployment/FastAPI_Intergrating_Intel_OpenVINO_for_optimization/openvino/openvino_model/resnet50_dog_cat.xml"
results_file = "C:/Users/user/Desktop/tensorrt-api-deployment/FastAPI_Intergrating_Intel_OpenVINO_for_optimization/openvino/results.txt"
image_folder = "C:/Users/user/Desktop/tensorrt-api-deployment/FastAPI_Intergrating_Intel_OpenVINO_for_optimization/openvino/test_images"

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
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
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
    images = np.vstack(images)  # Convert to batch format
    return images, image_files[:batch_size]

def run_inference(session, images, input_name):
    """Run inference on ONNX model."""
    return session.run(None, {input_name: images})[0]

def run_openvino_inference(compiled_model, images):
    """Run inference on OpenVINO model."""
    infer_request = compiled_model.create_infer_request()
    infer_request.infer({compiled_model.input(0): images})
    return infer_request.get_output_tensor(0).data

def benchmark_model(session, images, input_name, model_name, num_runs=50):
    """Benchmark model performance."""
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
            f.write("🔹 **Model Performance Comparison** 🔹\n\n")
            f.write(f"{'Metric':<25}{'ONNX Model':<20}{'OpenVINO Model'}\n")
            f.write("="*70 + "\n")
            f.write(f"{'Batch Size':<25}{batch_size:<20}{batch_size}\n")
            f.write(f"{'Avg Latency (ms)':<25}{onnx_metrics[0]:<20.2f}{openvino_metrics[0]:.2f}\n")
            f.write(f"{'Throughput (img/sec)':<25}{onnx_metrics[1]:<20.2f}{openvino_metrics[1]:.2f}\n")
            f.write(f"{'Avg CPU Usage (%)':<25}{onnx_metrics[2]:<20.2f}{openvino_metrics[2]:.2f}\n")
            f.write(f"{'Avg Memory Usage (MB)':<25}{onnx_metrics[3]:<20.2f}{openvino_metrics[3]:.2f}\n")

        print(f"\n✅ Benchmark results saved to: {results_file}")
    else:
        print("\n❌ Benchmark failed due to errors.")

