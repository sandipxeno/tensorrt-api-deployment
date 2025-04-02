import time
import psutil
import numpy as np
import onnxruntime as ort

# Load ONNX model
onnx_session = ort.InferenceSession("C:/Users/user/Desktop/tensorrt-api-deployment/Data_Labelling_Conversion/onnx/resnet50_dog_cat.onnx", providers=["CPUExecutionProvider"])

# Define input shape
IMG_SIZE = 224  # Modify this if needed

def preprocess_dummy_image():
    """Create a dummy image and preprocess it to match model input."""
    image = np.random.rand(IMG_SIZE, IMG_SIZE, 3).astype(np.float32)  # Random image
    image = image / 255.0  # Normalize
    image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def benchmark_onnx_runtime(num_iterations=100):
    """Benchmark ONNX inference speed, CPU, and memory usage."""
    
    latencies = []
    cpu_usages = []
    memory_usages = []

    input_name = onnx_session.get_inputs()[0].name

    print(f"Running {num_iterations} inferences on ONNX Runtime (CPU)...")

    for _ in range(num_iterations):
        # Preprocess input
        image = preprocess_dummy_image()
        
        # Measure CPU and memory usage
        process = psutil.Process()
        cpu_before = psutil.cpu_percent(interval=None)
        memory_before = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

        # Measure inference time
        start_time = time.time()
        _ = onnx_session.run(None, {input_name: image})
        end_time = time.time()

        # Get CPU and memory usage after inference
        cpu_after = psutil.cpu_percent(interval=None)
        memory_after = process.memory_info().rss / (1024 * 1024)

        # Store metrics
        latencies.append((end_time - start_time) * 1000)  # Convert to ms
        cpu_usages.append(cpu_after - cpu_before)
        memory_usages.append(memory_after - memory_before)

    # Compute statistics
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    min_latency = np.min(latencies)
    avg_cpu_usage = np.mean(cpu_usages)
    avg_memory_usage = np.mean(memory_usages)
    throughput = num_iterations / (sum(latencies) / 1000)  # Inferences per second

    # Print results
    print("\n==== ONNX Runtime (CPU) Benchmark Results ====")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Min Latency: {min_latency:.2f} ms")
    print(f"Max Latency: {max_latency:.2f} ms")
    print(f"Average CPU Usage: {avg_cpu_usage:.2f} %")
    print(f"Average Memory Usage: {avg_memory_usage:.2f} MB")
    print(f"Throughput: {throughput:.2f} inferences per second")

if __name__ == "__main__":
    benchmark_onnx_runtime(num_iterations=100)
