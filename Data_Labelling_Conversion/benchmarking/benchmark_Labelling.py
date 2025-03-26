import time
import numpy as np
import onnxruntime as ort

# Define model paths
fp32_model_path = "C:/Users/user/Desktop/Pytorch_onnx/Data_Labelling_Conversion/onnx/resnet50_dog_cat.onnx"
int8_model_path = "C:/Users/user/Desktop/Pytorch_onnx/Data_Labelling_Conversion/onnx/resnet50_quantized.onnx"

# Load a sample input tensor (modify shape as per your model)
input_shape = (1, 3, 224, 224)  # Example for ResNet50
dummy_input = np.random.randn(*input_shape).astype(np.float32)

# Log file path
log_file = "C:/Users/user/Desktop/Pytorch_onnx/Data_Labelling_Conversion/onnx/benchmark_results.txt"

def benchmark_model(model_path, input_data):
    """Runs inference and logs performance metrics."""
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    # Warm-up run
    session.run(None, {input_name: input_data})

    # Measure inference time
    start_time = time.time()
    for _ in range(100):  # Run 100 iterations
        session.run(None, {input_name: input_data})
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / 100
    return avg_latency

# Run benchmarking and log results
with open(log_file, "w") as log:
    log.write("ONNX Model Benchmark Results\n")
    log.write("=" * 40 + "\n")

    # Benchmark FP32 model
    print("\nBenchmarking FP32 Model...")
    fp32_latency = benchmark_model(fp32_model_path, dummy_input)
    log.write(f"FP32 Model: {fp32_model_path}\n")
    log.write(f"Avg Latency per inference: {fp32_latency:.6f} seconds\n\n")

    # Benchmark Quantized INT8 model
    print("\nBenchmarking Quantized INT8 Model...")
    int8_latency = benchmark_model(int8_model_path, dummy_input)
    log.write(f"Quantized INT8 Model: {int8_model_path}\n")
    log.write(f"Avg Latency per inference: {int8_latency:.6f} seconds\n")

print(f"\n Benchmarking completed! Results saved to: {log_file}")
