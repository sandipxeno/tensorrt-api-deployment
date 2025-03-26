from onnxruntime.quantization import quantize_dynamic, QuantType

# Apply dynamic quantization
quantized_model_path = "C:/Users/user/Desktop/Pytorch_onnx/Data_Labelling_Conversion/onnx/resnet50_quantized.onnx"
quantize_dynamic("C:/Users/user/Desktop/Pytorch_onnx/Data_Labelling_Conversion/onnx/resnet50_dog_cat.onnx", quantized_model_path, weight_type=QuantType.QUInt8)

print(f"Quantized model saved at {quantized_model_path}")
