import torch

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 10

# Paths
TRAIN_DATA_PATH = "C:/Users/user/Desktop/Pytorch_onnx/Data_labelling_model/training_set"
TEST_DATA_PATH = "C:/Users/user/Desktop/Pytorch_onnx/Data_labelling_model/test_set"
MODEL_SAVE_PATH = "resnet50_binary_classification.pth"
