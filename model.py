import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms


# Define the MesoNet architecture (CNN used for deepfake detection)
class MesoNet(nn.Module):
    def __init__(self):
        super(MesoNet, self).__init__()

        # First convolution layer
        # Takes RGB image (3 channels) and extracts 8 basic features (edges, textures)
        # Important: This is where the model starts learning visual patterns
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)

        # Batch Normalization
        # Important: Stabilizes training and speeds up convergence
        self.bn1 = nn.BatchNorm2d(8)

        # Max Pooling layer
        # Reduces image size and keeps important features
        # Important: Reduces computation and prevents overfitting
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Second convolution layer
        # Learns more complex patterns from previous features
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(8)

        # Third convolution layer
        # Increases depth (8 → 16), meaning more complex feature extraction
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(16)

        # Fourth convolution layer
        # Learns high-level features (like facial inconsistencies in deepfakes)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm2d(16)

        # Fully connected layer
        # Converts extracted features into a smaller representation
        # Important: Helps in final decision making
        self.fc1 = nn.Linear(16 * 16 * 16, 16)

        # Output layer
        # Produces a single value (probability)
        # Important: Used for binary classification (real vs fake)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        # Conv1 → ReLU → BatchNorm → Pooling
        # Important: Extracts basic features and reduces size
        x = self.pool(self.bn1(nn.functional.relu(self.conv1(x))))

        # Conv2 block
        # Important: Learns more detailed patterns
        x = self.pool(self.bn2(nn.functional.relu(self.conv2(x))))

        # Conv3 block
        # Important: Captures higher-level features
        x = self.pool(self.bn3(nn.functional.relu(self.conv3(x))))

        # Conv4 block
        # Important: Detects complex patterns like inconsistencies in images
        x = self.pool(self.bn4(nn.functional.relu(self.conv4(x))))

        # Flatten the feature maps into a vector
        # Important: Required before passing into fully connected layers
        x = x.view(-1, 16 * 16 * 16)

        # First dense layer with ReLU
        # Important: Introduces non-linearity and helps learning complex relationships
        x = nn.functional.relu(self.fc1(x))

        # Final output with sigmoid activation
        # Important: Converts output into probability (0 to 1)
        x = torch.sigmoid(self.fc2(x))

        return x


# Image preprocessing (same as training)
# Important: Ensures consistency between training and testing data
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),  # Important: Model expects fixed input size
        transforms.ToTensor(),  # Important: Converts image to numerical tensor
        transforms.Normalize(  # Important: Improves model performance and stability
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


# Function to load trained model weights
def load_model(model, model_path="mesonet_model.pth"):

    # Select device (GPU if available, else CPU)
    # Important: GPU speeds up computation significantly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to selected device
    model = model.to(device)

    # Load saved weights
    # Important: Reuses learned knowledge without retraining
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")

    return model


# Function to make prediction
def predict(image_path, model):

    # Define device again for inference
    # Important: Ensures model and data are on same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load image and convert to RGB
    # Important: Ensures compatibility with model input
    image = Image.open(image_path).convert("RGB")

    # Apply preprocessing and add batch dimension
    # Important: Model expects batch input (even for single image)
    image = transform(image).unsqueeze(0)

    # Move image to device
    image = image.to(device)

    # Disable gradient calculation
    # Important: Saves memory and speeds up inference
    with torch.no_grad():
        outputs = model(image)  # Forward pass

        # Convert output tensor to numpy
        # Important: Easier to interpret result outside PyTorch
        probs = outputs[0].cpu().detach().numpy()

        # Final output: probability value
        # Important: Used to classify image as real or fake
        return probs
