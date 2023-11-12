import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.init as init

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load dataset
dataset = ImageFolder(root='./Agricultural-crops', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the number of classes
num_classes = len(dataset.classes)

# CNN Model with ReLU activation
class CNNModelReLU(nn.Module):
    def __init__(self):
        super(CNNModelReLU, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# CNN Model with Sigmoid activation
class CNNModelSigmoid(nn.Module):
    def __init__(self):
        super(CNNModelSigmoid, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.sigmoid(self.conv1(x)))
        x = self.pool(F.sigmoid(self.conv2(x)))
        x = x.view(-1, 32 * 64 * 64)
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

# Training loop with visualization
def train_model(model, dataloader, num_epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Visualize some predictions
        if epoch % 5 == 0:
            visualize_predictions(model, dataloader)

# Function to visualize predictions
def visualize_predictions(model, dataloader, num_images=5):
    model.eval()
    images, labels = next(iter(dataloader))
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)

    fig, axs = plt.subplots(1, num_images, figsize=(15, 3))

    for i in range(num_images):
        img = images[i].permute(1, 2, 0).numpy()
        label = dataset.classes[labels[i]]
        prediction = dataset.classes[predictions[i]]
        
        axs[i].imshow(img)
        axs[i].set_title(f'Label: {label}\nPrediction: {prediction}')
        axs[i].axis('off')

    plt.show()
    model.train()

# Train models
model_relu = CNNModelReLU()
model_sigmoid = CNNModelSigmoid()

train_model(model_relu, dataloader)
train_model(model_sigmoid, dataloader)

# CNN Model with ReLU activation and Batch Normalization
class CNNModelReLUWithBatchNorm(nn.Module):
    def __init__(self):
        super(CNNModelReLUWithBatchNorm, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 64 * 64, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = x.view(-1, 32 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# CNN Model with ReLU activation and Dropout
class CNNModelReLUWithDropout(nn.Module):
    def __init__(self):
        super(CNNModelReLUWithDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 64 * 64, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.dropout1(self.conv1(x))))
        x = self.pool(F.relu(self.dropout2(self.conv2(x))))
        x = x.view(-1, 32 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize weights with Xavier initialization
def xavier_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)

# Training loop with validation
def train_model_with_validation(model, dataloader, num_epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Apply weight initialization
    model.apply(xavier_init)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in dataloader:  # You need to create a validation DataLoader
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            accuracy = correct / total
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Validation Accuracy: {accuracy:.2%}')

# Validation DataLoader
validation_dataset = ImageFolder(root='./Agricultural-crops-validation', transform=transform)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

# Train models with validation
model_relu_with_batch_norm = CNNModelReLUWithBatchNorm()
model_relu_with_dropout = CNNModelReLUWithDropout()

train_model_with_validation(model_relu_with_batch_norm, dataloader)
train_model_with_validation(model_relu_with_dropout, dataloader)
