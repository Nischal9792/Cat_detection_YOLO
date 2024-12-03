import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torch import nn, optim
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import os

# Paths
PROJECT_DIR = 'E:/cat_detection/yolov5'
DATASET_DIR = 'E:/cat_detection/dataset'
RUNS_DIR = 'E:/cat_detection/runs'
Path(RUNS_DIR).mkdir(parents=True, exist_ok=True)

# Dataset
class CatBreedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))  # Ensure consistent ordering of classes
        for label, breed in enumerate(self.classes):
            breed_dir = os.path.join(root_dir, breed)
            for image_name in os.listdir(breed_dir):
                image_path = os.path.join(breed_dir, image_name)
                # Verify the image before adding it to the samples list
                try:
                    with Image.open(image_path) as img:
                        img.verify()  # Verify the image to ensure it's not corrupted
                    self.samples.append((image_path, label))
                except (UnidentifiedImageError, OSError):
                    print(f"Error reading image {image_path}: corrupted image, skipping.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# DataLoader
train_dataset = CatBreedDataset(DATASET_DIR, transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Model
num_classes = len(train_dataset.classes)  # Includes "Not a Cat"
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Optimizer and Loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training
def train(model, loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(loader):.4f}")
        checkpoint_path = os.path.join(RUNS_DIR, f"epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), checkpoint_path)

train(model, train_loader, optimizer, criterion)
