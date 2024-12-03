import os
import shutil
from tqdm import tqdm
import yaml

# Directories for dataset, annotations, and image storage
dataset_dir = "dataset"  # Directory where your dataset is stored
labels_dir = "data/labels"  # Where YOLO annotations will be saved
images_dir = "data/images"  # Where images will be saved
train_dir = "data/train"
valid_dir = "data/valid"

# Create directories for the dataset split (train, valid)
os.makedirs(labels_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# Create subdirectories for train/valid images and labels
os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
os.makedirs(os.path.join(valid_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(valid_dir, 'labels'), exist_ok=True)

# Class mapping: automatically map breed folders to integers
breeds = os.listdir(dataset_dir)
class_mapping = {breed: idx for idx, breed in enumerate(breeds)}

# Prepare annotations and dataset
for breed, breed_idx in class_mapping.items():
    breed_dir = os.path.join(dataset_dir, breed)
    for img_name in tqdm(os.listdir(breed_dir)):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(breed_dir, img_name)
            label_name = img_name.split('.')[0] + '.txt'
            label_path = os.path.join(labels_dir, label_name)

            # Create the annotation file
            with open(label_path, 'w') as label_file:
                label_file.write(f"{breed_idx} 0.5 0.5 1.0 1.0\n")  # Full image as one bounding box

            # Copy image to the images folder
            shutil.copy(img_path, images_dir)

# Split dataset into train and valid
img_files = os.listdir(images_dir)
train_size = int(0.8 * len(img_files))
valid_size = len(img_files) - train_size

train_images = img_files[:train_size]
valid_images = img_files[train_size:]

# Copy files to train and valid directories
for img_name in tqdm(train_images):
    label_name = img_name.split('.')[0] + '.txt'
    shutil.copy(os.path.join(images_dir, img_name), os.path.join(train_dir, 'images', img_name))
    shutil.copy(os.path.join(labels_dir, label_name), os.path.join(train_dir, 'labels', label_name))

for img_name in tqdm(valid_images):
    label_name = img_name.split('.')[0] + '.txt'
    shutil.copy(os.path.join(images_dir, img_name), os.path.join(valid_dir, 'images', img_name))
    shutil.copy(os.path.join(labels_dir, label_name), os.path.join(valid_dir, 'labels', label_name))

# Create YAML file for YOLOv5
yaml_data = {
    'train': 'data/train/images',
    'val': 'data/valid/images',
    'nc': len(class_mapping),
    'names': list(class_mapping.keys())  # List of breed names
}

yaml_file = "data/cat_breed_dataset.yaml"
with open(yaml_file, 'w') as f:
    yaml.dump(yaml_data, f)

print(f"Dataset YAML file created: {yaml_file}")
