import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, render_template
from PIL import Image

app = Flask(__name__)

# Load the pre-trained ResNet18 model and modify the final layer
def load_model(num_classes=67):
    # Load ResNet18 with pre-trained weights
    model = models.resnet18(pretrained=True)
    
    # Modify the final fully connected layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load the trained weights (make sure the checkpoint corresponds to the correct architecture)
    checkpoint_path = 'E:/cat_detection/runs/epoch_10.pt'  # Update path if needed
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    return model

# Initialize the model
num_classes = 67  # Update with the actual number of classes in your dataset
model = load_model(num_classes=num_classes)

# Define the preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Match ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization for ResNet
])

# Class names (update this with your actual class labels)
class_names = [
    'Abyssinian', 'American Bobtail', 'American Curl', 'American Shorthair', 
    'American Wirehair', 'Applehead Siamese', 'Balinese', 'Bengal', 'Birman', 
    'Bombay', 'British Shorthair', 'Burmese', 'Burmilla', 'Calico', 
    'Canadian Hairless', 'Chartreux', 'Chausie', 'Chinchilla', 'Cornish Rex', 
    'Cymric', 'Devon Rex', 'Dilute Calico', 'Dilute Tortoiseshell', 
    'Domestic Long Hair', 'Domestic Medium Hair', 'Domestic Short Hair', 
    'Egyptian Mau', 'Exotic Shorthair', 
    'Extra-Toes Cat - Hemingway Polydactyl', 'Havana', 'Himalayan', 
    'Japanese Bobtail', 'Javanese', 'Korat', 'LaPerm', 'Maine Coon', 'Manx', 
    'Munchkin', 'Nebelung', 'Norwegian Forest Cat', 'Ocicat', 'Oriental Long Hair', 
    'Oriental Short Hair', 'Oriental Tabby', 'Persian', 'Pixiebob', 'Ragamuffin', 
    'Ragdoll', 'Russian Blue', 'Scottish Fold', 'Selkirk Rex', 'Siamese', 
    'Siberian', 'Silver', 'Singapura', 'Snowshoe', 'Somali', 
    'Sphynx - Hairless Cat', 'Tabby', 'Tiger', 'Tonkinese', 'Torbie', 
    'Tortoiseshell', 'Turkish Angora', 'Turkish Van', 'Tuxedo', 'York Chocolate'
]  # Replace with real breed names

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')  # Ensure you have an `index.html` template

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    try:
        # Load and preprocess the image
        image = Image.open(file).convert('RGB')  # Ensure the image is RGB
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        # Get the predicted class label
        predicted_class = class_names[predicted.item()]
        return f"Predicted Breed: {predicted_class}"
    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"Error during prediction: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
