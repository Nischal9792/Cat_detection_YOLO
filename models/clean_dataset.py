import os
from PIL import Image

# Define the directory containing your dataset
dataset_dir = "E:/cat_detection/dataset/Somali"

# Function to check if the image is corrupted and remove it if so
def remove_corrupted_images(directory):
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        image_path = os.path.join(directory, filename)
        
        # Only process files with image extensions (jpg, jpeg, png)
        if filename.lower().endswith(('jpg', 'jpeg', 'png')):
            try:
                # Try to open the image to check if it's valid
                with Image.open(image_path) as img:
                    img.verify()  # Verify the image is not corrupted
            except Exception as e:
                # If an error occurs (corrupted image), print a message and remove it
                print(f"Error reading image {filename}: {e}")
                os.remove(image_path)  # Remove the corrupted image
                print(f"Removed corrupted image: {filename}")

# Run the cleaning function
remove_corrupted_images(dataset_dir)
