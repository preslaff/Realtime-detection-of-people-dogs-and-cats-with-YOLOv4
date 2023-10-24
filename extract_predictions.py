import os
import csv
import time
from roboflow import Roboflow
from tqdm import tqdm

yolo_version = int(input("Please input model version: "))

rf = Roboflow(api_key="VUZ4GpmqhfyJIQU6enkZ")
project = rf.workspace().project("dashcam-detector")
model = project.version(yolo_version).model

# Directory containing your imagesc downloaded in 
image_directory = "data_split_v4/test/"

# Allowed image extensions
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}

# Create or open the CSV file for writing
csv_file = "predictions_v4.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file,
                            fieldnames=["image_name", "x", "y", "width", "height", "confidence", "class", "image_path",
                                        "prediction_type"])
    writer.writeheader()

    # Get a list of all images in directory
all_images = [img for img in os.listdir(image_directory) if os.path.splitext(img)[1].lower() in image_extensions]

    # Using tqdm to wrap the iteration and show progress
    for image_name in tqdm(all_images, desc="Processing images", unit="image"):
        image_path = os.path.join(image_directory, image_name)

        # Predict for the current image
        prediction = model.predict(image_path, confidence=40, overlap=30).json()

        # Add the image name to prediction for better tracking
        for pred in prediction["predictions"]:
            pred["image_name"] = image_name
            writer.writerow(pred)

        # Delay for a bit to avoid hitting API rate limits
        time.sleep(1)  # Sleep for 1 second

print(f"\nPredictions saved to {csv_file}")
