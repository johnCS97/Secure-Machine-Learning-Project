import os
import tarfile
import urllib.request
import os
import numpy as np
import tensorflow as tf



# Define dataset URL and file name
LFW_URL = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))  # Grandparent directory
DATASET_DIR = os.path.join(PARENT_DIR, "datasets")  # Dataset folder in the grandparent directory
ARCHIVE_NAME = os.path.join(DATASET_DIR, "lfw-deepfunneled.tgz")  # Archive file in the grandparent directory
EXTRACT_TO=(DATASET_DIR+("/lfw-deepfunneled"))
OUTPUT_DIR = DATASET_DIR+"/processed_lfw_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TARGET_SIZE = (224, 224)

# Download the dataset
if not os.path.exists(ARCHIVE_NAME):
    print("Downloading LFW dataset...")
    urllib.request.urlretrieve(LFW_URL, ARCHIVE_NAME)
    print("Download complete.")

# Extract the dataset
if not os.path.exists(EXTRACT_TO):
    print("Extracting dataset...")
    with tarfile.open(ARCHIVE_NAME, "r:gz") as tar:
        tar.extractall(DATASET_DIR)
    print("Extraction complete.")
else:
    print("Dataset already exists.")

# Function to process a single image
def process_image(image_path, target_size):
    # Load the image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)

    # Resize the image to the target size
    img = tf.image.resize(img, target_size)

    # Normalize the image to range [0, 1]
    img = img / 255.0

    return img

# Process all images in the dataset
def process_dataset(dataset_dir, output_dir, target_size):
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file)
                
                # Process the image
                processed_img = process_image(file_path, target_size)
                
                # Save the processed image as a TensorFlow tensor
                output_path = os.path.join(output_dir, file)
                tf.io.write_file(
                    output_path,
                    tf.image.encode_jpeg(tf.cast(processed_img * 255, tf.uint8))
                )
    print("Processing complete. Processed images saved to:", output_dir)

# Run the processing function
process_dataset(EXTRACT_TO, OUTPUT_DIR, TARGET_SIZE)