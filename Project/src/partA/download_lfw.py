import os
import tarfile
import urllib.request

# Define dataset URL and file name
LFW_URL = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))  # Grandparent directory
DATASET_DIR = os.path.join(PARENT_DIR, "datasets")  # Dataset folder in the grandparent directory
ARCHIVE_NAME = os.path.join(DATASET_DIR, "lfw-deepfunneled.tgz")  # Archive file in the grandparent directory


# Download the dataset
if not os.path.exists(ARCHIVE_NAME):
    print("Downloading LFW dataset...")
    urllib.request.urlretrieve(LFW_URL, ARCHIVE_NAME)
    print("Download complete.")

# Extract the dataset
if not os.path.exists(DATASET_DIR):
    print("Extracting dataset...")
    with tarfile.open(ARCHIVE_NAME, "r:gz") as tar:
        tar.extractall(DATASET_DIR)
    print("Extraction complete.")
else:
    print("Dataset already exists.")
