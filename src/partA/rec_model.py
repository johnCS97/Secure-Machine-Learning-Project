from deepface import DeepFace
import os

# Path to the LFW dataset
lfw_path = "../../datasets/lfw_deepfunneled"

# List all people in the dataset
people = os.listdir(lfw_path)
print(people)
# Verify two images of the same person
img1_path = os.path.join(lfw_path, people[0], os.listdir(os.path.join(lfw_path, people[0]))[0])
img2_path = os.path.join(lfw_path, people[0], os.listdir(os.path.join(lfw_path, people[0]))[1])

result = DeepFace.verify(img1_path, img2_path, model_name="VGG-Face")
print("Verification Result:", result)
