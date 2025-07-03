import os
import pymongo
import pandas as pd

# MongoDB connection (Update with your MongoDB Atlas link)
MONGO_URI = "mongodb://localhost:27017/"  # Use MongoDB Atlas URI if needed
DATABASE_NAME = "eyediseaseDB"
COLLECTION_NAME = "images"

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Define dataset folder path
DATASET_PATH = "dataset/"  # Ensure dataset is stored here

# Prepare dataset for MongoDB
image_data = []

for disease in os.listdir(DATASET_PATH):
    disease_path = os.path.join(DATASET_PATH, disease)
    if os.path.isdir(disease_path):  # Ensure it's a folder
        for image_name in os.listdir(disease_path):
            image_path = os.path.join(disease_path, image_name)
            image_data.append({
                "filename": image_name,
                "label": disease,
                "path": image_path
            })

# Insert into MongoDB
if image_data:
    collection.insert_many(image_data)
    print(f"Inserted {len(image_data)} images into MongoDB.")

# Verify data insertion
print("Sample Data:", collection.find_one())
