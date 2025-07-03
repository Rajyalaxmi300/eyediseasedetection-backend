import os
import pymongo

# MongoDB Connection
MONGO_URI = "mongodb://localhost:27017/"
client = pymongo.MongoClient(MONGO_URI)
db = client["EyeDiseaseDB"]
collection = db["images"]

# Dataset Path (Update this!)
DATASET_PATH = r"C:\Users\hp\OneDrive\Desktop\random2\innovatrix\dataset\Eye_diseases"

# Clear old data (optional)
collection.delete_many({})  

# Store image metadata
image_data = []

for disease in os.listdir(DATASET_PATH):
    disease_path = os.path.join(DATASET_PATH, disease)

    if os.path.isdir(disease_path):  # Ensure it's a folder
        for image_name in os.listdir(disease_path):
            if image_name.lower().endswith((".jpg", ".jpeg", ".png")):  # Only images
                image_path = os.path.join(disease_path, image_name)

                image_data.append({
                    "filename": image_name,
                    "label": disease,  # Disease category
                    "path": image_path
                })

# Insert data into MongoDB
if image_data:
    collection.insert_many(image_data)
    print(f"✅ Successfully uploaded {len(image_data)} images to MongoDB!")
else:
    print("❌ No images found!")

