import cv2
import face_recognition
import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from uuid import uuid4
from tqdm import tqdm

# Connect to Qdrant instance
client = QdrantClient(host="localhost", port=6334, prefer_grpc=True)

# Collection parameters
collection_name = "embedding_collection1"
vector_size = 128
distance_metric = Distance.COSINE

# Create the collection
def create_collection():
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance_metric),
        )
        print(f"Collection '{collection_name}' created successfully.")
    else:
        print(f"Collection '{collection_name}' already exists.")

# Insert data into Qdrant
def insert_data(name, embedding):
    id = str(uuid4())
    points = [PointStruct(id=id, vector=embedding, payload={"name": name})]
    client.upsert(collection_name=collection_name, points=points)
    print(f"Added embedding for {name} with ID {id}.")

# Extract embeddings for an image
def get_embedding(image_path):
    img = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    if not face_locations:
        return []
    face_encodings = face_recognition.face_encodings(rgb_image, known_face_locations=face_locations)
    return face_encodings

# Train embeddings for the dataset
def train_data(dataset_path):
    if not os.path.exists(dataset_path):
        print("Dataset path does not exist.")
        return
    
    for root, dirs, files in tqdm(os.walk(dataset_path), desc="Processing Folders"):
        folder_name = os.path.basename(root)
        embeddings_list = []
        
        for file in tqdm(files, desc=f"Processing Files in {folder_name}"):
            full_path = os.path.join(root, file)
            embeddings = get_embedding(full_path)
            if embeddings:
                # embeddings_list.append(embeddings[0])
        
        # if embeddings_list:
        #     avg_embedding = np.mean(embeddings_list, axis=0)
                insert_data(folder_name, embeddings[0])

if __name__ == "__main__":
    create_collection()
    train_data(dataset_path=r"C:\Users\sijan\Desktop\jojo\yoyo")
