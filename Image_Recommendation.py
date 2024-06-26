import os
import base64
import cv2
import numpy as np
import pickle
from collections import namedtuple
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from io import BytesIO
from PIL import Image

app = FastAPI()

# Load the pre-trained EfficientNetB0 model
base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

# Define a named tuple to represent a case
Case = namedtuple('Case', ['features', 'label', 'image_path'])

# Function to preprocess an image and extract features
def extract_features(image):
    img = cv2.resize(image, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = base_model.predict(img)
    features = features.flatten()
    return features

# Function to load the case base from a file
def load_case_base(case_base_file):
    with open(case_base_file, 'rb') as f:
        return pickle.load(f)

# Function to save the case base to a file
def save_case_base(case_base, case_base_file):
    with open(case_base_file, 'wb') as f:
        pickle.dump(case_base, f)

# Function to create the case base
def create_case_base(dataset_directory):
    case_base = []
    for part_number in os.listdir(dataset_directory):
        part_path = os.path.join(dataset_directory, part_number)
        if not os.path.isdir(part_path):
            continue
        for file_name in os.listdir(part_path):
            if file_name.endswith((".jpg", ".jpeg", ".png", ".webp")):
                image_path = os.path.join(part_path, file_name)
                img = cv2.imread(image_path)
                features = extract_features(img)
                case = Case(features, part_number, image_path)# Case('features', 'label', 'image_path')
                case_base.append(case)
    return case_base

# Function to retrieve similar cases from the case base
def retrieve_similar_cases(query_features, case_base, n_neighbors=5):
    X = [case.features for case in case_base]
    y = [case.label for case in case_base]
    image_paths = [case.image_path for case in case_base]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    query_features_scaled = scaler.transform([query_features])
    neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    neigh.fit(X_scaled)
    distances, indices = neigh.kneighbors(query_features_scaled)
    similar_cases = [(case_base[idx].label, case_base[idx].image_path, 1 - dist) for idx, dist in zip(indices[0], distances[0])]
    return similar_cases

# Load or create the case base
case_base_file = 'case_base.pkl'
dataset_directory = './dataset/dataset'
case_base = load_case_base(case_base_file) if os.path.exists(case_base_file) else create_case_base(dataset_directory)

@app.post("/image_similarity_search/")
async def image_similarity_search(query_image: UploadFile = File(...)):
    # Read the uploaded image
    contents = await query_image.read()
    nparr = np.frombuffer(contents, np.uint8)
    query_image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Extract features from the query image
    query_features = extract_features(query_image_np)

    # Retrieve similar cases
    similar_cases = retrieve_similar_cases(query_features, case_base)

    # Filter cases with a similarity score of 0.6 or higher (corresponding to 60% similarity)
    filtered_cases = [case for case in similar_cases if case[2] >= 0.6]

    # Get the top similar images and their accuracy (based on the similarity score)
    top_similar_images = []
    for case in filtered_cases:
        label, image_path, similarity_score = case
        accuracy = similarity_score * 100
        top_similar_images.append({'image_path': image_path, 'accuracy': accuracy})

    # Return the top similar images and their accuracy
    response_data = {
        'similar_images': top_similar_images
    }
    return JSONResponse(content=response_data)