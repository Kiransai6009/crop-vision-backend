import os
import csv
import random
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor

DATASET_PATH = "dataset.csv"
MODEL_PATH = "yield_model.pkl"

def generate_dummy_dataset(filename=DATASET_PATH, num_samples=1000):
    """Generates a dummy dataset if it doesn't exist."""
    if os.path.exists(filename):
        return
        
    np.random.seed(42)
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["NDVI", "Rainfall", "Temperature", "Humidity", "Yield"])
        
        for _ in range(num_samples):
            ndvi = round(random.uniform(0.1, 0.9), 3)
            temp = round(random.uniform(15.0, 40.0), 1)
            rain = round(random.uniform(20.0, 300.0), 1)
            hum = round(random.uniform(30.0, 90.0), 1)
            
            # Simple dummy relationship
            y = 4.5 * ndvi - 0.05 * abs(temp - 28)**1.5 + 0.012 * rain + 0.015 * hum + random.gauss(0, 0.3)
            y = round(max(0.5, min(y, 12.0)), 2)
            
            writer.writerow([ndvi, rain, temp, hum, y])

def train_model():
    """Trains the RandomForestRegressor on dataset.csv and saves to yield_model.pkl"""
    generate_dummy_dataset(DATASET_PATH)
    
    X = []
    y = []
    
    with open(DATASET_PATH, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            X.append([
                float(row["NDVI"]),
                float(row["Rainfall"]),
                float(row["Temperature"]),
                float(row["Humidity"])
            ])
            y.append(float(row["Yield"]))
            
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X, y)
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
        
    print("Model trained and saved to", MODEL_PATH)

def predict_yield(ndvi, rainfall, temperature, humidity):
    """Loads the model and predicts crop yield."""
    if not os.path.exists(MODEL_PATH):
        train_model()
        
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        
    features = np.array([[ndvi, rainfall, temperature, humidity]])
    prediction = model.predict(features)[0]
    return float(round(prediction, 2))

if __name__ == "__main__":
    train_model()
