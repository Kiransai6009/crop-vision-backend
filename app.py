"""
CropVision – Unified Backend (MongoDB Migration)
============================================================
Consolidated Flask API using MongoDB for data storage and custom JWT Auth.

Routes (Auth):
  POST /api/auth/signup    -> Create new account
  POST /api/auth/login     -> Authenticate & get token
  GET  /api/auth/me        -> Get current user info

Routes (Data):
  POST /api/predict        -> Predict crop yield (ML formula)
  POST /api/profit         -> Calculate profit/income
  GET  /api/history        -> Get stored predictions from MongoDB
  POST /api/history        -> Save a new prediction to MongoDB
  POST /api/fertilizer     -> Fertilizer recommendations
  POST /api/risk           -> Risk alert evaluation

General:
  GET  /api/health         -> Health check
============================================================
"""

import os
import json
import random
import math
import hashlib
import uuid
import bcrypt
import jwt
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np
from dotenv import load_dotenv
from functools import wraps
from pymongo import MongoClient
from bson.objectid import ObjectId

from ml_model import predict_yield
from ndvi import calculate_ndvi, get_live_ndvi

load_dotenv()

app = Flask(__name__)
# Refined CORS: origins MUST be specific if supports_credentials=True
CORS(app, resources={r"/*": {"origins": ["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:5173"]}}, supports_credentials=True)

# --- MongoDB & Fallback Data Store ---
MONGODB_URI = os.getenv("MONGODB_URI", "")
JWT_SECRET = os.getenv("JWT_SECRET", "fallback_secret_key")

db = None
users_col = None
preds_col = None

# In-memory mock storage for testing when DB is unavailable
mock_users = []
mock_preds = []

class MockCollection:
    def __init__(self, data, name):
        self.data = data
        self.name = name
    def find_one(self, query):
        for item in self.data:
            if all(item.get(k) == v for k, v in query.items()):
                return item
        return None
    def insert_one(self, doc):
        self.data.append(doc)
        return doc
    def update_one(self, query, update):
        item = self.find_one(query)
        if item:
            item.update(update.get("$set", {}))
        return item
    def find(self, query):
        res = [item for item in self.data if all(item.get(k) == v for k, v in query.items())]
        class Cursor:
            def __init__(self, items): self.items = items
            def sort(self, *args): return self
            def limit(self, *args): return self
            def __iter__(self): return iter(self.items)
        return Cursor(res)

if MONGODB_URI:
    print("Initializing MongoDB connection...")
    try:
        final_uri = MONGODB_URI.replace("<db_password>", os.getenv("DB_PASSWORD", "db_password"))
        client = MongoClient(final_uri, serverSelectionTimeoutMS=2000)
        client.admin.command('ping')
        db = client["crop-insight-hub"]
        users_col = db["users"]
        preds_col = db["predictions"]
        print("Successfully connected to MongoDB Atlas")
    except Exception as e:
        print(f"MongoDB connection failed: {e}. Switching to In-Memory Mock mode.")
        users_col = MockCollection(mock_users, "users")
        preds_col = MockCollection(mock_preds, "predictions")
else:
    print("No MONGODB_URI found. Using In-Memory Mock mode.")
    users_col = MockCollection(mock_users, "users")
    preds_col = MockCollection(mock_preds, "predictions")

# --- Configuration & Keys ---
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# --- Auth Middleware ---
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        request.user = {"id": "guest_id", "email": "guest@cropvision.io"} 
        
        if not auth_header or not auth_header.startswith('Bearer '):
            return f(*args, **kwargs)
        
        token = auth_header.split(' ')[1]
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            request.user = payload
        except Exception as e:
            print(f"Auth verification failed: {e}")
            
        return f(*args, **kwargs)
    return decorated

@app.route("/api/faq")
def get_faq():
    return jsonify([
        {"id": "1", "question": "How does CropVision predict yields?", "answer": "We use a Random Forest ML model that analyzes parameters like NDVI, precipitation, and temperature for your specific district.", "category": "General"},
        {"id": "2", "question": "What is NDVI?", "answer": "NDVI (Normalized Difference Vegetation Index) is a spectral measurement of plant health. Higher values (0.6 - 0.9) indicate healthy, green vegetation.", "category": "Analysis"},
        {"id": "3", "question": "How often is satellite data updated?", "answer": "Sentinel-2 satellite imagery is typically updated every 5-10 days depending on cloud cover and pass schedules.", "category": "Technical"},
        {"id": "4", "question": "Can I use CropVision offline?", "answer": "CropVision requires an internet connection to fetch real-time satellite and weather data.", "category": "Usage"}
    ])

@app.route("/api/support", methods=["POST"])
@require_auth
def submit_support():
    data = request.get_json() or {}
    user_id = request.user.get("sub")
    if not user_id or user_id == "guest_id":
        return jsonify({"error": "Unauthorized"}), 401
    
    ticket = {
        "user_id": user_id,
        "subject": data.get("subject"),
        "description": data.get("description"),
        "status": "Open",
        "created_at": datetime.utcnow()
    }
    if db is not None:
        db["support_tickets"].insert_one(ticket)
    
    return jsonify({"message": "Ticket submitted", "id": str(ticket.get("_id", "local"))}), 201

# --- Auth Routes ---

@app.route("/api/auth/signup", methods=["POST"])
def auth_signup():
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    display_name = data.get("display_name", email.split('@')[0])

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    if users_col.find_one({"email": email}):
        return jsonify({"error": "Email already registered"}), 409

    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    user_id = str(uuid.uuid4())
    users_col.insert_one({
        "user_id": user_id,
        "email": email,
        "password": hashed,
        "display_name": display_name,
        "created_at": datetime.utcnow()
    })

    # Generate token
    token = jwt.encode({
        "sub": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(days=7)
    }, JWT_SECRET, algorithm="HS256")

    return jsonify({
        "token": token,
        "user": {"id": user_id, "email": email, "display_name": display_name}
    }), 201

@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    user = users_col.find_one({"email": email})
    if not user or not bcrypt.checkpw(password.encode('utf-8'), user["password"]):
        return jsonify({"error": "Invalid login credentials"}), 401

    token = jwt.encode({
        "sub": user["user_id"],
        "email": email,
        "exp": datetime.utcnow() + timedelta(days=7)
    }, JWT_SECRET, algorithm="HS256")

    return jsonify({
        "token": token,
        "user": {"id": user["user_id"], "email": email, "display_name": user.get("display_name")}
    })

@app.route("/api/auth/me", methods=["GET"])
@require_auth
def auth_me():
    user_id = request.user.get("sub")
    if not user_id or user_id == "guest_id":
        return jsonify({"error": "Unauthorized"}), 401
    
    user = users_col.find_one({"user_id": user_id})
    if not user:
        return jsonify({"error": "User not found"}), 404
        
    return jsonify({
        "id": user["user_id"],
        "email": user["email"],
        "display_name": user.get("display_name")
    })

@app.route("/api/profile/update", methods=["POST"])
@require_auth
def update_profile():
    data = request.get_json() or {}
    user_id = request.user.get("sub")
    if not user_id or user_id == "guest_id":
        return jsonify({"error": "Unauthorized"}), 401
    
    display_name = data.get("display_name")
    if not display_name:
        return jsonify({"error": "Display name required"}), 400
        
    if users_col is not None:
        users_col.update_one({"user_id": user_id}, {"$set": {"display_name": display_name}})
    
    return jsonify({"message": "Profile updated", "display_name": display_name})

# --- Prediction Logic ---

def save_prediction(user_id, crop, ndvi, yield_result):
    if preds_col is not None:
        try:
            preds_col.insert_one({
                "user_id": user_id,
                "crop": crop,
                "ndvi_value": float(ndvi),
                "predicted_yield": float(yield_result),
                "created_at": datetime.utcnow()
            })
        except Exception as e:
            print(f"MongoDB insert error: {e}")

# --- Location Database (Mock) ---
LOCATIONS = {
    "India": {
        "Andhra Pradesh": {
            "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "crop": "Rice"},
            "Vijayawada":    {"lat": 16.5062, "lon": 80.6480, "crop": "Cotton"},
            "Guntur":        {"lat": 16.3067, "lon": 80.4365, "crop": "Chilli"},
            "Tirupati":      {"lat": 13.6288, "lon": 79.4192, "crop": "Groundnut"},
            "Kurnool":       {"lat": 15.8281, "lon": 78.0373, "crop": "Jowar"},
            "Nellore":       {"lat": 14.4426, "lon": 79.9865, "crop": "Rice"},
            "Kadapa":        {"lat": 14.4674, "lon": 78.8241, "crop": "Tomato"},
        },
        "Telangana": {
            "Hyderabad":     {"lat": 17.3850, "lon": 78.4867, "crop": "Maize"},
            "Warangal":      {"lat": 17.9689, "lon": 79.5941, "crop": "Cotton"},
        },
        "Maharashtra": {
            "Pune":          {"lat": 18.5204, "lon": 73.8567, "crop": "Sugarcane"},
            "Nashik":        {"lat": 19.9975, "lon": 73.7898, "crop": "Grapes"},
            "Nagpur":        {"lat": 21.1458, "lon": 79.0882, "crop": "Orange"},
        }
    }
}

# --- Shared Logic ---
def get_weather(lat: float, lon: float):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,weather_code"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        d = resp.json()["current"]
        return {
            "temperature": round(d["temperature_2m"], 1),
            "humidity": d["relative_humidity_2m"],
            "rainfall": round(d["precipitation"] * 30, 1), 
            "description": "Live Data",
            "source": "Open-Meteo"
        }
    except Exception:
        return {"temperature": 28.5, "humidity": 65, "rainfall": 120.0, "description": "Simulated", "source": "Fallback"}

def compute_ndvi_sim(lat: float, lon: float, month: int) -> float:
    is_monsoon = 6 <= month <= 9
    base = 0.55 if abs(lat) < 23.5 else 0.45
    seas = 0.15 if is_monsoon else 0.05
    return round(max(0.05, min(0.95, base + seas + random.uniform(-0.05, 0.05))), 3)

def compute_soil_moisture(temp: float, humidity: float, rainfall: float) -> float:
    return round(max(0.1, min(0.9, 0.1 + humidity / 200 + rainfall / 600)), 3)

@app.route("/api/health")
def health_check():
    return jsonify({"status": "healthy", "service": "MongoDB Backend", "version": "3.0.0"})

@app.route("/api/dashboard", methods=["GET"])
def get_dashboard():
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    if lat is None or lon is None: return jsonify({"error":"Missing coords"}), 400
    
    w = get_weather(lat, lon)
    month = datetime.now().month
    ndvi = compute_ndvi_sim(lat, lon, month)
    moist = compute_soil_moisture(w["temperature"], w["humidity"], w["rainfall"])
    
    pred = predict_yield(ndvi, w["rainfall"], w["temperature"], w["humidity"])
    
    return jsonify({
        "ndvi": {"current": ndvi, "soil_moisture": moist},
        "yield": {"predicted": round(pred, 2), "confidence": 85},
        "weather": w,
        "generated_at": datetime.utcnow().isoformat()
    })

@app.route("/api/predict", methods=["POST"])
@require_auth
def api_predict():
    data = request.get_json() or {}
    rain = float(data.get("rainfall", 100))
    temp = float(data.get("temperature", 28))
    hum  = float(data.get("humidity", 65))
    ndvi = float(data.get("ndvi", 0.5))
    crop = data.get("crop", "Unknown")
    
    pred = predict_yield(ndvi, rain, temp, hum)
    
    user_id = request.user.get("sub")
    if user_id and user_id != "guest_id":
        save_prediction(user_id, crop, ndvi, round(pred, 2))
    
    return jsonify({
        "predicted_yield": round(pred, 2),
        "confidence": 85.0
    })

@app.route("/api/history", methods=["GET"])
@require_auth
def api_get_history():
    user_id = request.user.get("sub")
    if user_id and user_id != "guest_id" and preds_col is not None:
        try:
            cursor = preds_col.find({"user_id": user_id}).sort("created_at", -1).limit(50)
            history = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                history.append(doc)
            return jsonify(history)
        except Exception as e:
            print(f"MongoDB history fetch error: {e}")
    
    return jsonify([])

@app.route("/api/history", methods=["POST"])
@require_auth
def api_save_history():
    data = request.get_json() or {}
    user_id = request.user.get("sub")
    if not user_id or user_id == "guest_id":
        return jsonify({"error": "Unauthorized"}), 401
        
    entry = {
        "user_id": user_id,
        "crop": data.get("crop", "Unknown"),
        "ndvi_value": float(data.get("ndvi", 0.5)),
        "predicted_yield": float(data.get("predicted_yield", 0)),
        "created_at": datetime.utcnow()
    }
    
    if preds_col is not None:
        preds_col.insert_one(entry)
    
    entry["_id"] = "local" # For JSON responsiveness
    return jsonify({"message": "Saved", "entry": entry}), 201

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port, host="0.0.0.0")
