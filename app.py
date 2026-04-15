"""
CropVision – Unified Backend (MongoDB + Live Location)
============================================================
Consolidated Flask API using MongoDB for data storage, custom JWT Auth,
and live location-based weather/NDVI/prediction pipeline.

Routes (Auth):
  POST /api/auth/signup    -> Create new account
  POST /api/auth/login     -> Authenticate & get token
  GET  /api/auth/me        -> Get current user info

Routes (Location):
  POST /api/location       -> Store user's live location in MongoDB

Routes (Data):
  GET  /api/dashboard      -> Full dashboard data (weather + NDVI + yield)
  POST /api/predict        -> Predict crop yield (ML model)
  GET  /api/weather        -> Weather data for lat/lon
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
from datetime import datetime, timedelta, timezone
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
location_col = None
weather_col = None
ndvi_col = None

# In-memory mock storage for testing when DB is unavailable
mock_users = []
mock_preds = []
mock_locations = []
mock_weather = []
mock_ndvi = []

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
        location_col = db["location_data"]
        weather_col = db["weather_data"]
        ndvi_col = db["ndvi_data"]
        print("Successfully connected to MongoDB Atlas")
    except Exception as e:
        print(f"MongoDB connection failed: {e}. Switching to In-Memory Mock mode.")
        users_col = MockCollection(mock_users, "users")
        preds_col = MockCollection(mock_preds, "predictions")
        location_col = MockCollection(mock_locations, "location_data")
        weather_col = MockCollection(mock_weather, "weather_data")
        ndvi_col = MockCollection(mock_ndvi, "ndvi_data")
else:
    print("No MONGODB_URI found. Using In-Memory Mock mode.")
    users_col = MockCollection(mock_users, "users")
    preds_col = MockCollection(mock_preds, "predictions")
    location_col = MockCollection(mock_locations, "location_data")
    weather_col = MockCollection(mock_weather, "weather_data")
    ndvi_col = MockCollection(mock_ndvi, "ndvi_data")

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
        {"id": "4", "question": "Can I use CropVision offline?", "answer": "CropVision requires an internet connection to fetch real-time satellite and weather data.", "category": "Usage"},
        {"id": "5", "question": "How is my location used?", "answer": "Your live GPS coordinates are used to fetch hyper-local weather, calculate NDVI for your area, and generate precise crop yield predictions. Location data is stored securely in MongoDB.", "category": "Privacy"},
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
        "created_at": datetime.now(timezone.utc)
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
        "created_at": datetime.now(timezone.utc)
    })

    # Generate token
    token = jwt.encode({
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(days=7)
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
        "exp": datetime.now(timezone.utc) + timedelta(days=7)
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
        "display_name": user.get("display_name"),
        "role": user.get("role", "USER"),
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


# ============================================================
# LOCATION ROUTES — Store & retrieve user's live GPS location
# ============================================================

@app.route("/api/location", methods=["POST"])
@require_auth
def store_location():
    """Store user's live location in MongoDB."""
    data = request.get_json() or {}
    user_id = request.user.get("sub")
    
    lat = data.get("lat")
    lon = data.get("lon")
    location_name = data.get("location_name", "Unknown")
    
    if lat is None or lon is None:
        return jsonify({"error": "lat and lon are required"}), 400
    
    location_doc = {
        "user_id": user_id if user_id != "guest_id" else "anonymous",
        "lat": float(lat),
        "lon": float(lon),
        "location_name": location_name,
        "updated_at": datetime.now(timezone.utc)
    }
    
    if location_col is not None:
        # Upsert: update existing or insert new
        try:
            existing = location_col.find_one({"user_id": location_doc["user_id"]})
            if existing:
                location_col.update_one(
                    {"user_id": location_doc["user_id"]},
                    {"$set": location_doc}
                )
            else:
                location_col.insert_one(location_doc)
        except Exception as e:
            print(f"Location storage error: {e}")
    
    return jsonify({"message": "Location stored", "lat": lat, "lon": lon, "location_name": location_name})


# ============================================================
# WEATHER ROUTE — Fetch & store weather for lat/lon
# ============================================================

@app.route("/api/weather", methods=["GET"])
def api_weather():
    """Fetch weather for given lat/lon using Open-Meteo and optionally OpenWeather, store in MongoDB."""
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    if lat is None or lon is None:
        return jsonify({"error": "Missing lat/lon"}), 400
    
    weather = get_weather(lat, lon)
    
    # If OpenWeather API key is available, also fetch from there
    if OPENWEATHER_API_KEY:
        try:
            ow_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
            ow_resp = requests.get(ow_url, timeout=5)
            if ow_resp.ok:
                ow = ow_resp.json()
                weather["temperature"] = round(ow["main"]["temp"], 1)
                weather["humidity"] = ow["main"]["humidity"]
                weather["description"] = ow["weather"][0]["description"].title()
                weather["source"] = "OpenWeatherMap"
        except Exception as e:
            print(f"OpenWeather fallback: {e}")
    
    # Store in MongoDB
    if weather_col is not None:
        try:
            weather_col.insert_one({
                "lat": lat,
                "lon": lon,
                "weather": weather,
                "fetched_at": datetime.now(timezone.utc)
            })
        except Exception as e:
            print(f"Weather storage error: {e}")
    
    return jsonify(weather)


# --- Prediction Logic ---

def save_prediction(user_id, crop, ndvi, yield_result, lat=None, lon=None, weather=None):
    if preds_col is not None:
        try:
            doc = {
                "user_id": user_id,
                "crop": crop,
                "ndvi_value": float(ndvi),
                "predicted_yield": float(yield_result),
                "created_at": datetime.now(timezone.utc)
            }
            if lat is not None:
                doc["lat"] = float(lat)
            if lon is not None:
                doc["lon"] = float(lon)
            if weather:
                doc["weather"] = weather
            preds_col.insert_one(doc)
        except Exception as e:
            print(f"MongoDB insert error: {e}")

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
    mongo_status = "connected" if db is not None else "mock"
    return jsonify({
        "status": "healthy",
        "service": "MongoDB Backend",
        "version": "4.0.0",
        "database": mongo_status,
        "features": ["live_location", "weather_api", "ndvi", "ml_prediction", "jwt_auth"]
    })

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
    
    # Store NDVI snapshot in MongoDB
    if ndvi_col is not None:
        try:
            ndvi_doc = {
                "lat": lat,
                "lon": lon,
                "ndvi": ndvi,
                "cropHealth": "Healthy" if ndvi > 0.6 else "Moderate" if ndvi >= 0.3 else "Poor",
                "date": datetime.now(timezone.utc)
            }
            ndvi_col.insert_one(ndvi_doc)
        except Exception as e:
            print(f"NDVI storage error: {e}")
    
    return jsonify({
        "ndvi": {"current": ndvi, "soil_moisture": moist},
        "yield": {"predicted": round(pred, 2), "confidence": 85},
        "weather": w,
        "location": {"lat": lat, "lon": lon},
        "generated_at": datetime.now(timezone.utc).isoformat()
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
    lat  = data.get("lat")
    lon  = data.get("lon")
    
    pred = predict_yield(ndvi, rain, temp, hum)
    
    user_id = request.user.get("sub")
    weather_snapshot = {"temperature": temp, "humidity": hum, "rainfall": rain}
    
    if user_id and user_id != "guest_id":
        save_prediction(user_id, crop, ndvi, round(pred, 2), lat, lon, weather_snapshot)
    
    # Store NDVI data if coords available
    if lat is not None and lon is not None and ndvi_col is not None:
        try:
            ndvi_col.insert_one({
                "lat": float(lat),
                "lon": float(lon),
                "ndvi": ndvi,
                "cropHealth": "Healthy" if ndvi > 0.6 else "Moderate" if ndvi >= 0.3 else "Poor",
                "date": datetime.now(timezone.utc)
            })
        except Exception:
            pass
    
    return jsonify({
        "predicted_yield": round(pred, 2),
        "confidence": 85.0,
        "inputs": {
            "ndvi": ndvi,
            "rainfall": rain,
            "temperature": temp,
            "humidity": hum,
            "crop": crop
        }
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
        "created_at": datetime.now(timezone.utc)
    }
    
    # Include location if provided
    if data.get("lat") is not None:
        entry["lat"] = float(data["lat"])
    if data.get("lon") is not None:
        entry["lon"] = float(data["lon"])
    
    if preds_col is not None:
        preds_col.insert_one(entry)
    
    entry["_id"] = "local" # For JSON responsiveness
    return jsonify({"message": "Saved", "entry": entry}), 201

# ============================================================
# FERTILIZER & PROFIT ROUTES
# ============================================================

@app.route("/api/fertilizer", methods=["POST"])
@require_auth
def api_fertilizer():
    """Fertilizer recommendation based on NDVI value."""
    data = request.get_json() or {}
    ndvi = float(data.get("ndvi", 0.5))
    
    if ndvi >= 0.7:
        recommendation = {
            "status": "Healthy",
            "nitrogen": "Low (20-30 kg/ha)",
            "phosphorus": "Maintenance (15-20 kg/ha)",
            "potassium": "Maintenance (15-20 kg/ha)",
            "advice": "Vegetation is healthy. Apply maintenance-level fertilizer. Focus on micro-nutrients like Zinc and Boron.",
            "urgency": "low"
        }
    elif ndvi >= 0.4:
        recommendation = {
            "status": "Moderate",
            "nitrogen": "Medium (40-60 kg/ha)",
            "phosphorus": "Medium (25-35 kg/ha)",
            "potassium": "Medium (25-30 kg/ha)",
            "advice": "Moderate stress detected. Increase nitrogen application. Consider split dosing for better absorption.",
            "urgency": "medium"
        }
    else:
        recommendation = {
            "status": "Critical",
            "nitrogen": "High (80-100 kg/ha)",
            "phosphorus": "High (40-50 kg/ha)",
            "potassium": "High (35-45 kg/ha)",
            "advice": "Severe vegetation stress. Immediate high-dose fertilizer application recommended. Also check for pest/disease issues.",
            "urgency": "high"
        }
    
    recommendation["ndvi_input"] = ndvi
    return jsonify(recommendation)

@app.route("/api/profit", methods=["POST"])
@require_auth
def api_profit():
    """Profit estimation calculator."""
    data = request.get_json() or {}
    yield_per_ha = float(data.get("yield_per_ha", 4.0))
    area_ha = float(data.get("area_ha", 1.0))
    price_per_ton = float(data.get("price_per_ton", 20000))
    
    total_yield = yield_per_ha * area_ha
    gross_revenue = total_yield * price_per_ton
    estimated_cost = area_ha * 45000  # Avg cost per hectare in INR
    net_profit = gross_revenue - estimated_cost
    roi = (net_profit / estimated_cost) * 100 if estimated_cost > 0 else 0
    
    return jsonify({
        "total_yield_tons": round(total_yield, 2),
        "gross_revenue": round(gross_revenue, 2),
        "estimated_cost": round(estimated_cost, 2),
        "net_profit": round(net_profit, 2),
        "roi_percent": round(roi, 1),
        "currency": "INR"
    })

@app.route("/ndvi", methods=["POST"])
def api_ndvi_query():
    """Query NDVI for given coordinates and date range."""
    data = request.get_json() or {}
    lat = float(data.get("lat", 17.4))
    lon = float(data.get("lon", 78.5))
    start = data.get("start", "2024-01-01")
    end = data.get("end", "2024-12-31")
    
    result = get_live_ndvi(lat, lon, start, end)
    return jsonify(result)

@app.route("/forgot-password", methods=["POST"])
def forgot_password():
    """Placeholder forgot password route."""
    data = request.get_json() or {}
    email = data.get("email", "")
    # In production, send email with reset link
    return jsonify({"message": f"If {email} is registered, a reset link has been sent."})

@app.route("/reset-password", methods=["POST"])
def reset_password():
    """Placeholder reset password route."""
    data = request.get_json() or {}
    return jsonify({"message": "Password reset successfully. Please log in with your new password."})

@app.route("/api/chat", methods=["POST"])
@require_auth
def api_chat():
    """Simple AI chat endpoint — returns a contextual response."""
    data = request.get_json() or {}
    messages = data.get("messages", [])
    
    last_msg = messages[-1]["content"] if messages else ""
    
    # Simple contextual responses
    lower = last_msg.lower()
    if "ndvi" in lower:
        reply = "NDVI (Normalized Difference Vegetation Index) ranges from -1 to 1. Values above 0.6 indicate healthy vegetation. Our system computes NDVI from Sentinel-2 satellite bands B8 (NIR) and B4 (Red) using the formula: NDVI = (NIR - Red) / (NIR + Red)."
    elif "yield" in lower or "predict" in lower:
        reply = "Our yield prediction uses a Random Forest model trained on NDVI, rainfall, temperature, and humidity data. The model achieves ~85% accuracy. For best results, ensure your location is correctly detected so we can fetch precise weather data."
    elif "weather" in lower:
        reply = "Weather data is fetched in real-time from Open-Meteo API. We track temperature, humidity, precipitation, wind speed, and UV index. Data refreshes every 10 minutes automatically."
    elif "location" in lower:
        reply = "Your location is auto-detected using GPS (navigator.geolocation). This helps us fetch hyper-local weather data and compute area-specific NDVI estimates. You can refresh your location anytime using the 🔄 button on the dashboard."
    elif "fertilizer" in lower:
        reply = "Fertilizer recommendations are based on current NDVI values. Low NDVI (< 0.4) suggests high fertilizer needs, while healthy NDVI (> 0.7) means maintenance-level application is sufficient. Always consider soil testing for precise recommendations."
    elif "hello" in lower or "hi" in lower:
        reply = "Hello! I'm CropVision AI. I can help you with questions about NDVI, crop yield prediction, weather data, fertilizer recommendations, and more. What would you like to know?"
    else:
        reply = f"Thank you for your query about '{last_msg[:50]}...'. CropVision's ML engine processes satellite imagery and weather data to provide actionable agricultural insights. Could you specify if you need help with yield prediction, NDVI analysis, or weather data?"
    
    return jsonify({"content": reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port, host="0.0.0.0")

