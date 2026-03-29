"""
CropVision – Unified Backend
============================================================
Consolidated Flask API supporting both the React frontend 
and the prototype dashboard.

Routes (React):
  POST /api/predict        -> Predict crop yield (ML formula)
  POST /api/profit         -> Calculate profit/income
  GET  /api/history        -> Get stored predictions
  POST /api/history        -> Save a new prediction
  POST /api/fertilizer     -> Fertilizer recommendations
  POST /api/risk           -> Risk alert evaluation

Routes (Prototype):
  GET  /api/locations      -> Get available locations
  GET  /api/dashboard      -> Detailed dashboard data (Weather, NDVI, Yield)

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
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np
from dotenv import load_dotenv
import jwt
from functools import wraps
from supabase import create_client, Client

from ml_model import predict_yield
from ndvi import calculate_ndvi, get_live_ndvi

load_dotenv()

# --- Config and Supabase Initialization ---
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")

supabase = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    except Exception as e:
        print(f"Supabase init error: {e}")

app = Flask(__name__)
# Enable CORS for all routes and origins
CORS(app)

# --- Configuration & Keys ---
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
history_store = []

@app.route("/api/health")
def health_check():
    return jsonify({"status": "healthy", "service": "Crop Intelligence Backend", "version": "2.0.0"})

@app.route("/api/faq")
def get_mock_faq():
    return jsonify([
        {"id": "1", "question": "What is Crop Intelligence?", "answer": "Advanced AI for harvest optimization.", "category": "General"},
        {"id": "2", "question": "Is satellite data real-time?", "answer": "Yes, updated every 5-10 days via Sentinel-2.", "category": "Technical"}
    ])

def save_prediction(user_id, crop, ndvi, yield_result):
    if supabase:
        try:
            supabase.table("predictions").insert({
                "user_id": user_id,
                "crop": crop,
                "ndvi_value": float(ndvi),
                "predicted_yield": float(yield_result),
                # omitted 'id' and 'created_at' as Supabase will generate them automatically if set in DB default, 
                # but if user constraints specifically mention them: "columns: id, user_id, crop, ndvi_value, predicted_yield, created_at".
                # Supabase handles UUID and timestamps via default values, so usually we don't pass them explicitly, but I'll add them if needed. (I'll let DB handle defaults)
            }).execute()
        except Exception as e:
            print(f"Supabase insert error: {e}")

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Unauthorized, missing token"}), 401
        
        token = auth_header.split(' ')[1]
        try:
            # We skip audience here for compatibility, or check 'authenticated'
            payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], options={"verify_audience": False})
            request.user = payload
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except Exception as e:
            return jsonify({"error": "Invalid token", "message": str(e)}), 401
        return f(*args, **kwargs)
    return decorated
history_store = []

# --- Location Database ---
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
            "Nizamabad":     {"lat": 18.6725, "lon": 78.0941, "crop": "Turmeric"},
            "Karimnagar":    {"lat": 18.4386, "lon": 79.1288, "crop": "Rice"},
            "Khammam":       {"lat": 17.2473, "lon": 80.1514, "crop": "Cotton"},
        },
        "Maharashtra": {
            "Pune":          {"lat": 18.5204, "lon": 73.8567, "crop": "Sugarcane"},
            "Nashik":        {"lat": 19.9975, "lon": 73.7898, "crop": "Grapes"},
            "Nagpur":        {"lat": 21.1458, "lon": 79.0882, "crop": "Orange"},
            "Aurangabad":    {"lat": 19.8762, "lon": 75.3433, "crop": "Cotton"},
            "Amravati":      {"lat": 20.9320, "lon": 77.7523, "crop": "Soybean"},
        },
        "Punjab": {
            "Ludhiana":      {"lat": 30.9010, "lon": 75.8573, "crop": "Wheat"},
            "Amritsar":      {"lat": 31.6340, "lon": 74.8723, "crop": "Wheat"},
            "Jalandhar":     {"lat": 31.3260, "lon": 75.5762, "crop": "Rice"},
            "Patiala":       {"lat": 30.3398, "lon": 76.3869, "crop": "Wheat"},
            "Bathinda":      {"lat": 30.2110, "lon": 74.9455, "crop": "Cotton"},
        },
        "Karnataka": {
            "Bengaluru":     {"lat": 12.9716, "lon": 77.5946, "crop": "Ragi"},
            "Mysuru":        {"lat": 12.2958, "lon": 76.6394, "crop": "Sugarcane"},
            "Hubli":         {"lat": 15.3647, "lon": 75.1240, "crop": "Cotton"},
            "Belagavi":      {"lat": 15.8497, "lon": 74.4977, "crop": "Sugarcane"},
            "Dharwad":       {"lat": 15.4589, "lon": 75.0078, "crop": "Soybean"},
        },
        "Tamil Nadu": {
            "Chennai":       {"lat": 13.0827, "lon": 80.2707, "crop": "Rice"},
            "Coimbatore":    {"lat": 11.0168, "lon": 76.9558, "crop": "Cotton"},
            "Madurai":       {"lat":  9.9252, "lon": 78.1198, "crop": "Banana"},
            "Salem":         {"lat": 11.6643, "lon": 78.1460, "crop": "Turmeric"},
            "Thanjavur":     {"lat": 10.7870, "lon": 79.1378, "crop": "Rice"},
        },
        "Uttar Pradesh": {
            "Lucknow":       {"lat": 26.8467, "lon": 80.9462, "crop": "Sugarcane"},
            "Kanpur":        {"lat": 26.4499, "lon": 80.3319, "crop": "Wheat"},
            "Agra":          {"lat": 27.1767, "lon": 78.0081, "crop": "Wheat"},
            "Varanasi":      {"lat": 25.3176, "lon": 82.9739, "crop": "Rice"},
            "Meerut":        {"lat": 28.9845, "lon": 77.7064, "crop": "Sugarcane"},
        },
    }
}

# Removed dynamic ML training, importing from ml_model.py

# --- Helpers ---
def stable_rand(seed_str: str, lo: float, hi: float) -> float:
    h = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
    return lo + (h % 10000) / 10000 * (hi - lo)

def compute_ndvi_sim(lat: float, lon: float, month: int) -> float:
    is_monsoon = 6 <= month <= 9
    base = 0.55 if abs(lat) < 23.5 else 0.45
    seas = 0.15 if is_monsoon else 0.05
    noise = stable_rand(f"{lat:.2f}{lon:.2f}{month}", -0.05, 0.05)
    return round(max(0.05, min(0.95, base + seas + noise)), 3)

def compute_soil_moisture(temp: float, humidity: float, rainfall: float) -> float:
    return round(max(0.1, min(0.9, 0.1 + humidity / 200 + rainfall / 600)), 3)

def get_weather(lat: float, lon: float):
    # Try Open-Meteo
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,weather_code"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        d = resp.json()["current"]
        return {
            "temperature": round(d["temperature_2m"], 1),
            "humidity": d["relative_humidity_2m"],
            "rainfall": round(d["precipitation"] * 30, 1), # Monthly estimate
            "description": "Live Data",
            "source": "Open-Meteo"
        }
    except Exception as e:
        return {
            "temperature": 28.5, "humidity": 65, "rainfall": 120.0,
            "description": f"Simulated (Error: {str(e)})", "source": "Fallback"
        }

@app.route("/api/weather", methods=["GET"])
def api_weather():
    lat = request.args.get("lat", type=float, default=20.0)
    lon = request.args.get("lon", type=float, default=80.0)
    return jsonify({
        "current": get_weather(lat, lon),
        "weekly_forecast": [],
        "irrigation_advisory": "Maintain regular watering schedule."
    })

def get_ndvi_trend(lat: float, lon: float) -> list:
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    return [{"month": m, "ndvi": compute_ndvi_sim(lat, lon, i + 1)} for i, m in enumerate(months)]

def get_rainfall_trend(lat: float, lon: float, base_rain: float) -> list:
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monsoon_mult = [0.3,0.2,0.3,0.5,0.8,1.8,2.5,2.3,1.6,0.7,0.4,0.3]
    return [{"month": m, "rainfall": round(base_rain * monsoon_mult[i] + stable_rand(f"{lat:.1f}{lon:.1f}{i}r", -20, 20), 1)} for i, m in enumerate(months)]

def get_yield_history(lat: float, lon: float, predicted_yield: float) -> list:
    return [
        {"year": 2020, "actual": round(predicted_yield * stable_rand(f"{lat}{lon}2020", 0.75, 0.95), 2),
                        "predicted": round(predicted_yield * stable_rand(f"{lat}{lon}p2020", 0.78, 0.95), 2)},
        {"year": 2021, "actual": round(predicted_yield * stable_rand(f"{lat}{lon}2021", 0.80, 1.00), 2),
                        "predicted": round(predicted_yield * stable_rand(f"{lat}{lon}p2021", 0.82, 0.98), 2)},
        {"year": 2022, "actual": round(predicted_yield * stable_rand(f"{lat}{lon}2022", 0.85, 1.05), 2),
                        "predicted": round(predicted_yield * stable_rand(f"{lat}{lon}p2022", 0.88, 1.05), 2)},
        {"year": 2023, "actual": round(predicted_yield * stable_rand(f"{lat}{lon}2023", 0.88, 1.08), 2),
                        "predicted": round(predicted_yield * stable_rand(f"{lat}{lon}p2023", 0.90, 1.06), 2)},
        {"year": 2024, "actual": round(predicted_yield * stable_rand(f"{lat}{lon}2024", 0.90, 1.10), 2),
                        "predicted": round(predicted_yield, 2)},
        {"year": 2025, "actual": None, "predicted": round(predicted_yield * 1.04, 2)},
    ]

# --- Shared Logic ---
def get_health_status(ndvi):
    if ndvi >= 0.75: return {"status":"Excellent","color":"#00e676","icon":"🌿","score":95}
    if ndvi >= 0.60: return {"status":"Good","color":"#69f0ae","icon":"🌱","score":80}
    if ndvi >= 0.45: return {"status":"Moderate","color":"#ffeb3b","icon":"🌾","score":65}
    if ndvi >= 0.30: return {"status":"Poor","color":"#ff9800","icon":"⚠️","score":45}
    return {"status":"Critical","color":"#f44336","icon":"🚨","score":20}

# --- ROUTES ---

@app.route("/", methods=["GET"])
def index():
    return "Crop Yield Prediction Backend Running"

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "CropVision Unified API is running"})

@app.route("/api/locations", methods=["GET"])
def get_locations():
    result = {}
    for country, states in LOCATIONS.items():
        result[country] = {}
        for state, districts in states.items():
            result[country][state] = [
                {"name": d, "lat": v["lat"], "lon": v["lon"], "crop": v["crop"]}
                for d, v in districts.items()
            ]
    return jsonify(result)

@app.route("/api/dashboard", methods=["GET"])
def get_dashboard():
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    if lat is None or lon is None: return jsonify({"error":"Missing coords"}), 400
    
    w = get_weather(lat, lon)
    month = datetime.now().month
    ndvi = compute_ndvi_sim(lat, lon, month)
    moist = compute_soil_moisture(w["temperature"], w["humidity"], w["rainfall"])
    
    # Predict
    pred = predict_yield(ndvi, w["rainfall"], w["temperature"], w["humidity"])
    
    h = get_health_status(ndvi)
    
    crop = request.args.get("crop", default="Mixed")
    
    # Return hybrid payload for both the specified requirement and exact frontend needs
    return jsonify({
        "crop_health": h["status"],
        "ndvi_value": ndvi,
        "rainfall": w["rainfall"],
        "temperature": w["temperature"],
        "humidity": w["humidity"],
        "location": {"lat": lat, "lon": lon, "name": request.args.get("name", "Selected"), "crop": crop},
        "weather": {**w, "feels_like": w["temperature"], "wind_speed": 12, "pressure": 1013},
        "ndvi": {"current": ndvi, "soil_moisture": moist, "trend": get_ndvi_trend(lat, lon)},
        "yield": {"predicted": round(pred, 2), "confidence": 85, "crop": crop, "history": get_yield_history(lat, lon, pred)},
        "health": h,
        "rainfall_trend": get_rainfall_trend(lat, lon, w["rainfall"]),
        "generated_at": datetime.now().isoformat()
    })

@app.route("/ndvi", methods=["POST"])
@require_auth
def api_ndvi_live():
    data = request.get_json() or {}
    lat = data.get("lat")
    lon = data.get("lon")
    start = data.get("start")
    end = data.get("end")
    
    if lat is None or lon is None or start is None or end is None:
        return jsonify({"error": "Missing parameters. Need lat, lon, start, end"}), 400

    # Local file-based caching
    cache_file = "ndvi_cache.json"
    cache_key = f"{lat}_{lon}_{start}_{end}"
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
                if cache_key in cache_data:
                    return jsonify({"mean_ndvi": cache_data[cache_key]["ndvi"], "source": "cache"})
        except Exception as e:
            print(f"Cache read error: {e}")

    # Fetch live NDVI
    result = get_live_ndvi(lat, lon, start, end)
    
    # Save to Cache
    try:
        cache_data = {}
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
        
        cache_data[cache_key] = {"ndvi": result["ndvi"], "source": result["source"], "cached_at": datetime.now().isoformat()}
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)
    except Exception as e:
        print(f"Cache write error: {e}")

    return jsonify({"mean_ndvi": result["ndvi"], "source": result["source"]})

@app.route("/api/predict", methods=["POST"])
@require_auth
def api_predict():
    data = request.get_json() or {}
    rain = float(data.get("rainfall", 100))
    temp = float(data.get("temperature", 28))
    hum  = float(data.get("humidity", 65))
    ndvi = float(data.get("ndvi", 0.5))
    crop = data.get("crop", "Unknown")
    
    moist = compute_soil_moisture(temp, hum, rain)
    pred = predict_yield(ndvi, rain, temp, hum)
    
    h = get_health_status(ndvi)
    
    # Save to supabase
    user_id = request.user.get("sub")
    if user_id:
        save_prediction(user_id, crop, ndvi, round(pred, 2))
    
    # Modified to match specified JSON requirement precisely
    return jsonify({
        "predicted_yield": round(pred, 2),
        "confidence": 85.0
    })

@app.route("/api/profit", methods=["POST"])
def api_profit():
    data = request.get_json() or {}
    yield_val = float(data.get("yield_per_ha", 3.0))
    area = float(data.get("area_ha", 1.0))
    price = float(data.get("price_per_ton", 20000))
    
    income = yield_val * area * price
    cost = 30000 * area # Simplified
    return jsonify({
        "total_yield_tons": round(yield_val * area, 2),
        "gross_income": round(income, 2),
        "total_cost": round(cost, 2),
        "net_profit": round(income - cost, 2)
    })

@app.route("/history", methods=["GET"])
@require_auth
def get_history_route():
    # User requested: "Add a GET /history route that fetches the last 10 predictions for the authenticated user from Supabase"
    user_id = request.user.get("sub")
    if supabase and user_id:
        try:
            resp = supabase.table("predictions").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(10).execute()
            return jsonify(resp.data)
        except Exception as e:
            print(f"Supabase History Error (Falling back): {e}")
            # Fallback to local store
    
    # Fallback/Local Intelligence if Supabase unreachable
    user_history = [entry for entry in history_store if entry.get("user_id") == user_id]
    return jsonify(user_history[-10:])

@app.route("/api/history", methods=["GET", "POST"])
def api_history():
    if request.method == "POST":
        data = request.get_json() or {}
        entry = {**data, "id": str(uuid.uuid4()), "date": datetime.now().isoformat()}
        history_store.append(entry)
        return jsonify(entry), 201
    return jsonify(history_store)

@app.route("/api/fertilizer", methods=["POST"])
def api_fertilizer():
    data = request.get_json() or {}
    ndvi = float(data.get("ndvi", 0.5))
    if ndvi < 0.3:
        recs = [{"name": "Urea", "dose": "120kg/ha"}]
    else:
        recs = [{"name": "NPK", "dose": "60kg/ha"}]
    return jsonify({"recommendations": recs})

# --- AI CHAT ASSISTANT ---

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json() or {}
    messages = data.get("messages", [])
    
    # SYSTEM PROMPT
    system_prompt = (
        "You are the CropVision AI Assistant. You help farmers and researchers understand "
        "crop yield predictions, NDVI spectral data, and weather impacts. "
        "Knowledge areas: Rice, Cotton, Chilli, Groundnut, Sugarcane, Grapes, Wheat, Turmeric. "
        "Tone: Helpful, professional, concise. Use markdown for lists or bold text."
    )

    # 1. Check for Gemini Key
    gemini_key = os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        try:
            # Simple direct API call to Gemini (non-streaming for simplicity in this bridge)
            # You can also use 'google-generativeai' library if it's in requirements
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}"
            headers = {"Content-Type": "application/json"}
            
            # Format history for Gemini
            contents = []
            for m in messages:
                role = "user" if m["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": m["content"]}]})
            
            # Prepend system instruction as a user message if needed, or use specific param
            body = {
                "contents": contents,
                "system_instruction": {"parts": [{"text": system_prompt}]}
            }
            
            resp = requests.post(url, headers=headers, json=body, timeout=10)
            if resp.ok:
                result = resp.json()
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                return jsonify({"role": "assistant", "content": text})
        except Exception as e:
            print(f"Gemini error: {e}")

    # 2. Fallback Intelligence (If Key missing or error)
    user_msg = messages[-1]["content"].lower() if messages else ""
    
    # Basic Rule-based Fallback
    if any(k in user_msg for k in ["hello", "hi", "who are you"]):
        reply = "Hello! I am your CropVision Assistant. How can I help you with your agriculture data today?"
    elif "ndvi" in user_msg:
        reply = "Normalized Difference Vegetation Index (NDVI) measures plant health by comparing reflected Red and Near-Infrared light. Healthy plants reflect more NIR. In CropVision, anything above 0.6 is considered healthy vegetation."
    elif any(k in user_msg for k in ["yield", "prediction", "predict"]):
        reply = "I use a Random Forest Regression model to predict yields. It takes into account your local Rainfall, Temperature, Humidity, and satellite-derived NDVI values."
    elif "weather" in user_msg:
        reply = "I integrate live data from Open-Meteo to provide real-time updates on temperature and precipitation for your specific coordinates."
    elif "crop" in user_msg:
        reply = "CropVision currently supports a wide variety of Indian crops including Rice, Cotton, Sugarcane, Wheat, and specialized crops like Turmeric and Orange depending on your selected state."
    else:
        reply = "That's an interesting question! While I'm in fallback mode, I can specifically help you with details about **NDVI analysis**, **Yield Predictions**, or **Weather Integration**. What would you like to know more about?"

    return jsonify({"role": "assistant", "content": reply})

@app.route("/api/risk", methods=["POST"])
def api_risk():
    data = request.get_json() or {}
    rain = float(data.get("rainfall", 100))
    alerts = []
    if rain < 50: alerts.append({"type":"Drought", "severity":"Critical"})
    return jsonify({"risk_count": len(alerts), "alerts": alerts})

# --- Passwords & Security ---
MOCK_TOKENS = {}

@app.route("/forgot-password", methods=["POST"])
def forgot_password():
    data = request.get_json() or {}
    email = data.get("email")
    if not email:
        return jsonify({"error": "Email is required"}), 400
    
    # In real app: Check if email exists in DB
    # Generate secure token
    token = hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()
    MOCK_TOKENS[token] = email
    
    # Real app: Send email with: http://localhost:8081/reset-password?token=token
    print(f"PASSWORD RESET LINK: http://localhost:8081/reset-password?token={token}")
    
    return jsonify({"message": "Password reset link sent to your email", "token_preview_for_dev": token})

@app.route("/reset-password", methods=["POST"])
def reset_password():
    data = request.get_json() or {}
    token = data.get("token")
    new_password = data.get("new_password")
    
    if not token or not new_password:
        return jsonify({"error": "Missing token or new_password"}), 400
    
    if token not in MOCK_TOKENS:
        return jsonify({"error": "Invalid or expired token"}), 400
    
    email = MOCK_TOKENS.pop(token)
    
    # Real app: Update password in DB
    print(f"Updating password for {email} to {new_password}")
    
    return jsonify({"message": "Password updated successfully"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port, host="0.0.0.0")
