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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
# Enable CORS for all routes and origins
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- Configuration & Keys ---
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
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

# --- ML Model Setup ---
def train_yield_model():
    np.random.seed(42)
    n_samples = 1000
    ndvi = np.random.uniform(0.1, 0.9, n_samples)
    temp = np.random.uniform(15, 40, n_samples)
    rain = np.random.uniform(20, 300, n_samples)
    hum  = np.random.uniform(30, 90, n_samples)
    moist = np.random.uniform(0.1, 0.8, n_samples)
    
    y = (4.5 * ndvi - 0.05 * np.abs(temp - 28)**1.5 + 0.012 * rain + 0.015 * hum + 2.0 * moist + np.random.normal(0, 0.3, n_samples))
    y = np.clip(y, 0.5, 12)
    
    X = np.column_stack([ndvi, temp, rain, hum, moist])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

MODEL, SCALER = train_yield_model()

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
    X = np.array([[ndvi, w["temperature"], w["rainfall"], w["humidity"], moist]])
    pred = MODEL.predict(SCALER.transform(X))[0]
    
    h = get_health_status(ndvi)
    
    crop = request.args.get("crop", default="Mixed")
    
    return jsonify({
        "location": {"lat": lat, "lon": lon, "name": request.args.get("name", "Selected"), "crop": crop},
        "weather": {**w, "feels_like": w["temperature"], "wind_speed": 12, "pressure": 1013},
        "ndvi": {"current": ndvi, "soil_moisture": moist, "trend": get_ndvi_trend(lat, lon)},
        "yield": {"predicted": round(pred, 2), "confidence": 85, "crop": crop, "history": get_yield_history(lat, lon, pred)},
        "health": h,
        "rainfall_trend": get_rainfall_trend(lat, lon, w["rainfall"]),
        "generated_at": datetime.now().isoformat()
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json() or {}
    rain = float(data.get("rainfall", 100))
    temp = float(data.get("temperature", 28))
    hum  = float(data.get("humidity", 65))
    ndvi = float(data.get("ndvi", 0.5))
    
    moist = compute_soil_moisture(temp, hum, rain)
    X = np.array([[ndvi, temp, rain, hum, moist]])
    pred = MODEL.predict(SCALER.transform(X))[0]
    
    h = get_health_status(ndvi)
    return jsonify({
        "ndvi": ndvi,
        "predicted_yield": round(pred, 2),
        "crop_status": h["status"],
        "confidence": 85
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

@app.route("/api/risk", methods=["POST"])
def api_risk():
    data = request.get_json() or {}
    rain = float(data.get("rainfall", 100))
    alerts = []
    if rain < 50: alerts.append({"type":"Drought", "severity":"Critical"})
    return jsonify({"risk_count": len(alerts), "alerts": alerts})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port, host="0.0.0.0")
