#!/usr/bin/env python3
"""
Predicta Europe ML v2 - Flask Backend
Full working version with Nesine integration + model handling
"""

import os
import logging
from datetime import date
from flask import Flask, jsonify, request
from flask_cors import CORS
from nesine_fetcher import fetch_today
from ml_prediction_engine import MLPredictionEngine

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FeatureEngineer")

# Config
APP_PORT = int(os.environ.get("PORT", 8000))
MODELS_DIR = os.environ.get("PREDICTA_MODELS_DIR", "data/ai_models_v2")
RAW_DIR = os.environ.get("PREDICTA_RAW_DIR", "data/raw")

app = Flask(__name__)
CORS(app)

# ML Engine
engine = MLPredictionEngine(model_path=MODELS_DIR)

@app.route("/")
def home():
    return jsonify({
        "status": "Predicta ML v2 active",
        "models_dir": MODELS_DIR,
        "port": APP_PORT,
        "trained": engine.is_trained
    })

@app.route("/api/matches/today", methods=["GET"])
def today_matches():
    """Fetch today’s matches from Nesine"""
    try:
        logger.info("📡 Fetching today’s matches from Nesine API...")
        matches = fetch_today(filter_leagues=False)
        return jsonify({
            "count": len(matches),
            "items": matches,
            "date": str(date.today())
        })
    except Exception as e:
        logger.error(f"❌ Nesine Fetch Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict/today", methods=["GET"])
def predict_today():
    """Predict today’s matches"""
    try:
        matches = fetch_today(filter_leagues=False)
        results = []
        for m in matches:
            pred = engine.predict_match(
                m["home_team"], m["away_team"], m.get("odds", {"1": 2.0, "X": 3.0, "2": 3.5}), m.get("league", "Unknown")
            )
            results.append({**m, "prediction": pred})
        return jsonify({"count": len(results), "items": results})
    except Exception as e:
        logger.error(f"Prediction Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("============================================================")
    logger.info("🎯 PREDICTA EUROPE ML v2")
    logger.info("============================================================")
    logger.info(f"📂 Models: {MODELS_DIR}")
    logger.info(f"📂 Raw Data: {RAW_DIR}")
    logger.info(f"🌐 Port: {APP_PORT}")
    logger.info(f"🤖 ML Model: {'✅ Trained' if engine.is_trained else '⚠️ Untrained'}")
    logger.info("============================================================")
    app.run(host="0.0.0.0", port=APP_PORT, debug=False)
