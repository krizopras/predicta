#!/usr/bin/env python3
"""
Predicta Europe ML v2 - Flask Backend (FULL WORKING VERSION)
Auto bulletin fetcher (Nesine) + ML predictions
"""

import os
import logging
from datetime import date
from flask import Flask, jsonify, request
from flask_cors import CORS

# ========== ML & Nesine imports ==========
from ml_prediction_engine import MLPredictionEngine
from nesine_fetcher import fetch_bulletin

# ========== Logging ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Predicta")

# ========== Config ==========
APP_PORT = int(os.environ.get("PORT", 8000))
MODELS_DIR = os.environ.get("PREDICTA_MODELS_DIR", "data/ai_models_v2")

# ========== Flask setup ==========
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ========== ML Engine ==========
engine = MLPredictionEngine(model_path=MODELS_DIR)

# ======================================================
# ROUTES
# ======================================================

@app.route("/")
def home():
    return jsonify({
        "status": "Predicta ML v2 aktif",
        "port": APP_PORT,
        "model_trained": engine.is_trained,
        "models_dir": MODELS_DIR,
        "author": "Olcay Arslan",
        "api_endpoints": [
            "/api/leagues",
            "/api/matches/today",
            "/api/predict/today",
            "/api/predictions/batch",
            "/api/debug/nesine-test"
        ]
    })


# ----------------- LIGLER -----------------
@app.route("/api/leagues", methods=["GET"])
def get_leagues():
    """Return league list from today's bulletin"""
    try:
        today = date.today()
        matches = fetch_bulletin(today, filter_leagues=False)
        leagues = sorted(set(m.get("league", "Unknown") for m in matches))
        return jsonify({"count": len(leagues), "items": leagues})
    except Exception as e:
        logger.error(f"/api/leagues error: {e}")
        return jsonify({"error": str(e), "items": []}), 500


# ----------------- TÜM MAÇLAR -----------------
@app.route("/api/matches/today", methods=["GET"])
def get_today_matches():
    """Get all matches from Nesine bulletin"""
    try:
        today = date.today()
        matches = fetch_bulletin(today, filter_leagues=False)
        return jsonify({"count": len(matches), "items": matches})
    except Exception as e:
        logger.error(f"/api/matches/today error: {e}")
        return jsonify({"error": str(e), "items": []}), 500


# ----------------- BUGÜNKÜ MAÇLAR + TAHMİN -----------------
@app.route("/api/predict/today", methods=["GET"])
def predict_today():
    """Predict all matches from today's bulletin"""
    try:
        today = date.today()
        matches = fetch_bulletin(today, filter_leagues=False)

        results = []
        for m in matches:
            try:
                home = m.get("home_team")
                away = m.get("away_team")
                odds = m.get("odds", {"1": 2.0, "X": 3.0, "2": 3.5})
                league = m.get("league", "Unknown")

                pred = engine.predict_match(home, away, odds, league)
                results.append({**m, "prediction": pred})
            except Exception as err:
                results.append({**m, "error": str(err)})

        return jsonify({"count": len(results), "items": results})
    except Exception as e:
        logger.error(f"/api/predict/today error: {e}")
        return jsonify({"error": str(e), "items": []}), 500


# ----------------- TOPLU TAHMİN -----------------
@app.route("/api/predictions/batch", methods=["POST"])
def batch_predictions():
    """Predict multiple matches from CSV"""
    data = request.get_json(silent=True) or {}
    csv = data.get("csv", "")
    matches = []

    if not csv.strip():
        return jsonify({"error": "CSV required"}), 400

    lines = [ln.strip() for ln in csv.splitlines() if ln.strip()]
    if lines and "," in lines[0].lower():
        for ln in lines[1:] if "home" in lines[0].lower() else lines:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) >= 6:
                matches.append({
                    "home_team": parts[0],
                    "away_team": parts[1],
                    "league": parts[2],
                    "odds": {"1": float(parts[3]), "X": float(parts[4]), "2": float(parts[5])}
                })

    results = []
    for m in matches:
        try:
            pred = engine.predict_match(
                m["home_team"], m["away_team"], m["odds"], m["league"]
            )
            results.append({**m, "prediction": pred})
        except Exception as e:
            results.append({**m, "error": str(e)})

    return jsonify({"count": len(results), "items": results})


# ----------------- DEBUG NESINE -----------------
@app.route("/api/debug/nesine-test", methods=["GET"])
def debug_nesine():
    try:
        today = date.today()
        matches = fetch_bulletin(today, filter_leagues=False)
        leagues = {}
        for m in matches:
            lg = m.get("league", "Unknown")
            leagues[lg] = leagues.get(lg, 0) + 1

        return jsonify({
            "status": "ok",
            "date": str(today),
            "total_matches": len(matches),
            "leagues": dict(sorted(leagues.items(), key=lambda x: x[1], reverse=True)),
            "api_url": f"https://cdnbulten.nesine.com/api/bulten/getprebultenfull?date={today}"
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("⚽ Predicta Europe ML v2 - Backend Active")
    logger.info("=" * 60)
    logger.info(f"📂 Models: {MODELS_DIR}")
    logger.info(f"🌐 Port: {APP_PORT}")
    logger.info(f"🤖 Trained: {'✅' if engine.is_trained else '⚠️ Not trained'}")
    logger.info("=" * 60)
    app.run(host="0.0.0.0", port=APP_PORT, debug=False)
