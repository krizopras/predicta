#!/usr/bin/env python3
"""
Predicta Europe ML v2 - Flask Backend (TRACKER FIXED)
Auto bulletin fetcher (Nesine) + ML predictions + Prediction Tracking
Author: Olcay Arslan
"""

import os
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS

# ========== ML & Nesine imports ==========
from ml_prediction_engine import MLPredictionEngine
from nesine_fetcher import (
    fetch_bulletin,
    fetch_today,
    fetch_yesterday_results,
    fetch_results_for_date
)
from historical_processor import HistoricalDataProcessor
from prediction_tracker import PredictionTracker

# ========== Logging ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Predicta")

# ========== Config ==========
APP_PORT = int(os.environ.get("PORT", 8000))
MODELS_DIR = os.environ.get("PREDICTA_MODELS_DIR", "data/ai_models_v2")
RAW_DATA_PATH = os.environ.get("PREDICTA_RAW_DATA", "data/raw")
CLUBS_PATH = os.environ.get("PREDICTA_CLUBS", "data/clubs")

# ========== Flask setup ==========
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ========== ML Engine ==========
try:
    engine = MLPredictionEngine(model_path=MODELS_DIR)
except Exception as e:
    logger.error(f"❌ Model engine yüklenemedi: {e}")
    engine = None

# ========== Historical Processor ==========
history_processor = HistoricalDataProcessor(
    raw_data_path=RAW_DATA_PATH,
    clubs_path=CLUBS_PATH
)

# ========== Prediction Tracker ==========
tracker = PredictionTracker(storage_dir="data/predictions")

# ======================================================
# ROUTES
# ======================================================

@app.route("/")
def home():
    return jsonify({
        "status": "Predicta ML v2 Active (Tracker Enabled)",
        "port": APP_PORT,
        "model_trained": getattr(engine, "is_trained", False),
        "models_dir": str(MODELS_DIR),
        "raw_data_path": str(RAW_DATA_PATH),
        "clubs_path": str(CLUBS_PATH),
        "author": "Olcay Arslan",
        "api_endpoints": [
            "/api/status",
            "/api/leagues",
            "/api/matches/today",
            "/api/predict/today",
            "/api/predictions/batch",
            "/api/predictions/update-result",
            "/api/predictions/auto-update-results",
            "/api/predictions/accuracy-report",
            "/api/predictions/export-csv",
            "/api/debug/nesine-test",
            "/api/reload",
            "/api/models/info",
            "/api/models/delete",
            "/api/history/load",
            "/api/training/start"
        ]
    })


@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify({
        "status": "ok",
        "model_trained": getattr(engine, "is_trained", False),
        "tracker_enabled": True,
        "timestamp": datetime.now().isoformat()
    })


# ======================================================
# LEAGUES
# ======================================================
@app.route("/api/leagues", methods=["GET"])
def get_leagues():
    try:
        filter_enabled = request.args.get('filter', 'false').lower() == 'true'
        today = date.today()
        matches = fetch_bulletin(today, filter_leagues=filter_enabled)

        leagues_dict = {}
        for m in matches:
            league_name = m.get("league", "Unknown")
            if league_name not in leagues_dict:
                leagues_dict[league_name] = {
                    "name": league_name,
                    "match_count": 0
                }
            leagues_dict[league_name]["match_count"] += 1

        leagues_list = sorted(leagues_dict.values(), key=lambda x: x["match_count"], reverse=True)
        return jsonify({
            "count": len(leagues_list),
            "items": leagues_list,
            "total_matches": len(matches)
        })

    except Exception as e:
        logger.error(f"/api/leagues error: {e}", exc_info=True)
        return jsonify({"error": str(e), "items": []}), 500


# ======================================================
# MATCHES TODAY
# ======================================================
@app.route("/api/matches/today", methods=["GET"])
def get_today_matches():
    try:
        filter_enabled = request.args.get('filter', 'false').lower() == 'true'
        today = date.today()
        matches = fetch_bulletin(today, filter_leagues=filter_enabled)

        logger.info(f"📊 Found {len(matches)} matches today (filter={filter_enabled})")
        return jsonify({
            "count": len(matches),
            "items": matches,
            "date": str(today),
            "filter_enabled": filter_enabled
        })

    except Exception as e:
        logger.error(f"/api/matches/today error: {e}", exc_info=True)
        return jsonify({"error": str(e), "items": []}), 500


# ======================================================
# PREDICT TODAY
# ======================================================
@app.route("/api/predict/today", methods=["GET"])
def predict_today():
    try:
        if engine is None:
            return jsonify({"error": "Model engine not loaded"}), 500

        filter_enabled = request.args.get('filter', 'false').lower() == 'true'
        league_filter = request.args.get('league', '').strip()
        save_predictions = request.args.get('save', 'true').lower() == 'true'

        today = date.today()
        matches = fetch_bulletin(today, filter_leagues=filter_enabled)
        logger.info(f"🎯 Predicting {len(matches)} matches...")

        results = []
        success_count = 0

        for m in matches:
            try:
                home = m.get("home_team")
                away = m.get("away_team")
                odds = m.get("odds", {"1": 2.0, "X": 3.0, "2": 3.5})
                league = m.get("league", "Unknown")

                if league_filter and league_filter.lower() not in league.lower():
                    continue

                for key in ["1", "X", "2"]:
                    if not odds.get(key):
                        odds[key] = 3.0

                pred = engine.predict_match(home, away, odds, league)
                result_item = {**m, "prediction": pred}
                results.append(result_item)
                success_count += 1

                if save_predictions:
                    tracker.save_prediction(m, pred, str(today))

            except Exception as err:
                logger.warning(f"⚠️ Prediction error ({home} - {away}): {err}")
                results.append({
                    **m,
                    "prediction": {
                        "error": str(err),
                        "prediction": "?",
                        "confidence": 0
                    }
                })

        logger.info(f"✅ {success_count}/{len(matches)} predictions successful")

        return jsonify({
            "count": len(results),
            "items": results,
            "date": str(today),
            "success_count": success_count,
            "filter_enabled": filter_enabled,
            "saved_to_tracker": save_predictions
        })

    except Exception as e:
        logger.error(f"/api/predict/today error: {e}", exc_info=True)
        return jsonify({"error": str(e), "items": []}), 500


# ======================================================
# MODELS INFO (FIXED)
# ======================================================
@app.route("/api/models/info", methods=["GET"])
def get_models_info():
    """Safe JSON model info (no PosixPath error)"""
    try:
        if engine is None:
            return jsonify({"error": "Engine not initialized"}), 500

        info = {
            "is_trained": engine.is_trained,
            "model_path": str(engine.model_path),
            "models": {}
        }

        for name, model in engine.models.items():
            if model is not None:
                info["models"][name] = {
                    "type": type(model).__name__,
                    "trained": hasattr(model, 'n_estimators')
                }

        if hasattr(engine, 'scaler') and engine.scaler is not None:
            info["scaler"] = {
                "type": type(engine.scaler).__name__,
                "fitted": hasattr(engine.scaler, 'mean_')
            }

        meta_path = Path(engine.model_path) / "training_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                info["training_metadata"] = metadata
            except Exception as e:
                logger.warning(f"⚠️ Metadata read error: {e}")

        # convert any Path objects to string
        def convert(o):
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, dict):
                return {k: convert(v) for k, v in o.items()}
            if isinstance(o, list):
                return [convert(v) for v in o]
            return o

        info = convert(info)
        return jsonify(info)

    except Exception as e:
        logger.error(f"/api/models/info error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ======================================================
# AUTO RESULT UPDATE (shortened to relevant fix)
# ======================================================
@app.route("/api/predictions/auto-update-results", methods=["POST"])
def auto_update_results():
    """Auto-update results and compute accuracy"""
    try:
        data = request.get_json(silent=True) or {}
        target_date = data.get("date")
        if target_date:
            target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
        else:
            target_date = date.today() - timedelta(days=1)

        results = fetch_results_for_date(target_date)
        if not results:
            return jsonify({
                "status": "no_data",
                "message": f"No results found for {target_date}"
            })

        updated_count = 0
        ms_correct = 0
        for r in results:
            success = tracker.update_actual_result(
                r['home_team'], r['away_team'], r['home_score'], r['away_score'], str(target_date)
            )
            if success:
                updated_count += 1

        return jsonify({
            "status": "ok",
            "updated_count": updated_count,
            "date": str(target_date)
        })
    except Exception as e:
        logger.error(f"/api/predictions/auto-update-results error: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


# ======================================================
# TRAINING
# ======================================================
@app.route("/api/training/start", methods=["POST", "GET"])
def start_training():
    """Retrain models in background"""
    import threading

    def train_thread():
        try:
            from model_trainer import ProductionModelTrainer
            trainer = ProductionModelTrainer(
                models_dir=MODELS_DIR,
                raw_data_path=RAW_DATA_PATH,
                clubs_path=CLUBS_PATH,
                min_matches=50,
                test_size=0.2,
                random_state=42,
                version_archive=False,
                verbose=True
            )
            result = trainer.run_full_pipeline()
            if result.get('success'):
                engine.load_models()
                logger.info("✅ Model retraining complete")
        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)

    if request.method == "POST":
        threading.Thread(target=train_thread, daemon=True).start()
        return jsonify({"status": "training_started"})
    else:
        return jsonify({"message": "Send POST to start training"})

# ======================================================
# ERROR HANDLING
# ======================================================
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Server error"}), 500

# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("⚽ Predicta Europe ML v2 - Backend Active (FIXED)")
    logger.info("=" * 60)
    logger.info(f"📂 Models: {MODELS_DIR}")
    logger.info(f"📂 Raw Data: {RAW_DATA_PATH}")
    logger.info(f"📂 Clubs: {CLUBS_PATH}")
    logger.info(f"🌐 Port: {APP_PORT}")
    logger.info(f"🤖 Trained: {'✅' if getattr(engine, 'is_trained', False) else '⚠️ Training required'}")
    logger.info("=" * 60)

    app.run(host="0.0.0.0", port=APP_PORT, debug=False)
