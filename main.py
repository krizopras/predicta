#!/usr/bin/env python3
"""
Predicta Europe ML v3 - FULL AUTO MODE
Otomatik model eğitimi ve yükleme sistemi
"""

import os
import json
import logging
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, jsonify, request
from flask_cors import CORS

# ML & Nesine imports
from ml_prediction_engine import MLPredictionEngine
from nesine_fetcher import fetch_bulletin, fetch_yesterday_results, fetch_results_for_date
from historical_processor import HistoricalDataProcessor
from prediction_tracker import PredictionTracker

# Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/auto_train_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Predicta Auto")

# Config
APP_PORT = int(os.environ.get("PORT", 8000))
MODELS_DIR = os.environ.get("PREDICTA_MODELS_DIR", "data/ai_models_v3")
RAW_DATA_PATH = os.environ.get("PREDICTA_RAW_DATA", "data/raw")
CLUBS_PATH = os.environ.get("PREDICTA_CLUBS", "data/clubs")

# Flask setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ML Engine
engine = MLPredictionEngine(model_dir=MODELS_DIR)

# Historical Processor
history_processor = HistoricalDataProcessor(
    raw_data_path=RAW_DATA_PATH,
    clubs_path=CLUBS_PATH
)

# Prediction Tracker
tracker = PredictionTracker(storage_dir="data/predictions")

# Auto-training state
auto_training_state = {
    "is_running": False,
    "current_step": "",
    "progress": 0,
    "message": "",
    "error": None
}


def auto_train_pipeline():
    """Otomatik eğitim pipeline'ı"""
    global auto_training_state
    
    try:
        auto_training_state["is_running"] = True
        auto_training_state["error"] = None
        
        # Step 1: History Load
        logger.info("🔄 Auto-training Step 1/3: Loading history...")
        auto_training_state["current_step"] = "history"
        auto_training_state["progress"] = 10
        auto_training_state["message"] = "Geçmiş veriler yükleniyor..."
        
        total_matches = 0
        if os.path.exists(RAW_DATA_PATH):
            for country_dir in Path(RAW_DATA_PATH).iterdir():
                if country_dir.is_dir():
                    try:
                        matches = history_processor.load_country_data(country_dir.name)
                        total_matches += len(matches)
                        logger.info(f"✅ {country_dir.name}: {len(matches)} maç")
                    except Exception as e:
                        logger.warning(f"⚠️ {country_dir.name}: {e}")
        
        auto_training_state["progress"] = 30
        auto_training_state["message"] = f"{total_matches} geçmiş maç yüklendi"
        logger.info(f"✅ History loaded: {total_matches} matches")
        
        # Step 2: Training
        logger.info("🔄 Auto-training Step 2/3: Training models...")
        auto_training_state["current_step"] = "training"
        auto_training_state["progress"] = 40
        auto_training_state["message"] = "AI modelleri eğitiliyor (5-10 dakika)..."
        
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
        
        if not result.get('success'):
            raise Exception(result.get('error', 'Training failed'))
        
        auto_training_state["progress"] = 80
        auto_training_state["message"] = "Eğitim tamamlandı"
        logger.info("✅ Training completed successfully")
        
        # Step 3: Reload
        logger.info("🔄 Auto-training Step 3/3: Reloading models...")
        auto_training_state["current_step"] = "reload"
        auto_training_state["progress"] = 90
        auto_training_state["message"] = "Modeller yükleniyor..."
        
        success = engine.load_models()
        
        if not success:
            raise Exception("Model loading failed")
        
        auto_training_state["progress"] = 100
        auto_training_state["message"] = "✅ Sistem hazır!"
        auto_training_state["is_running"] = False
        logger.info("✅ Auto-training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Auto-training error: {e}", exc_info=True)
        auto_training_state["is_running"] = False
        auto_training_state["error"] = str(e)
        auto_training_state["message"] = f"❌ Hata: {str(e)}"


def check_and_auto_train():
    """Model kontrolü ve otomatik eğitim başlatma"""
    if engine.is_trained:
        logger.info("✅ Models already trained, skipping auto-training")
        return
    
    logger.info("⚠️ Models not found, starting auto-training pipeline...")
    thread = threading.Thread(target=auto_train_pipeline, daemon=True)
    thread.start()


# ========== ROUTES ==========

@app.route("/")
def home():
    return jsonify({
        "status": "Predicta ML v3 - AUTO MODE",
        "port": APP_PORT,
        "model_trained": engine.is_trained,
        "auto_training": auto_training_state,
        "author": "Olcay Arslan"
    })


@app.route("/api/status", methods=["GET"])
def get_status():
    """Backend status + auto-training status"""
    return jsonify({
        "status": "ok",
        "model_trained": engine.is_trained,
        "auto_training": auto_training_state,
        "timestamp": datetime.now().isoformat()
    })


@app.route("/api/leagues", methods=["GET"])
def get_leagues():
    """Get league list"""
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


@app.route("/api/matches/today", methods=["GET"])
def get_today_matches():
    """Fetch today's matches"""
    try:
        filter_enabled = request.args.get('filter', 'false').lower() == 'true'
        today = date.today()
        matches = fetch_bulletin(today, filter_leagues=filter_enabled)
        
        return jsonify({
            "count": len(matches),
            "items": matches,
            "date": str(today)
        })
    except Exception as e:
        logger.error(f"/api/matches/today error: {e}", exc_info=True)
        return jsonify({"error": str(e), "items": []}), 500


@app.route("/api/predict/today", methods=["GET"])
def predict_today():
    """Predict today's matches"""
    try:
        filter_enabled = request.args.get('filter', 'false').lower() == 'true'
        save_predictions = request.args.get('save', 'true').lower() == 'true'
        
        today = date.today()
        matches = fetch_bulletin(today, filter_leagues=filter_enabled)
        
        results = []
        success_count = 0
        
        for m in matches:
            try:
                home = m.get("home_team")
                away = m.get("away_team")
                odds = m.get("odds", {"1": 2.0, "X": 3.0, "2": 3.5})
                league = m.get("league", "Unknown")
                
                if not odds.get("1"):
                    odds["1"] = 2.0
                if not odds.get("X"):
                    odds["X"] = 3.0
                if not odds.get("2"):
                    odds["2"] = 3.5
                
                pred = engine.predict_match(home, away, odds, league)
                
                result_item = {**m, "prediction": pred}
                results.append(result_item)
                success_count += 1
                
                if save_predictions:
                    tracker.save_prediction(m, pred, str(today))
                
            except Exception as err:
                logger.warning(f"⚠️ Prediction error: {err}")
                results.append({
                    **m,
                    "prediction": {
                        "error": str(err),
                        "prediction": "?",
                        "confidence": 0
                    }
                })
        
        return jsonify({
            "count": len(results),
            "items": results,
            "success_count": success_count
        })
        
    except Exception as e:
        logger.error(f"/api/predict/today error: {e}", exc_info=True)
        return jsonify({"error": str(e), "items": []}), 500


@app.route("/api/predictions/auto-update-results", methods=["POST"])
def auto_update_results():
    """Auto-update match results from Nesine"""
    try:
        data = request.get_json(silent=True) or {}
        
        target_date = data.get("date")
        if target_date:
            from datetime import datetime as dt
            target_date = dt.strptime(target_date, "%Y-%m-%d").date()
        else:
            target_date = date.today() - timedelta(days=1)
        
        filter_enabled = data.get("filter", False)
        results = fetch_results_for_date(target_date, filter_leagues=filter_enabled)
        
        if not results:
            return jsonify({
                "status": "no_data",
                "message": f"No results for {target_date}",
                "updated_count": 0
            })
        
        updated_count = 0
        ms_correct = 0
        score_correct = 0
        
        for result in results:
            try:
                success = tracker.update_actual_result(
                    result['home_team'],
                    result['away_team'],
                    result['home_score'],
                    result['away_score'],
                    str(target_date)
                )
                
                if success:
                    updated_count += 1
                    
            except Exception as e:
                logger.warning(f"Update error: {e}")
        
        return jsonify({
            "status": "ok",
            "message": f"{updated_count} results updated",
            "updated_count": updated_count
        })
        
    except Exception as e:
        logger.error(f"/api/predictions/auto-update-results error: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/predictions/accuracy-report", methods=["GET"])
def get_accuracy_report():
    """Get accuracy report"""
    try:
        days = int(request.args.get('days', 7))
        report = tracker.get_accuracy_report(days=days)
        return jsonify(report)
    except Exception as e:
        logger.error(f"Accuracy report error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/reload", methods=["GET", "POST"])
def reload_models():
    """Reload models"""
    try:
        success = engine.load_models()
        
        return jsonify({
            "status": "ok" if success else "partial",
            "message": "Models reloaded" if success else "Some models missing",
            "is_trained": engine.is_trained,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Reload error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/history/load", methods=["POST"])
def load_history():
    """Load historical data"""
    try:
        if not os.path.exists(RAW_DATA_PATH):
            return jsonify({
                "status": "error",
                "message": f"Data folder not found: {RAW_DATA_PATH}",
                "matches_loaded": 0
            }), 404
        
        countries = [
            d for d in os.listdir(RAW_DATA_PATH)
            if os.path.isdir(os.path.join(RAW_DATA_PATH, d))
        ]
        
        total_matches = 0
        loaded_countries = []
        
        for country in countries:
            try:
                matches = history_processor.load_country_data(country)
                if matches:
                    total_matches += len(matches)
                    loaded_countries.append(f"{country} ({len(matches)})")
            except Exception as e:
                logger.warning(f"⚠️ {country}: {e}")
        
        return jsonify({
            "status": "ok",
            "matches_loaded": total_matches,
            "countries_loaded": loaded_countries
        })
        
    except Exception as e:
        logger.error(f"History load error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/training/start", methods=["POST"])
def start_training():
    """Start training"""
    if auto_training_state["is_running"]:
        return jsonify({
            "status": "already_running",
            "message": "Training already in progress"
        }), 400
    
    thread = threading.Thread(target=auto_train_pipeline, daemon=True)
    thread.start()
    
    return jsonify({
        "status": "ok",
        "message": "Training started"
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Server error"}), 500


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("⚽ Predicta Europe ML v3 - FULL AUTO MODE")
    logger.info("=" * 60)
    logger.info(f"📂 Models: {MODELS_DIR}")
    logger.info(f"📂 Data: {RAW_DATA_PATH}")
    logger.info(f"🌐 Port: {APP_PORT}")
    
    # Auto-check and train on startup
    check_and_auto_train()
    
    logger.info("=" * 60)
    
    app.run(host="0.0.0.0", port=APP_PORT, debug=False)
