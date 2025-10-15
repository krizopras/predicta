#!/usr/bin/env python3
"""
Predicta Europe ML v2 - Flask Backend (TRACKER INTEGRATED)
Auto bulletin fetcher (Nesine) + ML predictions + Prediction Tracking
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
from nesine_fetcher import fetch_bulletin, fetch_today, fetch_yesterday_results, fetch_results_for_date
from historical_processor import HistoricalDataProcessor
from prediction_tracker import PredictionTracker

# ========== Logging ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Predicta Europe ML v3")

# ========== Config ==========
APP_PORT = int(os.environ.get("PORT", 8000))
MODELS_DIR = os.environ.get("PREDICTA_MODELS_DIR", "data/ai_models_v3")
RAW_DATA_PATH = os.environ.get("PREDICTA_RAW_DATA", "data/raw")
CLUBS_PATH = os.environ.get("PREDICTA_CLUBS", "data/clubs")

# ========== Flask setup ==========
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ========== ML Engine ==========
engine = MLPredictionEngine(models_dir=MODELS_DIR)

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
        "model_trained": engine.is_trained,
        "models_dir": str(MODELS_DIR),  # ✅ str() conversion
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
    """Backend status check"""
    return jsonify({
        "status": "ok",
        "model_trained": engine.is_trained,
        "tracker_enabled": True,
        "timestamp": datetime.now().isoformat()
    })


@app.route("/api/leagues", methods=["GET"])
def get_leagues():
    """Get league list from today's bulletin"""
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
    """Fetch all matches from Nesine bulletin"""
    try:
        filter_enabled = request.args.get('filter', 'false').lower() == 'true'
        today = date.today()
        matches = fetch_bulletin(today, filter_leagues=filter_enabled)
        
        logger.info(f"📊 Found {len(matches)} matches today (filter: {filter_enabled})")
        
        return jsonify({
            "count": len(matches),
            "items": matches,
            "date": str(today),
            "filter_enabled": filter_enabled
        })
    except Exception as e:
        logger.error(f"/api/matches/today error: {e}", exc_info=True)
        return jsonify({"error": str(e), "items": []}), 500


@app.route("/api/predict/today", methods=["GET"])
def predict_today():
    """Predict all today's matches and save"""
    try:
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
                
                if not odds.get("1"):
                    odds["1"] = 2.0
                if not odds.get("X"):
                    odds["X"] = 3.0
                if not odds.get("2"):
                    odds["2"] = 3.5
                
                pred = engine.predict_match(home, away, odds, league)
                
                result_item = {
                    **m,
                    "prediction": pred
                }
                
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
        
        logger.info(f"✅ {success_count}/{len(matches)} matches predicted successfully")
        
        if save_predictions:
            logger.info(f"💾 {success_count} predictions saved")
        
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


@app.route("/api/reload", methods=["GET", "POST"])
def reload_models():
    """Reload models"""
    try:
        success = engine.load_models()
        
        return jsonify({
            "status": "ok" if success else "partial",
            "message": "Models reloaded successfully" if success else "Some models missing",
            "is_trained": engine.is_trained,
            "models_dir": str(engine.models_dir),  # ✅ String conversion
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"/api/reload error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route("/api/models/info", methods=["GET"])
def get_models_info():
    """Get model details"""
    try:
        info = {
            "is_trained": engine.is_trained,
            "models_dir": str(engine.models_dir),  # ✅ String conversion
            "models": {}
        }
        
        for name, model in engine.models.items():
            if model is not None:
                info["models"][name] = {
                    "type": type(model).__name__,
                    "trained": hasattr(model, 'n_estimators')
                }
        
        info["scaler"] = {
            "type": type(engine.scaler).__name__,
            "fitted": hasattr(engine.scaler, 'mean_') and engine.scaler.mean_ is not None
        }
        
        if hasattr(engine, 'score_predictor') and engine.score_predictor:
            info["score_model"] = {
                "available": engine.score_predictor.model is not None,
                "classes": len(engine.score_predictor.space) if engine.score_predictor.space else 0
            }
        
        meta_path = Path(engine.models_dir) / "training_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                info["training_metadata"] = {
                    "date": metadata.get("training_date"),
                    "version": metadata.get("version"),
                    "total_matches": metadata.get("total_clean_matches"),
                    "countries": metadata.get("countries_count"),
                    "features": metadata.get("features_count"),
                    "ms_accuracies": metadata.get("ms_accuracies"),
                    "score_accuracy": metadata.get("score_accuracy")
                }
            except:
                pass
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"/api/models/info error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route("/api/history/load", methods=["POST"])
def load_history():
    """Load historical data - OPTIMIZED for Railway"""
    try:
        if engine.feature_engineer is None:
            return jsonify({
                "status": "error",
                "message": "Feature Engineer not available",
                "matches_loaded": 0
            }), 500
        
        if not os.path.exists(RAW_DATA_PATH):
            return jsonify({
                "status": "error",
                "message": f"Historical data folder not found: {RAW_DATA_PATH}",
                "matches_loaded": 0
            }), 404
        
        countries = [
            d for d in os.listdir(RAW_DATA_PATH)
            if os.path.isdir(os.path.join(RAW_DATA_PATH, d)) and d.lower() != "clubs"
        ]
        
        if not countries:
            return jsonify({
                "status": "warning",
                "message": f"No country folders found in {RAW_DATA_PATH}",
                "matches_loaded": 0
            }), 200
        
        total_matches = 0
        loaded_countries = []
        failed_countries = []
        
        logger.info(f"📂 Detected {len(countries)} countries: {', '.join(countries)}")
        
        # Disable auto-save during bulk loading
        if hasattr(engine.feature_engineer, '_auto_save'):
            original_auto_save = engine.feature_engineer._auto_save
            engine.feature_engineer._auto_save = False
            logger.info("🔧 Auto-save disabled for bulk loading")
        else:
            original_auto_save = None
        
        try:
            for country in countries:
                try:
                    logger.info(f"📄 Loading {country}...")
                    
                    if not history_processor:
                        logger.error("❌ History processor not available")
                        failed_countries.append(f"{country} (processor unavailable)")
                        continue
                    
                    matches = history_processor.load_country_data(country)
                    
                    if not matches:
                        failed_countries.append(f"{country} (no data)")
                        continue
                    
                    # Process matches in batches
                    batch_size = 500
                    for i in range(0, len(matches), batch_size):
                        batch = matches[i:i+batch_size]
                        
                        for m in batch:
                            try:
                                result = m.get('result', '?')
                                
                                if result not in ('1', 'X', '2'):
                                    try:
                                        hg = int(m.get('home_score', 0))
                                        ag = int(m.get('away_score', 0))
                                        result = '1' if hg > ag else ('X' if hg == ag else '2')
                                    except:
                                        result = 'X'
                                
                                if hasattr(engine.feature_engineer, 'update_match_result'):
                                    engine.feature_engineer.update_match_result(
                                        m['home_team'], m['away_team'],
                                        m.get('league', 'Unknown'),
                                        result,
                                        int(m.get('home_score', 0)),
                                        int(m.get('away_score', 0)),
                                        m.get('date', ''),
                                        'home', 'away'
                                    )
                                
                            except Exception as e:
                                continue
                        
                        logger.info(f"   ⏳ Processed {min(i+batch_size, len(matches))}/{len(matches)} matches")
                    
                    total_matches += len(matches)
                    loaded_countries.append(f"{country} ({len(matches)} matches)")
                    logger.info(f"✅ {country}: {len(matches)} matches loaded")
                    
                except Exception as e:
                    logger.warning(f"⚠️ {country} could not be loaded: {e}")
                    failed_countries.append(f"{country} (error: {str(e)[:50]})")
                    continue
            
            # Save once at the end
            logger.info("💾 Saving feature data (single batch save)...")
            try:
                if hasattr(engine.feature_engineer, '_save_data'):
                    engine.feature_engineer._save_data()
                    logger.info("✅ Feature data saved successfully")
                elif hasattr(engine.feature_engineer, 'manual_save'):
                    engine.feature_engineer.manual_save()
                    logger.info("✅ Feature data saved (manual)")
            except Exception as e:
                logger.error(f"❌ Feature data save error: {e}")
            
        finally:
            # Re-enable auto-save
            if original_auto_save is not None:
                engine.feature_engineer._auto_save = original_auto_save
                logger.info("🔧 Auto-save re-enabled")
        
        response = {
            "status": "ok",
            "message": f"Historical data loaded from {len(loaded_countries)} countries",
            "matches_loaded": total_matches,
            "countries_loaded": loaded_countries,
            "countries_failed": failed_countries,
            "total_countries": len(countries),
            "success_rate": f"{len(loaded_countries)}/{len(countries)}"
        }
        
        logger.info(f"🎯 Total {total_matches} matches added to feature memory")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"/api/history/load error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e),
            "matches_loaded": 0
        }), 500


@app.route("/api/models/delete", methods=["POST"])
def delete_models():
    """Delete existing models to force retraining"""
    try:
        import shutil
        
        deleted_files = []
        models_dir = Path(MODELS_DIR)
        
        if not models_dir.exists():
            return jsonify({
                "status": "no_models",
                "message": "No models directory found"
            })
        
        model_files = [
            "ensemble_models.pkl",
            "scaler.pkl",
            "score_model.pkl",
            "training_metadata.json"
        ]
        
        for file in model_files:
            file_path = models_dir / file
            if file_path.exists():
                file_path.unlink()
                deleted_files.append(file)
                logger.info(f"🗑️ Deleted: {file}")
        
        if deleted_files:
            return jsonify({
                "status": "ok",
                "message": f"Deleted {len(deleted_files)} model files",
                "deleted_files": deleted_files,
                "hint": "Now run: POST /api/training/start"
            })
        else:
            return jsonify({
                "status": "no_files",
                "message": "No model files found to delete"
            })
            
    except Exception as e:
        logger.error(f"/api/models/delete error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route("/api/training/start", methods=["GET", "POST"])
def start_training():
    """Start model training in background"""
    
    if request.method == "GET":
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Predicta Training</title>
    <meta charset="utf-8">
    <style>
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            max-width: 700px; 
            margin: 50px auto; 
            padding: 30px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { color: #333; margin-top: 0; }
        button { 
            background: #4CAF50; 
            color: white; 
            padding: 15px 40px; 
            border: none; 
            border-radius: 8px; 
            font-size: 18px; 
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
            width: 100%;
        }
        button:hover { 
            background: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .info { 
            background: #e3f2fd; 
            padding: 20px; 
            border-radius: 8px; 
            margin: 20px 0;
            border-left: 4px solid #2196F3;
        }
        .info p { margin: 8px 0; }
        .warning {
            background: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #ffc107;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Training Start</h1>
        
        <div class="info">
            <p><strong>Status:</strong> Ready</p>
            <p><strong>Railway Mode:</strong> Active</p>
            <p><strong>Estimated Time:</strong> 10-15 minutes</p>
        </div>
        
        <div class="warning">
            <strong>Warning:</strong> After clicking, check Railway logs. Training will continue in background.
        </div>
        
        <form method="POST" action="/api/training/start" 
              onsubmit="return confirm('Are you sure you want to start training?\\n\\nThis will take 10-15 minutes.');">
            <button type="submit">Start Training</button>
        </form>
    </div>
</body>
</html>"""
        return html_content
    
    import threading
    
  def train_background():
    try:
        logger.info("Training started")
        
        from model_trainer import ProductionModelTrainer
        
        trainer = ProductionModelTrainer(
            models_dir=MODELS_DIR,
            raw_data_path=RAW_DATA_PATH,
            clubs_path=CLUBS_PATH,           
            test_size=0.2,
            random_state=42,
            version_archive=False,
            verbose=True
        )
        
        result = trainer.run_full_pipeline()
        
        if result.get('success'):
            logger.info("Training completed!")
            engine.load_models()
            logger.info("Models reloaded successfully")
        else:
            logger.error(f"Training error: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
    
    # Run in background
    thread = threading.Thread(target=train_background, daemon=True)
    thread.start()
    
    # HTML response for browser
    success_html = """<!DOCTYPE html>
<html>
<head>
    <title>Training Started</title>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="5;url=/api/training/start">
    <style>
        body { 
            font-family: 'Segoe UI', Arial; 
            max-width: 600px; 
            margin: 100px auto; 
            text-align: center;
            padding: 30px;
            background: #f5f5f5;
        }
        .success {
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .icon { font-size: 64px; margin-bottom: 20px; }
        h1 { color: #28a745; margin: 20px 0; }
        p { color: #666; line-height: 1.8; }
        .highlight { 
            background: #e3f2fd; 
            padding: 15px; 
            border-radius: 8px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="success">
        <div class="icon">Success</div>
        <h1>Training Started!</h1>
        
        <div class="highlight">
            <p><strong>Estimated Time:</strong> 10-15 minutes</p>
            <p><strong>Status:</strong> Running in background</p>
        </div>
        
        <p>Check Railway dashboard logs for progress</p>
        <p><a href="/api/models/info" target="_blank">Check model status</a></p>
        
        <p><small style="color: #999;">Redirecting in 5 seconds...</small></p>
    </div>
</body>
</html>"""
    return success_html


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
    logger.info("Backend Active - Predicta Europe ML v2")
    logger.info("=" * 60)
    logger.info(f"Models: {MODELS_DIR}")
    logger.info(f"Raw Data: {RAW_DATA_PATH}")
    logger.info(f"Clubs: {CLUBS_PATH}")
    logger.info(f"Predictions: data/predictions")
    logger.info(f"Port: {APP_PORT}")
    logger.info(f"Trained: {'YES' if engine.is_trained else 'NO - Training required'}")
    logger.info("=" * 60)
    
    app.run(host="0.0.0.0", port=APP_PORT, debug=False)
