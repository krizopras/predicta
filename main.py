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
engine = MLPredictionEngine(model_path=MODELS_DIR)

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
        "models_dir": MODELS_DIR,
        "raw_data_path": RAW_DATA_PATH,
        "clubs_path": CLUBS_PATH,
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


# ----------------- LEAGUES -----------------
@app.route("/api/leagues", methods=["GET"])
def get_leagues():
    """Get league list from today's bulletin"""
    try:
        filter_enabled = request.args.get('filter', 'false').lower() == 'true'
        today = date.today()
        matches = fetch_bulletin(today, filter_leagues=filter_enabled)
        
        # Collect leagues
        leagues_dict = {}
        for m in matches:
            league_name = m.get("league", "Unknown")
            if league_name not in leagues_dict:
                leagues_dict[league_name] = {
                    "name": league_name,
                    "match_count": 0
                }
            leagues_dict[league_name]["match_count"] += 1
        
        # Convert to list and sort
        leagues_list = sorted(leagues_dict.values(), key=lambda x: x["match_count"], reverse=True)
        
        return jsonify({
            "count": len(leagues_list),
            "items": leagues_list,
            "total_matches": len(matches)
        })
    except Exception as e:
        logger.error(f"/api/leagues error: {e}", exc_info=True)
        return jsonify({"error": str(e), "items": []}), 500


# ----------------- ALL MATCHES -----------------
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


# ----------------- TODAY'S MATCHES + PREDICTION (WITH TRACKER) -----------------
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
                
                # League filter
                if league_filter and league_filter.lower() not in league.lower():
                    continue
                
                # Use default odds if missing
                if not odds.get("1"):
                    odds["1"] = 2.0
                if not odds.get("X"):
                    odds["X"] = 3.0
                if not odds.get("2"):
                    odds["2"] = 3.5
                
                # Make prediction
                pred = engine.predict_match(home, away, odds, league)
                
                result_item = {
                    **m,
                    "prediction": pred
                }
                
                results.append(result_item)
                success_count += 1
                
                # Save predictions (optional)
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


# ----------------- BATCH PREDICTION -----------------
@app.route("/api/predictions/batch", methods=["POST"])
def batch_predictions():
    """Batch match prediction (CSV format)"""
    data = request.get_json(silent=True) or {}
    csv = data.get("csv", "")
    matches = []

    if not csv.strip():
        return jsonify({"error": "CSV required"}), 400

    try:
        lines = [ln.strip() for ln in csv.splitlines() if ln.strip()]
        
        # Skip header row
        start_index = 1 if lines and "home" in lines[0].lower() else 0
        
        for ln in lines[start_index:]:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) >= 6:
                try:
                    matches.append({
                        "home_team": parts[0],
                        "away_team": parts[1],
                        "league": parts[2],
                        "odds": {
                            "1": float(parts[3]),
                            "X": float(parts[4]),
                            "2": float(parts[5])
                        }
                    })
                except ValueError as e:
                    logger.warning(f"⚠️ CSV row skipped: {ln} ({e})")
                    continue

        results = []
        for m in matches:
            try:
                pred = engine.predict_match(
                    m["home_team"], m["away_team"], m["odds"], m["league"]
                )
                results.append({**m, "prediction": pred})
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                results.append({**m, "error": str(e)})

        return jsonify({
            "count": len(results),
            "items": results
        })
        
    except Exception as e:
        logger.error(f"/api/predictions/batch error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ----------------- PREDICTION RESULT UPDATE -----------------
@app.route("/api/predictions/update-result", methods=["POST"])
def update_prediction_result():
    """Update match result and calculate accuracy"""
    try:
        data = request.get_json(silent=True) or {}
        
        home_team = data.get("home_team")
        away_team = data.get("away_team")
        home_score = data.get("home_score")
        away_score = data.get("away_score")
        match_date = data.get("match_date")  # Optional
        
        if not all([home_team, away_team, 
                   home_score is not None, away_score is not None]):
            return jsonify({
                "error": "home_team, away_team, home_score, away_score required"
            }), 400
        
        success = tracker.update_actual_result(
            home_team, away_team,
            int(home_score), int(away_score),
            match_date
        )
        
        if success:
            return jsonify({
                "status": "ok",
                "message": f"Result updated: {home_team} {home_score}-{away_score} {away_team}"
            })
        else:
            return jsonify({
                "status": "not_found",
                "message": "Prediction not found or update failed"
            }), 404
            
    except Exception as e:
        logger.error(f"/api/predictions/update-result error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ----------------- AUTO RESULT UPDATE -----------------
@app.route("/api/predictions/auto-update-results", methods=["POST"])
def auto_update_results():
    """
    Fetch yesterday's match results from Nesine and auto-update predictions
    + Real-time accuracy analysis
    """
    try:
        data = request.get_json(silent=True) or {}
        
        # Date parameter (optional)
        target_date = data.get("date")
        if target_date:
            from datetime import datetime as dt
            target_date = dt.strptime(target_date, "%Y-%m-%d").date()
        else:
            # Default: yesterday
            target_date = date.today() - timedelta(days=1)
        
        filter_enabled = data.get("filter", False)
        
        logger.info(f"🔄 Starting auto result update: {target_date}")
        
        # Fetch results from Nesine
        results = fetch_results_for_date(target_date, filter_leagues=filter_enabled)
        
        if not results:
            return jsonify({
                "status": "no_data",
                "message": f"No results found for {target_date}",
                "date": str(target_date),
                "updated_count": 0
            })
        
        logger.info(f"📊 Found {len(results)} finished matches")
        
        # Send each result to tracker
        updated_count = 0
        failed_count = 0
        failed_matches = []
        
        # Real-time accuracy metrics
        ms_correct = 0
        score_correct = 0
        updated_matches_detail = []
        
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
                    
                    # Check accuracy from prediction file
                    pred_file = tracker.storage_dir / f"predictions_{target_date}.json"
                    if pred_file.exists():
                        with open(pred_file, 'r', encoding='utf-8') as f:
                            predictions = json.load(f)
                            
                        for pred in predictions:
                            if (pred['match']['home_team'].lower() == result['home_team'].lower() and
                                pred['match']['away_team'].lower() == result['away_team'].lower()):
                                
                                # MS accuracy
                                predicted_ms = pred['prediction']['ms_prediction']
                                actual_ms = result['result']
                                ms_match = (predicted_ms == actual_ms)
                                if ms_match:
                                    ms_correct += 1
                                
                                # Score accuracy
                                predicted_score = pred['prediction']['score_prediction']
                                actual_score = result['score']
                                score_match = (predicted_score == actual_score)
                                if score_match:
                                    score_correct += 1
                                
                                updated_matches_detail.append({
                                    'match': f"{result['home_team']} - {result['away_team']}",
                                    'predicted_ms': predicted_ms,
                                    'actual_ms': actual_ms,
                                    'ms_correct': ms_match,
                                    'predicted_score': predicted_score,
                                    'actual_score': actual_score,
                                    'score_correct': score_match,
                                    'confidence': pred['prediction']['confidence']
                                })
                                break
                    
                    logger.info(f"   ✅ {result['home_team']} {result['score']} {result['away_team']}")
                else:
                    failed_count += 1
                    failed_matches.append(f"{result['home_team']} - {result['away_team']}")
                    
            except Exception as e:
                failed_count += 1
                failed_matches.append(f"{result['home_team']} - {result['away_team']} (error: {str(e)[:30]})")
                logger.warning(f"   ⚠️ Update error: {result['home_team']} - {result['away_team']}: {e}")
        
        # Success rates
        ms_accuracy = (ms_correct / updated_count * 100) if updated_count > 0 else 0
        score_accuracy = (score_correct / updated_count * 100) if updated_count > 0 else 0
        
        logger.info(f"✅ Update completed: {updated_count} successful, {failed_count} failed")
        logger.info(f"📊 MS Accuracy: {ms_accuracy:.1f}% ({ms_correct}/{updated_count})")
        logger.info(f"⚽ Score Accuracy: {score_accuracy:.1f}% ({score_correct}/{updated_count})")
        
        return jsonify({
            "status": "ok",
            "message": f"{updated_count}/{len(results)} match results updated",
            "date": str(target_date),
            "total_results": len(results),
            "updated_count": updated_count,
            "failed_count": failed_count,
            "failed_matches": failed_matches[:10],  # First 10 failures
            "accuracy": {
                "ms_correct": ms_correct,
                "ms_accuracy": round(ms_accuracy, 2),
                "score_correct": score_correct,
                "score_accuracy": round(score_accuracy, 2)
            },
            "details": updated_matches_detail[:20]  # First 20 details
        })
        
    except Exception as e:
        logger.error(f"/api/predictions/auto-update-results error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


# ----------------- ACCURACY REPORT -----------------
@app.route("/api/predictions/accuracy-report", methods=["GET"])
def get_accuracy_report():
    """Prediction accuracy report"""
    try:
        days = int(request.args.get('days', 7))
        report = tracker.get_accuracy_report(days=days)
        return jsonify(report)
        
    except Exception as e:
        logger.error(f"/api/predictions/accuracy-report error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ----------------- CSV EXPORT -----------------
@app.route("/api/predictions/export-csv", methods=["GET"])
def export_predictions_csv():
    """Export predictions as CSV"""
    try:
        days = int(request.args.get('days', 30))
        output_file = tracker.export_to_csv(days=days)
        
        if output_file:
            return jsonify({
                "status": "ok",
                "file": output_file,
                "message": "CSV export successful"
            })
        else:
            return jsonify({
                "status": "no_data",
                "message": "No data to export"
            }), 404
            
    except Exception as e:
        logger.error(f"/api/predictions/export-csv error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ----------------- DEBUG NESINE -----------------
@app.route("/api/debug/nesine-test", methods=["GET"])
def debug_nesine():
    """Nesine API test endpoint"""
    try:
        today = date.today()
        
        # All matches without filter
        all_matches = fetch_bulletin(today, filter_leagues=False)
        
        # Group leagues
        leagues = {}
        for m in all_matches:
            lg = m.get("league", "Unknown")
            leagues[lg] = leagues.get(lg, 0) + 1

        # Filtered matches
        filtered_matches = fetch_bulletin(today, filter_leagues=True)

        return jsonify({
            "status": "ok",
            "date": str(today),
            "total_matches": len(all_matches),
            "filtered_matches": len(filtered_matches),
            "leagues": dict(sorted(leagues.items(), key=lambda x: x[1], reverse=True)),
            "api_url": f"https://cdnbulten.nesine.com/api/bulten/getprebultenfull?date={today}",
            "sample_matches": all_matches[:5] if all_matches else []
        })
    except Exception as e:
        logger.error(f"/api/debug/nesine-test error: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


# ----------------- MODEL MANAGEMENT -----------------
@app.route("/api/reload", methods=["GET", "POST"])
def reload_models():
    """Reload models"""
    try:
        success = engine.load_models()
        
        return jsonify({
            "status": "ok" if success else "partial",
            "message": "Models reloaded successfully" if success else "Some models missing",
            "is_trained": engine.is_trained,
            "model_path": engine.model_path,
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
            "model_path": engine.model_path,
            "models": {}
        }
        
        # MS models
        for name, model in engine.models.items():
            if model is not None:
                info["models"][name] = {
                    "type": type(model).__name__,
                    "trained": hasattr(model, 'n_estimators')
                }
        
        # Scaler
        info["scaler"] = {
            "type": type(engine.scaler).__name__,
            "fitted": hasattr(engine.scaler, 'mean_') and engine.scaler.mean_ is not None
        }
        
        # Score model
        if hasattr(engine, 'score_predictor') and engine.score_predictor:
            info["score_model"] = {
                "available": engine.score_predictor.model is not None,
                "classes": len(engine.score_predictor.space) if engine.score_predictor.space else 0
            }
        
        # Info from metadata file
        meta_path = Path(engine.model_path) / "training_metadata.json"
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
    """Load historical data and add as features to model"""
    try:
        if not os.path.exists(RAW_DATA_PATH):
            return jsonify({
                "status": "error",
                "message": f"Historical data folder not found: {RAW_DATA_PATH}",
                "matches_loaded": 0,
                "hint": "Create data/raw folder and add country-based CSV/TXT files"
            }), 404
        
        # Detect all countries
        countries = [
            d for d in os.listdir(RAW_DATA_PATH)
            if os.path.isdir(os.path.join(RAW_DATA_PATH, d)) and d.lower() != "clubs"
        ]
        
        if not countries:
            return jsonify({
                "status": "warning",
                "message": f"No country folders found in {RAW_DATA_PATH}",
                "matches_loaded": 0,
                "hint": "Example: create folders like data/raw/turkey, data/raw/england"
            }), 200
        
        total_matches = 0
        loaded_countries = []
        failed_countries = []
        
        logger.info(f"📂 Detected {len(countries)} countries: {', '.join(countries)}")
        
        for country in countries:
            try:
                logger.info(f"📄 Loading {country}...")
                matches = history_processor.load_country_data(country)
                
                if not matches:
                    failed_countries.append(f"{country} (no data)")
                    continue
                
                # Add to feature engineer
                for m in matches:
                    result = m.get('result', '?')
                    
                    # Determine result
                    if result not in ('1', 'X', '2'):
                        try:
                            hg = int(m.get('home_score', 0))
                            ag = int(m.get('away_score', 0))
                            result = '1' if hg > ag else ('X' if hg == ag else '2')
                        except:
                            result = 'X'
                    
                    # Home team history
                    home_result = 'W' if result == '1' else ('D' if result == 'X' else 'L')
                    engine.feature_engineer.update_team_history(
                        m['home_team'],
                        {
                            'result': home_result,
                            'goals_for': int(m.get('home_score', 0)),
                            'goals_against': int(m.get('away_score', 0)),
                            'date': m.get('date', ''),
                            'venue': 'home'
                        }
                    )
                    
                    # Away team history
                    away_result = 'L' if result == '1' else ('D' if result == 'X' else 'W')
                    engine.feature_engineer.update_team_history(
                        m['away_team'],
                        {
                            'result': away_result,
                            'goals_for': int(m.get('away_score', 0)),
                            'goals_against': int(m.get('home_score', 0)),
                            'date': m.get('date', ''),
                            'venue': 'away'
                        }
                    )
                    
                    # H2H history
                    engine.feature_engineer.update_h2h_history(
                        m['home_team'], m['away_team'],
                        {
                            'result': result,
                            'home_goals': int(m.get('home_score', 0)),
                            'away_goals': int(m.get('away_score', 0))
                        }
                    )
                    
                    # League statistics
                    engine.feature_engineer.update_league_results(
                        m.get('league', 'Unknown'), result
                    )
                
                total_matches += len(matches)
                loaded_countries.append(f"{country} ({len(matches)} matches)")
                logger.info(f"✅ {country}: {len(matches)} matches loaded")
                
            except Exception as e:
                logger.warning(f"⚠️ {country} could not be loaded: {e}")
                failed_countries.append(f"{country} (error: {str(e)[:50]})")
                continue
        
        # Save feature data
        engine.feature_engineer._save_data()
        
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
        models_path = Path(MODELS_DIR)
        
        if not models_path.exists():
            return jsonify({
                "status": "no_models",
                "message": "No models directory found"
            })
        
        # Delete model files
        model_files = [
            "ensemble_models.pkl",
            "scaler.pkl",
            "score_model.pkl",
            "training_metadata.json"
        ]
        
        for file in model_files:
            file_path = models_path / file
            if file_path.exists():
                file_path.unlink()
                deleted_files.append(file)
                logger.info(f"🗑️  Deleted: {file}")
        
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


@app.route("/api/training/start", methods=["POST"])
def start_training():
    """Start model training in background"""
    import threading
    
    def train_background():
        try:
            logger.info("🎯 Training started")
            
            # ✅ Import from model_trainer
            from model_trainer import ProductionModelTrainer
            
            # ✅ Correct parameters matching ProductionModelTrainer.__init__
            trainer = ProductionModelTrainer(
                models_dir=MODELS_DIR,
                raw_data_path=RAW_DATA_PATH,
                clubs_path=CLUBS_PATH,
                min_matches=50,
                test_size=0.2,
                random_state=42,
                version_archive=False,  # Don't create versioned backup
                verbose=True
            )
            
            result = trainer.run_full_pipeline()
            
            if result.get('success'):
                logger.info("✅ Training completed!")
                # Reload models
                engine.load_models()
                logger.info("✅ Models reloaded successfully")
            else:
                logger.error(f"❌ Training error: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"❌ Training error: {e}", exc_info=True)
    
    # Run in background
    thread = threading.Thread(target=train_background, daemon=True)
    thread.start()
    
    return jsonify({
        "status": "ok",
        "message": "Training started in background",
        "hint": "May take 10-15 minutes. Check with /api/models/info"
    })


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
    logger.info("⚽ Predicta Europe ML v2 - Backend Active (TRACKER)")
    logger.info("=" * 60)
    logger.info(f"📂 Models: {MODELS_DIR}")
    logger.info(f"📂 Raw Data: {RAW_DATA_PATH}")
    logger.info(f"📂 Clubs: {CLUBS_PATH}")
    logger.info(f"📊 Predictions: data/predictions")
    logger.info(f"🌐 Port: {APP_PORT}")
    logger.info(f"🤖 Trained: {'✅' if engine.is_trained else '⚠️ Training required'}")
    logger.info("=" * 60)
    
    app.run(host="0.0.0.0", port=APP_PORT, debug=False)
