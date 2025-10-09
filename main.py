#!/usr/bin/env python3
"""
Predicta Europe ML v2 - Flask Backend (DÜZELTILMIŞ VERSIYON)
Auto bulletin fetcher (Nesine) + ML predictions
"""

import os
import logging
from datetime import date
from flask import Flask, jsonify, request
from flask_cors import CORS

# ========== ML & Nesine imports ==========
from ml_prediction_engine import MLPredictionEngine
from nesine_fetcher import fetch_bulletin, fetch_today

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
            "/api/debug/nesine-test",
            "/api/reload",
            "/api/history/load",
            "/api/training/start"
        ]
    })


# ----------------- LIGLER -----------------
@app.route("/api/leagues", methods=["GET"])
def get_leagues():
    """Bugünkü bültenden lig listesi döndür"""
    try:
        filter_enabled = request.args.get('filter', 'false').lower() == 'true'
        today = date.today()
        matches = fetch_bulletin(today, filter_leagues=filter_enabled)
        
        # Ligleri topla
        leagues_dict = {}
        for m in matches:
            league_name = m.get("league", "Bilinmeyen")
            if league_name not in leagues_dict:
                leagues_dict[league_name] = {
                    "name": league_name,
                    "match_count": 0
                }
            leagues_dict[league_name]["match_count"] += 1
        
        # Listeye çevir ve sırala
        leagues_list = sorted(leagues_dict.values(), key=lambda x: x["match_count"], reverse=True)
        
        return jsonify({
            "count": len(leagues_list),
            "items": leagues_list,
            "total_matches": len(matches)
        })
    except Exception as e:
        logger.error(f"/api/leagues error: {e}", exc_info=True)
        return jsonify({"error": str(e), "items": []}), 500


# ----------------- TÜM MAÇLAR -----------------
@app.route("/api/matches/today", methods=["GET"])
def get_today_matches():
    """Nesine bülteninden tüm maçları getir"""
    try:
        filter_enabled = request.args.get('filter', 'false').lower() == 'true'
        today = date.today()
        matches = fetch_bulletin(today, filter_leagues=filter_enabled)
        
        logger.info(f"📊 Bugün {len(matches)} maç bulundu (filtre: {filter_enabled})")
        
        return jsonify({
            "count": len(matches),
            "items": matches,
            "date": str(today),
            "filter_enabled": filter_enabled
        })
    except Exception as e:
        logger.error(f"/api/matches/today error: {e}", exc_info=True)
        return jsonify({"error": str(e), "items": []}), 500


# ----------------- BUGÜNKÜ MAÇLAR + TAHMİN -----------------
@app.route("/api/predict/today", methods=["GET"])
def predict_today():
    """Bugünkü tüm maçları tahmin et"""
    try:
        filter_enabled = request.args.get('filter', 'false').lower() == 'true'
        league_filter = request.args.get('league', '').strip()
        
        today = date.today()
        matches = fetch_bulletin(today, filter_leagues=filter_enabled)
        
        logger.info(f"🎯 {len(matches)} maç için tahmin yapılıyor...")

        results = []
        success_count = 0
        
        for m in matches:
            try:
                home = m.get("home_team")
                away = m.get("away_team")
                odds = m.get("odds", {"1": 2.0, "X": 3.0, "2": 3.5})
                league = m.get("league", "Unknown")
                
                # Lig filtresi
                if league_filter and league_filter.lower() not in league.lower():
                    continue
                
                # Oranlar eksikse varsayılan değerler kullan
                if not odds.get("1"):
                    odds["1"] = 2.0
                if not odds.get("X"):
                    odds["X"] = 3.0
                if not odds.get("2"):
                    odds["2"] = 3.5
                
                # Tahmin yap
                pred = engine.predict_match(home, away, odds, league)
                
                results.append({
                    **m,
                    "prediction": pred
                })
                success_count += 1
                
            except Exception as err:
                logger.warning(f"⚠️ Tahmin hatası ({home} - {away}): {err}")
                results.append({
                    **m,
                    "prediction": {
                        "error": str(err),
                        "prediction": "?",
                        "confidence": 0
                    }
                })
        
        logger.info(f"✅ {success_count}/{len(matches)} maç başarıyla tahmin edildi")
        
        return jsonify({
            "count": len(results),
            "items": results,
            "date": str(today),
            "success_count": success_count,
            "filter_enabled": filter_enabled
        })
        
    except Exception as e:
        logger.error(f"/api/predict/today error: {e}", exc_info=True)
        return jsonify({"error": str(e), "items": []}), 500


# ----------------- TOPLU TAHMİN -----------------
@app.route("/api/predictions/batch", methods=["POST"])
def batch_predictions():
    """Toplu maç tahmini (CSV formatı)"""
    data = request.get_json(silent=True) or {}
    csv = data.get("csv", "")
    matches = []

    if not csv.strip():
        return jsonify({"error": "CSV gerekli"}), 400

    try:
        lines = [ln.strip() for ln in csv.splitlines() if ln.strip()]
        
        # Başlık satırını atla
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
                    logger.warning(f"⚠️ CSV satırı atlandı: {ln} ({e})")
                    continue

        results = []
        for m in matches:
            try:
                pred = engine.predict_match(
                    m["home_team"], m["away_team"], m["odds"], m["league"]
                )
                results.append({**m, "prediction": pred})
            except Exception as e:
                logger.error(f"Tahmin hatası: {e}")
                results.append({**m, "error": str(e)})

        return jsonify({
            "count": len(results),
            "items": results
        })
        
    except Exception as e:
        logger.error(f"/api/predictions/batch error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ----------------- DEBUG NESINE -----------------
@app.route("/api/debug/nesine-test", methods=["GET"])
def debug_nesine():
    """Nesine API test endpoint"""
    try:
        today = date.today()
        
        # Filtresiz tüm maçlar
        all_matches = fetch_bulletin(today, filter_leagues=False)
        
        # Ligleri grupla
        leagues = {}
        for m in all_matches:
            lg = m.get("league", "Unknown")
            leagues[lg] = leagues.get(lg, 0) + 1

        # Filtrelenmiş maçlar
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


# ----------------- MODEL YÖNETİMİ -----------------
@app.route("/api/reload", methods=["GET", "POST"])
def reload_models():
    """Modelleri yeniden yükle"""
    try:
        engine.load_models()
        return jsonify({
            "status": "ok",
            "message": "Modeller yeniden yüklendi",
            "is_trained": engine.is_trained
        })
    except Exception as e:
        logger.error(f"/api/reload error: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/history/load", methods=["POST"])
def load_history():
    """Geçmiş verileri yükle (placeholder)"""
    return jsonify({
        "status": "ok",
        "message": "Bu özellik henüz aktif değil",
        "matches_loaded": 0
    })


@app.route("/api/training/start", methods=["POST"])
def start_training():
    """Model eğitimini başlat (placeholder)"""
    return jsonify({
        "status": "ok",
        "message": "Eğitim başlatıldı (background process)",
        "training_started": True
    })


# ======================================================
# HATA YÖNETİMİ
# ======================================================
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint bulunamadı"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Sunucu hatası"}), 500


# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("⚽ Predicta Europe ML v2 - Backend Aktif")
    logger.info("=" * 60)
    logger.info(f"📂 Models: {MODELS_DIR}")
    logger.info(f"🌐 Port: {APP_PORT}")
    logger.info(f"🤖 Trained: {'✅' if engine.is_trained else '⚠️ Eğitim gerekli'}")
    logger.info("=" * 60)
    
    app.run(host="0.0.0.0", port=APP_PORT, debug=False)
