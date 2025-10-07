#!/usr/bin/env python3
"""
Predicta Europe ML v2 - Flask Backend (TAMAMLANMIŞ VERSİYON)
Geçmiş maç verileri + Nesine canlı bülten entegrasyonu
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List
from flask import Flask, request, jsonify
from flask_cors import CORS

from ml_prediction_engine import MLPredictionEngine
from nesine_fetcher import fetch_today, fetch_matches_for_date

# Geçmiş veri işleyici
try:
    from historical_processor import HistoricalDataProcessor
    HISTORICAL_AVAILABLE = True
except Exception:
    HISTORICAL_AVAILABLE = False

try:
    from model_trainer import train_all
except Exception:
    train_all = None

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Config
APP_PORT = int(os.environ.get("PORT", "8000"))
MODELS_DIR = os.environ.get("PREDICTA_MODELS_DIR", "data/ai_models_v2")
RAW_DIR = os.environ.get("PREDICTA_RAW_DIR", "data/raw")

app = Flask(__name__)
CORS(app)

# ML Engine (global)
engine = MLPredictionEngine(model_path=MODELS_DIR)

# Geçmiş veri işleyici
historical_processor = None
if HISTORICAL_AVAILABLE:
    historical_processor = HistoricalDataProcessor(raw_data_path=RAW_DIR)
    logger.info("✅ Geçmiş veri işleyici aktif")

# ----------------- League Scanner -----------------
LEAGUE_CODE_MAP = {
    "tr1": "Türkiye Süper Lig",
    "tr2": "Türkiye 1. Lig",
    "en1": "İngiltere Premier League",
    "en2": "İngiltere Championship",
    "es1": "İspanya LaLiga",
    "es2": "İspanya LaLiga 2",
    "it1": "İtalya Serie A",
    "it2": "İtalya Serie B",
    "de1": "Almanya Bundesliga",
    "de2": "Almanya 2. Bundesliga",
    "fr1": "Fransa Ligue 1",
    "fr2": "Fransa Ligue 2",
    "nl1": "Hollanda Eredivisie",
    "pt1": "Portekiz Primeira Liga",
    "be1": "Belçika Pro League",
    "at1": "Avusturya Bundesliga",
    "ch1": "İsviçre Super League",
    "sc1": "İskoçya Premiership",
    "pl1": "Polonya Ekstraklasa",
    "ro1": "Romanya Liga I",
    "cz1": "Çekya 1. Liga",
    "gr1": "Yunanistan Super League",
    "se1": "İsveç Allsvenskan",
    "no1": "Norveç Eliteserien",
    "dk1": "Danimarka Superliga",
    "ie1": "İrlanda Premier Division",
    "hu1": "Macaristan NB I",
    "bg1": "Bulgaristan First League",
    "rs1": "Sırbistan SuperLiga",
    "hr1": "Hırvatistan HNL",
    "si1": "Slovenya PrvaLiga",
    "sk1": "Slovakya Super Liga"
}
FILE_CODE_REGEX = re.compile(r".*_(?P<code>[a-z]{2,3}\d?)\.txt$", re.IGNORECASE)

def scan_raw(root_dir: str = "data/raw") -> List[Dict]:
    """Raw klasöründeki ligleri tara"""
    out = []
    base = Path(root_dir)
    if not base.exists():
        return out
    for country_dir in sorted([d for d in base.iterdir() if d.is_dir()]):
        leagues = []
        for file in country_dir.glob("*.txt"):
            m = FILE_CODE_REGEX.match(file.name)
            if m:
                code = m.group("code").lower()
                leagues.append({
                    "code": code,
                    "name": LEAGUE_CODE_MAP.get(code, code.upper())
                })
        out.append({"country": country_dir.name, "leagues": leagues})
    return out

def load_historical_data():
    """Geçmiş maç verilerini yükle ve feature engineer'a besle"""
    if not historical_processor:
        logger.warning("⚠️ Geçmiş veri işleyici yok")
        return 0
    
    try:
        logger.info("📊 Geçmiş maç verileri yükleniyor...")
        matches, team_stats = historical_processor.process_all_countries()
        
        # Feature engineer'a geçmiş verileri besle
        for match in matches:
            try:
                # MS (Maç Sonucu) belirleme
                home_score = match.get('home_score', 0)
                away_score = match.get('away_score', 0)
                
                if home_score > away_score:
                    result = '1'
                elif home_score < away_score:
                    result = '2'
                else:
                    result = 'X'
                
                # Feature engineer'a ekle
                engine.feature_engineer.update_team_history(match['home_team'], {
                    'result': 'W' if result == '1' else ('D' if result == 'X' else 'L'),
                    'goals_for': home_score,
                    'goals_against': away_score,
                    'date': match.get('date', ''),
                    'venue': 'home'
                })
                
                engine.feature_engineer.update_team_history(match['away_team'], {
                    'result': 'L' if result == '1' else ('D' if result == 'X' else 'W'),
                    'goals_for': away_score,
                    'goals_against': home_score,
                    'date': match.get('date', ''),
                    'venue': 'away'
                })
                
                engine.feature_engineer.update_h2h_history(
                    match['home_team'],
                    match['away_team'],
                    {
                        'result': result,
                        'home_goals': home_score,
                        'away_goals': away_score
                    }
                )
                
                engine.feature_engineer.update_league_results(
                    match.get('league', 'Unknown'),
                    result
                )
            except Exception as e:
                logger.warning(f"Maç işleme hatası: {e}")
                continue
        
        logger.info(f"✅ {len(matches)} geçmiş maç yüklendi")
        return len(matches)
        
    except Exception as e:
        logger.error(f"❌ Geçmiş veri yükleme hatası: {e}")
        return 0

# ----------------- CORS Headers -----------------
@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

# ----------------- Health Check -----------------
@app.route("/")
def health():
    return jsonify({
        "status": "Predicta ML v2 aktif",
        "models_dir": MODELS_DIR,
        "raw_dir": RAW_DIR,
        "port": APP_PORT,
        "model_trained": engine.is_trained,
        "historical_available": HISTORICAL_AVAILABLE
    })

# ----------------- Static Data -----------------
@app.route("/api/leagues", methods=["GET"])
def leagues():
    data = scan_raw(RAW_DIR)
    return jsonify({"count": len(data), "items": data})

@app.route("/api/reload", methods=["GET"])
def reload_models():
    """Modelleri yeniden yükle"""
    engine._load_models()
    return jsonify({"status": "Models reloaded", "is_trained": engine.is_trained})

@app.route("/api/history/load", methods=["POST"])
def load_history():
    """Geçmiş verileri yükle"""
    count = load_historical_data()
    return jsonify({
        "status": "ok" if count > 0 else "error",
        "matches_loaded": count
    })

@app.route("/api/training/start", methods=["POST"])
def train_api():
    """Model eğitimi başlat"""
    if train_all is None:
        return jsonify({"error": "model_trainer.py bulunamadı"}), 500
    
    body = request.get_json(silent=True) or {}
    top_k = int(body.get("top_scores_k", 20))
    result = train_all(raw_path=RAW_DIR, top_scores_k=top_k)
    
    # Eğitimden sonra modelleri yeniden yükle
    engine._load_models()
    
    return jsonify(result)

# ----------------- Prediction Endpoints -----------------
@app.route("/api/predict", methods=["POST"])
def predict():
    """Tek maç tahmini"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    home = data.get("home_team")
    away = data.get("away_team")
    league = data.get("league", "Unknown")
    odds = data.get("odds", {"1": 2.0, "X": 3.0, "2": 3.5})

    if not home or not away:
        return jsonify({"error": "home_team & away_team required"}), 400

    try:
        prediction = engine.predict_match(home, away, odds, league)
        return jsonify(prediction)
    except Exception as e:
        logger.error(f"Tahmin hatası: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/predictions/batch", methods=["POST"])
def predict_batch():
    """Toplu tahmin"""
    data = request.get_json()
    if not data or "matches" not in data:
        # CSV formatı varsa dönüştür
        csv = (data or {}).get("csv")
        if csv:
            matches = []
            lines = [ln.strip() for ln in csv.splitlines() if ln.strip()]
            if lines and "," in lines[0].lower():
                # Başlık satırı varsa atla
                for ln in lines[1:] if "home" in lines[0].lower() else lines:
                    parts = [p.strip() for p in ln.split(",")]
                    if len(parts) >= 6:
                        matches.append({
                            "home_team": parts[0],
                            "away_team": parts[1],
                            "league": parts[2],
                            "odds": {"1": float(parts[3]), "X": float(parts[4]), "2": float(parts[5])}
                        })
            data = {"matches": matches}
        else:
            return jsonify({"error": "matches list required"}), 400

    results = []
    for match in data["matches"]:
        try:
            home = match["home_team"]
            away = match["away_team"]
            league = match.get("league", "Unknown")
            odds = match.get("odds", {"1": 2.0, "X": 3.0, "2": 3.5})
            pred = engine.predict_match(home, away, odds, league)
            results.append({
                "home_team": home,
                "away_team": away,
                "league": league,
                "prediction": pred
            })
        except Exception as e:
            logger.error(f"Batch tahmin hatası: {e}")
            results.append({"error": str(e), "match": match})

    return jsonify({"count": len(results), "items": results})

# ----------------- Nesine Canlı Maçlar -----------------
@app.route("/api/matches/today", methods=["GET"])
def today_matches():
    """Bugünkü maçları Nesine'den çek"""
    league_filter = (request.args.get("league") or "").strip().lower()
    
    try:
        out = fetch_today()
        
        if league_filter:
            out = [m for m in out if league_filter in (m.get("league", "").lower())]
        
        logger.info(f"✅ {len(out)} maç çekildi (filtre: {league_filter or 'yok'})")
        return jsonify({"count": len(out), "items": out})
    
    except Exception as e:
        logger.error(f"❌ Nesine fetch hatası: {e}", exc_info=True)
        return jsonify({"error": str(e), "count": 0, "items": []}), 500

@app.route("/api/matches/date", methods=["GET"])
def date_matches():
    """Belirli bir tarihteki maçları çek"""
    d = request.args.get("d")
    if not d:
        return jsonify({"error": "use ?d=YYYY-MM-DD"}), 400
    
    try:
        from datetime import date as _date
        year, month, day = map(int, d.split("-"))
        items = fetch_matches_for_date(_date(year, month, day))
        return jsonify({"count": len(items), "items": items})
    except Exception as e:
        logger.error(f"Tarih hatası: {e}")
        return jsonify({"error": str(e)}), 400

@app.route("/api/predict/today", methods=["GET"])
def predict_today():
    """Bugünkü maçlar için tahmin yap"""
    league_filter = (request.args.get("league") or "").strip().lower()
    
    try:
        # Maçları çek
        items = fetch_today()
        
        if league_filter:
            items = [m for m in items if league_filter in (m.get("league", "").lower())]
        
        logger.info(f"📊 {len(items)} maç için tahmin yapılıyor...")
        
        results = []
        for i, m in enumerate(items):
            try:
                logger.info(f"[{i+1}/{len(items)}] {m['home_team']} - {m['away_team']}")
                
                pred = engine.predict_match(
                    m["home_team"],
                    m["away_team"],
                    m["odds"],
                    m.get("league", "Unknown")
                )
                
                results.append({
                    **m,
                    "prediction": pred
                })
                
            except Exception as e:
                logger.error(f"❌ {m['home_team']} - {m['away_team']} hatası: {e}")
                results.append({
                    **m,
                    "error": str(e),
                    "prediction": {
                        "prediction": "X",
                        "confidence": 0,
                        "score_prediction": "?-?"
                    }
                })
        
        logger.info(f"✅ {len(results)} tahmin tamamlandı")
        return jsonify({"count": len(results), "items": results})
    
    except Exception as e:
        logger.error(f"❌ Toplu tahmin hatası: {e}", exc_info=True)
        return jsonify({"error": str(e), "count": 0, "items": []}), 500

# ----------------- Debug Endpoints -----------------
@app.route("/api/debug/feature-history", methods=["GET"])
def debug_feature_history():
    """Feature engineer'daki geçmiş verileri kontrol et"""
    team = request.args.get("team", "")
    
    if not team:
        return jsonify({
            "teams_count": len(engine.feature_engineer.team_history),
            "h2h_count": len(engine.feature_engineer.h2h_history),
            "leagues_count": len(engine.feature_engineer.league_results)
        })
    
    history = engine.feature_engineer.team_history.get(team, [])
    return jsonify({
        "team": team,
        "matches_count": len(history),
        "recent_matches": history[-5:] if history else []
    })

@app.route("/api/debug/nesine-test", methods=["GET"])
def debug_nesine():
    """Nesine API'yi test et"""
    try:
        from datetime import date
        import json
        
        today = date.today()
        matches = fetch_matches_for_date(today)
        
        return jsonify({
            "status": "ok" if matches else "no_data",
            "date": str(today),
            "matches_count": len(matches),
            "sample": matches[:3] if matches else [],
            "api_url": f"https://cdnbulten.nesine.com/api/bulten/getprebultenfull?date={today}"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

# ----------------- Startup Actions -----------------
@app.before_first_request
def startup():
    """Uygulama başlarken geçmiş verileri yükle"""
    logger.info("🚀 Predicta ML v2 başlatılıyor...")
    
    # Geçmiş verileri yükle
    if HISTORICAL_AVAILABLE:
        try:
            count = load_historical_data()
            logger.info(f"✅ {count} geçmiş maç yüklendi")
        except Exception as e:
            logger.error(f"⚠️ Geçmiş veri yükleme hatası: {e}")
    
    # Model durumunu kontrol et
    if engine.is_trained:
        logger.info("✅ ML modelleri hazır")
    else:
        logger.warning("⚠️ Modeller eğitilmemiş - temel tahmin modu aktif")
    
    logger.info("🎯 Sistem hazır!")

# ----------------- Run -----------------
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🎯 PREDICTA EUROPE ML v2")
    logger.info("=" * 60)
    logger.info(f"📂 Models: {MODELS_DIR}")
    logger.info(f"📂 Raw Data: {RAW_DIR}")
    logger.info(f"🌐 Port: {APP_PORT}")
    logger.info(f"🤖 ML Model: {'✅ Eğitilmiş' if engine.is_trained else '⚠️ Eğitilmemiş'}")
    logger.info(f"📊 Geçmiş Veri: {'✅ Aktif' if HISTORICAL_AVAILABLE else '❌ Kapalı'}")
    logger.info("=" * 60)
    
    app.run(host="0.0.0.0", port=APP_PORT, debug=False)