# ==============================================================
#  Predicta Europe ML v2 - Flask Backend
#  Olcay Arslan | Railway Deployment Ready | Port 8000
# ==============================================================
#  Özellikler:
#   ✅ Tüm Avrupa lig arşivlerini tarar (data/raw/)
#   ✅ Model eğitimi (model_trainer.py üzerinden)
#   ✅ Tek maç ve toplu maç tahmini (MLPredictionEngine)
#   ✅ CORS tam açık (GitHub Pages erişimi)
#   ✅ Railway uyumlu (PORT=8000)
# ==============================================================

import os
import re
import json
from pathlib import Path
from typing import Dict, List
from flask import Flask, request, jsonify
from flask_cors import CORS

# ML Prediction Engine
from ml_prediction_engine import MLPredictionEngine

# Eğitim fonksiyonu (varsa)
try:
    from model_trainer import train_all
except Exception:
    train_all = None

# --- Ortam değişkenleri ---
APP_PORT = int(os.environ.get("PORT", "8000"))
MODELS_DIR = os.environ.get("PREDICTA_MODELS_DIR", "data/ai_models_v2")
RAW_DIR = os.environ.get("PREDICTA_RAW_DIR", "data/raw")

# --- Flask uygulaması ---
app = Flask(__name__)
CORS(app)

# --- Prediction Engine yükle ---
engine = MLPredictionEngine(model_path=MODELS_DIR)

# ==============================================================
#  LEAGUE SCANNER
# ==============================================================

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
    "ru1": "Rusya Premier League",
    "ua1": "Ukrayna Premier League",
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
    """data/raw altındaki ülke klasörlerini tarar, lig kodlarını çıkarır."""
    output = []
    base = Path(root_dir)
    if not base.exists():
        return output

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
        output.append({
            "country": country_dir.name,
            "leagues": leagues
        })
    return output


# ==============================================================
#  CORS HEADERS (GITHUB PAGES İÇİN)
# ==============================================================

@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response


# ==============================================================
#  API ROUTES
# ==============================================================

@app.route("/")
def health():
    return jsonify({
        "status": "Predicta ML v2 aktif",
        "models_dir": MODELS_DIR,
        "raw_dir": RAW_DIR,
        "port": APP_PORT
    })


@app.route("/api/leagues", methods=["GET"])
def leagues():
    """Tüm Avrupa liglerini listeler"""
    data = scan_raw(RAW_DIR)
    return jsonify({"count": len(data), "items": data})


@app.route("/api/reload", methods=["GET"])
def reload_models():
    """Model dosyalarını yeniden yükler"""
    engine._load_models()
    return jsonify({"status": "Models reloaded"})


@app.route("/api/training/start", methods=["POST"])
def train_api():
    """Yeni model eğitimi başlatır"""
    if train_all is None:
        return jsonify({"error": "model_trainer.py bulunamadı"}), 500
    try:
        body = request.get_json(silent=True) or {}
        top_k = int(body.get("top_scores_k", 20))
    except Exception:
        top_k = 20
    result = train_all(raw_path=RAW_DIR, top_scores_k=top_k)
    return jsonify(result)


@app.route("/api/predict", methods=["POST"])
def predict():
    """Tek maç tahmini yapar"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    home = data.get("home_team")
    away = data.get("away_team")
    league = data.get("league", "Unknown")
    odds = data.get("odds", {"1": 2.0, "X": 3.0, "2": 3.5})

    if not home or not away:
        return jsonify({"error": "home_team & away_team required"}), 400

    prediction = engine.predict_match(home, away, odds, league)
    return jsonify(prediction)


@app.route("/api/predictions/batch", methods=["POST"])
def predict_batch():
    """Birden fazla maçı toplu olarak işler"""
    data = request.get_json()
    if not data or "matches" not in data:
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
            results.append({"error": str(e), "match": match})

    return jsonify({"count": len(results), "items": results})


# ==============================================================
#  UYGULAMA ÇALIŞTIR
# ==============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=APP_PORT)

