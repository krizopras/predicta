# main.py
# Predicta Europe ML v2 — Flask backend
# Özellikler:
#  - /api/training/start: Tüm Avrupa arşiviyle modeli eğitir (MS + Skor)
#  - /api/reload: Diskteki modelleri yeniden yükler
#  - /api/predict: Tek maç için tahmin (MS + skor + value bet)
#  - /api/predictions/batch: Çoklu maç tahmini (CSV/JSON)
#  - /api/leagues: data/raw içindeki ülkeleri ve lig kodlarını listeler (index.html dropdown için)

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from ml_prediction_engine import MLPredictionEngine
# Eğitim tetiklemek için:
try:
    from model_trainer import train_all
except Exception:
    train_all = None  # Eğitim dosyası yoksa graceful davran

APP_PORT = int(os.environ.get("PORT", "5000"))
MODELS_DIR = os.environ.get("PREDICTA_MODELS_DIR", "data/ai_models_v2")
RAW_DIR = os.environ.get("PREDICTA_RAW_DIR", "data/raw")

app = Flask(__name__, static_folder="templates", static_url_path="/templates")
CORS(app)

engine = MLPredictionEngine(model_path=MODELS_DIR)

# ----------------------- Yardımcılar -----------------------------

LEAGUE_CODE_MAP = {
    # yaygın kod -> okunur ad (genişletilebilir)
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
    "sk1": "Slovakya Super Liga",
    "tr":  "Türkiye (genel)",
    "en":  "İngiltere (genel)",
    "es":  "İspanya (genel)",
    "it":  "İtalya (genel)",
    "de":  "Almanya (genel)",
    "fr":  "Fransa (genel)",
}

FILE_CODE_REGEX = re.compile(r".*_(?P<code>[a-z]{2,3}\d?)\.txt$", re.IGNORECASE)

def scan_europe_raw(raw_root: str) -> List[Dict]:
    """
    data/raw/ altındaki tüm ülke klasörlerini ve içindeki *_<code>.txt dosyalarını tarar.
    Örn: 2018-19_tr1.txt -> code = tr1
    Dönüş: [{country:'turkey', leagues:[{code:'tr1', name:'Türkiye Süper Lig'}, ...]}, ...]
    """
    out = []
    root = Path(raw_root)
    if not root.exists():
        return out

    for country_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        country_name = country_dir.name
        codes = set()
        for f in country_dir.glob("*.txt"):
            m = FILE_CODE_REGEX.match(f.name)
            if m:
                codes.add(m.group("code").lower())

        leagues = []
        for code in sorted(codes):
            readable = LEAGUE_CODE_MAP.get(code, code.upper())
            leagues.append({"code": code, "name": readable})

        out.append({"country": country_name, "leagues": leagues})
    return out

def odds_from_payload(data: Dict) -> Dict:
    odds = data.get("odds", {})
    # needle case normalize
    return {
        "1": float(odds.get("1", odds.get("MS1", 2.0))),
        "X": float(odds.get("X", odds.get("MS0", 3.0))),
        "2": float(odds.get("2", odds.get("MS2", 3.5))),
    }

# -------------------------- Routes --------------------------------

@app.route("/")
def health():
    return jsonify({
        "status": "Predicta Europe ML v2 ready",
        "models_dir": MODELS_DIR,
        "raw_dir": RAW_DIR
    })

@app.route("/templates/<path:path>")
def serve_templates(path):
    # index.html gibi dosyaları servis edebilmek için
    return send_from_directory("templates", path)

@app.route("/api/leagues", methods=["GET"])
def api_leagues():
    # data/raw/ altını tarayıp tüm Avrupa ülkeleri + lig kodlarını döner
    data = scan_europe_raw(RAW_DIR)
    return jsonify({"count": len(data), "items": data})

@app.route("/api/reload", methods=["GET"])
def api_reload():
    engine._load_models()
    return jsonify({"status": "reloaded"})

@app.route("/api/training/start", methods=["POST", "GET"])
def api_training_start():
    if train_all is None:
        return jsonify({"error": "model_trainer bulunamadı"}), 500
    top_k = None
    try:
        body = request.get_json(silent=True) or {}
        top_k = int(body.get("top_scores_k", 20))
    except Exception:
        top_k = 20
    summary = train_all(raw_path=RAW_DIR, top_scores_k=top_k)
    return jsonify(summary)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400
    home = data.get("home_team")
    away = data.get("away_team")
    league = data.get("league", "Unknown")
    if not home or not away:
        return jsonify({"error": "home_team & away_team required"}), 400
    odds = odds_from_payload(data)
    pred = engine.predict_match(home, away, odds, league)
    return jsonify(pred)

@app.route("/api/predictions/batch", methods=["POST"])
def api_predictions_batch():
    """
    Body iki formattan birinde olabilir:
      1) {"matches":[{"home_team":"...","away_team":"...","league":"...","odds":{"1":2.0,"X":3.0,"2":3.5}}, ...]}
      2) {"csv":"home,away,league,1,X,2\nGalatasaray,Fenerbahçe,Super Lig,2.1,3.2,3.4\n..."}
    """
    data = request.get_json()
    results = []
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    if "matches" in data and isinstance(data["matches"], list):
        for m in data["matches"]:
            try:
                home = m["home_team"]; away = m["away_team"]
                league = m.get("league", "Unknown")
                odds = odds_from_payload(m)
                results.append({
                    "home_team": home,
                    "away_team": away,
                    "league": league,
                    "odds": odds,
                    "prediction": engine.predict_match(home, away, odds, league)
                })
            except Exception as e:
                results.append({"error": str(e), "input": m})
        return jsonify({"count": len(results), "items": results})

    if "csv" in data and isinstance(data["csv"], str):
        lines = [ln.strip() for ln in data["csv"].splitlines() if ln.strip()]
        # header bekleniyor: home,away,league,1,X,2
        start = 1 if lines and lines[0].lower().startswith("home") else 0
        for ln in lines[start:]:
            try:
                home, away, league, o1, ox, o2 = [p.strip() for p in ln.split(",")]
                odds = {"1": float(o1), "X": float(ox), "2": float(o2)}
                results.append({
                    "home_team": home,
                    "away_team": away,
                    "league": league,
                    "odds": odds,
                    "prediction": engine.predict_match(home, away, odds, league)
                })
            except Exception as e:
                results.append({"error": str(e), "line": ln})
        return jsonify({"count": len(results), "items": results})

    return jsonify({"error": "Provide 'matches' array or 'csv' string"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=APP_PORT, debug=True)
