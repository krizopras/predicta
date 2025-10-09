#!/usr/bin/env python3
"""
Predicta Europe ML v2 - Flask Backend (FINAL VERSION)
Historical match data + Nesine live bulletin integration
"""

import os
import re
import logging
from pathlib import Path
from datetime import date
from typing import Dict, List
from flask import Flask, request, jsonify
from flask_cors import CORS

from ml_prediction_engine import MLPredictionEngine
from nesine_fetcher import fetch_today, fetch_matches_for_date, fetch_bulletin

# Historical data processor
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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Config
APP_PORT = int(os.environ.get("PORT", "8000"))
MODELS_DIR = os.environ.get("PREDICTA_MODELS_DIR", "data/ai_models_v2")
RAW_DIR = os.environ.get("PREDICTA_RAW_DIR", "data/raw")

app = Flask(__name__)

# CORS
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": False,
        "max_age": 3600
    }
})

# ML Engine
engine = MLPredictionEngine(model_path=MODELS_DIR)

# Historical Processor
historical_processor = None
if HISTORICAL_AVAILABLE:
    historical_processor = HistoricalDataProcessor(raw_data_path=RAW_DIR)
    logger.info("✅ Historical processor active")

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
    """Scan leagues from the raw data folder"""
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
    """Load historical match data and feed to the feature engineer"""
    if not historical_processor:
        logger.warning("⚠️ No historical processor found")
        return 0

    try:
        logger.info("📜 Loading historical match data...")
        matches, team_stats = historical_processor.process_all_countries()

        for match in matches:
            try:
                home_score = match.get("home_score", 0)
                away_score = match.get("away_score", 0)

                if home_score > away_score:
                    result = "1"
                elif home_score < away_score:
                    result = "2"
                else:
                    result = "X"

                engine.feature_engineer.update_team_history(match["home_team"], {
                    "result": "W" if result == "1" else ("D" if result == "X" else "L"),
                    "goals_for": home_score,
                    "goals_against": away_score,
                    "date": match.get("date", ""),
                    "venue": "home"
                })

                engine.feature_engineer.update_team_history(match["away_team"], {
                    "result": "L" if result == "1" else ("D" if result == "X" else "W"),
                    "goals_for": away_score,
                    "goals_against": home_score,
                    "date": match.get("date", ""),
                    "venue": "away"
                })

                engine.feature_engineer.update_h2h_history(
                    match["home_team"], match["away_team"],
                    {"result": result, "home_goals": home_score, "away_goals": away_score}
                )

                engine.feature_engineer.update_league_results(
                    match.get("league", "Unknown"), result
                )
            except Exception as e:
                logger.warning(f"Match processing error: {e}")
                continue

        logger.info(f"✅ Loaded {len(matches)} historical matches")
        return len(matches)

    except Exception as e:
        logger.error(f"❌ Error loading historical data: {e}")
        return 0

# ----------------- Flask Routes -----------------

@app.route("/")
def health():
    return jsonify({
        "status": "Predicta ML v2 active",
        "models_dir": MODELS_DIR,
        "raw_dir": RAW_DIR,
        "port": APP_PORT,
        "model_trained": engine.is_trained,
        "historical_available": HISTORICAL_AVAILABLE
    })


@app.route("/api/matches/today")
def today_matches():
    """Fetch today's matches from Nesine"""
    filter_enabled = request.args.get("filter", "false").lower() == "true"

    try:
        logger.info(f"💡 Fetching today's matches (filter={filter_enabled})...")
        out = fetch_today(filter_leagues=filter_enabled)

        if not out:
            return jsonify({
                "count": 0,
                "items": [],
                "message": "No matches found for today."
            })

        logger.info(f"✅ {len(out)} matches retrieved")
        return jsonify({"count": len(out), "items": out})

    except Exception as e:
        logger.error(f"❌ Nesine fetch error: {e}", exc_info=True)
        return jsonify({"error": str(e), "count": 0, "items": []})


@app.route("/api/predict/today")
def predict_today():
    """Generate predictions for today's matches"""
    filter_enabled = request.args.get("filter", "false").lower() == "true"
    try:
        logger.info(f"💡 Fetching today's matches (filter={filter_enabled})...")
        items = fetch_today(filter_leagues=filter_enabled)

        if not items:
            return jsonify({"count": 0, "items": [], "message": "No matches found."})

        results = []
        for i, m in enumerate(items):
            try:
                pred = engine.predict_match(
                    m.get("home_team", "Unknown"),
                    m.get("away_team", "Unknown"),
                    m.get("odds", {"1": 2.0, "X": 3.0, "2": 3.5}),
                    m.get("league", "Unknown")
                )
                results.append({**m, "prediction": pred})
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                results.append({**m, "error": str(e)})

        logger.info(f"✅ {len(results)} predictions completed")
        return jsonify({"count": len(results), "items": results})

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}", exc_info=True)
        return jsonify({"error": str(e), "count": 0, "items": []})


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🎯 PREDICTA EUROPE ML v2")
    logger.info("=" * 60)
    logger.info(f"📂 Models: {MODELS_DIR}")
    logger.info(f"📂 Raw Data: {RAW_DIR}")
    logger.info(f"🌐 Port: {APP_PORT}")
    logger.info(f"🤖 ML Model: {'✅ Trained' if engine.is_trained else '⚠️ Untrained'}")
    logger.info(f"📊 Historical Data: {'✅ Active' if HISTORICAL_AVAILABLE else '❌ Disabled'}")
    logger.info("=" * 60)
    app.run(host="0.0.0.0", port=APP_PORT, debug=False)
