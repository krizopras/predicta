import os
from flask import Flask, jsonify, request
from datetime import date
from nesine_fetcher import fetch_bulletin

app = Flask(__name__)

# === Config ===
APP_PORT = int(os.environ.get("PORT", 8000))
MODELS_DIR = os.environ.get("PREDICTA_MODELS_DIR", "data/ai_models_v2")

@app.route("/")
def home():
    return jsonify({
        "status": "Predicta ML v2 active",
        "port": APP_PORT,
        "models_dir": MODELS_DIR,
        "model_trained": os.path.exists(MODELS_DIR),
    })


# ✅ NESINE – Today’s matches (no league filtering)
@app.route("/api/predict/today")
def predict_today():
    filter_param = request.args.get("filter", "false").lower() == "true"
    matches = fetch_bulletin(date.today(), filter_leagues=filter_param)
    return jsonify({"items": matches, "count": len(matches)})


# ✅ Get all leagues (used by sidebar)
@app.route("/api/leagues")
def get_leagues():
    matches = fetch_bulletin(date.today(), filter_leagues=False)
    leagues = sorted(set(m["league"] for m in matches if "league" in m))
    items = [{"name": l} for l in leagues]
    return jsonify({"items": [{"leagues": items}], "count": len(leagues)})


# ✅ Debug test endpoint
@app.route("/api/debug/nesine-test")
def nesine_debug():
    matches = fetch_bulletin(date.today(), filter_leagues=False)
    leagues = {}
    for m in matches:
        leagues[m["league"]] = leagues.get(m["league"], 0) + 1

    return jsonify({
        "date": str(date.today()),
        "total_matches": len(matches),
        "filtered_matches": len(matches),
        "leagues": leagues,
        "api_url": "https://cdnbulten.nesine.com/api/bulten/getprebultenfull",
    })


# ✅ Favicon fix (prevent browser 404)
@app.route("/favicon.ico")
def favicon():
    return "", 204


if __name__ == "__main__":
    print("🚀 Predicta Europe ML Backend started")
    app.run(host="0.0.0.0", port=APP_PORT)
