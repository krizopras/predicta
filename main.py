#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predicta Europe ML - Ana API Sunucusu (detaylı + Railway uyumlu)
"""

import os
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS

# ==============================================================
# LOG AYARLARI
# ==============================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [API] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# ==============================================================
# FLASK BAŞLAT
# ==============================================================
app = Flask(__name__)
CORS(app)

REQUIRED_DIRS = ["data", "data/raw", "data/ai_models_v2", "data/clubs", "logs"]

for d in REQUIRED_DIRS:
    os.makedirs(d, exist_ok=True)
    log.info(f"📂 Klasör kontrolü: {d} → OK")

# ==============================================================
# MODEL MOTORU YÜKLE
# ==============================================================
try:
    from ml_prediction_engine import MLPredictionEngine
    engine = MLPredictionEngine()
    log.info("✅ MLPredictionEngine başarıyla yüklendi.")
except Exception as e:
    engine = None
    log.error(f"❌ Tahmin motoru yüklenemedi: {e}")

# ==============================================================
# ROOT ENDPOINT
# ==============================================================
@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "message": "Predicta Europe ML API çalışıyor 🚀",
        "engine_loaded": engine is not None
    })

# ==============================================================
# MODEL YENİDEN YÜKLE
# ==============================================================
@app.route("/api/reload", methods=["POST", "GET"])
def reload_models():
    try:
        if engine is None:
            return jsonify({"status": "error", "message": "Engine yüklenemedi"}), 500
        if hasattr(engine, "_load_models"):
            engine._load_models()
            log.info("✅ Modeller yeniden yüklendi (_load_models).")
        elif hasattr(engine, "load_models"):
            engine.load_models()
            log.info("✅ Modeller yeniden yüklendi (load_models).")
        else:
            return jsonify({"status": "error", "message": "Model yükleyici bulunamadı"}), 500
        return jsonify({"status": "ok", "message": "Modeller başarıyla yeniden yüklendi"})
    except Exception as e:
        log.error(f"/api/reload hata: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

# ==============================================================
# MODEL EĞİTİMİ BAŞLAT
# ==============================================================
@app.route("/api/training/start", methods=["POST"])
def start_training():
    try:
        from model_trainer import train_with_progress
        log.info("🎯 Eğitim isteği alındı...")
        success = train_with_progress(auto_confirm=True)
        if success:
            log.info("✅ Eğitim tamamlandı")
            return jsonify({"status": "ok", "message": "Model eğitimi tamamlandı"})
        else:
            log.warning("⚠️ Eğitim başarısız veya iptal edildi")
            return jsonify({"status": "error", "message": "Eğitim başarısız veya iptal edildi"}), 500
    except Exception as e:
        log.error(f"❌ Eğitim hatası: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

# ==============================================================
# MAÇ TAHMİNİ ENDPOINT
# ==============================================================
@app.route("/api/predict", methods=["POST"])
def predict_match():
    try:
        data = request.get_json(force=True)
        if engine is None:
            return jsonify({"status": "error", "message": "Model motoru yüklenemedi"}), 500
        result = engine.predict_match(
            data.get("home_team"),
            data.get("away_team"),
            data.get("odds", {})
        )
        return jsonify({"status": "ok", "prediction": result})
    except Exception as e:
        log.error(f"/api/predict hata: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

# ==============================================================
# SUNUCU BAŞLAT
# ==============================================================
if __name__ == "__main__":
    log.info("🚀 Predicta Europe ML API başlatılıyor...")
    app.run(host="0.0.0.0", port=8080)
