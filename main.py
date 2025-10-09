#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predicta Europe ML - Ana API (Railway uyumlu tam sürüm)
- Flask tabanlı API sunucusu
- Eğitim, tahmin ve model reload işlemleri
- Otomatik klasör kontrolü
"""

import os
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS

# ============================================================
# 🧩 Flask ayarları
# ============================================================
app = Flask(__name__)
CORS(app)

# Logging ayarları (Railway loglarında görünsün)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# 📁 Gerekli dizinleri oluştur
# ============================================================
REQUIRED_DIRS = ["data", "data/raw", "data/ai_models_v2", "data/clubs", "logs"]

for d in REQUIRED_DIRS:
    os.makedirs(d, exist_ok=True)
    logger.info(f"📂 Klasör kontrolü: {d} → OK")

# ============================================================
# 🧠 Model motorunu yükle
# ============================================================
try:
    from ml_prediction_engine import MLPredictionEngine
    engine = MLPredictionEngine()
    logger.info("✅ MLPredictionEngine başarıyla başlatıldı.")
except Exception as e:
    logger.error(f"❌ Tahmin motoru yüklenemedi: {e}")
    engine = None

# ============================================================
# 🧩 API: Test / Healthcheck
# ============================================================
@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "message": "Predicta Europe ML API aktif 🚀",
        "engine_loaded": engine is not None
    })

# ============================================================
# 🧩 API: Model reload
# ============================================================
@app.route("/api/reload", methods=["GET", "POST"])
def reload_models():
    """Mevcut modelleri yeniden yükler"""
    try:
        if engine is None:
            return jsonify({"status": "error", "message": "Engine yüklenemedi"}), 500

        if hasattr(engine, "_load_models"):
            engine._load_models()
            msg = "Yeni nesil model yükleyici (_load_models) çağrıldı"
        elif hasattr(engine, "load_models"):
            engine.load_models()
            msg = "Klasik load_models() çağrıldı"
        else:
            return jsonify({"status": "error", "message": "Model yükleyici bulunamadı"}), 500

        logger.info("✅ Modeller yeniden yüklendi.")
        return jsonify({"status": "ok", "message": msg})
    except Exception as e:
        logger.error(f"/api/reload error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

# ============================================================
# 🧩 API: Model Eğitimi
# ============================================================
@app.route("/api/training/start", methods=["POST"])
def start_training():
    """Gerçek model eğitimini başlat"""
    try:
        from model_trainer import train_with_progress, check_data_structure, check_dependencies

        logger.info("🧠 Eğitim isteği alındı...")

        # Klasörleri garantiye al
        for d in REQUIRED_DIRS:
            os.makedirs(d, exist_ok=True)

        # Bağımlılık ve veri yapısı kontrolü
        if not check_dependencies():
            logger.error("Eksik ML bağımlılıkları tespit edildi.")
            return jsonify({
                "status": "error",
                "message": "Eksik ML bağımlılıkları. Lütfen requirements.txt yükleyin."
            }), 500

        if not check_data_structure():
            logger.error("Veri yapısı hatalı (data/raw eksik veya boş).")
            return jsonify({
                "status": "error",
                "message": "Veri yapısı uygun değil. data/raw dizinini kontrol edin."
            }), 500

        # Eğitim süreci
        logger.info("🎯 MODEL EĞİTİMİ BAŞLATILIYOR...")
        success = train_with_progress(auto_confirm=True)

        if success:
            logger.info("✅ Eğitim başarıyla tamamlandı.")
            return jsonify({
                "status": "ok",
                "message": "Model eğitimi başarıyla tamamlandı.",
                "models_dir": "data/ai_models_v2"
            })
        else:
            logger.warning("⚠️ Eğitim başarısız veya iptal edildi.")
            return jsonify({
                "status": "error",
                "message": "Eğitim başarısız veya iptal edildi."
            }), 500

    except Exception as e:
        import traceback
        logger.error(f"❌ Eğitim başlatılamadı: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        }), 500

# ============================================================
# 🧩 API: Tahmin
# ============================================================
@app.route("/api/predict", methods=["POST"])
def predict_match():
    """Tek maç tahmini üret"""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "JSON body boş"}), 400

        if engine is None:
            return jsonify({"status": "error", "message": "Model motoru yüklenemedi"}), 500

        prediction = engine.predict_match(
            data.get("home_team"),
            data.get("away_team"),
            data.get("odds", {})
        )

        return jsonify({
            "status": "ok",
            "prediction": prediction
        })
    except Exception as e:
        logger.error(f"/api/predict hata: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

# ============================================================
# 🚀 Ana Çalıştırıcı
# ============================================================
if __name__ == "__main__":
    logger.info("🚀 Predicta Europe ML API başlatılıyor...")
    app.run(host="0.0.0.0", port=8080)
