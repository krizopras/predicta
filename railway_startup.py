#!/usr/bin/env python3
"""
railway_startup.py - Railway GitHub Deploy için Otomatik Kurulum
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("RailwayStartup")

def check_models_exist():
    """Eğitilmiş modeller var mı?"""
    models_dir = Path("data/ai_models_v2")
    required = ["ensemble_models.pkl", "score_model.pkl", "scaler.pkl"]
    
    if not models_dir.exists():
        return False
    
    for file in required:
        path = models_dir / file
        if not path.exists() or path.stat().st_size < 1000:
            return False
    
    return True

def check_raw_data():
    """Ham veri var mı?"""
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        return False
    
    # Ülke klasörleri var mı?
    countries = [d for d in raw_dir.iterdir() if d.is_dir() and d.name != "clubs"]
    return len(countries) > 0

def start_flask():
    """Flask'ı başlat"""
    logger.info("=" * 70)
    logger.info("🚀 FLASK BAŞLATILIYOR")
    logger.info("=" * 70)
    
    port = int(os.environ.get("PORT", 8000))
    
    try:
        from main import app
        app.run(host="0.0.0.0", port=port, debug=False)
    except Exception as e:
        logger.error(f"❌ Flask hatası: {e}")
        sys.exit(1)

def main():
    logger.info("=" * 70)
    logger.info("🚂 RAILWAY STARTUP - PREDICTA ML")
    logger.info("=" * 70)
    
    # Ortam kontrolü
    is_railway = os.getenv('RAILWAY_ENVIRONMENT') is not None
    logger.info(f"Railway: {'✅' if is_railway else '❌'}")
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"Port: {os.environ.get('PORT', '8000')}")
    
    # Model kontrolü
    has_models = check_models_exist()
    has_data = check_raw_data()
    
    logger.info(f"Modeller: {'✅ Mevcut' if has_models else '❌ Yok'}")
    logger.info(f"Ham Veri: {'✅ Mevcut' if has_data else '❌ Yok'}")
    
    if not has_models:
        logger.warning("⚠️  MODELLER YOK!")
        logger.info("💡 Frontend'den şu adımları takip edin:")
        logger.info("   1. 'History' butonu → Veri yükle")
        logger.info("   2. 'Train' butonu → Model eğit (10-15 dk)")
        logger.info("   3. Hazır!")
    else:
        logger.info("✅ Modeller mevcut, sistem hazır!")
    
    # Flask'ı başlat
    start_flask()

if __name__ == "__main__":
    main()
