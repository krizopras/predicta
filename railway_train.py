# railway_train.py
"""
Railway'de ML Eğitimi için Özel Script
- Bellek optimizasyonu
- Progress tracking
- Hata yönetimi
- Model eğitimini otomatik başlatır
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# ========== Logging ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("railway_training.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


# ========== Ortam Kontrolü ==========
def check_environment():
    """Railway ortamını kontrol et"""
    logger.info("=" * 70)
    logger.info("🚂 RAILWAY ENVIRONMENT CHECK")
    logger.info("=" * 70)

    is_railway = os.getenv("RAILWAY_ENVIRONMENT") is not None
    logger.info(f"Railway Environment: {is_railway}")

    if is_railway:
        logger.info(f"Railway Project: {os.getenv('RAILWAY_PROJECT_NAME', 'N/A')}")
        logger.info(f"Railway Service: {os.getenv('RAILWAY_SERVICE_NAME', 'N/A')}")
        logger.info(f"Railway Environment ID: {os.getenv('RAILWAY_ENVIRONMENT', 'N/A')}")

    # Bellek durumu
    try:
        import psutil
        mem = psutil.virtual_memory()
        logger.info(f"Total Memory: {mem.total / (1024 ** 3):.2f} GB")
        logger.info(f"Available Memory: {mem.available / (1024 ** 3):.2f} GB")
        logger.info(f"Memory Usage: {mem.percent}%")
    except ImportError:
        logger.warning("⚠️ psutil not installed, skipping memory check")

    # Disk ve klasör kontrolü
    models_dir = Path("data/ai_models_v3")
    raw_dir = Path("data/raw")

    logger.info(f"Models directory exists: {models_dir.exists()}")
    logger.info(f"Raw data directory exists: {raw_dir.exists()}")

    if raw_dir.exists():
        countries = [p.name for p in raw_dir.iterdir() if p.is_dir()]
        logger.info(f"Detected countries: {countries}")
    else:
        logger.error("❌ Raw data directory not found! Please upload your data/raw folder.")
        sys.exit(1)


# ========== Model Eğitimi ==========
def run_training():
    """Model eğitimini başlat"""
    from model_trainer import ProductionModelTrainer

    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("🚀 STARTING ML TRAINING ON RAILWAY")
    logger.info("=" * 70)

    try:
        trainer = ProductionModelTrainer(
            models_dir="data/ai_models_v2",
            raw_data_path="data/raw",
            clubs_path="data/clubs",
            min_matches=100,
            test_size=0.2,
            verbose=True
        )

        result = trainer.run_full_pipeline()

        if result.get("success", False):
            logger.info("✅ TRAINING SUCCESSFUL")
            logger.info(f"Total Clean Matches: {result.get('total_clean_matches')}")
        else:
            logger.warning(f"⚠️ Training returned error: {result.get('error')}")

    except Exception as e:
        logger.exception(f"❌ TRAINING FAILED: {e}")
    finally:
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"⏱️ Duration: {duration:.2f} seconds")
        logger.info("🚀 Training process finished")


# ========== Ana Akış ==========
if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("PREDICTA - RAILWAY TRAINING SCRIPT INITIATED")
    logger.info("=" * 70)

    check_environment()
    run_training()

    logger.info("=" * 70)
    logger.info("✅ RAILWAY TRAINING COMPLETED")
    logger.info("=" * 70)
