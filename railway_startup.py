#!/usr/bin/env python3
"""
railway_startup.py - Railway GitHub Deploy için Otomatik Kurulum
"""
import os
import sys
import subprocess
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


def auto_train_models():
    """Railway'de otomatik model eğitimi"""
    logger.info("=" * 70)
    logger.info("🎓 OTOMATİK MODEL EĞİTİMİ BAŞLATILIYOR")
    logger.info("=" * 70)
    
    try:
        # Railway için optimize edilmiş parametreler
        result = subprocess.run(
            [sys.executable, "model_trainer.py", 
             "--min-matches", "10",       # ✅ Düşük minimum
             "--max-matches", "50000",    # ✅ Railway bellek limiti
             "--batch-size", "500",       # ✅ Bellek optimizasyonu
             "--test-size", "0.2",
             "--seed", "42",
             "--verbose"],
            capture_output=True,
            text=True,
            timeout=1800  # 30 dakika max
        )
        
        # Çıktıları logla
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"[TRAIN] {line}")
        
        if result.stderr:
            for line in result.stderr.splitlines():
                logger.warning(f"[TRAIN-ERR] {line}")
        
        if result.returncode == 0:
            logger.info("✅ Model eğitimi başarılı!")
            return True
        else:
            logger.error(f"❌ Model eğitimi başarısız! (exit code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Eğitim timeout! (>30 dakika)")
        return False
    except Exception as e:
        logger.error(f"❌ Eğitim hatası: {e}")
        return False


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
    auto_train = os.getenv('AUTO_TRAIN', 'false').lower() == 'true'
    
    logger.info(f"Railway: {'✅' if is_railway else '❌'}")
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"Port: {os.environ.get('PORT', '8000')}")
    logger.info(f"Auto-Train: {'✅' if auto_train else '❌'}")
    
    # Model kontrolü
    has_models = check_models_exist()
    has_data = check_raw_data()
    
    logger.info(f"Modeller: {'✅ Mevcut' if has_models else '❌ Yok'}")
    logger.info(f"Ham Veri: {'✅ Mevcut' if has_data else '❌ Yok'}")
    
    # Otomatik eğitim karar mantığı
    should_train = False
    
    if not has_models:
        logger.warning("⚠️  MODELLER YOK!")
        
        if has_data and auto_train:
            logger.info("🎯 AUTO_TRAIN=true, veri mevcut → Otomatik eğitim başlıyor!")
            should_train = True
        elif has_data and not auto_train:
            logger.info("💡 Veri mevcut ama AUTO_TRAIN=false")
            logger.info("   Ortam değişkenine ekleyin: AUTO_TRAIN=true")
            logger.info("   Ya da frontend'den manuel eğitin:")
            logger.info("   1. 'Train' butonu → Model eğit (10-15 dk)")
        else:
            logger.info("💡 Frontend'den şu adımları takip edin:")
            logger.info("   1. 'History' butonu → Veri yükle")
            logger.info("   2. 'Train' butonu → Model eğit (10-15 dk)")
    else:
        logger.info("✅ Modeller mevcut, sistem hazır!")
    
    # Otomatik eğitim varsa çalıştır
    if should_train:
        success = auto_train_models()
        
        if not success:
            logger.warning("⚠️  Otomatik eğitim başarısız!")
            logger.info("💡 Flask yine de başlatılıyor (manuel eğitim yapabilirsiniz)")
        else:
            # Eğitim sonrası kontrol
            if check_models_exist():
                logger.info("✅ Modeller başarıyla oluşturuldu!")
            else:
                logger.error("❌ Eğitim tamamlandı ama modeller bulunamadı!")
    
    # Her durumda Flask'ı başlat
    logger.info("\n" + "=" * 70)
    start_flask()


if __name__ == "__main__":
    main()
