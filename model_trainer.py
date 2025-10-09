#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predicta Europe ML - Detaylı Model Eğitici (Railway uyumlu)
Tam otomatik model eğitim sistemi:
 - Bağımlılık ve veri yapısı kontrolü
 - MS (1X2) ve Skor modellerinin eğitimi
 - Eğitim sonuçlarının detaylı loglanması
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime

# ==========================================================
# 🧱 LOG AYARLARI
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [TRAIN] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# ==========================================================
# 📁 KLASÖR OLUŞTURUCU
# ==========================================================
def ensure_directories():
    """Tüm gerekli klasörleri oluşturur"""
    required_dirs = [
        "data",
        "data/raw",
        "data/ai_models_v2",
        "data/clubs",
        "logs"
    ]
    for d in required_dirs:
        os.makedirs(d, exist_ok=True)
        log.info(f"📂 Klasör kontrolü: {d} → OK")

# ==========================================================
# 🔍 BAĞIMLILIK KONTROLÜ
# ==========================================================
def check_dependencies():
    log.info("📦 Gerekli Python kütüphaneleri kontrol ediliyor...")
    missing = []
    for lib in ["numpy", "pandas", "sklearn", "xgboost"]:
        try:
            __import__(lib)
            log.info(f"✅ {lib}")
        except ImportError:
            missing.append(lib)
            log.warning(f"❌ {lib} eksik!")
    if missing:
        log.error(f"⚠️ Eksik kütüphaneler: {', '.join(missing)}")
        log.info(f"Kurulum: pip install {' '.join(missing)}")
        return False
    return True

# ==========================================================
# 📊 VERİ KONTROLÜ
# ==========================================================
def check_data_structure():
    """Veri yapısını kontrol eder"""
    log.info("📁 Veri yapısı kontrol ediliyor...")
    if not os.path.exists("data/raw"):
        log.error("❌ data/raw klasörü bulunamadı!")
        return False

    countries = [d for d in os.listdir("data/raw") if os.path.isdir(os.path.join("data/raw", d))]
    if not countries:
        log.error("⚠️ data/raw klasöründe ülke klasörü yok!")
        return False

    total_files = 0
    for c in countries:
        files = [f for f in os.listdir(os.path.join("data/raw", c)) if f.endswith(('.csv', '.txt'))]
        log.info(f"   • {c}: {len(files)} dosya")
        total_files += len(files)

    if total_files == 0:
        log.warning("⚠️ Veri dosyası bulunamadı")
        return False

    log.info(f"✅ Toplam {total_files} veri dosyası bulundu ({len(countries)} ülke)")
    return True

# ==========================================================
# 🧠 TÜM MAÇLARI YÜKLE
# ==========================================================
def load_all_matches(raw_path="data/raw"):
    matches = []
    for country in os.listdir(raw_path):
        cpath = os.path.join(raw_path, country)
        if not os.path.isdir(cpath):
            continue
        for file in os.listdir(cpath):
            if not file.endswith(('.csv', '.txt')):
                continue
            fpath = os.path.join(cpath, file)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()[1:]
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) < 6:
                        continue
                    try:
                        home, away, result, hg, ag, league = parts[:6]
                        matches.append({
                            "home_team": home.strip(),
                            "away_team": away.strip(),
                            "result": result.strip(),
                            "home_goals": int(hg or 0),
                            "away_goals": int(ag or 0),
                            "league": league.strip()
                        })
                    except Exception:
                        continue
    return matches

# ==========================================================
# 🎯 MODEL EĞİTİMİ
# ==========================================================
def train_all(raw_path="data/raw", clubs_path="data/clubs", top_scores_k=24):
    from ml_prediction_engine import MLPredictionEngine
    from score_predictor import ScorePredictor

    ensure_directories()
    matches = load_all_matches(raw_path)
    log.info(f"📊 {len(matches)} geçmiş maç yüklendi")

    if len(matches) < 100:
        raise ValueError(f"Yetersiz veri ({len(matches)}). Minimum 100 maç gerekli.")

    log.info("🎓 MS (1-X-2) modeli eğitiliyor...")
    ml_engine = MLPredictionEngine()
    ms_result = ml_engine.train(matches)
    log.info("✅ MS modeli tamamlandı")

    log.info("⚽ Skor tahmin modeli eğitiliyor...")
    score_predictor = ScorePredictor("data/ai_models_v2")
    score_result = score_predictor.train(matches, top_k=top_scores_k)
    log.info("✅ Skor modeli tamamlandı")

    summary = {
        "success": True,
        "samples_total": len(matches),
        "timestamp": datetime.now().isoformat(),
        "ms": ms_result,
        "score": score_result
    }

    with open("data/ai_models_v2/training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info("💾 Sonuçlar kaydedildi: data/ai_models_v2/training_summary.json")
    return summary

# ==========================================================
# 🚀 ANA ÇALIŞTIRICI
# ==========================================================
def train_with_progress(auto_confirm=False):
    try:
        ensure_directories()
        if not check_dependencies():
            raise RuntimeError("Eksik bağımlılıklar tespit edildi.")
        if not check_data_structure():
            raise RuntimeError("Veri yapısı eksik veya hatalı.")
        result = train_all()
        log.info("✅ Eğitim başarıyla tamamlandı!")
        return result.get("success", False)
    except Exception as e:
        log.error(f"❌ Eğitim hatası: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    train_with_progress(auto_confirm=True)
