#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREDICTA AI v6.0 - ENTEgre TAHMİN SİSTEMİ
Nesine'den canlı maçları çeker, geçmiş verilerle eğitir ve tahmin üretir
"""

import os
import json
import time
import logging
import threading
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import requests
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ----------------------------------------------------
# Import Gelişmiş Modüller
# ----------------------------------------------------
try:
    from improved_prediction_engine import ProfessionalPredictionEngine
    from advanced_feature_engineer import AdvancedFeatureEngineer
    from txt_data_processor import TXTDataProcessor
    from enhanced_predictor import EnhancedPredictor
    from database_manager import AIDatabaseManager
except ImportError as e:
    logging.warning(f"Bazı modüller yüklenemedi: {e}")
    
    # Fallback basit motor
    class ProfessionalPredictionEngine:
        def predict_match(self, home_team, away_team, odds, league="default"):
            return {
                "prediction": "1",
                "confidence": 65.5,
                "probabilities": {"home_win": 65.5, "draw": 25.0, "away_win": 9.5},
                "score_prediction": "2-1",
                "value_bet": {"value_index": 0.12, "rating": "GOOD"},
                "risk_level": "LOW",
                "recommendation": "✅ RECOMMENDED"
            }

# ----------------------------------------------------
# Global Ayarlar
# ----------------------------------------------------
DATA_DIR = "./data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(DATA_DIR, "ai_models")
MATCHES_FILE = os.path.join(DATA_DIR, "future_matches.json")

DAYS_AHEAD = 7
REFRESH_INTERVAL = 4 * 3600  # 4 saat
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# ----------------------------------------------------
# Logger
# ----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("predicta_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("predicta-v6")

# ----------------------------------------------------
# FastAPI App
# ----------------------------------------------------
app = FastAPI(
    title="PREDICTA AI v6.0 - Entegre Tahmin Sistemi",
    description="Nesine'den canlı maçları çeker, geçmiş verilerle eğitir ve AI tahminleri üretir",
    version="6.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files için
app.mount("/static", StaticFiles(directory="static"), name="static")

# ----------------------------------------------------
# Global Değişkenler
# ----------------------------------------------------
_cached_matches: List[Dict[str, Any]] = []
_cached_predictions: List[Dict[str, Any]] = []
_cache_lock = threading.Lock()
_is_training = False
_training_progress = 0
_last_training_time = None

# Tahmin motorları
try:
    predictor = EnhancedPredictor()
    logger.info("✅ EnhancedPredictor başarıyla yüklendi")
except:
    predictor = ProfessionalPredictionEngine()
    logger.info("✅ ProfessionalPredictionEngine (fallback) yüklendi")

db_manager = AIDatabaseManager()

# ----------------------------------------------------
# Yardımcı Fonksiyonlar
# ----------------------------------------------------
def ensure_directories():
    """Gerekli dizinleri oluştur"""
    for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
        os.makedirs(directory, exist_ok=True)

def save_matches_to_disk(matches: List[Dict[str, Any]]):
    """Maçları diske kaydet"""
    ensure_directories()
    try:
        with open(MATCHES_FILE, "w", encoding="utf-8") as f:
            json.dump(matches, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 {len(matches)} maç diske kaydedildi")
    except Exception as e:
        logger.error(f"❌ Maç kaydetme hatası: {e}")

def load_matches_from_disk() -> List[Dict[str, Any]]:
    """Diskten maçları yükle"""
    if not os.path.exists(MATCHES_FILE):
        return []
    try:
        with open(MATCHES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Maç yükleme hatası: {e}")
        return []

def parse_nesine_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Nesine JSON verisini parse et"""
    matches = []
    if not isinstance(data, dict):
        return matches
        
    sg = data.get("sg", {})
    ea = sg.get("EA", [])
    ca = sg.get("CA", [])
    
    for m in (ea + ca):
        if m.get("GT") != 1:  # Sadece futbol
            continue
            
        home = m.get("HN", "").strip()
        away = m.get("AN", "").strip()
        if not home or not away:
            continue
            
        # Oranları bul
        odds = {"1": 2.0, "X": 3.0, "2": 3.5}
        for bahis in m.get("MA", []):
            if bahis.get("MTID") == 1:  # Maç Sonucu
                oranlar = bahis.get("OCA", [])
                if len(oranlar) >= 3:
                    try:
                        odds["1"] = float(oranlar[0].get("O", 2.0))
                        odds["X"] = float(oranlar[1].get("O", 3.0))
                        odds["2"] = float(oranlar[2].get("O", 3.5))
                    except:
                        pass
                break
                
        matches.append({
            "home_team": home,
            "away_team": away,
            "league": m.get("LC", "Bilinmeyen"),
            "match_id": m.get("C", ""),
            "date": m.get("D", datetime.now().strftime("%Y-%m-%d")),
            "time": m.get("T", "20:00"),
            "odds": odds,
            "is_live": m.get("S") == 1
        })
        
    return matches

def fetch_future_matches(days_ahead: int = DAYS_AHEAD) -> List[Dict[str, Any]]:
    """Nesine'den gelecek maçları çek"""
    all_matches = []
    s = requests.Session()
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    
    for i in range(days_ahead):
        date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
        url = f"https://cdnbulten.nesine.com/api/bulten/getprebultenfull?date={date}"
        
        try:
            r = s.get(url, headers=headers, timeout=20)
            if r.status_code == 200:
                data = r.json()
                parsed = parse_nesine_json(data)
                all_matches.extend(parsed)
                logger.info(f"📅 {date}: {len(parsed)} maç")
            else:
                logger.warning(f"❌ {url} -> {r.status_code}")
        except Exception as e:
            logger.warning(f"❌ {url} bağlantı hatası: {e}")
            continue
            
    # Duplicate filtreleme
    seen = set()
    unique_matches = []
    for m in all_matches:
        key = f"{m['home_team']}|{m['away_team']}|{m['date']}"
        if key not in seen:
            seen.add(key)
            unique_matches.append(m)
            
    unique_matches.sort(key=lambda x: (x["date"], x["time"]))
    logger.info(f"✅ Toplam {len(unique_matches)} oynanmamış maç bulundu")
    return unique_matches

async def process_historical_data():
    """Geçmiş verileri işle ve modele yükle"""
    global _is_training, _training_progress, _last_training_time
    
    if _is_training:
        logger.info("⏳ Eğitim zaten devam ediyor...")
        return
        
    _is_training = True
    _training_progress = 0
    
    try:
        logger.info("🔄 Geçmiş veriler işleniyor...")
        _training_progress = 10
        
        # TXT verilerini işle
        processor = TXTDataProcessor(RAW_DATA_DIR)
        _training_progress = 30
        
        result = await processor.process_all_countries()
        _training_progress = 70
        
        logger.info(f"✅ {len(result['matches'])} geçmiş maç işlendi")
        logger.info(f"✅ {len(result['team_stats'])} takım istatistiği hesaplandı")
        
        # Enhanced predictor'ı güncelle
        if hasattr(predictor, '_load_historical_data'):
            predictor._load_historical_data()
            
        _training_progress = 100
        _last_training_time = datetime.now()
        
        logger.info("🎯 Model eğitimi tamamlandı!")
        
    except Exception as e:
        logger.error(f"❌ Eğitim hatası: {e}")
    finally:
        _is_training = False

def generate_predictions(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Maçlar için tahmin üret"""
    predictions = []
    
    for i, match in enumerate(matches):
        try:
            # Gelişmiş tahmin motoru kullan
            prediction = predictor.predict_match(
                match["home_team"],
                match["away_team"], 
                match["odds"],
                match["league"]
            )
            
            # Maç bilgileriyle birleştir
            enhanced_match = match.copy()
            enhanced_match["ai_prediction"] = prediction
            enhanced_match["prediction_id"] = f"pred_{i}_{int(time.time())}"
            
            predictions.append(enhanced_match)
            
            # Veritabanına kaydet
            db_manager.save_match_prediction(enhanced_match)
            
        except Exception as e:
            logger.error(f"❌ Tahmin hatası {match['home_team']} vs {match['away_team']}: {e}")
            
    logger.info(f"✅ {len(predictions)} maç için tahmin üretildi")
    return predictions

def background_refresh():
    """Arka planda veri yenileme"""
    global _cached_matches, _cached_predictions
    
    while True:
        try:
            logger.info("🔄 Arka plan yenileme başlatılıyor...")
            
            # 1. Maçları çek
            matches = fetch_future_matches(DAYS_AHEAD)
            
            # 2. Tahmin üret
            predictions = generate_predictions(matches)
            
            # 3. Cache'e kaydet
            with _cache_lock:
                _cached_matches = matches
                _cached_predictions = predictions
                
            # 4. Diske kaydet
            save_matches_to_disk(matches)
            
            logger.info(f"✅ Yenileme tamamlandı: {len(predictions)} tahmin")
            
        except Exception as e:
            logger.error(f"❌ Arka plan yenileme hatası: {e}")
            
        time.sleep(REFRESH_INTERVAL)

# ----------------------------------------------------
# FastAPI Routes
# ----------------------------------------------------
@app.on_event("startup")
async def on_startup():
    """Uygulama başlangıcında çalışacak kod"""
    global _cached_matches, _cached_predictions
    
    ensure_directories()
    
    # Diskten maçları yükle
    _cached_matches = load_matches_from_disk()
    
    # Geçmiş verileri işle (async)
    asyncio.create_task(process_historical_data())
    
    # Arka plan thread'i başlat
    t = threading.Thread(target=background_refresh, daemon=True)
    t.start()
    
    logger.info("🚀 PREDICTA AI v6.0 başlatıldı!")

@app.get("/")
async def root():
    return {
        "service": "PREDICTA AI v6.0 - Entegre Tahmin Sistemi",
        "status": "active",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "matches": "/api/matches/upcoming",
            "predictions": "/api/predictions/upcoming", 
            "training": "/api/training/status",
            "value_bets": "/api/predictions/value-bets"
        }
    }

@app.get("/api/matches/upcoming")
async def get_upcoming_matches(limit: Optional[int] = 0, league: Optional[str] = None):
    """Yaklaşan maçları getir"""
    with _cache_lock:
        matches = list(_cached_matches)
        
    if league and league != "all":
        matches = [m for m in matches if league.lower() in m["league"].lower()]
        
    if limit and limit > 0:
        matches = matches[:limit]
        
    return {
        "success": True,
        "count": len(matches),
        "matches": matches,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/predictions/upcoming")
async def get_predictions(
    limit: Optional[int] = 50,
    league: Optional[str] = None,
    min_confidence: Optional[float] = 0
):
    """AI tahminlerini getir"""
    with _cache_lock:
        predictions = list(_cached_predictions)
    
    # Filtreleme
    if league and league != "all":
        predictions = [p for p in predictions if league.lower() in p["league"].lower()]
        
    if min_confidence > 0:
        predictions = [p for p in predictions if p["ai_prediction"]["confidence"] >= min_confidence]
    
    if limit and limit > 0:
        predictions = predictions[:limit]
        
    return {
        "success": True,
        "count": len(predictions),
        "predictions": predictions,
        "filters": {
            "league": league,
            "min_confidence": min_confidence,
            "limit": limit
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/predictions/value-bets")
async def get_value_bets(
    min_value: float = 0.1,
    min_confidence: float = 60,
    limit: int = 20
):
    """Değerli bahisleri getir"""
    with _cache_lock:
        predictions = list(_cached_predictions)
    
    value_bets = []
    for pred in predictions:
        ai_pred = pred["ai_prediction"]
        value_data = ai_pred.get("value_bet", {})
        
        if (value_data.get("value_index", 0) >= min_value and 
            ai_pred.get("confidence", 0) >= min_confidence):
            value_bets.append(pred)
    
    if limit and limit > 0:
        value_bets = value_bets[:limit]
        
    return {
        "success": True,
        "count": len(value_bets),
        "value_bets": value_bets,
        "criteria": {
            "min_value_index": min_value,
            "min_confidence": min_confidence
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/training/start")
async def start_training(background_tasks: BackgroundTasks):
    """Model eğitimini başlat"""
    if _is_training:
        raise HTTPException(status_code=400, detail="Eğitim zaten devam ediyor")
    
    background_tasks.add_task(process_historical_data)
    
    return {
        "success": True,
        "message": "Eğitim başlatıldı",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/training/status")
async def get_training_status():
    """Eğitim durumunu getir"""
    return {
        "is_training": _is_training,
        "progress": _training_progress,
        "last_training_time": _last_training_time,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/system/health")
async def system_health():
    """Sistem sağlık durumu"""
    with _cache_lock:
        match_count = len(_cached_matches)
        prediction_count = len(_cached_predictions)
    
    return {
        "status": "healthy",
        "matches_cached": match_count,
        "predictions_cached": prediction_count,
        "last_refresh": datetime.now().isoformat(),
        "version": "6.0",
        "engine": predictor.__class__.__name__
    }

@app.post("/api/matches/refresh")
async def manual_refresh():
    """Manuel yenileme"""
    try:
        matches = fetch_future_matches(DAYS_AHEAD)
        predictions = generate_predictions(matches)
        
        with _cache_lock:
            global _cached_matches, _cached_predictions
            _cached_matches = matches
            _cached_predictions = predictions
            
        save_matches_to_disk(matches)
        
        return {
            "success": True,
            "matches_updated": len(matches),
            "predictions_updated": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------------------------
# Çalıştırma
# ----------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    
    ensure_directories()
    
    # İlk veri çekme
    try:
        logger.info("🔄 İlk veri çekme başlatılıyor...")
        initial_matches = fetch_future_matches(DAYS_AHEAD)
        initial_predictions = generate_predictions(initial_matches)
        
        with _cache_lock:
            _cached_matches = initial_matches
            _cached_predictions = initial_predictions
            
        save_matches_to_disk(initial_matches)
        logger.info("✅ İlk veri çekme tamamlandı")
        
    except Exception as e:
        logger.error(f"❌ İlk veri çekme hatası: {e}")
    
    # Sunucuyu başlat
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
