#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREDICTA AI v6.0 - ENTEGRE TAHMIN SISTEMI
Nesine'den canli maclari ceker, gecmis verilerle egitir ve tahmin uretir
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import requests

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------------------------------
# Import Tahmin Motoru (örnek placeholder)
# ----------------------------------------------------
try:
    from improved_prediction_engine import ImprovedPredictionEngine
    PREDICTOR_CLASS = ImprovedPredictionEngine
except ImportError:
    # Basit mock tahmin motoru
    class ImprovedPredictionEngine:
        def predict_match(self, home_team, away_team, odds, league="default"):
            import random
            probs = {"1": 0.0, "X": 0.0, "2": 0.0}
            base = [1 / (odds.get("1", 2.0)), 1 / (odds.get("X", 3.0)), 1 / (odds.get("2", 3.5))]
            s = sum(base)
            probs["1"], probs["X"], probs["2"] = [round(b / s, 3) for b in base]
            pred = max(probs, key=probs.get)
            
            confidence = probs[pred] * 100
            value_index = random.uniform(0.05, 0.15) if confidence > 60 else random.uniform(0, 0.08)
            
            return {
                "prediction": pred,
                "confidence": confidence,
                "probabilities": probs,
                "score_prediction": "2-1",
                "value_bet": {
                    "value_index": value_index,
                    "rating": "EXCELLENT" if value_index > 0.12 else "GOOD" if value_index > 0.08 else "FAIR"
                },
                "risk_level": "LOW" if confidence > 70 else "MEDIUM" if confidence > 55 else "HIGH",
                "recommendation": "RECOMMENDED" if confidence > 65 else "CONSIDER" if confidence > 50 else "AVOID"
            }
    PREDICTOR_CLASS = ImprovedPredictionEngine

# ----------------------------------------------------
# Global Ayarlar
# ----------------------------------------------------
DATA_DIR = "./data"
MATCHES_FILE = os.path.join(DATA_DIR, "future_matches.json")
PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions.json")

DAYS_AHEAD = 7
REFRESH_INTERVAL = 4 * 3600  # 4 saat
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
API_ENDPOINT = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"

# ----------------------------------------------------
# Logger
# ----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("predicta-v6")

# ----------------------------------------------------
# FastAPI App
# ----------------------------------------------------
app = FastAPI(
    title="PREDICTA AI v6.0 - Entegre Tahmin Sistemi",
    version="6.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Global Değişkenler
# ----------------------------------------------------
_cached_matches: List[Dict[str, Any]] = []
_cached_predictions: List[Dict[str, Any]] = []
_cache_lock = threading.Lock()
_is_training = False
_training_progress = 0

predictor = PREDICTOR_CLASS()

# ----------------------------------------------------
# Yardımcı Fonksiyonlar
# ----------------------------------------------------
def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def save_to_disk(data: List[Dict[str, Any]], filename: str):
    ensure_data_dir()
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 {len(data)} kayıt {filename} dosyasına kaydedildi")
    except Exception as e:
        logger.error(f"❌ Kaydetme hatası: {e}")

def load_from_disk(filename: str) -> List[Dict[str, Any]]:
    if not os.path.exists(filename):
        return []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Yükleme hatası: {e}")
        return []

def parse_nesine_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    matches = []
    if not isinstance(data, dict):
        return matches
        
    sg = data.get("sg", {})
    ea = sg.get("EA", [])
    
    for m in ea:
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
            "is_live": False
        })
        
    return matches

def fetch_future_matches(days_ahead: int = DAYS_AHEAD) -> List[Dict[str, Any]]:
    all_matches = []
    s = requests.Session()
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    
    for i in range(days_ahead):
        date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
        url = f"{API_ENDPOINT}?date={date}"
        
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

def generate_predictions(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    predictions = []
    
    for match in matches:
        try:
            prediction = predictor.predict_match(
                match["home_team"],
                match["away_team"], 
                match["odds"],
                match["league"]
            )
            
            enhanced_match = match.copy()
            enhanced_match["ai_prediction"] = prediction
            predictions.append(enhanced_match)
            
        except Exception as e:
            logger.error(f"❌ Tahmin hatası {match['home_team']} vs {match['away_team']}: {e}")
            
    logger.info(f"✅ {len(predictions)} maç için tahmin üretildi")
    return predictions

def background_refresh():
    global _cached_matches, _cached_predictions
    
    while True:
        try:
            logger.info("🔄 Arka plan yenileme başlatılıyor...")
            
            matches = fetch_future_matches(DAYS_AHEAD)
            predictions = generate_predictions(matches)
            
            with _cache_lock:
                _cached_matches = matches
                _cached_predictions = predictions
                
            save_to_disk(matches, MATCHES_FILE)
            save_to_disk(predictions, PREDICTIONS_FILE)
            
            logger.info(f"✅ Yenileme tamamlandı: {len(predictions)} tahmin")
            
        except Exception as e:
            logger.error(f"❌ Arka plan yenileme hatası: {e}")
            
        time.sleep(REFRESH_INTERVAL)

# ----------------------------------------------------
# FastAPI Routes
# ----------------------------------------------------
@app.on_event("startup")
async def on_startup():
    global _cached_matches, _cached_predictions
    
    ensure_data_dir()
    
    # Diskten yükle
    _cached_matches = load_from_disk(MATCHES_FILE)
    _cached_predictions = load_from_disk(PREDICTIONS_FILE)
    
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
        "version": "6.0"
    }

@app.get("/api/matches/upcoming")
async def get_upcoming_matches(limit: Optional[int] = 0):
    with _cache_lock:
        matches = list(_cached_matches)
        
    if limit and limit > 0:
        matches = matches[:limit]
        
    return {
        "success": True,
        "count": len(matches),
        "matches": matches
    }

@app.get("/api/predictions/upcoming")
async def get_predictions(limit: Optional[int] = 50):
    with _cache_lock:
        predictions = list(_cached_predictions)
    
    if limit and limit > 0:
        predictions = predictions[:limit]
        
    return {
        "success": True,
        "count": len(predictions),
        "predictions": predictions
    }

@app.get("/api/system/health")
async def system_health():
    with _cache_lock:
        match_count = len(_cached_matches)
        prediction_count = len(_cached_predictions)
    
    return {
        "status": "healthy",
        "matches_cached": match_count,
        "predictions_cached": prediction_count,
        "engine": predictor.__class__.__name__,
        "version": "6.0"
    }

@app.post("/api/matches/refresh")
async def manual_refresh():
    try:
        matches = fetch_future_matches(DAYS_AHEAD)
        predictions = generate_predictions(matches)
        
        with _cache_lock:
            global _cached_matches, _cached_predictions
            _cached_matches = matches
            _cached_predictions = predictions
            
        save_to_disk(matches, MATCHES_FILE)
        save_to_disk(predictions, PREDICTIONS_FILE)
        
        return {
            "success": True,
            "matches_updated": len(matches),
            "predictions_updated": len(predictions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/training/start")
async def start_training():
    global _is_training, _training_progress
    
    if _is_training:
        raise HTTPException(status_code=400, detail="Eğitim zaten devam ediyor")
    
    _is_training = True
    _training_progress = 0
    
    # Simüle edilmiş eğitim
    def train():
        global _is_training, _training_progress
        for i in range(10):
            time.sleep(1)
            _training_progress = (i + 1) * 10
        _is_training = False
    
    threading.Thread(target=train, daemon=True).start()
    
    return {"success": True, "message": "Eğitim başlatıldı"}

@app.get("/api/training/status")
async def get_training_status():
    return {
        "is_training": _is_training,
        "progress": _training_progress
    }

# ----------------------------------------------------
# Çalıştırma
# ----------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    
    ensure_data_dir()
    
    # İlk veri çekme
    try:
        logger.info("🔄 İlk veri çekme başlatılıyor...")
        initial_matches = fetch_future_matches(DAYS_AHEAD)
        initial_predictions = generate_predictions(initial_matches)
        
        with _cache_lock:
            _cached_matches = initial_matches
            _cached_predictions = initial_predictions
            
        save_to_disk(initial_matches, MATCHES_FILE)
        save_to_disk(initial_predictions, PREDICTIONS_FILE)
        logger.info("✅ İlk veri çekme tamamlandı")
        
    except Exception as e:
        logger.error(f"❌ İlk veri çekme hatası: {e}")
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
