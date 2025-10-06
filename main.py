#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREDICTA AI v6.1 - ML ENTEGRE TAHMIN SISTEMI
Nesine'den canli maclari ceker, gecmis verilerle ML modelini egitir ve tahmin uretir
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
# ML Tahmin Motoru Import
# ----------------------------------------------------
try:
    from ml_prediction_engine import MLPredictionEngine
    PREDICTOR_CLASS = MLPredictionEngine
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    # Fallback: Basit tahmin motoru
    try:
        from improved_prediction_engine import ImprovedPredictionEngine
        PREDICTOR_CLASS = ImprovedPredictionEngine
    except ImportError:
        class BasicPredictor:
            def predict_match(self, home_team, away_team, odds, league="default"):
                import random
                pred = random.choice(['1', 'X', '2'])
                return {
                    "prediction": pred,
                    "confidence": random.uniform(50, 80),
                    "probabilities": {"home_win": 33.3, "draw": 33.3, "away_win": 33.3},
                    "model": "Basic Random"
                }
        PREDICTOR_CLASS = BasicPredictor

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
logger = logging.getLogger("predicta-ml")

# ----------------------------------------------------
# FastAPI App
# ----------------------------------------------------
app = FastAPI(
    title="PREDICTA AI v6.1 - ML Entegre Tahmin Sistemi",
    version="6.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Global Degiskenler
# ----------------------------------------------------
_cached_matches: List[Dict[str, Any]] = []
_cached_predictions: List[Dict[str, Any]] = []
_cache_lock = threading.Lock()
_is_training = False
_training_progress = 0
_model_accuracy = 0.0

predictor = None

# ----------------------------------------------------
# Yardimci Fonksiyonlar
# ----------------------------------------------------
def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "ai_models_v2"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)

def save_to_disk(data: List[Dict[str, Any]], filename: str):
    ensure_data_dir()
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Kaydedildi: {len(data)} kayit -> {filename}")
    except Exception as e:
        logger.error(f"Kaydetme hatasi: {e}")

def load_from_disk(filename: str) -> List[Dict[str, Any]]:
    if not os.path.exists(filename):
        return []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Yukleme hatasi: {e}")
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
            
        # Oranlari bul
        odds = {"1": 2.0, "X": 3.0, "2": 3.5}
        for bahis in m.get("MA", []):
            if bahis.get("MTID") == 1:  # Mac Sonucu
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
                logger.info(f"{date}: {len(parsed)} mac")
            else:
                logger.warning(f"{url} -> {r.status_code}")
        except Exception as e:
            logger.warning(f"{url} baglanti hatasi: {e}")
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
    logger.info(f"Toplam {len(unique_matches)} oynanmamis mac bulundu")
    return unique_matches

def generate_predictions(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    global predictor
    
    if predictor is None:
        logger.warning("Tahmin motoru hazir degil, baslatiliyor...")
        predictor = PREDICTOR_CLASS()
    
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
            logger.error(f"Tahmin hatasi {match['home_team']} vs {match['away_team']}: {e}")
            
    logger.info(f"{len(predictions)} mac icin tahmin uretildi")
    return predictions

# ----------------------------------------------------
# Gecmis Veri Yukleme ve Model Egitimi
# ----------------------------------------------------
def load_historical_data_and_train():
    """Gecmis verileri yukle ve modeli egit"""
    global predictor, _is_training, _training_progress, _model_accuracy
    
    if not ML_AVAILABLE:
        logger.warning("ML kutuphaneleri yuklu degil, temel tahmin kullanilacak")
        logger.info("Kurulum: pip install scikit-learn xgboost")
        predictor = PREDICTOR_CLASS()
        return
    
    try:
        _is_training = True
        _training_progress = 10
        
        logger.info("Gecmis veriler yukleniyor...")
        
        # TXTDataProcessor ile gecmis verileri yukle
        try:
            from txt_data_processor import TXTDataProcessor
            import asyncio
            
            processor = TXTDataProcessor(raw_data_path="data/raw")
            
            _training_progress = 20
            
            # Async fonksiyonu calistir
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(processor.process_all_countries())
            loop.close()
            
            historical_matches = result.get('matches', [])
            logger.info(f"{len(historical_matches)} gecmis mac yuklendi")
            
            _training_progress = 40
            
        except ImportError:
            logger.warning("TXTDataProcessor bulunamadi, islenmis veriler deneniyor...")
            
            # Islenmis verilerden yukle
            processed_file = os.path.join(DATA_DIR, "processed", "matches.json")
            if os.path.exists(processed_file):
                with open(processed_file, 'r', encoding='utf-8') as f:
                    historical_matches = json.load(f)
                logger.info(f"{len(historical_matches)} gecmis mac yuklendi (cache)")
            else:
                logger.warning("Gecmis veri bulunamadi")
                historical_matches = []
            
            _training_progress = 40
        
        # Model egitimi
        if len(historical_matches) >= 100:
            logger.info(f"Model egitimi basliyor... ({len(historical_matches)} mac)")
            
            ml_predictor = MLPredictionEngine()
            
            _training_progress = 50
            
            training_result = ml_predictor.train(historical_matches)
            
            _training_progress = 90
            
            if training_result['success']:
                predictor = ml_predictor
                _model_accuracy = training_result.get('ensemble_accuracy', 0) * 100
                logger.info(f"Model egitildi! Dogruluk: %{_model_accuracy:.2f}")
                logger.info(f"Modeller: {training_result.get('accuracies', {})}")
            else:
                logger.warning(f"Model egitilemedi: {training_result.get('error')}")
                predictor = PREDICTOR_CLASS()
                
        else:
            logger.warning(f"Yetersiz veri ({len(historical_matches)} mac, min 100), temel tahmin kullanilacak")
            predictor = PREDICTOR_CLASS()
        
        _training_progress = 100
        
    except Exception as e:
        logger.error(f"Gecmis veri yukleme/egitim hatasi: {e}")
        predictor = PREDICTOR_CLASS()
    
    finally:
        _is_training = False

def background_refresh():
    global _cached_matches, _cached_predictions
    
    while True:
        try:
            logger.info("Arka plan yenileme baslatiyor...")
            
            matches = fetch_future_matches(DAYS_AHEAD)
            predictions = generate_predictions(matches)
            
            with _cache_lock:
                _cached_matches = matches
                _cached_predictions = predictions
                
            save_to_disk(matches, MATCHES_FILE)
            save_to_disk(predictions, PREDICTIONS_FILE)
            
            logger.info(f"Yenileme tamamlandi: {len(predictions)} tahmin")
            
        except Exception as e:
            logger.error(f"Arka plan yenileme hatasi: {e}")
            
        time.sleep(REFRESH_INTERVAL)

# ----------------------------------------------------
# FastAPI Routes
# ----------------------------------------------------
@app.on_event("startup")
async def on_startup():
    global _cached_matches, _cached_predictions
    
    ensure_data_dir()
    
    # Diskten yukle
    _cached_matches = load_from_disk(MATCHES_FILE)
    _cached_predictions = load_from_disk(PREDICTIONS_FILE)
    
    # ML modelini egit (arka planda)
    training_thread = threading.Thread(target=load_historical_data_and_train, daemon=True)
    training_thread.start()
    
    # Arka plan yenileme thread'i baslat
    refresh_thread = threading.Thread(target=background_refresh, daemon=True)
    refresh_thread.start()
    
    logger.info("PREDICTA AI v6.1 (ML) baslatildi!")

@app.get("/")
async def root():
    return {
        "service": "PREDICTA AI v6.1 - ML Entegre Tahmin Sistemi",
        "status": "active",
        "ml_enabled": ML_AVAILABLE,
        "model_accuracy": round(_model_accuracy, 2),
        "timestamp": datetime.now().isoformat(),
        "version": "6.1"
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
    
    model_type = predictor.__class__.__name__ if predictor else "Not Initialized"
    
    return {
        "status": "healthy",
        "matches_cached": match_count,
        "predictions_cached": prediction_count,
        "engine": model_type,
        "ml_enabled": ML_AVAILABLE,
        "model_accuracy": round(_model_accuracy, 2),
        "is_training": _is_training,
        "version": "6.1"
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
async def start_training(background_tasks: BackgroundTasks):
    global _is_training
    
    if _is_training:
        raise HTTPException(status_code=400, detail="Egitim zaten devam ediyor")
    
    if not ML_AVAILABLE:
        raise HTTPException(status_code=400, detail="ML kutuphaneleri yuklu degil")
    
    # Arka planda egit
    background_tasks.add_task(load_historical_data_and_train)
    
    return {"success": True, "message": "Egitim baslatildi"}

@app.get("/api/training/status")
async def get_training_status():
    return {
        "is_training": _is_training,
        "progress": _training_progress,
        "model_accuracy": round(_model_accuracy, 2),
        "ml_available": ML_AVAILABLE
    }

# ----------------------------------------------------
# Calistirma
# ----------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    
    ensure_data_dir()
    
    # Ilk veri cekme
    try:
        logger.info("Ilk veri cekme baslatiyor...")
        initial_matches = fetch_future_matches(DAYS_AHEAD)
        
        with _cache_lock:
            _cached_matches = initial_matches
            
        save_to_disk(initial_matches, MATCHES_FILE)
        logger.info("Ilk veri cekme tamamlandi")
        
    except Exception as e:
        logger.error(f"Ilk veri cekme hatasi: {e}")
    
    # ML egitimini baslat (arka planda)
    training_thread = threading.Thread(target=load_historical_data_and_train, daemon=True)
    training_thread.start()
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
