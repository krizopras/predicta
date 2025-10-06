#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREDICTA AI v6.2 - ML ENTEGRE TAHMIN SISTEMI
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
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------------------------------
# ML Tahmin Motoru Import
# ----------------------------------------------------
try:
    from ml_prediction_engine import MLPredictionEngine
    PREDICTOR_CLASS = MLPredictionEngine
    ML_AVAILABLE = True
    logger.info("ML Prediction Engine başarıyla yüklendi")
except ImportError as e:
    logger.warning(f"ML Prediction Engine yüklenemedi: {e}")
    ML_AVAILABLE = False
    # Fallback: Basit tahmin motoru
    try:
        from improved_prediction_engine import ImprovedPredictionEngine
        PREDICTOR_CLASS = ImprovedPredictionEngine
        logger.info("Improved Prediction Engine yüklendi")
    except ImportError as e:
        logger.warning(f"Improved Prediction Engine yüklenemedi: {e}")
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
        logger.info("Basic Predictor kullanılıyor")

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
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(DATA_DIR, "predicta_ai.log"), encoding='utf-8')
    ]
)
logger = logging.getLogger("predicta-ml")

# ----------------------------------------------------
# FastAPI App
# ----------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    ensure_data_dir()
    
    # Diskten yükle
    global _cached_matches, _cached_predictions
    _cached_matches = load_from_disk(MATCHES_FILE)
    _cached_predictions = load_from_disk(PREDICTIONS_FILE)
    
    # ML modelini eğit (arka planda)
    training_thread = threading.Thread(target=load_historical_data_and_train, daemon=True)
    training_thread.start()
    
    # Arka plan yenileme thread'i başlat
    refresh_thread = threading.Thread(target=background_refresh, daemon=True)
    refresh_thread.start()
    
    logger.info("PREDICTA AI v6.2 (ML) başlatıldı!")
    
    yield  # App çalışıyor
    
    # Shutdown
    logger.info("PREDICTA AI kapatılıyor...")

app = FastAPI(
    title="PREDICTA AI v6.2 - ML Entegre Tahmin Sistemi",
    version="6.2",
    lifespan=lifespan
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
_last_refresh = None

predictor = None

# ----------------------------------------------------
# Yardimci Fonksiyonlar
# ----------------------------------------------------

def ensure_data_dir():
    """Veri dizinlerini oluştur"""
    try:
        os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, "logs"), exist_ok=True)
        logger.info("Veri dizinleri oluşturuldu/doğrulandı")
    except Exception as e:
        logger.error(f"Dizin oluşturma hatası: {e}")

def save_to_disk(data: List[Dict[str, Any]], filename: str):
    """Veriyi diske kaydet"""
    ensure_data_dir()
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Kaydedildi: {len(data)} kayıt -> {filename}")
        return True
    except Exception as e:
        logger.error(f"Kaydetme hatası {filename}: {e}")
        return False

def load_from_disk(filename: str) -> List[Dict[str, Any]]:
    """Diskten veri yükle"""
    if not os.path.exists(filename):
        return []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Yüklendi: {len(data)} kayıt <- {filename}")
        return data
    except Exception as e:
        logger.error(f"Yükleme hatası {filename}: {e}")
        return []

def parse_nesine_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Nesine JSON verisini ayrıştır"""
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
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Oran dönüşüm hatası: {e}")
                break
                
        match_date = m.get("D", datetime.now().strftime("%Y-%m-%d"))
        match_time = m.get("T", "20:00")
        
        # Tarih kontrolü (geçmiş maçları filtrele)
        try:
            match_datetime = datetime.strptime(f"{match_date} {match_time}", "%Y-%m-%d %H:%M")
            if match_datetime < datetime.now() - timedelta(hours=5):  # 5 saat öncesine kadar
                continue
        except ValueError:
            pass
                
        matches.append({
            "home_team": home,
            "away_team": away,
            "league": m.get("LC", "Bilinmeyen"),
            "match_id": m.get("C", ""),
            "date": match_date,
            "time": match_time,
            "odds": odds,
            "is_live": False,
            "timestamp": datetime.now().isoformat()
        })
        
    return matches

def fetch_future_matches(days_ahead: int = DAYS_AHEAD) -> List[Dict[str, Any]]:
    """Gelecek maçları getir"""
    all_matches = []
    s = requests.Session()
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    
    for i in range(days_ahead):
        date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
        url = f"{API_ENDPOINT}?date={date}"
        
        try:
            r = s.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                data = r.json()
                parsed = parse_nesine_json(data)
                all_matches.extend(parsed)
                logger.info(f"{date}: {len(parsed)} maç")
            else:
                logger.warning(f"{url} -> {r.status_code}")
        except requests.exceptions.Timeout:
            logger.warning(f"{url} zaman aşımı")
        except Exception as e:
            logger.warning(f"{url} bağlantı hatası: {e}")
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
    logger.info(f"Toplam {len(unique_matches)} oynanmamış maç bulundu")
    return unique_matches

def generate_predictions(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tahminleri oluştur"""
    global predictor
    
    if predictor is None:
        logger.warning("Tahmin motoru hazır değil, basit tahmin kullanılıyor...")
        predictor = PREDICTOR_CLASS()
    
    predictions = []
    successful_predictions = 0
    
    for i, match in enumerate(matches):
        try:
            prediction = predictor.predict_match(
                match["home_team"],
                match["away_team"], 
                match["odds"],
                match["league"]
            )
            
            enhanced_match = match.copy()
            enhanced_match["ai_prediction"] = prediction
            enhanced_match["prediction_timestamp"] = datetime.now().isoformat()
            predictions.append(enhanced_match)
            successful_predictions += 1
            
            # İlerleme günlüğü
            if (i + 1) % 10 == 0:
                logger.info(f"Tahmin {i+1}/{len(matches)} tamamlandı")
                
        except Exception as e:
            logger.error(f"Tahmin hatası {match['home_team']} vs {match['away_team']}: {e}")
            # Hata durumunda maçı tahminsiz ekle
            match["ai_prediction"] = {
                "prediction": "U",
                "confidence": 0,
                "probabilities": {"home_win": 0, "draw": 0, "away_win": 0},
                "model": "Error",
                "error": str(e)
            }
            predictions.append(match)
            
    logger.info(f"{successful_predictions}/{len(matches)} maç için tahmin üretildi")
    return predictions

# ----------------------------------------------------
# Geçmiş Veri Yükleme ve Model Egitimi
# ----------------------------------------------------
def load_historical_data_and_train():
    """Geçmiş verileri yükle ve modeli eğit"""
    global predictor, _is_training, _training_progress, _model_accuracy
    
    if not ML_AVAILABLE:
        logger.warning("ML kütüphaneleri yüklü değil, temel tahmin kullanılacak")
        logger.info("Kurulum: pip install scikit-learn xgboost pandas numpy")
        predictor = PREDICTOR_CLASS()
        return
    
    try:
        _is_training = True
        _training_progress = 10
        
        logger.info("Geçmiş veriler yükleniyor...")
        
        historical_matches = []
        
        # TXTDataProcessor ile geçmiş verileri yükle
        try:
            from txt_data_processor import TXTDataProcessor
            import asyncio
            
            _training_progress = 20
            
            processor = TXTDataProcessor(raw_data_path="data/raw")
            
            # Async fonksiyonu çalıştır
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(processor.process_all_countries())
            loop.close()
            
            historical_matches = result.get('matches', [])
            logger.info(f"{len(historical_matches)} geçmiş maç yüklendi (TXTProcessor)")
            
            _training_progress = 40
            
        except ImportError as e:
            logger.warning(f"TXTDataProcessor bulunamadı: {e}, işlenmiş veriler deneniyor...")
            
            # İşlenmiş verilerden yükle
            processed_file = os.path.join(DATA_DIR, "processed", "matches.json")
            if os.path.exists(processed_file):
                with open(processed_file, 'r', encoding='utf-8') as f:
                    historical_matches = json.load(f)
                logger.info(f"{len(historical_matches)} geçmiş maç yüklendi (cache)")
            else:
                logger.warning("Geçmiş veri bulunamadı")
                historical_matches = []
            
            _training_progress = 40
        
        # Model eğitimi
        if len(historical_matches) >= 100:
            logger.info(f"Model eğitimi başlıyor... ({len(historical_matches)} maç)")
            
            _training_progress = 50
            
            ml_predictor = MLPredictionEngine()
            training_result = ml_predictor.train(historical_matches)
            
            _training_progress = 90
            
            if training_result.get('success', False):
                predictor = ml_predictor
                _model_accuracy = training_result.get('ensemble_accuracy', 0) * 100
                logger.info(f"Model eğitildi! Doğruluk: %{_model_accuracy:.2f}")
                
                # Model detaylarını logla
                accuracies = training_result.get('accuracies', {})
                for model_name, accuracy in accuracies.items():
                    logger.info(f"  - {model_name}: %{accuracy*100:.2f}")
                    
            else:
                error_msg = training_result.get('error', 'Bilinmeyen hata')
                logger.warning(f"Model eğitilemedi: {error_msg}")
                predictor = PREDICTOR_CLASS()
                
        else:
            logger.warning(f"Yetersiz veri ({len(historical_matches)} maç, min 100), temel tahmin kullanılacak")
            predictor = PREDICTOR_CLASS()
        
        _training_progress = 100
        
    except Exception as e:
        logger.error(f"Geçmiş veri yükleme/eğitim hatası: {e}")
        predictor = PREDICTOR_CLASS()
    
    finally:
        _is_training = False
        logger.info("Model eğitim süreci tamamlandı")

def background_refresh():
    """Arka plan yenileme döngüsü"""
    global _cached_matches, _cached_predictions, _last_refresh
    
    # İlk çalışmada 30 saniye bekle (model eğitimi için zaman tanı)
    time.sleep(30)
    
    while True:
        try:
            logger.info("Arka plan yenileme başlatılıyor...")
            
            matches = fetch_future_matches(DAYS_AHEAD)
            predictions = generate_predictions(matches)
            
            with _cache_lock:
                _cached_matches = matches
                _cached_predictions = predictions
                _last_refresh = datetime.now().isoformat()
                
            save_to_disk(matches, MATCHES_FILE)
            save_to_disk(predictions, PREDICTIONS_FILE)
            
            logger.info(f"Yenileme tamamlandı: {len(predictions)} tahmin")
            
        except Exception as e:
            logger.error(f"Arka plan yenileme hatası: {e}")
            
        logger.info(f"Sonraki yenileme için {REFRESH_INTERVAL} saniye bekleniyor...")
        time.sleep(REFRESH_INTERVAL)

# ----------------------------------------------------
# FastAPI Routes
# ----------------------------------------------------
@app.get("/")
async def root():
    return {
        "service": "PREDICTA AI v6.2 - ML Entegre Tahmin Sistemi",
        "status": "active",
        "ml_enabled": ML_AVAILABLE,
        "model_accuracy": round(_model_accuracy, 2),
        "last_refresh": _last_refresh,
        "timestamp": datetime.now().isoformat(),
        "version": "6.2"
    }

@app.get("/api/matches/upcoming")
async def get_upcoming_matches(limit: Optional[int] = 0, league: Optional[str] = None):
    with _cache_lock:
        matches = list(_cached_matches)
        
    # Lig filtreleme
    if league:
        matches = [m for m in matches if league.lower() in m.get("league", "").lower()]
        
    if limit and limit > 0:
        matches = matches[:limit]
        
    return {
        "success": True,
        "count": len(matches),
        "matches": matches
    }

@app.get("/api/predictions/upcoming")
async def get_predictions(limit: Optional[int] = 50, min_confidence: Optional[float] = 0):
    with _cache_lock:
        predictions = list(_cached_predictions)
    
    # Güven filtresi
    if min_confidence > 0:
        predictions = [
            p for p in predictions 
            if p.get("ai_prediction", {}).get("confidence", 0) >= min_confidence
        ]
    
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
        "training_progress": _training_progress,
        "last_refresh": _last_refresh,
        "version": "6.2"
    }

@app.post("/api/matches/refresh")
async def manual_refresh(background_tasks: BackgroundTasks):
    """Manuel yenileme endpoint'i"""
    
    async def refresh_task():
        try:
            matches = fetch_future_matches(DAYS_AHEAD)
            predictions = generate_predictions(matches)
            
            with _cache_lock:
                global _cached_matches, _cached_predictions, _last_refresh
                _cached_matches = matches
                _cached_predictions = predictions
                _last_refresh = datetime.now().isoformat()
                
            save_to_disk(matches, MATCHES_FILE)
            save_to_disk(predictions, PREDICTIONS_FILE)
            
            logger.info("Manuel yenileme tamamlandı")
            
        except Exception as e:
            logger.error(f"Manuel yenileme hatası: {e}")
    
    background_tasks.add_task(refresh_task)
    
    return {
        "success": True,
        "message": "Yenileme başlatıldı",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/training/start")
async def start_training(background_tasks: BackgroundTasks):
    global _is_training
    
    if _is_training:
        raise HTTPException(status_code=400, detail="Eğitim zaten devam ediyor")
    
    if not ML_AVAILABLE:
        raise HTTPException(status_code=400, detail="ML kütüphaneleri yüklü değil")
    
    # Arka planda eğit
    background_tasks.add_task(load_historical_data_and_train)
    
    return {
        "success": True, 
        "message": "Eğitim başlatıldı",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/training/status")
async def get_training_status():
    return {
        "is_training": _is_training,
        "progress": _training_progress,
        "model_accuracy": round(_model_accuracy, 2),
        "ml_available": ML_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/leagues")
async def get_leagues():
    """Mevcut ligleri listele"""
    with _cache_lock:
        matches = list(_cached_matches)
    
    leagues = sorted(list(set(m.get("league", "Bilinmeyen") for m in matches)))
    
    return {
        "success": True,
        "count": len(leagues),
        "leagues": leagues
    }

# ----------------------------------------------------
# Calistirma
# ----------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    
    ensure_data_dir()
    
    # İlk veri çekme
    try:
        logger.info("İlk veri çekme başlatılıyor...")
        initial_matches = fetch_future_matches(DAYS_AHEAD)
        
        with _cache_lock:
            _cached_matches = initial_matches
            
        save_to_disk(initial_matches, MATCHES_FILE)
        logger.info("İlk veri çekme tamamlandı")
        
    except Exception as e:
        logger.error(f"İlk veri çekme hatası: {e}")
    
    # ML eğitimini başlat (arka planda)
    training_thread = threading.Thread(target=load_historical_data_and_train, daemon=True)
    training_thread.start()
    
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"PREDICTA AI v6.2 {port} portunda başlatılıyor...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port, 
        log_level="info",
        access_log=True
    )
