# main_railway.py
#!/usr/bin/env python3
import os
import time
import asyncio
import threading
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import modüller
from txt_data_processor import TXTDataProcessor
from improved_prediction_engine import AdvancedNesineFetcher, ProfessionalPredictionEngine

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Predicta AI Railway", version="7.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global state
class AppState:
    def __init__(self):
        self.is_ready = False
        self.is_processing = False
        self.matches = []
        self.team_stats = {}
        self.processor = TXTDataProcessor()
        self.fetcher = AdvancedNesineFetcher()
        self.predictor = ProfessionalPredictionEngine()

app_state = AppState()

@app.on_event("startup")
async def startup():
    """Uygulama başlangıcında veri işleme"""
    # Arka planda veri işleme başlat
    threading.Thread(target=initialize_data, daemon=True).start()

def initialize_data():
    """Veri işlemeyi başlat (thread içinde)"""
    try:
        app_state.is_processing = True
        logger.info("🔄 Geçmiş veriler işleniyor...")
        
        # Sync wrapper for async function
        async def process_data():
            return await app_state.processor.process_all_countries()
        
        # Event loop oluştur
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(process_data())
        loop.close()
        
        app_state.team_stats = result["team_stats"]
        logger.info(f"✅ {len(app_state.team_stats)} takım istatistiği yüklendi")
        
        # Maçları çek
        refresh_matches()
        
        app_state.is_ready = True
        app_state.is_processing = False
        logger.info("🚀 Sistem hazır!")
        
    except Exception as e:
        logger.error(f"❌ Başlangıç hatası: {e}")
        app_state.is_processing = False

def refresh_matches():
    """Maç verilerini güncelle"""
    try:
        matches = app_state.fetcher.fetch_matches()
        app_state.matches = matches
        logger.info(f"📅 {len(matches)} maç güncellendi")
    except Exception as e:
        logger.error(f"Maç güncelleme hatası: {e}")

@app.get("/")
async def root():
    return {
        "service": "Predicta AI Railway v7.0",
        "status": "ready" if app_state.is_ready else "initializing",
        "features": ["TXT veri işleme", "Geçmiş analiz", "Canlı tahmin"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/system/status")
async def system_status():
    """Sistem durumu"""
    return {
        "is_ready": app_state.is_ready,
        "is_processing": app_state.is_processing,
        "team_stats_loaded": len(app_state.team_stats),
        "cached_matches": len(app_state.matches),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/predictions/railway")
async def get_predictions(limit: int = 50):
    """Tahminleri getir"""
    if not app_state.is_ready:
        raise HTTPException(503, "Sistem hazır değil. Lütfen bekleyin...")
    
    if not app_state.matches:
        refresh_matches()
    
    predictions = []
    for match in app_state.matches[:limit]:
        try:
            pred = app_state.predictor.predict_match(
                match['home_team'],
                match['away_team'],
                match['odds'],
                match['league']
            )
            match_with_pred = match.copy()
            match_with_pred['ai_prediction'] = pred
            predictions.append(match_with_pred)
        except Exception as e:
            logger.error(f"Tahmin hatası: {e}")
    
    return {
        "success": True,
        "count": len(predictions),
        "matches": predictions,
        "engine": "RailwayOptimized",
        "historical_data_used": len(app_state.team_stats) > 0,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/system/refresh")
async def refresh_data(background_tasks: BackgroundTasks):
    """Verileri manuel yenile"""
    background_tasks.add_task(refresh_matches)
    return {"message": "Yenileme başlatıldı", "timestamp": datetime.now().isoformat()}

@app.get("/api/team/{team_name}")
async def get_team_stats(team_name: str):
    """Takım istatistiklerini getir"""
    if team_name in app_state.team_stats:
        return {
            "team": team_name,
            "stats": app_state.team_stats[team_name],
            "found": True
        }
    else:
        return {
            "team": team_name, 
            "stats": {},
            "found": False,
            "message": "Takım bulunamadı"
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main_railway:app", host="0.0.0.0", port=port, log_level="info")
