#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREDICTA AI - RAILWAY UYUMLU VERSİYON (Selenium yok, sadece requests)
"""
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uvicorn
import random
import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, Any, List

# YENİ: Fetcher sınıfı yerine, tahmin motorunu import ediyoruz
from prediction_engine import NesineAdvancedPredictor 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Predicta AI API", version="4.1")

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== SINIF BAŞLATMA ====================
# Tüm veri çekme ve tahmin mantığı NesineAdvancedPredictor'da birleşti.
predictor = NesineAdvancedPredictor()

# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "status": "Predicta AI v4.1 - ENTEGRE VERSIYON",
        "engine": "requests + BeautifulSoup",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "method": "requests"}

@app.get("/api/nesine/live-predictions")
async def get_live_predictions(league: str = "all", limit: int = 250):
    try:
        logger.info(f"Endpoint çağrıldı. Lig: {league}, Limit: {limit}")
        
        # Prediction Engine'den verileri çek ve tahminleri al
        matches: List[Dict[str, Any]] = await predictor.fetch_nesine_matches(league_filter=league, limit=limit) 
        
        matches_with_predictions = []
        for match in matches:
            # Tahmin yap
            prediction_data = predictor.predict_match_comprehensive(match)
            
            # API yanıtını oluştur
            matches_with_predictions.append({
                'home_team': match['home_team'],
                'away_team': match['away_team'],
                'league': match['league'],
                'match_id': match.get('match_id', ''),
                'time': match.get('time', '20:00'),
                'date': match.get('date', datetime.now().strftime('%Y-%m-%d')),
                'odds': match['odds'],
                'is_live': match.get('is_live', False),
                'ai_prediction': prediction_data
            })
        
        source = "API/Fallback"
        logger.info(f"✅ {len(matches_with_predictions)} mac hazir - Kaynak: {source}")
        
        return {
            "success": True,
            "matches": matches_with_predictions,
            "count": len(matches_with_predictions),
            "source": source,
            "nesine_live": len(matches_with_predictions) > 0,
            "ai_powered": True,
            "message": f"{len(matches_with_predictions)} mac bulundu ({source})",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Endpoint hatasi: {e}")
        raise HTTPException(status_code=500, detail=f"Sunucu hatası: {str(e)}")

@app.get("/api/nesine/matches")
async def get_matches(league: str = "all", limit: int = 250):
    # Bu endpoint'i ana endpoint'e yönlendiriyoruz
    return await get_live_predictions(league, limit)

@app.get("/api/nesine/raw")
async def get_raw_nesine_data():
    """Ham Nesine API verisi (debug için)"""
    try:
        api_url = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"
        
        api_headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nesine.com/",
            "Origin": "https://www.nesine.com",
        }
        
        response = requests.get(api_url, headers=api_headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            # Özet istatistikler
            ea_count = len(data.get("sg", {}).get("EA", []))
            ca_count = len(data.get("sg", {}).get("CA", []))
            
            return {
                "success": True,
                "status_code": response.status_code,
                "prematch_count": ea_count,
                "live_count": ca_count,
                "total_matches": ea_count + ca_count,
                "raw_data": data,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "status_code": response.status_code,
                "error": "API yanit vermedi"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
