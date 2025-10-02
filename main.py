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

from prediction_engine import ImprovedPredictionEngine   # ✅ Yeni AI motoru

from fetcher import AdvancedNesineFetcher  # eğer fetcher ayrı dosyada değilse, mevcut kod içinden taşıyabiliriz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Predicta AI API", version="5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== NESINE FETCHER & PREDICTOR ====================
fetcher = AdvancedNesineFetcher()
predictor = ImprovedPredictionEngine(seed=42)   # ✅ gelişmiş tahmin motoru

# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "status": "Predicta AI v5.0 - Railway Optimized",
        "engine": "BeautifulSoup + requests (no Selenium) + ImprovedPredictionEngine",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "method": "requests"}

@app.get("/api/nesine/live-predictions")
async def get_live_predictions(league: str = "all", limit: int = 250):
    try:
        logger.info("Nesine CDN API'den veri cekiliyor...")

        matches = fetcher.get_nesine_official_api()
        if not matches:
            logger.warning("API basarisiz, HTML parsing deneniyor...")
            html_content = fetcher.get_page_content()
            matches = fetcher.extract_matches(html_content) if html_content else []

        # Lig filtresi
        if league != "all":
            matches = [m for m in matches if league.lower() in m['league'].lower()]

        valid_matches = []
        for match in matches:
            home = match['home_team'].lower()
            away = match['away_team'].lower()
            if home == away:
                continue
            valid_matches.append(match)

        # AI tahminleri ekle
        matches_with_predictions = []
        for match in valid_matches[:limit]:
            prediction = predictor.predict_match({
                "home_team": match['home_team'],
                "away_team": match['away_team'],
                "odds": match['odds']
            })

            matches_with_predictions.append({
                **match,
                "ai_prediction": prediction
            })

        source = "CDN API" if matches else "HTML Fallback"
        logger.info(f"{len(matches_with_predictions)} mac hazir - Kaynak: {source}")

        return {
            "success": True,
            "matches": matches_with_predictions,
            "count": len(matches_with_predictions),
            "source": source,
            "nesine_live": len(matches_with_predictions) > 0,
            "ai_powered": True,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Endpoint hatasi: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/nesine/matches")
async def get_matches(league: str = "all", limit: int = 250):
    return await get_live_predictions(league, limit)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
