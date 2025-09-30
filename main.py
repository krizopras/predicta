#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREDICTA - BASİT VE ÇALIŞAN VERSİYON
"""
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uvicorn

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Predicta AI", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/")
async def root():
    return {"message": "Predicta API ÇALIŞIYOR!", "status": "healthy"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0"
    }

# Simple matches endpoint
@app.get("/api/nesine/matches")
async def get_matches(league: str = "all", limit: int = 10):
    try:
        # Sample data
        sample_matches = [
            {
                "home_team": "Galatasaray",
                "away_team": "Fenerbahçe",
                "league": "Süper Lig",
                "odds": {"1": 2.10, "X": 3.40, "2": 3.20},
                "ai_prediction": {
                    "prediction": "1",
                    "confidence": 72.5,
                    "home_win_prob": 52.1,
                    "draw_prob": 28.3,
                    "away_win_prob": 19.6
                }
            },
            {
                "home_team": "Beşiktaş",
                "away_team": "Trabzonspor",
                "league": "Süper Lig",
                "odds": {"1": 2.30, "X": 3.20, "2": 3.10},
                "ai_prediction": {
                    "prediction": "X",
                    "confidence": 65.8,
                    "home_win_prob": 41.2,
                    "draw_prob": 35.8,
                    "away_win_prob": 23.0
                }
            }
        ]
        
        return {
            "success": True,
            "matches": sample_matches[:limit],
            "count": len(sample_matches[:limit]),
            "message": "Backend çalışıyor!"
        }
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return {
            "success": False,
            "matches": [],
            "message": f"Error: {str(e)}"
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
