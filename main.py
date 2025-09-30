#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREDICTA AI - TAM ENTEGRE SİSTEM
"""
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uvicorn

# Modüllerimizi import ediyoruz
from database_manager import AIDatabaseManager
from ai_engine import EnhancedSuperLearningAI
from prediction_engine import NesineAdvancedPredictor

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Predicta AI", version="2.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
db_manager = AIDatabaseManager()
ai_engine = EnhancedSuperLearningAI(db_manager)
predictor = NesineAdvancedPredictor()

# Health check
@app.get("/")
async def root():
    return {
        "message": "Predicta AI Tam Entegre Sistem ÇALIŞIYOR!",
        "status": "healthy", 
        "version": "2.0",
        "modules": ["AI Engine", "Database", "Prediction Engine"]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0",
        "ai_status": ai_engine.get_detailed_performance()
    }

# Ana endpoint - Canlı verilerle birlikte
@app.get("/api/nesine/matches")
async def get_matches(league: str = "all", limit: int = 20):
    try:
        # 1. Önce canlı verileri çekmeyi dene
        live_matches = await predictor.fetch_nesine_matches(league)
        
        matches_with_predictions = []
        
        for match in live_matches[:limit]:
            # 2. AI tahmini oluştur
            if league == "all" or league.lower() in match.get('league', '').lower():
                prediction = predictor.predict_match_comprehensive(match)
                
                match_data = {
                    "home_team": match['home_team'],
                    "away_team": match['away_team'], 
                    "league": match['league'],
                    "time": match.get('time', '19:00'),
                    "odds": match.get('odds', {'1': 2.0, 'X': 3.0, '2': 3.5}),
                    "ai_prediction": {
                        "prediction": prediction['result_prediction'],
                        "confidence": prediction['confidence'],
                        "home_win_prob": prediction['probabilities'].get('1', 33.3),
                        "draw_prob": prediction['probabilities'].get('X', 33.3),
                        "away_win_prob": prediction['probabilities'].get('2', 33.3),
                        "score_prediction": prediction['score_prediction'],
                        "ai_powered": True
                    }
                }
                
                # 3. Veritabanına kaydet
                db_manager.save_match_prediction(match_data)
                matches_with_predictions.append(match_data)
        
        # Eğer canlı veri yoksa, veritabanındakileri kullan
        if not matches_with_predictions:
            matches_with_predictions = db_manager.get_recent_matches(league, limit)
            # AI tahmini ekle
            for match in matches_with_predictions:
                if 'ai_prediction' not in match:
                    prediction = ai_engine.predict_with_confidence(match)
                    match['ai_prediction'] = prediction
        
        return {
            "success": True,
            "matches": matches_with_predictions,
            "count": len(matches_with_predictions),
            "source": "live" if live_matches else "database",
            "ai_powered": True,
            "message": f"🤖 {len(matches_with_predictions)} maç AI tahmini ile yüklendi"
        }
        
    except Exception as e:
        logger.error(f"Error in get_matches: {e}")
        # Fallback: Basit sample data
        return await get_sample_matches(league, limit)

# Sample data fallback
async def get_sample_matches(league: str = "all", limit: int = 10):
    """Örnek maç verileri"""
    sample_matches = [
        {
            "home_team": "Galatasaray",
            "away_team": "Fenerbahçe", 
            "league": "Süper Lig",
            "time": "20:00",
            "odds": {"1": 2.10, "X": 3.40, "2": 3.20},
            "ai_prediction": {
                "prediction": "1",
                "confidence": 72.5,
                "home_win_prob": 52.1,
                "draw_prob": 28.3, 
                "away_win_prob": 19.6,
                "score_prediction": "2-1",
                "ai_powered": True
            }
        },
        {
            "home_team": "Beşiktaş",
            "away_team": "Trabzonspor",
            "league": "Süper Lig",
            "time": "19:00",
            "odds": {"1": 2.30, "X": 3.20, "2": 3.10},
            "ai_prediction": {
                "prediction": "X", 
                "confidence": 65.8,
                "home_win_prob": 41.2,
                "draw_prob": 35.8,
                "away_win_prob": 23.0,
                "score_prediction": "1-1",
                "ai_powered": True
            }
        },
        {
            "home_team": "Manchester City",
            "away_team": "Liverpool", 
            "league": "Premier League",
            "time": "18:30",
            "odds": {"1": 1.90, "X": 3.60, "2": 3.80},
            "ai_prediction": {
                "prediction": "1",
                "confidence": 68.2,
                "home_win_prob": 48.5, 
                "draw_prob": 30.1,
                "away_win_prob": 21.4,
                "score_prediction": "2-0",
                "ai_powered": True
            }
        }
    ]
    
    # Lig filtresi uygula
    if league != "all":
        filtered_matches = [m for m in sample_matches if league.lower() in m['league'].lower()]
    else:
        filtered_matches = sample_matches
    
    return {
        "success": True,
        "matches": filtered_matches[:limit],
        "count": len(filtered_matches[:limit]),
        "source": "sample",
        "ai_powered": True,
        "message": "Örnek verilerle devam ediliyor"
    }

# AI performans endpoint'i
@app.get("/api/ai/performance")
async def get_ai_performance():
    """AI model performansını getir"""
    try:
        performance = ai_engine.get_detailed_performance()
        db_manager.save_ai_performance(performance)
        
        return {
            "success": True,
            "performance": performance,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"AI performance error: {e}")
        return {
            "success": False,
            "performance": {"status": "unknown", "accuracy": 0},
            "message": str(e)
        }

# Takım istatistikleri
@app.get("/api/team/{team_name}")
async def get_team_stats(team_name: str, league: str = "Süper Lig"):
    """Takım istatistiklerini getir"""
    try:
        stats = db_manager.get_team_stats(team_name, league)
        return {
            "success": True,
            "team": team_name,
            "stats": stats,
            "league": league
        }
    except Exception as e:
        logger.error(f"Team stats error: {e}")
        return {
            "success": False,
            "team": team_name,
            "stats": {},
            "message": str(e)
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=port,
        log_level="info",
        reload=True  # Geliştirme için auto-reload
    )
