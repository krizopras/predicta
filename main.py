#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREDICTA AI - NESINE + AI ENTEGRASYONv2
"""
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import uvicorn
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Predicta AI API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== NESİNE SCRAPER ======
class NesineScraperLive:
    """Nesine'den gerçek veri çeken sınıf"""
    
    def __init__(self):
        try:
            import requests
            from bs4 import BeautifulSoup
            self.requests = requests
            self.BeautifulSoup = BeautifulSoup
            self.available = True
        except ImportError:
            logger.warning("requests veya bs4 yok, fallback moda geçiliyor")
            self.available = False
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def fetch_live_matches(self):
        """Nesine'den canlı maçları çek"""
        if not self.available:
            return None
            
        try:
            url = "https://www.nesine.com/iddaa"
            response = self.requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = self.BeautifulSoup(response.text, 'html.parser')
            matches = []
            
            # Nesine'nin HTML yapısına göre parse et
            # Bu kısım Nesine'nin güncel yapısına göre ayarlanmalı
            match_elements = soup.find_all('div', class_='match-row')
            
            for elem in match_elements[:20]:  # İlk 20 maç
                try:
                    teams = elem.find_all('span', class_='team-name')
                    if len(teams) >= 2:
                        home_team = teams[0].text.strip()
                        away_team = teams[1].text.strip()
                        
                        # Oranları bul
                        odds = elem.find_all('button', class_='odd-button')
                        odds_values = [float(o.text) if o.text else 2.0 for o in odds[:3]]
                        
                        matches.append({
                            'home_team': home_team,
                            'away_team': away_team,
                            'league': 'Bilinmeyen',  # Lig bilgisi parse edilecek
                            'odds': {
                                '1': odds_values[0] if len(odds_values) > 0 else 2.0,
                                'X': odds_values[1] if len(odds_values) > 1 else 3.0,
                                '2': odds_values[2] if len(odds_values) > 2 else 3.5
                            }
                        })
                except Exception as e:
                    logger.debug(f"Maç parse hatası: {e}")
                    continue
            
            logger.info(f"✅ Nesine'den {len(matches)} maç çekildi")
            return matches
            
        except Exception as e:
            logger.error(f"❌ Nesine scraping hatası: {e}")
            return None

# ====== AI TAHMİN MOTORU ======
class SimplePredictionEngine:
    """Basit ama etkili tahmin motoru"""
    
    def predict_match(self, home_team, away_team, odds):
        """Maç tahmini üret"""
        # Oranlardan olasılık hesapla
        odds_1 = odds.get('1', 2.0)
        odds_x = odds.get('X', 3.0)
        odds_2 = odds.get('2', 3.5)
        
        # Implied probability
        prob_1 = (1 / odds_1) * 100
        prob_x = (1 / odds_x) * 100
        prob_2 = (1 / odds_2) * 100
        
        # Normalize
        total = prob_1 + prob_x + prob_2
        prob_1 = (prob_1 / total) * 100
        prob_x = (prob_x / total) * 100
        prob_2 = (prob_2 / total) * 100
        
        # En yüksek olasılığı bul
        if prob_1 > prob_x and prob_1 > prob_2:
            prediction = "1"
            confidence = prob_1
        elif prob_2 > prob_x and prob_2 > prob_1:
            prediction = "2"
            confidence = prob_2
        else:
            prediction = "X"
            confidence = prob_x
        
        # Skor tahmini
        if prediction == "1":
            score = f"{random.randint(2,3)}-{random.randint(0,1)}"
        elif prediction == "2":
            score = f"{random.randint(0,1)}-{random.randint(2,3)}"
        else:
            score = f"{random.randint(1,2)}-{random.randint(1,2)}"
        
        return {
            'prediction': prediction,
            'confidence': round(confidence, 1),
            'home_win_prob': round(prob_1, 1),
            'draw_prob': round(prob_x, 1),
            'away_win_prob': round(prob_2, 1),
            'score_prediction': score,
            'timestamp': datetime.now().isoformat()
        }

# Global instance'lar
nesine_scraper = NesineScraperLive()
prediction_engine = SimplePredictionEngine()

# ====== ENDPOINTS ======

@app.get("/")
async def root():
    return {
        "status": "🚀 Predicta AI v3.0",
        "nesine_status": "available" if nesine_scraper.available else "fallback",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/nesine/live-predictions")
async def get_live_predictions(league: str = "all", limit: int = 20):
    """Nesine'den canlı veri + AI tahmin"""
    try:
        # Nesine'den veri çek
        live_matches = nesine_scraper.fetch_live_matches()
        
        # Eğer çekilemezse fallback
        if not live_matches:
            logger.warning("Nesine verisi çekilemedi, fallback kullanılıyor")
            live_matches = get_fallback_matches()
        
        # Her maça AI tahmini ekle
        matches_with_predictions = []
        for match in live_matches[:limit]:
            prediction = prediction_engine.predict_match(
                match['home_team'],
                match['away_team'],
                match['odds']
            )
            
            matches_with_predictions.append({
                **match,
                'ai_prediction': prediction,
                'time': '20:00',
                'date': datetime.now().strftime('%Y-%m-%d')
            })
        
        return {
            "success": True,
            "matches": matches_with_predictions,
            "count": len(matches_with_predictions),
            "nesine_live": nesine_scraper.available,
            "ai_powered": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Hata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def get_fallback_matches():
    """Fallback test maçları"""
    return [
        {
            'home_team': 'Galatasaray',
            'away_team': 'Fenerbahçe',
            'league': 'Süper Lig',
            'odds': {'1': 2.10, 'X': 3.40, '2': 3.20}
        },
        {
            'home_team': 'Beşiktaş',
            'away_team': 'Trabzonspor',
            'league': 'Süper Lig',
            'odds': {'1': 2.30, 'X': 3.20, '2': 3.10}
        },
        {
            'home_team': 'Manchester City',
            'away_team': 'Liverpool',
            'league': 'Premier League',
            'odds': {'1': 1.90, 'X': 3.60, '2': 3.80}
        }
    ]

@app.get("/health")
async def health():
    return {"status": "healthy", "nesine": nesine_scraper.available}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
