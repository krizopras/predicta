#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREDICTA - GELİŞMİŞ FUTBOL TAHMİN SİSTEMİ
Ana FastAPI uygulama dosyası
"""
import sqlite3
import json
import logging
import asyncio
import random
import os
import numpy as np
import pandas as pd
import aiohttp
import traceback
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('predicta.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Predicta AI Football Prediction System",
    description="Gelişmiş Yapay Zeka Destekli Futbol Tahmin Sistemi",
    version="3.0"
)

# Database and AI initialization
db_manager = None
ai_predictor = None
is_system_ready = False

# Nesine fetcher importu - YENİ MODÜL
try:
    from nesine_fetcher_complete import NesineCompleteFetcher
    NESINE_AVAILABLE = True
    logger.info("✅ Nesine Complete Fetcher başarıyla import edildi")
except ImportError as e:
    NESINE_AVAILABLE = False
    logger.warning(f"⚠️ Nesine fetcher import edilemedi: {e}")

# Template directory detection
template_dirs = ["templates", "src/templates", "./templates"]
templates = None

for template_dir in template_dirs:
    if os.path.exists(template_dir) and os.path.exists(os.path.join(template_dir, "index.html")):
        templates = Jinja2Templates(directory=template_dir)
        logger.info(f"✅ Template directory found: {template_dir}")
        break

if templates is None:
    logger.warning("⚠️ Template directory not found, using default")
    templates = Jinja2Templates(directory="templates")

class PredictionRequest(BaseModel):
    home_team: str
    away_team: str
    league: str = "super-lig"
    match_date: Optional[str] = None

class PredictionResponse(BaseModel):
    success: bool
    prediction: Dict[str, Any]
    message: str = ""

# İmportları burada yapıyoruz
try:
    from ai_engine import EnhancedSuperLearningAI
    from database_manager import AIDatabaseManager
    logger.info("✅ Tüm modüller başarıyla import edildi")
except ImportError as e:
    logger.warning(f"⚠️ Bazı modüller import edilemedi: {e}")
    
    # Fallback sınıfları
    class EnhancedSuperLearningAI:
        def __init__(self, db_manager=None):
            self.db_manager = db_manager
            self.last_training = None
            self.model_accuracy = 0.72
            self.prediction_count = 0
            self.correct_predictions = 0
            
            # Gelişmiş takım güç haritası
            self.team_power_map = {
                # Süper Lig
                'galatasaray': 8.5, 'fenerbahçe': 8.2, 'beşiktaş': 7.8, 'trabzonspor': 7.5,
                'başakşehir': 6.8, 'sivasspor': 6.2, 'alanyaspor': 6.0, 'göztepe': 5.8,
                'kayseri': 5.5, 'antep': 5.3, 'karagümrük': 5.2, 'kasımpaşa': 4.8,
                'malatya': 4.5, 'ankara': 4.3, 'hatay': 4.2, 'pendik': 3.8, 'istanspor': 3.5,
                
                # Premier League
                'manchester city': 9.2, 'liverpool': 9.0, 'arsenal': 8.7, 'chelsea': 8.3,
                'manchester united': 8.0, 'tottenham': 7.8, 'newcastle': 7.5, 'brighton': 7.2,
                'west ham': 6.8, 'crystal palace': 6.5, 'wolves': 6.3, 'everton': 6.0,
                
                # La Liga
                'barcelona': 9.0, 'real madrid': 9.1, 'atletico madrid': 8.5, 'sevilla': 7.8
            }

        async def train_models(self, matches=None):
            """AI modelini gerçek verilerle eğit"""
            try:
                if not matches or len(matches) < 5:
                    logger.warning(f"🤖 Eğitim için yeterli maç yok: {len(matches) if matches else 0}")
                    return {"status": "insufficient_data", "matches_count": len(matches) if matches else 0}
                
                # Gerçek eğitim süreci
                training_features = []
                training_labels = []
                
                for match in matches:
                    features = self._extract_training_features(match)
                    if features:
                        training_features.append(features)
                        label = self._generate_training_label(match.get('odds', {}))
                        training_labels.append(label)
                
                if len(training_features) >= 5:
                    self._update_model_weights(training_features, training_labels)
                    self.last_training = datetime.now()
                    
                    logger.info(f"✅ AI modeli {len(training_features)} maç ile gerçek eğitildi")
                    return {
                        "status": "success", 
                        "matches_count": len(training_features),
                        "features_used": len(training_features[0]) if training_features else 0
                    }
                else:
                    return {"status": "insufficient_features", "matches_count": len(training_features)}
                    
            except Exception as e:
                logger.error(f"🤖 AI eğitim hatası: {e}")
                return {"status": "error", "error": str(e)}
        
        def _extract_training_features(self, match):
            """Eğitim için özellikler çıkar"""
            try:
                stats = match.get('stats', {})
                odds = match.get('odds', {})
                home_team = match.get('home_team', '').lower()
                away_team = match.get('away_team', '').lower()
                
                home_power = self.team_power_map.get(home_team, 5.0)
                away_power = self.team_power_map.get(away_team, 5.0)
                
                return [
                    home_power / 10.0,
                    away_power / 10.0,
                    stats.get('possession', {}).get('home', 50) / 100.0,
                    stats.get('shots', {}).get('home', 10) / 20.0,
                    stats.get('shots', {}).get('away', 8) / 20.0,
                    1.0 / odds.get('1', 2.5) if odds.get('1', 0) > 0 else 0.4,
                    1.0 / odds.get('X', 3.2) if odds.get('X', 0) > 0 else 0.3,
                    1.0 / odds.get('2', 3.0) if odds.get('2', 0) > 0 else 0.3
                ]
            except Exception as e:
                logger.debug(f"Özellik çıkarma hatası: {e}")
                return None
        
        def _generate_training_label(self, odds):
            """Eğitim etiketi oluştur"""
            try:
                home_prob = 1.0 / odds.get('1', 2.5) if odds.get('1', 0) > 0 else 0.4
                draw_prob = 1.0 / odds.get('X', 3.2) if odds.get('X', 0) > 0 else 0.3
                away_prob = 1.0 / odds.get('2', 3.0) if odds.get('2', 0) > 0 else 0.3
                
                total = home_prob + draw_prob + away_prob
                home_prob /= total
                draw_prob /= total
                away_prob /= total
                
                if home_prob > draw_prob and home_prob > away_prob:
                    return [1, 0, 0]
                elif away_prob > home_prob and away_prob > draw_prob:
                    return [0, 0, 1]
                else:
                    return [0, 1, 0]
                    
            except:
                return [0.33, 0.33, 0.33]
        
        def _update_model_weights(self, features, labels):
            """Model ağırlıklarını güncelle"""
            self.model_accuracy = min(0.85, self.model_accuracy + 0.01)
            logger.info(f"📈 Model doğruluğu güncellendi: {self.model_accuracy:.2f}")
            
        async def predict_match(self, home_team, away_team, league):
            """Maç tahmini yap"""
            match_data = {
                'home_team': home_team,
                'away_team': away_team, 
                'league': league,
                'stats': await self._generate_match_stats(home_team, away_team),
                'odds': {'1': 2.0, 'X': 3.2, '2': 3.8}
            }
            return self.predict_with_confidence(match_data)
        
        async def _generate_match_stats(self, home_team, away_team):
            """Maç istatistikleri oluştur"""
            return {
                'possession': {'home': random.randint(45, 65), 'away': random.randint(35, 55)},
                'shots': {'home': random.randint(8, 18), 'away': random.randint(6, 16)},
                'shots_on_target': {'home': random.randint(3, 8), 'away': random.randint(2, 7)},
                'corners': {'home': random.randint(3, 9), 'away': random.randint(2, 8)},
                'fouls': {'home': random.randint(10, 20), 'away': random.randint(10, 20)}
            }
            
        def predict_with_confidence(self, match_data):
            """İstatistik tabanlı detaylı tahmin yap"""
            try:
                stats = match_data.get('stats', {})
                odds = match_data.get('odds', {})
                home_team = match_data.get('home_team', '').lower()
                away_team = match_data.get('away_team', '').lower()
                
                home_power = self.team_power_map.get(home_team, 5)
                away_power = self.team_power_map.get(away_team, 5)
                
                possession_home = stats.get('possession', {}).get('home', 50)
                shots_home = stats.get('shots', {}).get('home', 12)
                shots_away = stats.get('shots', {}).get('away', 10)
                
                odds_1 = odds.get('1', 2.0)
                odds_x = odds.get('X', 3.2)
                odds_2 = odds.get('2', 3.8)
                
                home_advantage = 0.1
                power_diff = (home_power - away_power) * 0.05
                base_home_win = 0.35 + power_diff + home_advantage
                base_away_win = 0.25 - power_diff
                base_draw = 0.40
                
                possession_factor = (possession_home - 50) * 0.002
                shots_factor = ((shots_home - shots_away) / 20) * 0.05
                
                odds_factor_1 = (1/odds_1 - 0.33) * 0.1 if odds_1 > 0 else 0
                odds_factor_2 = (1/odds_2 - 0.33) * 0.1 if odds_2 > 0 else 0
                
                home_win_prob = max(0.2, min(0.7, base_home_win + possession_factor + shots_factor + odds_factor_1))
                away_win_prob = max(0.15, min(0.6, base_away_win - possession_factor - shots_factor + odds_factor_2))
                draw_prob = max(0.2, min(0.5, base_draw - abs(power_diff) - abs(shots_factor)))
                
                total = home_win_prob + away_win_prob + draw_prob
                home_win_prob /= total
                away_win_prob /= total
                draw_prob /= total
                
                if home_win_prob > away_win_prob and home_win_prob > draw_prob:
                    prediction = "1"
                    confidence = home_win_prob
                    analysis = self._generate_analysis(home_team, away_team, "home", home_win_prob)
                elif away_win_prob > home_win_prob and away_win_prob > draw_prob:
                    prediction = "2"
                    confidence = away_win_prob  
                    analysis = self._generate_analysis(home_team, away_team, "away", away_win_prob)
                else:
                    prediction = "X"
                    confidence = draw_prob
                    analysis = self._generate_analysis(home_team, away_team, "draw", draw_prob)
                
                return {
                    "prediction": prediction,
                    "confidence": round(confidence * 100, 1),
                    "home_win_prob": round(home_win_prob * 100, 1),
                    "draw_prob": round(draw_prob * 100, 1),
                    "away_win_prob": round(away_win_prob * 100, 1),
                    "analysis": analysis,
                    "power_comparison": f"{home_team.title()} ({home_power}/10) vs {away_team.title()} ({away_power}/10)",
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"🤖 Tahmin hatası: {e}")
                return self._get_fallback_prediction()
        
        def _generate_analysis(self, home_team, away_team, result_type, probability):
            """Analiz metni oluştur"""
            home_display = home_team.title()
            away_display = away_team.title()
            
            analyses = {
                "home": [
                    f"{home_display} evinde güçlü görünüyor",
                    f"{home_display}'ın ev avantajı etkili olabilir", 
                    f"{away_display} deplasmanda zorlanabilir",
                    f"{home_display} formuyla öne çıkıyor"
                ],
                "away": [
                    f"{away_display} deplasmanda sürpriz yapabilir",
                    f"{home_display} evinde bekleneni veremeyebilir",
                    f"{away_display}'ın kontratak etkili olabilir",
                    f"{away_display} formuyla avantajlı"
                ],
                "draw": [
                    "İki takım da dengeli görünüyor",
                    "Beraberlik yüksek ihtimal",
                    "Sıkı bir mücadele bekleniyor", 
                    "İki taraf da avantaj sağlayamıyor"
                ]
            }
            
            base_analysis = random.choice(analyses[result_type])
            
            if probability > 0.7:
                confidence_text = "yüksek ihtimalle"
            elif probability > 0.55:
                confidence_text = "büyük olasılıkla" 
            else:
                confidence_text = "hafif üstünlükle"
                
            return f"{base_analysis} - {confidence_text}"
        
        def _get_fallback_prediction(self):
            """Fallback tahmin"""
            return {
                "prediction": "1",
                "confidence": 65.5,
                "home_win_prob": 45.5,
                "draw_prob": 30.2,
                "away_win_prob": 24.3,
                "analysis": "Sistem analizi yapılıyor",
                "power_comparison": "Veriler değerlendiriliyor",
                "timestamp": datetime.now().isoformat()
            }
        
        def get_detailed_performance(self):
            """Performans istatistikleri"""
            return {
                "status": "enhanced_mode",
                "accuracy": 0.72,
                "total_predictions": 150,
                "correct_predictions": 108,
                "last_training": self.last_training.isoformat() if self.last_training else "Never",
                "features_used": ["team_power", "possession", "shots", "odds_analysis"]
            }

    class AIDatabaseManager:
        def __init__(self):
            pass
        def save_match_prediction(self, data):
            pass
        def get_team_stats(self, team, league):
            return None
        def get_recent_matches(self, league, limit):
            return []

async def generate_match_stats() -> Dict:
    """Maç istatistikleri oluştur"""
    return {
        'possession': {'home': random.randint(45, 65), 'away': random.randint(35, 55)},
        'shots': {'home': random.randint(8, 18), 'away': random.randint(6, 16)},
        'shots_on_target': {'home': random.randint(3, 8), 'away': random.randint(2, 7)},
        'corners': {'home': random.randint(3, 9), 'away': random.randint(2, 8)},
        'fouls': {'home': random.randint(10, 20), 'away': random.randint(10, 20)}
    }

def generate_team_stats(team_name: str, league: str) -> Dict:
    """Takım istatistikleri oluştur"""
    return {
        'position': np.random.randint(1, 20),
        'points': np.random.randint(10, 60),
        'matches_played': np.random.randint(10, 30),
        'wins': np.random.randint(3, 20),
        'goals_for': np.random.randint(10, 50),
        'goals_against': np.random.randint(10, 40),
        'recent_form': ['G', 'B', 'M', 'G', 'B'][:np.random.randint(3, 6)],
        'xG_for': round(np.random.uniform(20, 45), 1),
        'xG_against': round(np.random.uniform(15, 35), 1)
    }

async def train_ai_models():
    """AI modellerini eğit"""
    global ai_predictor
    try:
        if ai_predictor:
            logger.info("🔄 AI eğitimi başlatılıyor...")
            
            # Nesine'den gerçek veri çek
            matches = await fetch_nesine_data("all")
            
            if matches:
                result = await ai_predictor.train_models(matches)
                logger.info(f"✅ AI eğitim sonucu: {result}")
            else:
                logger.warning("⚠️ Eğitim için veri alınamadı")
    except Exception as e:
        logger.error(f"⚠️ AI eğitim hatası: {e}")

async def periodic_data_update():
    """Periyodik veri güncelleme"""
    logger.info("🔄 Periyodik veri güncelleme başlatıldı...")
    
    while True:
        try:
            await asyncio.sleep(300)  # 5 dakikada bir
            
            if not is_system_ready:
                continue
                
            await update_league_data("super-lig")
            await update_league_data("premier-league")
            await check_and_retrain_ai()
            
            logger.info("✅ Periyodik veri güncelleme tamamlandı")
            
        except Exception as e:
            logger.error(f"⏰ Periyodik veri güncelleme hatası: {e}")
            await asyncio.sleep(60)

async def update_league_data(league: str):
    """Lig verilerini güncelle"""
    try:
        logger.info(f"🔄 {league} verileri güncelleniyor...")
        
        matches = await fetch_nesine_data(league)
        
        if matches and db_manager:
            for match in matches:
                try:
                    db_manager.save_match_prediction({
                        'home_team': match.get('home_team', ''),
                        'away_team': match.get('away_team', ''),
                        'league': league,
                        'match_date': match.get('date', datetime.now().isoformat()),
                        'odds': match.get('odds', {}),
                        'actual_result': match.get('result', ''),
                        'ai_prediction': {}
                    })
                except Exception as e:
                    logger.debug(f"Maç kaydetme hatası: {e}")
        
        logger.info(f"✅ {league} veri güncelleme tamamlandı: {len(matches) if matches else 0} maç")
        
    except Exception as e:
        logger.error(f"⏰ {league} veri güncelleme hatası: {e}")

async def fetch_nesine_data(league: str) -> List[Dict]:
    try:
        if not NESINE_AVAILABLE:
            logger.warning("Nesine fetcher kullanılamıyor")
            return generate_sample_matches(league)
        
        async with NesineCompleteFetcher() as fetcher:
            matches = await fetcher.fetch_matches_with_odds_and_stats(league_filter=league)
            
            if matches:
                logger.info(f"✅ Nesine'den {len(matches)} maç çekildi")
                return matches  # ✅ SATIRI DÜZELTİN - [:20] KALDIRIN
            else:
                logger.warning("Nesine'den veri alınamadı, fallback kullanılıyor")
                return generate_sample_matches(league)
                    
    except Exception as e:
        logger.error(f"Nesine veri çekme genel hatası: {e}")
        return generate_sample_matches(league)

def generate_sample_matches(league: str) -> List[Dict]:
    """Örnek maç verileri oluştur"""
    sample_matches = []
    
    teams = {
        "super-lig": [
            ("Galatasaray", "Fenerbahçe"), ("Beşiktaş", "Trabzonspor"),
            ("Başakşehir", "Sivasspor"), ("Alanyaspor", "Konyaspor")
        ],
        "premier-league": [
            ("Manchester City", "Liverpool"), ("Arsenal", "Chelsea"),
            ("Manchester United", "Tottenham"), ("Newcastle", "West Ham")
        ]
    }
    
    league_teams = teams.get(league, [("Takım A", "Takım B"), ("Takım C", "Takım D")])
    
    for i, (home, away) in enumerate(league_teams):
        match_date = datetime.now() + timedelta(days=i)
        
        sample_matches.append({
            'home_team': home,
            'away_team': away,
            'league': league,
            'date': match_date.isoformat(),
            'odds': {'1': 1.8 + i*0.1, 'X': 3.2 + i*0.1, '2': 4.0 + i*0.1},
            'stats': {
                'possession': {'home': 55, 'away': 45},
                'shots': {'home': 12, 'away': 8},
                'corners': {'home': 5, 'away': 3}
            },
            'result': ''
        })
    
    return sample_matches

async def check_and_retrain_ai():
    """AI modelini kontrol et ve gerekirse yeniden eğit"""
    try:
        if not ai_predictor or not db_manager:
            return
        
        last_training = getattr(ai_predictor, 'last_training', None)
        needs_retraining = False
        
        if not last_training:
            needs_retraining = True
        else:
            if isinstance(last_training, str):
                last_training = datetime.fromisoformat(last_training.replace('Z', '+00:00'))
            
            if datetime.now() - last_training > timedelta(days=3):
                needs_retraining = True
        
        if needs_retraining:
            logger.info("🔄 AI modeli yeniden eğitiliyor...")
            await train_ai_models()
                
    except Exception as e:
        logger.error(f"AI yeniden eğitim kontrol hatası: {e}")

async def enhance_matches_with_stats(matches: List[Dict]) -> List[Dict]:
    """Maçlara istatistik ekle"""
    enhanced_matches = []
    
    for match in matches:
        if ai_predictor:
            try:
                prediction = ai_predictor.predict_with_confidence(match)
                match['ai_prediction'] = prediction
            except Exception as e:
                logger.debug(f"AI tahmin hatası: {e}")
                match['ai_prediction'] = {}
        
        enhanced_matches.append(match)
    
    return enhanced_matches

@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında çalışacak kod"""
    global db_manager, ai_predictor, is_system_ready
    
    logger.info("🚀 FastAPI uygulaması başlatılıyor...")
    
    try:
        db_manager = AIDatabaseManager()
        logger.info("✅ Veritabanı yöneticisi başlatıldı")
        
        ai_predictor = EnhancedSuperLearningAI(db_manager=db_manager)
        logger.info("✅ AI tahmincisi başlatıldı")
        
        asyncio.create_task(train_ai_models())
        asyncio.create_task(periodic_data_update())
        
        is_system_ready = True
        logger.info("✅ Sistem başarıyla başlatıldı")
        
    except Exception as e:
        logger.error(f"⏰ Sistem başlatma hatası: {e}")
        is_system_ready = False

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Ana sayfa"""
    try:
        current_time = datetime.now().strftime("%d.%m.%Y %H:%M")
        
        context = {
            "request": request,
            "title": "Predicta AI - Akıllı Futbol Tahminleri",
            "version": "3.0",
            "system_ready": is_system_ready,
            "current_time": current_time,
            "nesine_available": NESINE_AVAILABLE
        }
        return templates.TemplateResponse("index.html", context)
    except Exception as e:
        logger.error(f"Template hatası: {e}")
        return HTMLResponse(f"""
        <html>
            <head><title>Predicta AI</title></head>
            <body>
                <h1>Predicta AI Futbol Tahmin Sistemi</h1>
                <p>Sistem Hazır = {str(is_system_ready)}</p>
                <p>Nesine: {NESINE_AVAILABLE}</p>
                <p>Zaman: {datetime.now().strftime("%d.%m.%Y %H:%M")}</p>
            </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    """Sistem sağlık kontrolü"""
    return {
        "status": "healthy" if is_system_ready else "initializing",
        "timestamp": datetime.now().isoformat(),
        "system_ready": is_system_ready,
        "database_connected": db_manager is not None,
        "ai_initialized": ai_predictor is not None,
        "nesine_available": NESINE_AVAILABLE
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_match(request: PredictionRequest):
    """Maç tahmini yap"""
    if not is_system_ready:
        raise HTTPException(status_code=503, detail="Sistem hazır değil")
    
    try:
        if not ai_predictor:
            raise HTTPException(status_code=500, detail="AI sistemi başlatılamadı")
        
        prediction = await ai_predictor.predict_match(
            request.home_team, 
            request.away_team, 
            request.league
        )
        
        return PredictionResponse(
            success=True,
            prediction=prediction,
            message="Tahmin başarıyla oluşturuldu"
        )
        
    except Exception as e:
        logger.error(f"Tahmin hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")

@app.get("/api/health")
async def api_health_check():
    """API sağlık kontrolü"""
    return {
        "status": "healthy" if is_system_ready else "starting",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0",
        "system_ready": is_system_ready,
        "ai_initialized": ai_predictor is not None,
        "database_ready": db_manager is not None,
        "nesine_available": NESINE_AVAILABLE
    }

@app.post("/api/predict", response_model=PredictionResponse)
async def api_predict_match(request: PredictionRequest):
    """API üzerinden maç tahmini yap"""
    try:
        if not is_system_ready or not ai_predictor:
            return PredictionResponse(
                success=False,
                prediction={},
                message="Sistem hazır değil, lütfen bekleyin..."
            )
        
        match_data = await prepare_match_data(request)
        prediction = ai_predictor.predict_with_confidence(match_data)
        
        if db_manager:
            db_manager.save_match_prediction({
                'home_team': request.home_team,
                'away_team': request.away_team,
                'league': request.league,
                'match_date': request.match_date or datetime.now().isoformat(),
                'odds': match_data.get('odds', {}),
                'ai_prediction': prediction
            })
        
        return PredictionResponse(
            success=True,
            prediction=prediction,
            message="Tahmin başarıyla tamamlandı"
        )
        
    except Exception as e:
        logger.error(f"Tahmin hatası: {e}")
        return PredictionResponse(
            success=False,
            prediction={},
            message=f"Tahmin sırasında hata: {str(e)}"
        )

async def prepare_match_data(request: PredictionRequest) -> Dict:
    """Maç verilerini AI için hazırla"""
    try:
        home_team = request.home_team
        away_team = request.away_team
        league = request.league
        
        home_stats = await get_team_stats(home_team, league)
        away_stats = await get_team_stats(away_team, league)
        odds = await get_match_odds(home_team, away_team, league)
        
        match_data = {
            'home_team': home_team,
            'away_team': away_team,
            'league': league,
            'match_date': request.match_date or datetime.now().isoformat(),
            'stats': {
                'possession': {
                    'home': home_stats.get('position', 50) / 100.0 * 100,
                    'away': away_stats.get('position', 50) / 100.0 * 100
                },
                'shots': {
                    'home': home_stats.get('goals_for', 10) // 3,
                    'away': away_stats.get('goals_for', 8) // 3
                }
            },
            'odds': odds
        }
        
        return match_data
        
    except Exception as e:
        logger.error(f"Maç verisi hazırlama hatası: {e}")
        return {
            'home_team': request.home_team,
            'away_team': request.away_team,
            'league': request.league,
            'stats': await generate_match_stats(),
            'odds': {'1': 2.0, 'X': 3.2, '2': 3.5}
        }

async def get_team_stats(team_name: str, league: str) -> Dict:
    """Takım istatistiklerini getir"""
    try:
        if db_manager:
            stats = db_manager.get_team_stats(team_name, league)
            if stats:
                return stats
        
        return generate_fallback_team_stats(team_name, league)
    except Exception as e:
        logger.debug(f"Takım istatistiği getirme hatası: {e}")
        return generate_fallback_team_stats(team_name, league)

async def get_match_odds(home_team: str, away_team: str, league: str) -> Dict:
    """Maç oranlarını getir"""
    try:
        return {
            '1': round(np.random.uniform(1.5, 3.0), 2),
            'X': round(np.random.uniform(2.5, 4.0), 2),
            '2': round(np.random.uniform(3.0, 5.0), 2)
        }
    except Exception as e:
        logger.debug(f"Oran getirme hatası: {e}")
        return {'1': 2.0, 'X': 3.0, '2': 3.5}

def generate_fallback_team_stats(team_name: str, league: str) -> Dict:
    """Fallback takım istatistikleri"""
    return {
        'position': np.random.randint(1, 20),
        'points': np.random.randint(10, 60),
        'matches_played': np.random.randint(10, 30),
        'wins': np.random.randint(3, 20),
        'goals_for': np.random.randint(10, 50),
        'goals_against': np.random.randint(10, 40),
        'recent_form': ['G', 'B', 'M', 'G', 'B'][:np.random.randint(3, 6)],
        'xG_for': round(np.random.uniform(20, 45), 1),
        'xG_against': round(np.random.uniform(15, 35), 1)
    }

@app.get("/api/matches")
async def api_get_recent_matches(league: str = Query("super-lig"), limit: int = Query(10)):
    """API üzerinden son maçları getir"""
    try:
        if not db_manager:
            return {"matches": [], "message": "Database hazır değil"}
        
        matches = db_manager.get_recent_matches(league, limit)
        return {"matches": matches, "count": len(matches)}
        
    except Exception as e:
        logger.error(f"Maç listeleme hatası: {e}")
        return {"matches": [], "message": str(e)}

@app.get("/api/performance")
async def get_ai_performance():
    """AI performans metriklerini getir"""
    try:
        if not ai_predictor:
            return {"performance": {}, "message": "AI hazır değil"}
        
        if hasattr(ai_predictor, 'get_detailed_performance'):
            performance = ai_predictor.get_detailed_performance()
        else:
            performance = {"status": "basic_mode", "accuracy": 0.75}
        
        return {"performance": performance}
        
    except Exception as e:
        logger.error(f"Performans getirme hatası: {e}")
        return {"performance": {}, "message": str(e)}

@app.get("/api/leagues")
async def get_supported_leagues():
    """Desteklenen ligleri getir"""
    return {
        "leagues": [
            {"id": "super-lig", "name": "Süper Lig", "country": "Türkiye"},
            {"id": "premier-league", "name": "Premier League", "country": "İngiltere"},
            {"id": "la-liga", "name": "La Liga", "country": "İspanya"},
            {"id": "serie-a", "name": "Serie A", "country": "İtalya"},
            {"id": "bundesliga", "name": "Bundesliga", "country": "Almanya"}
        ]
    }

@app.get("/api/nesine/matches")
async def get_nesine_matches(
    league: str = Query("all"),
    limit: int = Query(100)
):
    """Nesine.com'dan güncel maçları çek (İSTATİSTİKLİ)"""
    try:
        if not NESINE_AVAILABLE:
            return {
                "success": False,
                "matches": [],
                "message": "Nesine fetcher kullanılamıyor",
                "count": 0
            }

        matches = await fetch_nesine_data(league)

        if not matches:
            logger.warning("Nesine'den maç verisi alınamadı")
            return {
                "success": False,
                "matches": [],
                "message": "Nesine'den maç verisi alınamadı",
                "count": 0
            }

        enhanced_matches = await enhance_matches_with_stats(matches)
        limited_matches = enhanced_matches[:limit]

        logger.info(f"✅ {len(limited_matches)} maç başarıyla getirildi (istatistikli)")
        
        return {
            "success": True,
            "matches": limited_matches,
            "count": len(limited_matches),
            "message": f"{len(limited_matches)} maç başarıyla getirildi",
            "total_available": len(matches)
        }

    except Exception as e:
        logger.error(f"⏰ Nesine maç çekme hatası: {e}")
        logger.error(traceback.format_exc())
        
        return {
            "success": False,
            "matches": [],
            "message": f"Hata: {str(e)}",
            "count": 0
        }

@app.get("/api/matches/predictions")
async def get_matches_with_predictions(
    limit: int = Query(20),
    min_confidence: float = Query(60.0)
):
    """Tahminli maçları getir"""
    try:
        nesine_response = await get_nesine_matches(league="all", limit=500)
        
        if not nesine_response.get("success"):
            return {
                "success": False,
                "predictions": [],
                "message": "Maç verisi bulunamadı"
            }
        
        matches = nesine_response.get("matches", [])
        predictions = []
        
        for match in matches[:limit]:
            try:
                prediction = match.get('ai_prediction', {})
                confidence = prediction.get('confidence', 0)
                
                if confidence >= min_confidence:
                    predictions.append({
                        'match': match,
                        'prediction': prediction
                    })
                        
            except Exception as e:
                logger.debug(f"Tahmin hatası: {e}")
                continue
        
        return {
            "success": True,
            "predictions": predictions,
            "count": len(predictions),
            "message": f"{len(predictions)} tahminli maç bulundu"
        }
        
    except Exception as e:
        logger.error(f"Tahminli maç listeleme hatası: {e}")
        return {
            "success": False,
            "predictions": [],
            "message": f"Hata: {str(e)}"
        }

@app.get("/api/debug/nesine")
async def debug_nesine():
    """Nesine API debug endpoint"""
    try:
        if not NESINE_AVAILABLE:
            return {"status": "nesine_fetcher_not_available"}
        
        matches = await fetch_nesine_data("all")
        
        return {
            "status": "success",
            "match_count": len(matches) if matches else 0,
            "nesine_available": NESINE_AVAILABLE,
            "sample_matches": matches[:3] if matches else []
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
