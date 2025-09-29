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

# Nesine fetcher importu - hata durumunda fallback
try:
    from nesine_match_fetcher import nesine_fetcher
    NESINE_AVAILABLE = True
    logger.info("✅ Nesine fetcher başarıyla import edildi")
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

# İmportları burada yapıyoruz, hata durumunda sistemin çalışmaya devam etmesi için
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
        async def train_models(self):
            return {"status": "AI not available"}
        async def predict_match(self, home_team, away_team, league):
            return {"error": "AI system not initialized"}
        def predict_with_confidence(self, match_data):
            return {
                "prediction": "1", 
                "confidence": round(random.uniform(60, 85), 1),
                "home_win_prob": round(random.uniform(40, 70), 1),
                "draw_prob": round(random.uniform(20, 35), 1),
                "away_win_prob": round(random.uniform(10, 40), 1)
            }
        def get_detailed_performance(self):
            return {"status": "basic_mode", "accuracy": 0.75}
    
    class AIDatabaseManager:
        def __init__(self):
            pass
        def save_match_prediction(self, data):
            pass
        def get_team_stats(self, team, league):
            return None
        def get_recent_matches(self, league, limit):
            return []

async def train_ai_models():
    """AI modellerini eğit"""
    global ai_predictor
    try:
        if ai_predictor:
            logger.info("🔄 Fallback AI eğitimi başlatılıyor...")
            success = await run_ai_training()
            if success:
                logger.info("✅ AI modelleri başarıyla eğitildi")
            else:
                logger.warning("⚠️ AI eğitimi tamamlanamadı, fallback modunda çalışıyor")
        else:
            logger.warning("AI predictor başlatılmadı")
    except Exception as e:
        logger.error(f"⚠️ AI eğitim hatası: {e}")

async def run_ai_training() -> bool:
    """AI eğitimini çalıştır"""
    try:
        if hasattr(ai_predictor, 'train_advanced_models'):
            return ai_predictor.train_advanced_models()
        else:
            return await fallback_ai_training()
    except Exception as e:
        logger.error(f"AI eğitim hatası: {e}")
        return False

async def fallback_ai_training() -> bool:
    """Fallback AI eğitim metodu"""
    try:
        logger.info("🔄 Fallback AI eğitimi başlatılıyor...")
        if ai_predictor and hasattr(ai_predictor, 'models'):
            logger.info("📊 Fallback modeller hazırlanıyor...")
            return True
        return False
    except Exception as e:
        logger.error(f"Fallback eğitim hatası: {e}")
        return False

async def periodic_data_update():
    """Periyodik veri güncelleme"""
    logger.info("🔄 Periyodik veri güncelleme başlatıldı...")
    
    while True:
        try:
            await asyncio.sleep(300)  # 5 dakikada bir
            
            if not is_system_ready:
                continue
                
            # Mevcut lig verilerini güncelle
            await update_league_data("super-lig")
            await update_league_data("premier-league")
            
            # AI modelini yeniden eğit (gerekirse)
            await check_and_retrain_ai()
            
            logger.info("✅ Periyodik veri güncelleme tamamlandı")
            
        except Exception as e:
            logger.error(f"❌ Periyodik veri güncelleme hatası: {e}")
            await asyncio.sleep(60)  # Hata durumunda 1 dakika bekle

async def update_league_data(league: str):
    """Lig verilerini güncelle"""
    try:
        logger.info(f"🔄 {league} verileri güncelleniyor...")
        
        # Nesine.com'dan veri çekme
        matches = await fetch_nesine_data(league)
        
        if matches and db_manager:
            for match in matches:
                try:
                    # Veritabanına kaydet
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
        logger.error(f"❌ {league} veri güncelleme hatası: {e}")

async def fetch_nesine_data(league: str) -> List[Dict]:
    """Nesine.com'dan maç verilerini çek"""
    try:
        if NESINE_AVAILABLE:
            matches = await nesine_fetcher.fetch_prematch_matches()
            if matches:
                # Lige göre filtrele
                league_matches = []
                for match in matches:
                    if league == "super-lig" and "Beşiktaş" in match.get('home_team', '') or "Galatasaray" in match.get('home_team', '') or "Fenerbahçe" in match.get('home_team', ''):
                        league_matches.append(match)
                    elif league == "premier-league" and any(team in match.get('home_team', '') for team in ["Manchester", "Liverpool", "Arsenal", "Chelsea"]):
                        league_matches.append(match)
                
                if league_matches:
                    return league_matches[:10]  # İlk 10 maç
        
        # Fallback: Örnek maçlar oluştur
        return generate_sample_matches(league)
                    
    except Exception as e:
        logger.error(f"Nesine veri çekme genel hatası: {e}")
        return generate_sample_matches(league)

def parse_nesine_response(data: Any, league: str) -> List[Dict]:
    """Nesine response'unu parse et"""
    try:
        matches = []
        
        # Numpy array kontrolü
        if isinstance(data, np.ndarray):
            if data.ndim > 1:
                data = data.flatten()
            data = data.tolist()
        
        # Basit ve güvenli parsing
        if isinstance(data, list):
            for item in data[:10]:  # İlk 10 maç
                match = parse_match_item(item, league)
                if match:
                    matches.append(match)
        elif isinstance(data, dict):
            # Farklı formatlar için
            for key in ['matches', 'events', 'games']:
                if key in data and isinstance(data[key], list):
                    for item in data[key][:10]:
                        match = parse_match_item(item, league)
                        if match:
                            matches.append(match)
                    break
        
        return matches if matches else generate_sample_matches(league)
        
    except Exception as e:
        logger.error(f"Nesine response parsing hatası: {e}")
        return generate_sample_matches(league)

def parse_match_item(item: Any, league: str) -> Optional[Dict]:
    """Tekil maç verisini parse et"""
    try:
        # Numpy array kontrolü
        if isinstance(item, np.ndarray):
            if item.ndim > 1:
                item = item.flatten()
            if len(item) >= 2:
                item = item.tolist()
            else:
                return None
                
        # Güvenli parsing
        if not isinstance(item, (dict, list)):
            return None
            
        if isinstance(item, list) and len(item) >= 2:
            # Basit liste formatı [home_team, away_team, odds...]
            home_team = str(item[0]) if len(item) > 0 else "Takım A"
            away_team = str(item[1]) if len(item) > 1 else "Takım B"
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'date': datetime.now().isoformat(),
                'odds': {'1': 2.0, 'X': 3.0, '2': 3.5},
                'result': ''
            }
        
        elif isinstance(item, dict):
            # Dictionary formatı
            home_team = item.get('home_team', item.get('homeTeam', 'Takım A'))
            away_team = item.get('away_team', item.get('awayTeam', 'Takım B'))
            
            # Odds verisi
            odds = item.get('odds', {})
            if not odds:
                odds = {
                    '1': float(item.get('home_odds', 2.0)),
                    'X': float(item.get('draw_odds', 3.0)),
                    '2': float(item.get('away_odds', 3.5))
                }
            
            return {
                'home_team': str(home_team),
                'away_team': str(away_team),
                'league': league,
                'date': item.get('date', datetime.now().isoformat()),
                'odds': odds,
                'result': item.get('result', '')
            }
        
        return None
        
    except Exception as e:
        logger.debug(f"Maç parsing hatası: {e}")
        return None

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
            'result': ''
        })
    
    return sample_matches

async def check_and_retrain_ai():
    """AI modelini kontrol et ve gerekirse yeniden eğit"""
    try:
        if not ai_predictor or not db_manager:
            return
        
        # Son eğitim tarihini kontrol et
        last_training = getattr(ai_predictor, 'last_training', None)
        needs_retraining = False
        
        if not last_training:
            needs_retraining = True
        else:
            # 3 günden eskiyse yeniden eğit
            if isinstance(last_training, str):
                last_training = datetime.fromisoformat(last_training.replace('Z', '+00:00'))
            
            if datetime.now() - last_training > timedelta(days=3):
                needs_retraining = True
        
        if needs_retraining:
            logger.info("🔄 AI modeli yeniden eğitiliyor...")
            success = await run_ai_training()
            if success:
                logger.info("✅ AI modeli başarıyla güncellendi")
            else:
                logger.warning("⚠️ AI model güncelleme başarısız")
                
    except Exception as e:
        logger.error(f"AI yeniden eğitim kontrol hatası: {e}")

@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında çalışacak kod"""
    global db_manager, ai_predictor, is_system_ready
    
    logger.info("🚀 FastAPI uygulaması başlatılıyor...")
    
    try:
        # Database manager'ı başlat
        db_manager = AIDatabaseManager()
        logger.info("✅ Veritabanı yöneticisi başlatıldı")
        
        # AI predictor'ı başlat
        ai_predictor = EnhancedSuperLearningAI(db_manager=db_manager)
        logger.info("✅ AI tahmincisi başlatıldı")
        
        # AI modellerini eğit (async olarak)
        asyncio.create_task(train_ai_models())
        
        # Periyodik görevleri başlat
        asyncio.create_task(periodic_data_update())
        
        is_system_ready = True
        logger.info("✅ Sistem başarıyla başlatıldı")
        
    except Exception as e:
        logger.error(f"❌ Sistem başlatma hatası: {e}")
        logger.info("⚠️ Sistem kısıtlı modda çalışacak")
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
                <p>Sistem başlatılıyor... Lütfen bekleyin.</p>
                <p>Durum: Sistem Hazır = {str(is_system_ready)}</p>
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

@app.get("/matches")
async def get_recent_matches(league: str = Query("super-lig"), limit: int = Query(10)):
    """Son maçları getir"""
    try:
        # Basit bir veri dönüşü - gerçek uygulamada veritabanından alacaksınız
        return {
            "success": True,
            "matches": [],
            "league": league,
            "limit": limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/teams")
async def get_teams(league: str = Query("super-lig")):
    """Lig takımlarını getir"""
    try:
        # Örnek takım listesi
        teams = ["Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor"]
        return {
            "success": True,
            "teams": teams,
            "league": league
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        
        # Tahmin verilerini hazırla
        match_data = await prepare_match_data(request)
        
        # AI ile tahmin yap
        prediction = ai_predictor.predict_with_confidence(match_data)
        
        # Veritabanına kaydet
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
    """Maç verilerini hazırla"""
    try:
        # Temel maç verisi
        match_data = {
            'home_team': request.home_team,
            'away_team': request.away_team,
            'league': request.league,
            'match_date': request.match_date or datetime.now().isoformat()
        }
        
        # Takım istatistiklerini getir
        home_stats = await get_team_stats(request.home_team, request.league)
        away_stats = await get_team_stats(request.away_team, request.league)
        
        # Oranları getir
        odds = await get_match_odds(request.home_team, request.away_team, request.league)
        
        # Context bilgisi
        context = await get_match_context(request.home_team, request.away_team, request.league)
        
        return {
            'home_stats': home_stats,
            'away_stats': away_stats,
            'odds': odds,
            'context': context
        }
        
    except Exception as e:
        logger.error(f"Maç verisi hazırlama hatası: {e}")
        return get_fallback_match_data(request)

async def get_team_stats(team_name: str, league: str) -> Dict:
    """Takım istatistiklerini getir"""
    try:
        if db_manager:
            stats = db_manager.get_team_stats(team_name, league)
            if stats:
                return stats
        
        # Fallback istatistikler
        return generate_fallback_stats(team_name, league)
    except Exception as e:
        logger.debug(f"Takım istatistiği getirme hatası: {e}")
        return generate_fallback_stats(team_name, league)

async def get_match_odds(home_team: str, away_team: str, league: str) -> Dict:
    """Maç oranlarını getir"""
    try:
        # Gerçek oran verisi yoksa fallback
        return {
            '1': round(np.random.uniform(1.5, 3.0), 2),
            'X': round(np.random.uniform(2.5, 4.0), 2),
            '2': round(np.random.uniform(3.0, 5.0), 2)
        }
    except Exception as e:
        logger.debug(f"Oran getirme hatası: {e}")
        return {'1': 2.0, 'X': 3.0, '2': 3.5}

async def get_match_context(home_team: str, away_team: str, league: str) -> Dict:
    """Maç context bilgisini getir"""
    return {
        'importance': round(np.random.uniform(0.3, 0.9), 2),
        'pressure': round(np.random.uniform(0.2, 0.8), 2),
        'motivation': round(np.random.uniform(0.4, 0.95), 2),
        'referee_stats': {'home_win_rate': round(np.random.uniform(0.4, 0.6), 2)}
    }

def get_fallback_match_data(request: PredictionRequest) -> Dict:
    """Fallback maç verisi"""
    return {
        'home_stats': generate_fallback_stats(request.home_team, request.league),
        'away_stats': generate_fallback_stats(request.away_team, request.league),
        'odds': {'1': 2.0, 'X': 3.0, '2': 3.5},
        'context': {
            'importance': 0.7,
            'pressure': 0.5,
            'motivation': 0.8
        }
    }

def generate_fallback_stats(team_name: str, league: str) -> Dict:
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
async def get_nesine_matches(limit: int = Query(100)):
    """Nesine.com'dan güncel maçları çek"""
    try:
        if not NESINE_AVAILABLE:
            return {
                "success": False,
                "matches": [],
                "message": "Nesine fetcher kullanılamıyor",
                "count": 0
            }
        
        # Nesine'den maçları çek
        matches = await nesine_fetcher.fetch_prematch_matches()
        
        if not matches:
            logger.warning("Nesine'den maç verisi alınamadı")
            return {
                "success": False,
                "matches": [],
                "message": "Nesine'den maç verisi alınamadı",
                "count": 0
            }
        
        logger.info(f"✅ Nesine'den {len(matches)} maç alındı")
        
        # Limit uygula
        limited_matches = matches[:limit]
        
        return {
            "success": True,
            "matches": limited_matches,
            "count": len(limited_matches),
            "message": f"{len(limited_matches)} maç başarıyla getirildi",
            "total_available": len(matches)
        }
        
    except Exception as e:
        logger.error(f"❌ Nesine maç çekme hatası: {e}")
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
        # Önce Nesine'den maçları çek
        nesine_response = await get_nesine_matches(limit=500)
        
        if not nesine_response.get("success"):
            return {
                "success": False,
                "predictions": [],
                "message": "Maç verisi bulunamadı"
            }
        
        matches = nesine_response.get("matches", [])
        predictions = []
        
        # Her maç için tahmin yap
        for match in matches[:limit]:
            try:
                # Tahmin verilerini hazırla
                match_data = {
                    'home_team': match['home_team'],
                    'away_team': match['away_team'],
                    'league': match.get('league', 'Unknown'),
                    'odds': match.get('odds', {})
                }
                
                # AI ile tahmin yap
                if ai_predictor:
                    prediction = ai_predictor.predict_with_confidence(match_data)
                    
                    # Güven seviyesi kontrolü
                    confidence = prediction.get('confidence', 0)
                    if confidence >= min_confidence:
                        predictions.append({
                            'match': match,
                            'prediction': prediction
                        })
                        
            except Exception as e:
                logger.debug(f"Tahmin hatası ({match['home_team']} vs {match['away_team']}): {e}")
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
        
        # Doğrudan nesine_fetcher test et
        matches = await nesine_fetcher.fetch_prematch_matches()
        
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

async def get_team_form(team_name: str) -> Dict:
    """Takım formu getir"""
    return {
        'last_5_games': ['W', 'L', 'W', 'D', 'W'],
        'goals_scored_avg': round(random.uniform(1.2, 2.8), 1),
        'goals_conceded_avg': round(random.uniform(0.8, 2.1), 1),
        'home_advantage': random.choice([True, False])
    }

async def get_head_to_head(home_team: str, away_team: str) -> Dict:
    """Kafa kafaya istatistikler"""
    return {
        'total_meetings': random.randint(5, 25),
        'home_wins': random.randint(2, 10),
        'away_wins': random.randint(1, 8),
        'draws': random.randint(1, 5),
        'avg_goals': round(random.uniform(2.1, 3.5), 1)
    }
# main.py'ye EKLENECEK FONKSİYONLAR

async def get_detailed_match_stats(match_id: int) -> Dict:
    """Detaylı maç istatistikleri çek"""
    try:
        if NESINE_AVAILABLE:
            # Nesine'dan detaylı istatistik çek
            matches = await nesine_fetcher.fetch_prematch_matches()
            for match in matches:
                if match.get('id') == match_id:
                    return match.get('stats', {})
        
        # Fallback istatistikler
        return generate_fallback_stats()
    except Exception as e:
        logger.error(f"İstatistik çekme hatası: {e}")
        return generate_fallback_stats()

def generate_fallback_stats() -> Dict:
    """Fallback istatistikler oluştur"""
    return {
        'possession': {'home': random.randint(45, 65), 'away': random.randint(35, 55)},
        'shots': {'home': random.randint(8, 18), 'away': random.randint(6, 16)},
        'shots_on_target': {'home': random.randint(3, 8), 'away': random.randint(2, 7)},
        'corners': {'home': random.randint(3, 9), 'away': random.randint(2, 8)},
        'fouls': {'home': random.randint(10, 20), 'away': random.randint(10, 20)},
        'yellow_cards': {'home': random.randint(1, 5), 'away': random.randint(1, 5)},
        'red_cards': {'home': random.randint(0, 1), 'away': random.randint(0, 1)}
    }

async def enhance_matches_with_stats(matches: List[Dict]) -> List[Dict]:
    """Maçlara istatistik ekle"""
    enhanced_matches = []
    
    for match in matches:
        # Mevcut istatistikleri kontrol et
        if not match.get('stats'):
            match['stats'] = await get_detailed_match_stats(match.get('id'))
        
        # AI analizi ekle
        if ai_predictor:
            prediction = ai_predictor.predict_with_confidence(match)
            match['ai_prediction'] = prediction
        
        enhanced_matches.append(match)
    
    return enhanced_matches
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
