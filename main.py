#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREDICTA - FASTAPI ANA GİRİŞ DOSYASI (RENDER UYUMLU)
Tüm modülleri birleştirir ve HTML arayüzünü sunar.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import logging
import os
from datetime import datetime
import threading
import time
import numpy as np
import asyncio
from contextlib import asynccontextmanager

# Kendi modüllerinizi import edin
from database_manager import DatabaseManager
from prediction_engine import NesineAdvancedPredictor

# --- Yapılandırma ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ortam değişkenleri
ALLOWED_ORIGINS = ["*"]

# Pydantic Modelleri (API yanıtları için gerekli)
class PredictionModel(BaseModel):
    result_prediction: str
    confidence: float
    iy_prediction: Optional[str] = None
    score_prediction: Optional[str] = None
    score_probabilities: Optional[List[Dict]] = None

class MatchModel(BaseModel):
    home_team: str
    away_team: str
    league: str
    date: str
    time: str
    odds: Dict[str, float]

class MatchWithPrediction(BaseModel):
    match: MatchModel
    prediction: PredictionModel

# Global değişkenler
db_manager: Optional[DatabaseManager] = None
predictor: Optional[NesineAdvancedPredictor] = None
background_task: Optional[asyncio.Task] = None

def initialize_system():
    """Sistem başlatma fonksiyonu"""
    global db_manager, predictor
    
    try:
        # Veritabanı klasörünü oluştur
        os.makedirs("data", exist_ok=True)
        
        # Modülleri başlat
        db_manager = DatabaseManager(db_path="data/nesine_advanced.db")
        
        # DÜZELTME YAPILDI: Sınıfın bir örneği oluşturuldu.
        predictor = NesineAdvancedPredictor() 
        
        logger.info("Sistem başarıyla başlatıldı")
        return True
        
    except Exception as e:
        logger.error(f"Sistem başlatma hatası: {e}")
        return False

async def periodic_data_update():
    """Verileri periyodik olarak çeken ve güncelleyen arka plan işlevi."""
    global predictor, db_manager
    
    logger.info("Periyodik veri güncelleme başlatıldı...")
    
    while True:
        try:
            if predictor is None:
                logger.warning("Predictor henüz hazır değil, 30 saniye bekleniyor...")
                await asyncio.sleep(30)
                continue
            
            # Maç verilerini çek ve tahmin et
            matches = predictor.fetch_nesine_matches()
            logger.info(f"{len(matches)} adet maç verisi çekildi")
            
            # Her maç için tahmin yap ve veritabanına kaydet
            predictions_saved = 0
            for match in matches:
                try:
                    prediction = predictor.predict_match_comprehensive(match)
                    
                    # Veritabanına kaydet
                    if db_manager and prediction.get('confidence', 0) > 30:
                        prediction_data = {
                            'home_team': match.get('home_team'),
                            'away_team': match.get('away_team'),
                            'league': match.get('league'),
                            'match_date': match.get('date'),
                            'match_time': match.get('time'),
                            'prediction_result': prediction.get('result_prediction'),
                            'prediction_iy': prediction.get('iy_prediction'),
                            'predicted_score': prediction.get('score_prediction'),
                            'confidence': prediction.get('confidence'),
                            'home_xg': prediction.get('home_xg'),
                            'away_xg': prediction.get('away_xg'),
                            'odds_1': match.get('odds', {}).get('1', 0.0),
                            'odds_x': match.get('odds', {}).get('X', 0.0),
                            'odds_2': match.get('odds', {}).get('2', 0.0),
                            'created_at': datetime.now().isoformat()
                        }
                        
                        # Veritabanına kaydet (DatabaseManager'da bu method olduğunu varsayıyorum)
                        try:
                            db_manager.save_prediction(prediction_data)
                            predictions_saved += 1
                        except Exception as db_error:
                            logger.warning(f"Veritabanına kaydetme hatası: {db_error}")
                
                except Exception as pred_error:
                    logger.warning(f"Tahmin hatası: {pred_error}")
                    continue
            
            logger.info(f"{predictions_saved} tahmin veritabanına kaydedildi")
            
            # 30 dakika bekle
            await asyncio.sleep(30 * 60)
            
        except Exception as e:
            logger.error(f"Periyodik veri güncelleme hatası: {e}")
            # Hata durumunda bekleme süresini kısaltıp tekrar dene
            await asyncio.sleep(5 * 60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("FastAPI uygulaması başlatılıyor...")
    
    # Sistem başlatma
    if not initialize_system():
        logger.error("Sistem başlatılamadı!")
    
    # Arka plan görevini başlat
    global background_task
    background_task = asyncio.create_task(periodic_data_update())
    
    yield
    
    # Shutdown
    logger.info("FastAPI uygulaması kapatılıyor...")
    if background_task:
        background_task.cancel()
        try:
            await background_task
        except asyncio.CancelledError:
            pass

# FastAPI app oluştur - lifespan context manager ile
app = FastAPI(
    title="Futbol Tahmin API",
    description="Render'da çalışan, tahminleri gösteren web arayüzü.",
    version="1.0.0",
    lifespan=lifespan
)

# Jinja2 Templates yapılandırması - DİNAMİK YOL BULMA
template_dir = None
possible_paths = [
    "predicta/templates",  # Normal yol
    "templates",           # Alternatif yol
    ".",                   # Root dizini
    "src/predicta/templates",  # Render specific
    "/opt/render/project/src/predicta/templates"  # Render absolute path
]

for path in possible_paths:
    if os.path.exists(path) and os.path.isdir(path):
        template_dir = path
        logger.info(f"Template directory found: {path}")
        # index.html kontrolü
        index_path = os.path.join(path, "index.html")
        if os.path.exists(index_path):
            logger.info(f"index.html found in: {path}")
        else:
            logger.warning(f"index.html NOT found in: {path}")
        break

if template_dir is None:
    logger.warning("No template directory found, using current directory")
    template_dir = "."

logger.info(f"Final template directory: {template_dir}")
templates = Jinja2Templates(directory=template_dir)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_fallback_html(request: Request, predictions_data: list, stats: dict, error_message: str = None):
    """Template bulunamazsa fallback HTML oluştur"""
    
    predictions_html = ""
    if predictions_data:
        for pred in predictions_data:
            predictions_html += f"""
            <div class="match-card">
                <h3>{pred['mac']}</h3>
                <p><strong>Lig:</strong> {pred['lig']} | <strong>Tarih:</strong> {pred['tarih']}</p>
                <p><strong>MS Tahmin:</strong> {pred['tahmin_ms']} | <strong>İY Tahmin:</strong> {pred['tahmin_iy']}</p>
                <p><strong>Skor Tahmini:</strong> {pred['skor']} | <strong>Güven:</strong> <span class="confidence">{pred['guven']:.1f}%</span></p>
            </div>
            """
    else:
        predictions_html = "<p>Henüz maç verisi bulunmamaktadır.</p>"
    
    error_html = f"<div class='error'><strong>Hata:</strong> {error_message}</div>" if error_message else ""
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>⚽ Predicta: Futbol Tahminleri</title>
        <style>
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                padding: 30px; 
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}
            .header {{ 
                text-align: center; 
                color: #2c3e50; 
                margin-bottom: 30px;
            }}
            .header h1 {{ 
                color: #e74c3c; 
                font-size: 2.5em;
                margin-bottom: 10px;
            }}
            .stats {{ 
                background: #ecf0f1; 
                padding: 15px; 
                border-radius: 8px; 
                margin: 20px 0;
                text-align: center;
            }}
            .match-card {{ 
                background: #f8f9fa; 
                margin: 15px 0; 
                padding: 20px; 
                border-radius: 10px;
                border-left: 5px solid #3498db;
                transition: transform 0.2s;
            }}
            .match-card:hover {{ 
                transform: translateY(-2px); 
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .confidence {{ 
                color: #e74c3c; 
                font-weight: bold; 
                font-size: 1.1em;
            }}
            .error {{ 
                background: #f8d7da; 
                color: #721c24; 
                padding: 15px; 
                border-radius: 5px; 
                margin: 20px 0;
            }}
            .info {{ 
                background: #d1ecf1; 
                color: #0c5460; 
                padding: 15px; 
                border-radius: 5px; 
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>⚽ Predicta</h1>
                <p>Güncel Futbol Maç Tahminleri</p>
                <p><em>Son Güncelleme: {datetime.now().strftime('%d.%m.%Y %H:%M')}</em></p>
            </div>
            
            {error_html}
            
            <div class="stats">
                <h3>📊 İstatistikler</h3>
                <p>Toplam Maç: <strong>{stats.get('total_matches', 0)}</strong> | Ortalama Güven: <strong>{stats.get('average_confidence', 0):.1f}%</strong></p>
            </div>
            
            <div id="predictions">
                <h3>🎯 Maç Tahminleri</h3>
                {predictions_html}
            </div>
            
            <div class="info">
                <p><strong>Not:</strong> Bu basit arayüz template dosyası bulunamadığı için gösterilmektedir.</p>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# --- ANA SAYFA ROTASI (index.html) ---

@app.get("/", response_class=HTMLResponse)
async def root(request: Request, min_confidence: Optional[float] = None):
    global predictor, db_manager
    
    # URL parametresinden min_confidence'ı al, yoksa varsayılan 50 kullan
    confidence_threshold = min_confidence if min_confidence is not None else 50.0
    
    if not predictor:
        return create_fallback_html(
            request, 
            [], 
            {}, 
            "Sistem başlatılıyor. Lütfen 2-3 dakika bekleyin..."
        )

    try:
        # Veritabanından belirlenen güven seviyesine göre tahminleri getir
        recent_predictions = []
        if db_manager:
            try:
                recent_predictions = db_manager.get_latest_predictions_from_db(min_confidence=confidence_threshold)
            except Exception as db_error:
                logger.warning(f"Veritabanı okuma hatası: {db_error}")
        
        predictions_data = []
        
        if recent_predictions:
            # Veritabanındaki tahminleri kullan
            for pred in recent_predictions:
                predictions_data.append({
                    "mac": f"{pred.get('home_team', 'N/A')} - {pred.get('away_team', 'N/A')}",
                    "lig": pred.get('league', 'Veritabanı'),
                    "tarih": pred.get('created_at', '')[:16] if pred.get('created_at') else '',
                    "tahmin_ms": pred.get('prediction_result', 'X'),
                    "tahmin_iy": pred.get('prediction_iy', 'X'),
                    "skor": pred.get('predicted_score', '1-1'),
                    "guven": pred.get('confidence', 50.0)
                })
        else:
            # Eğer veritabanında tahmin yoksa, fresh data çekmeyi dene
            try:
                matches = predictor.fetch_nesine_matches()
                
                for match in matches[:10]:  # Sadece ilk 10 maçı işle
                    prediction = predictor.predict_match_comprehensive(match)
                    
                    if prediction.get('confidence', 0) >= confidence_threshold:
                        predictions_data.append({
                            "mac": f"{match.get('home_team')} - {match.get('away_team')}",
                            "lig": match.get('league', 'Bilinmeyen'),
                            "tarih": f"{match.get('date')} {match.get('time')}",
                            "tahmin_ms": prediction.get('result_prediction', 'X'),
                            "tahmin_iy": prediction.get('iy_prediction', 'X'),
                            "skor": prediction.get('score_prediction', '1-1'),
                            "guven": prediction.get('confidence', 50.0)
                        })
            except Exception as fresh_error:
                logger.warning(f"Fresh data çekme hatası: {fresh_error}")

        # İstatistikleri hesapla
        if predictions_data:
            stats = {
                "total_matches": len(predictions_data),
                "average_confidence": np.mean([p['guven'] for p in predictions_data]),
                "current_threshold": confidence_threshold
            }
        else:
            stats = {
                "total_matches": 0,
                "average_confidence": 0,
                "current_threshold": confidence_threshold
            }

        # Önce template'i deneyelim
        try:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "title": "⚽ Predicta: Güncel Maç Tahminleri",
                    "predictions": predictions_data,
                    "stats": stats,
                    "current_time": datetime.now(),
                    "error_message": None,
                    "confidence_threshold": confidence_threshold
                }
            )
        except Exception as template_error:
            logger.warning(f"Template bulunamadı, fallback HTML kullanılıyor: {template_error}")
            return create_fallback_html(request, predictions_data, stats)

    except Exception as e:
        logger.error(f"Ana sayfa hatası: {e}")
        return create_fallback_html(
            request, 
            [], 
            {"total_matches": 0, "average_confidence": 0, "current_threshold": confidence_threshold}, 
            f"Sistem hatası: {str(e)}"
        )

# Yeni endpoint: Güven seviyesini dinamik olarak ayarlama
@app.get("/confidence/{confidence_level}", response_class=HTMLResponse)
async def get_predictions_by_confidence(request: Request, confidence_level: float):
    """Belirli güven seviyesindeki tahminleri getir"""
    if confidence_level < 0 or confidence_level > 100:
        raise HTTPException(status_code=400, detail="Güven seviyesi 0-100 arasında olmalıdır")
    
    return await root(request, min_confidence=confidence_level)

@app.get("/matches/predictions", response_model=List[MatchWithPrediction])
async def get_matches_json(min_confidence: float = 60.0):
    """API: Yüksek güvenli tahminleri JSON olarak döndürür."""
    if not predictor:
        raise HTTPException(status_code=503, detail="Sistem henüz başlatılıyor. Lütfen bekleyin.")
    
    try:
        # Veritabanından tahminleri getir
        recent_predictions = []
        if db_manager:
            recent_predictions = db_manager.get_latest_predictions_from_db(min_confidence=min_confidence)
        
        formatted_list = []
        for pred in recent_predictions:
            try:
                match_with_pred = MatchWithPrediction(
                    match=MatchModel(
                        home_team=pred.get('home_team', 'N/A'),
                        away_team=pred.get('away_team', 'N/A'),
                        league=pred.get('league', 'API'),
                        date=pred.get('created_at', '')[:10] if pred.get('created_at') else '',
                        time=pred.get('created_at', '')[11:16] if pred.get('created_at') else '',
                        odds={"1": pred.get('odds_1', 2.0), "X": pred.get('odds_x', 3.0), "2": pred.get('odds_2', 3.5)}
                    ),
                    prediction=PredictionModel(
                        result_prediction=pred.get('prediction_result', 'X'),
                        confidence=pred.get('confidence', 50.0),
                        iy_prediction=pred.get('prediction_iy', 'X'),
                        score_prediction=pred.get('predicted_score', '1-1')
                    )
                )
                formatted_list.append(match_with_pred)
            except Exception as format_error:
                logger.warning(f"JSON format hatası: {format_error}")
                continue
        
        return formatted_list

    except Exception as e:
        logger.error(f"JSON API hatası: {e}")
        raise HTTPException(status_code=500, detail="Tahmin verilerini çekerken bir sunucu hatası oluştu.")

# --- Sağlık Kontrolü ---
@app.get("/health")
async def health_check():
    """Sistem sağlık kontrolü"""
    return {
        "status": "healthy" if predictor else "starting",
        "timestamp": datetime.now().isoformat(),
        "db_connected": db_manager is not None,
        "template_dir": template_dir
    }

# --- Lokal Test Başlatma ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
