#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREDICTA - FASTAPI ANA GİRİŞ DOSYASI (RENDER UYUMLU)
Süper Öğrenen Yapay Zeka ile geliştirilmiş tahmin sistemi
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
from ai_engine import EnhancedSuperLearningAI  # YENİ: AI motoru

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
    ai_powered: Optional[bool] = False
    certainty_index: Optional[float] = None
    risk_factors: Optional[Dict] = None

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

class AIPerformanceModel(BaseModel):
    current_accuracy: float
    training_samples: int
    feature_count: int
    cross_validation_score: float
    model_stability: float
    last_training: str
    adaptation_status: str
    certainty_trend: float
    risk_profile: Dict

# Global değişkenler
db_manager: Optional[DatabaseManager] = None
predictor: Optional[NesineAdvancedPredictor] = None
ai_predictor: Optional[EnhancedSuperLearningAI] = None
background_task: Optional[asyncio.Task] = None

def initialize_system():
    """Sistem başlatma fonksiyonu"""
    global db_manager, predictor, ai_predictor
    
    try:
        # Veritabanı klasörünü oluştur
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/ai_models_v2", exist_ok=True)
        
        # Modülleri başlat
        db_manager = DatabaseManager(db_path="data/nesine_advanced.db")
        predictor = NesineAdvancedPredictor() 
        
        # YENİ: AI predictor'ı başlat
        ai_predictor = EnhancedSuperLearningAI(db_manager=db_manager)
        logger.info("🤖 Süper Öğrenen AI başlatıldı")
        
        # AI modellerini eğit (eğer yoksa)
        try:
            ai_predictor.train_models()
            logger.info("✅ AI modelleri eğitildi")
        except Exception as train_error:
            logger.warning(f"⚠️ AI eğitim hatası: {train_error}")
        
        logger.info("✅ Sistem başarıyla başlatıldı")
        return True
        
    except Exception as e:
        logger.error(f"❌ Sistem başlatma hatası: {e}")
        return False

async def periodic_data_update():
    """Verileri periyodik olarak çeken ve güncelleyen arka plan işlevi."""
    global predictor, ai_predictor
    
    logger.info("🔄 Periyodik veri güncelleme başlatıldı...")
    
    while True:
        try:
            if predictor is None or ai_predictor is None:
                logger.warning("⚠️ Predictor henüz hazır değil, 30 saniye bekleniyor...")
                await asyncio.sleep(30)
                continue
            
            # Maç verilerini çek
            matches = predictor.fetch_nesine_matches()
            logger.info(f"📊 {len(matches)} adet maç verisi çekildi")
            
            # Her maç için AI tahmini yap ve veritabanına kaydet
            ai_predictions = []
            for match in matches:
                try:
                    # AI ile tahmin yap
                    ai_prediction = ai_predictor.predict_with_confidence(match)
                    
                    # Veritabanına kaydet
                    db_manager.save_prediction(
                        match_data=match,
                        prediction_data=ai_prediction,
                        source="ai_enhanced"
                    )
                    
                    ai_predictions.append({
                        'match': f"{match['home_team']} vs {match['away_team']}",
                        'prediction': ai_prediction['result_prediction'],
                        'confidence': ai_prediction['confidence'],
                        'ai_powered': ai_prediction.get('ai_powered', False)
                    })
                    
                except Exception as match_error:
                    logger.error(f"⏹️ Maç tahmin hatası: {match_error}")
                    continue
            
            logger.info(f"🤖 AI tahminleri tamamlandı: {len(ai_predictions)} maç")
            
            # AI modelini güncelle (öğrenme)
            try:
                ai_predictor.real_time_adaptation()
                logger.info("🔧 AI gerçek zamanlı adaptasyon tamamlandı")
            except Exception as adaptation_error:
                logger.warning(f"🔧 AI adaptasyon hatası: {adaptation_error}")
            
            # 30 dakika bekle
            await asyncio.sleep(30 * 60)
            
        except Exception as e:
            logger.error(f"❌ Periyodik veri güncelleme hatası: {e}")
            await asyncio.sleep(5 * 60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 FastAPI uygulaması başlatılıyor...")
    
    # Sistem başlatma
    if not initialize_system():
        logger.error("❌ Sistem başlatılamadı!")
    
    # Arka plan görevini başlat
    global background_task
    background_task = asyncio.create_task(periodic_data_update())
    
    yield
    
    # Shutdown
    logger.info("🛑 FastAPI uygulaması kapatılıyor...")
    if background_task:
        background_task.cancel()
        try:
            await background_task
        except asyncio.CancelledError:
            pass

# FastAPI app oluştur - lifespan context manager ile
app = FastAPI(
    title="Futbol Tahmin AI API",
    description="Süper Öğrenen Yapay Zeka ile Render'da çalışan tahmin sistemi.",
    version="2.0.0",
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
            ai_badge = " 🤖" if pred.get('ai_powered', False) else ""
            certainty_badge = f" 🎯{pred.get('certainty', 0.5):.1f}" if pred.get('certainty') else ""
            
            predictions_html += f"""
            <div class="match-card {'ai-powered' if pred.get('ai_powered') else ''}">
                <h3>{pred['mac']}{ai_badge}{certainty_badge}</h3>
                <p><strong>Lig:</strong> {pred['lig']} | <strong>Tarih:</strong> {pred['tarih']}</p>
                <p><strong>MS Tahmin:</strong> {pred['tahmin_ms']} | <strong>İY Tahmin:</strong> {pred['tahmin_iy']}</p>
                <p><strong>Skor Tahmini:</strong> {pred['skor']} | <strong>Güven:</strong> <span class="confidence">{pred['guven']:.1f}%</span></p>
            </div>
            """
    else:
        predictions_html = "<p>Henüz maç verisi bulunmamaktadır.</p>"
    
    error_html = f"<div class='error'><strong>Hata:</strong> {error_message}</div>" if error_message else ""
    
    # AI istatistikleri
    ai_stats_html = ""
    if stats.get('ai_performance'):
        ai_perf = stats['ai_performance']
        ai_stats_html = f"""
        <div class="ai-stats">
            <h4>🤖 AI Performansı</h4>
            <p>Doğruluk: <strong>{ai_perf.get('current_accuracy', 0)*100:.1f}%</strong> | 
               Eğitim Örnekleri: <strong>{ai_perf.get('training_samples', 0)}</strong> | 
               Durum: <strong>{ai_perf.get('adaptation_status', 'unknown')}</strong></p>
        </div>
        """
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🤖 Predicta AI: Futbol Tahminleri</title>
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
            .ai-stats {{
                background: #d4edda;
                color: #155724;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                border-left: 5px solid #28a745;
            }}
            .match-card {{ 
                background: #f8f9fa; 
                margin: 15px 0; 
                padding: 20px; 
                border-radius: 10px;
                border-left: 5px solid #3498db;
                transition: transform 0.2s;
            }}
            .match-card.ai-powered {{
                border-left: 5px solid #28a745;
                background: #f0fff4;
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
            .nav {{
                display: flex;
                gap: 10px;
                margin: 20px 0;
            }}
            .nav a {{
                padding: 10px 20px;
                background: #3498db;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                transition: background 0.3s;
            }}
            .nav a:hover {{
                background: #2980b9;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Predicta AI</h1>
                <p>Süper Öğrenen Yapay Zeka ile Futbol Tahminleri</p>
                <p><em>Son Güncelleme: {datetime.now().strftime('%d.%m.%Y %H:%M')}</em></p>
            </div>
            
            <div class="nav">
                <a href="/">🏠 Ana Sayfa</a>
                <a href="/ai/predictions">🤖 AI Tahminler</a>
                <a href="/api/ai/performance">📊 AI Performans</a>
            </div>
            
            {error_html}
            
            <div class="stats">
                <h3>📊 İstatistikler</h3>
                <p>Toplam Maç: <strong>{stats.get('total_matches', 0)}</strong> | 
                   AI Maçlar: <strong>{stats.get('ai_matches', 0)}</strong> | 
                   Ortalama Güven: <strong>{stats.get('average_confidence', 0):.1f}%</strong></p>
            </div>
            
            {ai_stats_html}
            
            <div id="predictions">
                <h3>🎯 Maç Tahminleri</h3>
                {predictions_html}
            </div>
            
            <div class="info">
                <p><strong>🤖 AI Sistemi Aktif:</strong> Yeşil çerçeveli kartlar AI tarafından üretilmiştir.</p>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# --- ANA SAYFA ROTASI (index.html) ---

@app.get("/", response_class=HTMLResponse)
async def root(request: Request, min_confidence: Optional[float] = None):
    global predictor, ai_predictor
    
    # URL parametresinden min_confidence'ı al, yoksa varsayılan 50 kullan
    confidence_threshold = min_confidence if min_confidence is not None else 50.0
    
    if not predictor or not ai_predictor:
        return create_fallback_html(
            request, 
            [], 
            {}, 
            "Sistem başlatılıyor. Lütfen 2-3 dakika bekleyin..."
        )

    try:
        # Veritabanından belirlenen güven seviyesine göre tahminleri getir
        recent_predictions = db_manager.get_latest_predictions_from_db(min_confidence=confidence_threshold)
        
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
                    "guven": pred.get('confidence', 50.0),
                    "ai_powered": pred.get('source') == 'ai_enhanced',
                    "certainty": pred.get('certainty_index', 0.5)
                })
        else:
            # Eğer veritabanında tahmin yoksa, fresh data çekmeyi dene
            try:
                matches = predictor.fetch_nesine_matches()
                
                for match in matches[:10]:  # Sadece ilk 10 maçı işle
                    # AI ile tahmin yap
                    prediction = ai_predictor.predict_with_confidence(match)
                    
                    if prediction.get('confidence', 0) >= confidence_threshold:
                        predictions_data.append({
                            "mac": f"{match.get('home_team')} - {match.get('away_team')}",
                            "lig": match.get('league', 'Bilinmeyen'),
                            "tarih": f"{match.get('date')} {match.get('time')}",
                            "tahmin_ms": prediction.get('result_prediction', 'X'),
                            "tahmin_iy": prediction.get('iy_prediction', 'X'),
                            "skor": prediction.get('score_prediction', '1-1'),
                            "guven": prediction.get('confidence', 50.0),
                            "ai_powered": prediction.get('ai_powered', False),
                            "certainty": prediction.get('certainty_index', 0.5)
                        })
            except Exception as fresh_error:
                logger.warning(f"Fresh data çekme hatası: {fresh_error}")

        # İstatistikleri hesapla
        if predictions_data:
            stats = {
                "total_matches": len(predictions_data),
                "ai_matches": len([p for p in predictions_data if p['ai_powered']]),
                "average_confidence": np.mean([p['guven'] for p in predictions_data]),
                "current_threshold": confidence_threshold,
                "ai_performance": ai_predictor.get_detailed_performance() if ai_predictor else {}
            }
        else:
            stats = {
                "total_matches": 0,
                "ai_matches": 0,
                "average_confidence": 0,
                "current_threshold": confidence_threshold,
                "ai_performance": {}
            }

        # Önce template'i deneyelim
        try:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "title": "🤖 Predicta AI: Güncel Maç Tahminleri",
                    "predictions": predictions_data,
                    "stats": stats,
                    "current_time": datetime.now(),
                    "error_message": None,
                    "confidence_threshold": confidence_threshold,
                    "ai_performance": stats.get('ai_performance', {})
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

# --- YENİ: AI ÖZELLİKLİ ROTALAR ---

@app.get("/ai/predictions", response_class=HTMLResponse)
async def ai_predictions(request: Request, min_confidence: float = 60.0, ai_only: bool = False):
    """AI tahminlerini gösteren özel sayfa"""
    global ai_predictor
    
    if not ai_predictor:
        return create_fallback_html(
            request, 
            [], 
            {}, 
            "🤖 AI sistemi başlatılıyor. Lütfen bekleyin..."
        )

    try:
        # AI tahminlerini getir
        predictions_data = []
        
        # Veritabanından AI tahminlerini al
        recent_predictions = db_manager.get_latest_predictions_from_db(min_confidence=min_confidence)
        
        for pred in recent_predictions:
            is_ai = pred.get('source') == 'ai_enhanced'
            
            # Eğer sadece AI tahminleri isteniyorsa ve bu AI değilse atla
            if ai_only and not is_ai:
                continue
                
            predictions_data.append({
                "mac": f"{pred.get('home_team', 'N/A')} - {pred.get('away_team', 'N/A')}",
                "lig": pred.get('league', 'Veritabanı'),
                "tarih": pred.get('created_at', '')[:16] if pred.get('created_at') else '',
                "tahmin_ms": pred.get('prediction_result', 'X'),
                "tahmin_iy": pred.get('prediction_iy', 'X'),
                "skor": pred.get('predicted_score', '1-1'),
                "guven": pred.get('confidence', 50.0),
                "ai_powered": is_ai,
                "certainty": pred.get('certainty_index', 0.5)
            })

        # İstatistikler
        stats = {
            "total_matches": len(predictions_data),
            "ai_matches": len([p for p in predictions_data if p['ai_powered']]),
            "average_confidence": np.mean([p['guven'] for p in predictions_data]) if predictions_data else 0,
            "current_threshold": min_confidence,
            "ai_performance": ai_predictor.get_detailed_performance()
        }

        # Template render
        try:
            return templates.TemplateResponse(
                "ai_predictions.html",
                {
                    "request": request,
                    "title": "🤖 AI Tahminleri",
                    "predictions": predictions_data,
                    "stats": stats,
                    "ai_performance": stats.get('ai_performance', {}),
                    "ai_only": ai_only,
                    "confidence_threshold": min_confidence
                }
            )
        except Exception as template_error:
            logger.warning(f"AI template bulunamadı, fallback kullanılıyor: {template_error}")
            return create_fallback_html(request, predictions_data, stats)

    except Exception as e:
        logger.error(f"AI tahmin sayfası hatası: {e}")
        return create_fallback_html(request, [], {}, f"AI sistemi hatası: {str(e)}")

@app.get("/api/ai/performance", response_model=AIPerformanceModel)
async def get_ai_performance():
    """AI performans metriklerini JSON olarak döndür"""
    global ai_predictor
    
    if not ai_predictor:
        raise HTTPException(status_code=503, detail="AI sistemi henüz başlatılmadı")
    
    try:
        performance = ai_predictor.get_detailed_performance()
        return AIPerformanceModel(**performance)
    except Exception as e:
        logger.error(f"AI performans API hatası: {e}")
        raise HTTPException(status_code=500, detail="AI performans verileri alınamadı")

@app.post("/api/ai/retrain")
async def retrain_ai_models():
    """AI modellerini manuel olarak yeniden eğit"""
    global ai_predictor
    
    if not ai_predictor:
        raise HTTPException(status_code=503, detail="AI sistemi henüz başlatılmadı")
    
    try:
        success = ai_predictor.train_models()
        if success:
            return {"status": "success", "message": "AI modelleri başarıyla yeniden eğitildi"}
        else:
            return {"status": "error", "message": "AI eğitimi başarısız oldu"}
    except Exception as e:
        logger.error(f"AI yeniden eğitim hatası: {e}")
        raise HTTPException(status_code=500, detail=f"AI eğitim hatası: {str(e)}")

@app.get("/api/ai/predict/{home_team}/{away_team}")
async def predict_specific_match(home_team: str, away_team: str):
    """Belirli bir maç için AI tahmini yap"""
    global ai_predictor, predictor
    
    if not ai_predictor or not predictor:
        raise HTTPException(status_code=503, detail="Sistem henüz başlatılmadı")
    
    try:
        # Maç verisi oluştur
        match_data = {
            'home_team': home_team,
            'away_team': away_team,
            'league': 'Özel Maç',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': '20:00',
            'odds': {'1': 2.0, 'X': 3.0, '2': 3.5},
            'home_stats': predictor._generate_team_stats(home_team, 'Özel Lig'),
            'away_stats': predictor._generate_team_stats(away_team, 'Özel Lig'),
            'importance': 1.0,
            'weather_impact': 1.0
        }
        
        # AI tahmini yap
        prediction = ai_predictor.predict_with_confidence(match_data)
        
        return {
            "match": f"{home_team} vs {away_team}",
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Özel maç tahmin hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")

# --- MEVCUT ROTALAR (GÜNCELLENMİŞ) ---

@app.get("/confidence/{confidence_level}", response_class=HTMLResponse)
async def get_predictions_by_confidence(request: Request, confidence_level: float):
    """Belirli güven seviyesindeki tahminleri getir"""
    if confidence_level < 0 or confidence_level > 100:
        raise HTTPException(status_code=400, detail="Güven seviyesi 0-100 arasında olmalıdır")
    
    return await root(request, min_confidence=confidence_level)

@app.get("/matches/predictions", response_model=List[MatchWithPrediction])
async def get_matches_json(min_confidence: float = 60.0):
    """API: Yüksek güvenli tahminleri JSON olarak döndürür."""
    global ai_predictor
    
    if not ai_predictor:
        raise HTTPException(status_code=503, detail="Sistem henüz başlatılıyor. Lütfen bekleyin.")
    
    try:
        # Veritabanından tahminleri getir
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
                        odds={"1": 2.0, "X": 3.0, "2": 3.5}
                    ),
                    prediction=PredictionModel(
                        result_prediction=pred.get('prediction_result', 'X'),
                        confidence=pred.get('confidence', 50.0),
                        iy_prediction=pred.get('prediction_iy', 'X'),
                        score_prediction=pred.get('predicted_score', '1-1'),
                        ai_powered=pred.get('source') == 'ai_enhanced',
                        certainty_index=pred.get('certainty_index', 0.5)
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

# --- Sağlık Kontrolü (GÜNCELLENMİŞ) ---

@app.get("/health")
async def health_check():
    """Sistem sağlık kontrolü"""
    ai_status = "active" if ai_predictor and ai_predictor.models_trained() else "inactive"
    
    return {
        "status": "healthy" if predictor else "starting",
        "ai_status": ai_status,
        "timestamp": datetime.now().isoformat(),
        "db_connected": db_manager is not None,
        "template_dir": template_dir,
        "version": "2.0.0-ai"
    }

# --- Lokal Test Başlatma ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
