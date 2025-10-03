#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREDICTA AI - v5.0 İyileştirilmiş Backend
Improved Prediction Engine ile entegre
"""
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uvicorn
import requests
from bs4 import BeautifulSoup
import re
import random

# Yeni tahmin motorunu import et
from improved_prediction_engine import ImprovedPredictionEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Predicta AI API",
    version="5.0",
    description="Gelişmiş hybrid tahmin motoru ile futbol analizi"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AdvancedNesineFetcher:
    """
    Nesine.com'dan maç verisi çeken sınıf
    1. Önce CDN API'yi dener
    2. Başarısız olursa HTML scraping yapar
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.nesine.com/',
            'Connection': 'keep-alive'
        }
        self.base_url = "https://www.nesine.com"
    
    def get_nesine_official_api(self):
        """
        Nesine'nin resmi CDN API'sinden veri çeker
        """
        try:
            api_url = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"
            
            response = self.session.get(
                api_url, 
                headers=self.headers, 
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info("Nesine CDN API başarılı")
                return self._parse_nesine_api_response(data)
            else:
                logger.warning(f"CDN API yanıt kodu: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"CDN API hatası: {e}")
            return None
    
    def _parse_nesine_api_response(self, data):
        """
        Nesine API yanıtını parse eder ve maç listesi döndürür
        """
        matches = []
        
        # EA: Erken Açılan (Prematch)
        ea_matches = data.get("sg", {}).get("EA", [])
        logger.info(f"Prematch maç sayısı: {len(ea_matches)}")
        
        for m in ea_matches:
            # Sadece futbol maçları (GT=1)
            if m.get("GT") != 1:
                continue
            
            match_data = self._format_nesine_match(m)
            if match_data and self._is_valid_match(match_data):
                matches.append(match_data)
        
        # CA: Canlı Maçlar
        ca_matches = data.get("sg", {}).get("CA", [])
        logger.info(f"Canlı maç sayısı: {len(ca_matches)}")
        
        for m in ca_matches:
            if m.get("GT") != 1:
                continue
            
            match_data = self._format_nesine_match(m)
            if match_data and self._is_valid_match(match_data):
                matches.append(match_data)
        
        logger.info(f"Toplam geçerli maç: {len(matches)}")
        return matches
    
    def _format_nesine_match(self, m):
        """
        Nesine API'den gelen match objesini standart formata çevirir
        """
        try:
            home_team = m.get("HN", "").strip()
            away_team = m.get("AN", "").strip()
            
            if not home_team or not away_team:
                return None
            
            # 1X2 oranlarını bul
            odds = {'1': 2.0, 'X': 3.0, '2': 3.5}  # Varsayılan
            
            for bahis in m.get("MA", []):
                # MTID 1 = Maç Sonucu (1X2)
                if bahis.get("MTID") == 1:
                    oranlar = bahis.get("OCA", [])
                    if len(oranlar) >= 3:
                        try:
                            odds['1'] = float(oranlar[0].get("O", 2.0))
                            odds['X'] = float(oranlar[1].get("O", 3.0))
                            odds['2'] = float(oranlar[2].get("O", 3.5))
                        except (ValueError, TypeError):
                            pass
                    break
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'league': m.get("LC", "Bilinmeyen Lig"),
                'league_id': m.get("LID", ""),
                'match_id': m.get("C", ""),
                'date': m.get("D", ""),
                'time': m.get("T", "20:00"),
                'odds': odds,
                'is_live': m.get("S") == 1  # 1 = canlı
            }
        except Exception as e:
            logger.debug(f"Match format hatası: {e}")
            return None
    
    def _is_valid_match(self, match):
        """
        Maçın geçerli olup olmadığını kontrol eder
        UI elementleri ve geçersiz isimleri filtreler
        """
        if not match:
            return False
        
        home = match['home_team'].lower()
        away = match['away_team'].lower()
        
        # UI elementlerini filtrele
        blacklist = [
            'hesabim', 'mesaj', 'cep tel', 'tekrar', 'tum talep',
            'menu', 'profil', 'ayarlar', 'favoriler', 'yardim',
            'iletisim', 'giris', 'kayit', 'casino', 'canli bahis'
        ]
        
        for word in blacklist:
            if word in home or word in away:
                return False
        
        # Aynı takım kontrolü
        if home == away:
            return False
        
        # Çok kısa isimler
        if len(home) < 3 or len(away) < 3:
            return False
        
        # Çok uzun isimler (muhtemelen hata)
        if len(home) > 50 or len(away) > 50:
            return False
        
        return True
    
    def get_page_content(self, url_path="/iddaa"):
        """
        Fallback: HTML sayfasını çeker
        """
        try:
            url = f"{self.base_url}{url_path}"
            response = self.session.get(url, headers=self.headers, timeout=20)
            response.raise_for_status()
            logger.info(f"HTML çekildi: {len(response.text)} karakter")
            return response.text
        except Exception as e:
            logger.error(f"HTML çekme hatası: {e}")
            return None
    
    def extract_matches_from_html(self, html_content):
        """
        HTML'den maç bilgilerini çıkarır (yedek yöntem)
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        matches = []
        seen = set()
        
        # Tüm metni al
        text = soup.get_text()
        
        # "TakımA vs TakımB" veya "TakımA - TakımB" formatını bul
        pattern = r'([A-Za-zÇçĞğİıÖöŞşÜü\s\.]{3,40})\s+(?:vs\.?|-)\s+([A-Za-zÇçĞğİıÖöŞşÜü\s\.]{3,40})'
        
        for match in re.finditer(pattern, text):
            home = match.group(1).strip()
            away = match.group(2).strip()
            
            key = f"{home}|{away}"
            if key in seen:
                continue
            
            match_dict = {
                'home_team': home,
                'away_team': away
            }
            
            if self._is_valid_match(match_dict):
                seen.add(key)
                matches.append({
                    'home_team': home,
                    'away_team': away,
                    'league': self._detect_league(home, away),
                    'odds': self._generate_realistic_odds(),
                    'match_id': '',
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'time': '20:00',
                    'is_live': False
                })
        
        logger.info(f"HTML'den {len(matches)} maç bulundu")
        return matches[:50]
    
    def _detect_league(self, home_team, away_team):
        """
        Takım isimlerinden lig tespit eder
        """
        text = f"{home_team} {away_team}".lower()
        
        leagues = {
            'Süper Lig': ['galatasaray', 'fenerbahce', 'besiktas', 'trabzonspor',
                         'basaksehir', 'sivasspor', 'konyaspor', 'alanyaspor'],
            'Premier League': ['arsenal', 'chelsea', 'liverpool', 'manchester', 
                              'city', 'united', 'tottenham', 'everton', 'leicester'],
            'La Liga': ['barcelona', 'madrid', 'atletico', 'sevilla', 'valencia',
                       'villarreal', 'athletic', 'betis', 'sociedad'],
            'Bundesliga': ['bayern', 'dortmund', 'leipzig', 'leverkusen',
                          'frankfurt', 'union', 'gladbach', 'wolfsburg'],
            'Serie A': ['milan', 'inter', 'juventus', 'roma', 'napoli',
                       'lazio', 'atalanta', 'fiorentina'],
            'Ligue 1': ['psg', 'marseille', 'lyon', 'monaco', 'lille', 'rennes'],
            'Eredivisie': ['ajax', 'psv', 'feyenoord', 'utrecht', 'alkmaar'],
            'Primeira Liga': ['porto', 'benfica', 'sporting', 'braga']
        }
        
        for league, keywords in leagues.items():
            if any(kw in text for kw in keywords):
                return league
        
        return 'Diğer Ligler'
    
    def _generate_realistic_odds(self):
        """
        Gerçekçi oranlar üretir
        """
        return {
            '1': round(random.uniform(1.8, 3.5), 2),
            'X': round(random.uniform(2.9, 3.7), 2),
            '2': round(random.uniform(1.8, 4.0), 2)
        }


# Global instance'lar
fetcher = AdvancedNesineFetcher()
predictor = ImprovedPredictionEngine()


# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """Ana endpoint - sistem bilgisi"""
    return {
        "status": "Predicta AI v5.0 - İyileştirilmiş Tahmin Motoru",
        "engine": "Hybrid Model (Odds + xG)",
        "data_source": "Nesine CDN API",
        "features": [
            "Overround düzeltmesi",
            "Matematiksel güven hesabı",
            "Risk değerlendirmesi",
            "Muhafazakar skor tahmini",
            "Gerçek zamanlı bülten"
        ],
        "disclaimer": "Bu tahminler eğitim amaçlıdır. Kumar bağımlılığı ciddi bir sorundur.",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health():
    """Sağlık kontrolü"""
    return {
        "status": "healthy",
        "engine": "ImprovedPredictionEngine v5.0",
        "data_source": "Nesine CDN API + HTML Fallback",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/nesine/live-predictions")
async def get_live_predictions(league: str = "all", limit: int = 50):
    """
    Ana endpoint: Nesine'den maçları çek ve gelişmiş tahmin yap
    
    Args:
        league: Lig filtresi (all, premier-league, super-lig, vb.)
        limit: Maksimum maç sayısı
    """
    try:
        logger.info(f"Maç verileri çekiliyor... (lig={league}, limit={limit})")
        
        # 1. Önce CDN API'yi dene
        matches = fetcher.get_nesine_official_api()
        source = "CDN API"
        
        # 2. API başarısızsa HTML parsing
        if not matches:
            logger.warning("API başarısız, HTML parsing deneniyor...")
            html_content = fetcher.get_page_content()
            
            if html_content:
                matches = fetcher.extract_matches_from_html(html_content)
                source = "HTML Fallback"
            else:
                matches = []
                source = "None"
        
        # 3. Lig filtresi uygula
        if league != "all" and matches:
            original_count = len(matches)
            matches = [m for m in matches 
                      if league.lower().replace('-', ' ') in m['league'].lower()]
            logger.info(f"Lig filtresi: {original_count} -> {len(matches)} maç")
        
        # 4. Her maç için gelişmiş tahmin yap
        matches_with_predictions = []
        
        for match in matches[:limit]:
            try:
                # İyileştirilmiş tahmin motoru
                prediction = predictor.predict_match(
                    home_team=match['home_team'],
                    away_team=match['away_team'],
                    odds=match['odds'],
                    league=match.get('league', 'default')
                )
                
                matches_with_predictions.append({
                    'home_team': match['home_team'],
                    'away_team': match['away_team'],
                    'league': match['league'],
                    'match_id': match.get('match_id', ''),
                    'time': match.get('time', '20:00'),
                    'date': match.get('date', datetime.now().strftime('%Y-%m-%d')),
                    'odds': match['odds'],
                    'is_live': match.get('is_live', False),
                    'ai_prediction': prediction
                })
                
            except Exception as e:
                logger.error(f"Tahmin hatası ({match['home_team']} vs {match['away_team']}): {e}")
                continue
        
        logger.info(f"{len(matches_with_predictions)} maç hazır - Kaynak: {source}")
        
        return {
            "success": True,
            "matches": matches_with_predictions,
            "count": len(matches_with_predictions),
            "source": source,
            "engine": "ImprovedPredictionEngine v5.0",
            "features": {
                "overround_correction": True,
                "confidence_calculation": True,
                "risk_assessment": True,
                "conservative_scores": True
            },
            "disclaimer": "Bu tahminler eğitim amaçlıdır. Gerçek bahis kararları için kullanmayın. Kumar bağımlılığı ciddi bir sorundur.",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Endpoint hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/nesine/matches")
async def get_matches(league: str = "all", limit: int = 50):
    """Alias endpoint - get_live_predictions ile aynı"""
    return await get_live_predictions(league, limit)


@app.get("/api/nesine/raw")
async def get_raw_nesine_data():
    """
    Ham Nesine API verisi (debug için)
    API'nin çalışıp çalışmadığını test etmek için kullanın
    """
    try:
        api_url = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"
        
        response = requests.get(
            api_url, 
            headers=fetcher.headers, 
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            
            ea_count = len(data.get("sg", {}).get("EA", []))
            ca_count = len(data.get("sg", {}).get("CA", []))
            
            # İlk maçı örnek olarak göster
            sample_match = None
            if ea_count > 0:
                sample_match = data.get("sg", {}).get("EA", [{}])[0]
            
            return {
                "success": True,
                "status_code": response.status_code,
                "prematch_count": ea_count,
                "live_count": ca_count,
                "total_matches": ea_count + ca_count,
                "sample_match": sample_match,
                "api_structure": {
                    "sg": "Sport groups",
                    "EA": "Early (Prematch)",
                    "CA": "Current (Live)",
                    "HN": "Home Name",
                    "AN": "Away Name",
                    "LC": "League Code",
                    "MA": "Market Array",
                    "MTID": "Market Type ID (1=1X2)"
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "status_code": response.status_code,
                "error": "API yanıt vermedi",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Raw API hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/debug/prediction-test")
async def test_prediction():
    """
    Tahmin motorunu farklı senaryolarla test eder
    """
    test_cases = [
        {
            "name": "Güçlü ev sahibi (düşük oran)",
            "odds": {'1': 1.50, 'X': 4.00, '2': 7.00},
            "expected": "Yüksek güven, 1 tahmini"
        },
        {
            "name": "Dengeli maç",
            "odds": {'1': 2.50, 'X': 3.20, '2': 2.80},
            "expected": "Orta güven, belirsiz sonuç"
        },
        {
            "name": "Güçlü deplasman (düşük oran)",
            "odds": {'1': 5.00, 'X': 3.80, '2': 1.65},
            "expected": "Yüksek güven, 2 tahmini"
        },
        {
            "name": "Belirsiz (yüksek oranlar)",
            "odds": {'1': 3.50, 'X': 3.30, '2': 2.20},
            "expected": "Düşük güven"
        }
    ]
    
    results = []
    
    for test in test_cases:
        prediction = predictor.predict_match(
            home_team="Test Home",
            away_team="Test Away",
            odds=test['odds'],
            league='Premier League'
        )
        
        results.append({
            "test_name": test['name'],
            "odds": test['odds'],
            "expected": test['expected'],
            "result": {
                "prediction": prediction['prediction'],
                "confidence": prediction['confidence'],
                "score": prediction['score_prediction'],
                "risk": prediction['risk_level'],
                "warning": prediction['warning'],
                "probabilities": {
                    "1": prediction['home_win_prob'],
                    "X": prediction['draw_prob'],
                    "2": prediction['away_win_prob']
                }
            }
        })
    
    return {
        "test_results": results,
        "engine": "ImprovedPredictionEngine v5.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/stats")
async def get_stats():
    """
    Sistem istatistikleri ve maç dağılımı
    """
    try:
        matches = fetcher.get_nesine_official_api()
        
        if matches:
            # Liglere göre dağılım
            leagues = {}
            for match in matches:
                league = match.get('league', 'Bilinmeyen')
                leagues[league] = leagues.get(league, 0) + 1
            
            # Canlı/prematch dağılımı
            live_count = sum(1 for m in matches if m.get('is_live', False))
            prematch_count = len(matches) - live_count
            
            return {
                "total_matches": len(matches),
                "live_matches": live_count,
                "prematch_matches": prematch_count,
                "leagues": leagues,
                "data_source": "Nesine CDN API",
                "engine": "ImprovedPredictionEngine v5.0",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "total_matches": 0,
                "error": "Veri çekilemedi",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Stats endpoint hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Predicta AI v5.0 başlatılıyor... Port: {port}")
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port, 
        log_level="info"
    )
