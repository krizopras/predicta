#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREDICTA AI - v5.3 İyileştirilmiş HTML Parser + API Debugger
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
import json

from improved_prediction_engine import ImprovedPredictionEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Predicta AI API",
    version="5.3",
    description="İyileştirilmiş HTML parser + API debugger"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FixedNesineScraper:
    """
    İyileştirilmiş Nesine scraper - Daha akıllı HTML parsing
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'tr-TR,tr;q=0.9',
            'Connection': 'keep-alive'
        }
    
    def get_matches(self):
        """Maçları çeker"""
        logger.info("Nesine verisi çekiliyor...")
        
        # 1. CDN API'yi dene ve yanıtı logla
        matches = self.try_cdn_api_debug()
        if matches and len(matches) > 5:
            logger.info(f"✅ CDN API'den {len(matches)} maç alındı")
            return matches
        
        # 2. HTML'den daha iyi parse et
        matches = self.scrape_html_improved()
        if matches and len(matches) > 5:
            logger.info(f"✅ HTML'den {len(matches)} maç alındı")
            return matches
        
        logger.error("❌ Yeterli maç bulunamadı - Demo veriler döndürülüyor")
        return self.get_demo_matches()
    
    def try_cdn_api_debug(self):
        """CDN API'yi dener ve yanıtı detaylı loglar"""
        try:
            url = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"
            
            response = self.session.get(url, headers=self.headers, timeout=10)
            
            logger.info(f"CDN API Status: {response.status_code}")
            logger.info(f"Content-Type: {response.headers.get('content-type')}")
            logger.info(f"Response length: {len(response.text)} karakter")
            
            # İlk 200 karakteri logla
            preview = response.text[:200].replace('\n', ' ')
            logger.info(f"Response preview: {preview}...")
            
            if response.status_code == 200:
                # JSON parse dene
                try:
                    data = response.json()
                    logger.info(f"JSON keys: {list(data.keys())}")
                    
                    # Veri yapısını kontrol et
                    if 'sg' in data:
                        ea = data.get('sg', {}).get('EA', [])
                        ca = data.get('sg', {}).get('CA', [])
                        logger.info(f"EA (Prematch): {len(ea)} maç")
                        logger.info(f"CA (Live): {len(ca)} maç")
                        
                        # İlk maçı logla
                        if ea:
                            first_match = ea[0]
                            logger.info(f"İlk maç örneği: {first_match.get('HN')} vs {first_match.get('AN')}")
                        
                        # Parse et
                        matches = self.parse_nesine_json(data)
                        return matches if matches else None
                    else:
                        logger.warning("JSON'da 'sg' key'i yok")
                        return None
                        
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse hatası: {e}")
                    logger.error(f"Ham yanıt ilk 500 karakter: {response.text[:500]}")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"CDN API hatası: {e}")
            return None
    
    def parse_nesine_json(self, data):
        """Nesine JSON'ını parse eder"""
        matches = []
        
        try:
            ea_matches = data.get('sg', {}).get('EA', [])
            ca_matches = data.get('sg', {}).get('CA', [])
            
            for m in ea_matches + ca_matches:
                # Sadece futbol (GT=1)
                if m.get('GT') != 1:
                    continue
                
                home = (m.get('HN') or '').strip()
                away = (m.get('AN') or '').strip()
                
                if not self.is_valid_match(home, away):
                    continue
                
                # Oranları bul
                odds = {'1': 2.0, 'X': 3.0, '2': 3.5}
                
                for bahis in m.get('MA', []):
                    if bahis.get('MTID') == 1:  # 1X2
                        oranlar = bahis.get('OCA', [])
                        if len(oranlar) >= 3:
                            try:
                                odds['1'] = float(oranlar[0].get('O', 2.0))
                                odds['X'] = float(oranlar[1].get('O', 3.0))
                                odds['2'] = float(oranlar[2].get('O', 3.5))
                            except:
                                pass
                        break
                
                matches.append({
                    'home_team': home,
                    'away_team': away,
                    'league': m.get('LC', 'Bilinmeyen'),
                    'odds': odds,
                    'time': m.get('T', '20:00'),
                    'date': m.get('D', datetime.now().strftime('%Y-%m-%d')),
                    'is_live': m.get('S') == 1
                })
            
            return matches if len(matches) > 5 else None
            
        except Exception as e:
            logger.error(f"JSON parse hatası: {e}")
            return None
    
    def scrape_html_improved(self):
        """İyileştirilmiş HTML scraping"""
        try:
            url = "https://www.nesine.com/iddaa"
            response = self.session.get(url, headers=self.headers, timeout=15)
            
            if response.status_code != 200:
                return None
            
            logger.info(f"HTML alındı: {len(response.text)} karakter")
            
            soup = BeautifulSoup(response.text, 'html.parser')
            matches = []
            seen = set()
            
            # Yöntem 1: Script tag'lerinde JSON ara
            for script in soup.find_all('script'):
                script_text = script.string or ''
                
                # JSON pattern ara
                json_patterns = [
                    r'window\.__INITIAL_STATE__\s*=\s*({.*?});',
                    r'window\.APP_DATA\s*=\s*({.*?});',
                    r'var\s+matches\s*=\s*(\[.*?\]);'
                ]
                
                for pattern in json_patterns:
                    match = re.search(pattern, script_text, re.DOTALL)
                    if match:
                        try:
                            json_data = json.loads(match.group(1))
                            logger.info(f"Script'te JSON bulundu: {list(json_data.keys())[:5]}")
                            # Recursive olarak maç bilgisi ara
                            found_matches = self.extract_matches_from_object(json_data)
                            if found_matches:
                                matches.extend(found_matches)
                        except:
                            pass
            
            # Yöntem 2: Tablolardan çek
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    text = ' '.join(cell.get_text(strip=True) for cell in cells)
                    
                    # Maç pattern'i
                    team_match = re.search(r'([A-ZÇĞİÖŞÜ][a-zçğıöşü\s\.]{2,25})\s+(?:-|vs)\s+([A-ZÇĞİÖŞÜ][a-zçğıöşü\s\.]{2,25})', text)
                    if team_match:
                        home = team_match.group(1).strip()
                        away = team_match.group(2).strip()
                        
                        if self.is_valid_match(home, away):
                            key = f"{home}|{away}"
                            if key not in seen:
                                seen.add(key)
                                matches.append({
                                    'home_team': home,
                                    'away_team': away,
                                    'league': self.detect_league(home, away),
                                    'odds': self.generate_odds(),
                                    'time': '20:00',
                                    'date': datetime.now().strftime('%Y-%m-%d'),
                                    'is_live': False
                                })
            
            # Yöntem 3: Div/span elementlerinde ara
            team_elements = soup.find_all(['div', 'span'], class_=re.compile(r'team|match|event', re.I))
            
            for i in range(len(team_elements) - 1):
                home_text = team_elements[i].get_text(strip=True)
                away_text = team_elements[i + 1].get_text(strip=True)
                
                if self.is_valid_match(home_text, away_text):
                    key = f"{home_text}|{away_text}"
                    if key not in seen:
                        seen.add(key)
                        matches.append({
                            'home_team': home_text,
                            'away_team': away_text,
                            'league': self.detect_league(home_text, away_text),
                            'odds': self.generate_odds(),
                            'time': '20:00',
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'is_live': False
                        })
            
            logger.info(f"HTML'den {len(matches)} potansiyel maç bulundu")
            return matches if len(matches) > 5 else None
            
        except Exception as e:
            logger.error(f"HTML scraping hatası: {e}")
            return None
    
    def extract_matches_from_object(self, obj, depth=0):
        """JSON objesinden recursive olarak maç bilgisi çıkarır"""
        if depth > 5:  # Sonsuz loop önleme
            return []
        
        matches = []
        
        if isinstance(obj, dict):
            # Maç benzeri key'ler ara
            if all(k in obj for k in ['home', 'away']):
                home = str(obj.get('home', ''))
                away = str(obj.get('away', ''))
                if self.is_valid_match(home, away):
                    matches.append({
                        'home_team': home,
                        'away_team': away,
                        'league': obj.get('league', 'Bilinmeyen'),
                        'odds': obj.get('odds', self.generate_odds()),
                        'time': obj.get('time', '20:00'),
                        'date': obj.get('date', datetime.now().strftime('%Y-%m-%d')),
                        'is_live': obj.get('is_live', False)
                    })
            
            # Recursive devam et
            for value in obj.values():
                matches.extend(self.extract_matches_from_object(value, depth + 1))
        
        elif isinstance(obj, list):
            for item in obj:
                matches.extend(self.extract_matches_from_object(item, depth + 1))
        
        return matches
    
    def is_valid_match(self, home, away):
        """Geçerli maç mı kontrol eder"""
        if not home or not away:
            return False
        
        # Uzunluk kontrol
        if len(home) < 3 or len(away) < 3:
            return False
        if len(home) > 40 or len(away) > 40:
            return False
        
        # Aynı takım
        if home.lower() == away.lower():
            return False
        
        # Blacklist - UI elementleri
        blacklist = [
            'hesabim', 'menu', 'giris', 'kayit', 'profil', 'mesaj', 'tel',
            'casino', 'canli', 'yardim', 'iletisim', 'kupon', 'bahis',
            'iddaa', 'nesine', 'oyna', 'kazanan', 'tutturan', 'cep',
            'tekrar', 'tum', 'talep', 'favori', 'ayarlar', 'bildir',
            'button', 'link', 'click', 'ara', 'bul', 'yukle'
        ]
        
        home_lower = home.lower()
        away_lower = away.lower()
        
        for word in blacklist:
            if word in home_lower or word in away_lower:
                return False
        
        # En az bir harf içermeli
        if not re.search(r'[a-zA-ZçğıöşüÇĞİÖŞÜ]', home) or not re.search(r'[a-zA-ZçğıöşüÇĞİÖŞÜ]', away):
            return False
        
        return True
    
    def detect_league(self, home, away):
        """Takım isimlerinden lig tahmin eder"""
        text = f"{home} {away}".lower()
        
        leagues = {
            'Süper Lig': ['galatasaray', 'fenerbahçe', 'beşiktaş', 'trabzonspor', 'başakşehir', 'sivasspor'],
            'Premier League': ['arsenal', 'chelsea', 'liverpool', 'manchester', 'city', 'united', 'spurs', 'everton'],
            'La Liga': ['barcelona', 'madrid', 'real', 'atletico', 'sevilla', 'valencia', 'villarreal'],
            'Bundesliga': ['bayern', 'dortmund', 'leipzig', 'leverkusen', 'frankfurt', 'wolfsburg'],
            'Serie A': ['milan', 'inter', 'juventus', 'roma', 'napoli', 'lazio', 'atalanta'],
            'Ligue 1': ['psg', 'marseille', 'lyon', 'monaco', 'lille', 'nice']
        }
        
        for league, keywords in leagues.items():
            if any(kw in text for kw in keywords):
                return league
        
        return 'Diğer Ligler'
    
    def generate_odds(self):
        """Gerçekçi oranlar üretir"""
        import random
        return {
            '1': round(random.uniform(1.80, 3.40), 2),
            'X': round(random.uniform(2.90, 3.70), 2),
            '2': round(random.uniform(1.80, 4.20), 2)
        }
    
    def get_demo_matches(self):
        """Gerçek takımlarla demo maçlar"""
        import random
        from datetime import timedelta
        
        realistic_matches = [
            ('Arsenal', 'Chelsea', 'Premier League'),
            ('Liverpool', 'Manchester City', 'Premier League'),
            ('Manchester United', 'Tottenham', 'Premier League'),
            ('Barcelona', 'Real Madrid', 'La Liga'),
            ('Atletico Madrid', 'Sevilla', 'La Liga'),
            ('Bayern Munich', 'Borussia Dortmund', 'Bundesliga'),
            ('Juventus', 'Inter Milan', 'Serie A'),
            ('AC Milan', 'Napoli', 'Serie A'),
            ('PSG', 'Marseille', 'Ligue 1'),
            ('Galatasaray', 'Fenerbahçe', 'Süper Lig'),
            ('Beşiktaş', 'Trabzonspor', 'Süper Lig'),
            ('Ajax', 'PSV', 'Eredivisie'),
            ('Porto', 'Benfica', 'Primeira Liga'),
            ('Celtic', 'Rangers', 'Scottish Premiership'),
            ('Leicester', 'Aston Villa', 'Premier League'),
            ('Real Sociedad', 'Athletic Bilbao', 'La Liga'),
            ('RB Leipzig', 'Bayer Leverkusen', 'Bundesliga'),
            ('Roma', 'Lazio', 'Serie A'),
            ('Lyon', 'Monaco', 'Ligue 1'),
            ('Başakşehir', 'Sivasspor', 'Süper Lig')
        ]
        
        matches = []
        random.shuffle(realistic_matches)
        
        for i, (home, away, league) in enumerate(realistic_matches[:15]):
            hour = 18 + (i % 6)
            minute = random.choice([0, 15, 30, 45])
            
            matches.append({
                'home_team': home,
                'away_team': away,
                'league': league,
                'odds': self.generate_odds(),
                'time': f"{hour:02d}:{minute:02d}",
                'date': (datetime.now() + timedelta(days=i % 3)).strftime('%Y-%m-%d'),
                'is_live': False
            })
        
        logger.info(f"✅ {len(matches)} demo maç oluşturuldu")
        return matches


# Global instances
scraper = FixedNesineScraper()
predictor = ImprovedPredictionEngine()


@app.get("/")
async def root():
    return {
        "status": "Predicta AI v5.3 - İyileştirilmiş HTML Parser",
        "engine": "Hybrid Model (Odds + xG)",
        "data_source": "Nesine CDN API + Advanced HTML Scraping + Demo Fallback",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "5.3"}


@app.get("/api/nesine/live-predictions")
async def get_live_predictions(league: str = "all", limit: int = 50):
    """Ana endpoint - Maçları çeker ve tahmin yapar"""
    try:
        logger.info(f"Maç verileri çekiliyor... (lig={league}, limit={limit})")
        
        matches = scraper.get_matches()
        
        if not matches:
            return {
                "success": False,
                "error": "Maç verisi çekilemedi",
                "matches": [],
                "count": 0
            }
        
        # Lig filtresi
        if league != "all":
            matches = [m for m in matches 
                      if league.lower().replace('-', ' ') in m['league'].lower()]
        
        # Tahmin yap
        matches_with_predictions = []
        
        for match in matches[:limit]:
            try:
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
                    'time': match.get('time', '20:00'),
                    'date': match.get('date', datetime.now().strftime('%Y-%m-%d')),
                    'odds': match['odds'],
                    'is_live': match.get('is_live', False),
                    'ai_prediction': prediction
                })
                
            except Exception as e:
                logger.error(f"Tahmin hatası: {e}")
                continue
        
        logger.info(f"✅ {len(matches_with_predictions)} maç hazır")
        
        return {
            "success": True,
            "matches": matches_with_predictions,
            "count": len(matches_with_predictions),
            "source": "Nesine (Gerçek/Demo)",
            "engine": "ImprovedPredictionEngine v5.0",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Endpoint hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/debug/cdn-api")
async def debug_cdn_api():
    """CDN API'yi debug eder"""
    try:
        url = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"
        response = requests.get(url, timeout=10)
        
        return {
            "status_code": response.status_code,
            "content_type": response.headers.get('content-type'),
            "response_length": len(response.text),
            "response_preview": response.text[:500],
            "is_json": response.headers.get('content-type', '').startswith('application/json'),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Predicta AI v5.3 başlatılıyor... Port: {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
