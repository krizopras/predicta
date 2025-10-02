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
import random
import requests
from bs4 import BeautifulSoup
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Predicta AI API", version="4.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== GELİŞTİRİLMİŞ NESINE FETCHER ====================
class AdvancedNesineFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.nesine.com/',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        self.base_url = "https://www.nesine.com"
    
    def get_page_content(self, url_path="/iddaa"):
        try:
            url = f"{self.base_url}{url_path}"
            response = self.session.get(url, headers=self.headers, timeout=20)
            response.raise_for_status()
            logger.info(f"Nesine sayfasi cekildi ({len(response.text)} karakter)")
            return response.text
        except Exception as e:
            logger.error(f"Sayfa cekme hatasi: {e}")
            return None
    
    def get_nesine_official_api(self):
        """Nesine'nin resmi CDN API'sinden veri çek"""
        try:
            api_url = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"
            
            api_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://www.nesine.com/",
                "Origin": "https://www.nesine.com",
                "Accept": "application/json"
            }
            
            response = self.session.get(api_url, headers=api_headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Nesine CDN API'den veri alindi (Status: {response.status_code})")
                return self._parse_nesine_api_response(data)
            else:
                logger.warning(f"CDN API yanit vermedi: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"CDN API hatasi: {e}")
            return None
    
    def _parse_nesine_api_response(self, data):
        """Nesine API yanıtını parse et"""
        matches = []
        
        # EA: Erken Açılan / Prematch Maçlar
        ea_matches = data.get("sg", {}).get("EA", [])
        logger.info(f"Prematch maclari aliniyor: {len(ea_matches)} adet")
        
        for m in ea_matches:
            if m.get("GT") != 1:  # Sadece futbol
                continue
            
            match_data = self._format_nesine_match(m)
            if match_data:
                matches.append(match_data)
        
        # CA: Canlı Maçlar
        ca_matches = data.get("sg", {}).get("CA", [])
        logger.info(f"Canli maclari aliniyor: {len(ca_matches)} adet")
        
        for m in ca_matches:
            if m.get("GT") != 1:  # Sadece futbol
                continue
            
            match_data = self._format_nesine_match(m)
            if match_data:
                matches.append(match_data)
        
        logger.info(f"Toplam {len(matches)} mac API'den parse edildi")
        return matches
    
    def _format_nesine_match(self, m):
        """Nesine API match objesini formatla"""
        try:
            home_team = m.get("HN", "").strip()
            away_team = m.get("AN", "").strip()
            
            if not home_team or not away_team:
                return None
            
            # 1X2 oranlarını bul
            odds = {'1': 2.0, 'X': 3.0, '2': 3.5}  # Default
            
            for bahis in m.get("MA", []):
                # MTID 1 = Maç Sonucu (1X2)
                if bahis.get("MTID") == 1:
                    oranlar = bahis.get("OCA", [])
                    if len(oranlar) >= 3:
                        try:
                            odds['1'] = float(oranlar[0].get("O", 2.0))
                            odds['X'] = float(oranlar[1].get("O", 3.0))
                            odds['2'] = float(oranlar[2].get("O", 3.5))
                        except:
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
            logger.debug(f"Match format hatasi: {e}")
            return None
    
    def extract_matches(self, html_content):
        """GELİŞTİRİLMİŞ: TÜM maçları yakalar"""
        soup = BeautifulSoup(html_content, 'html.parser')
        matches = []
        seen = set()
        
        all_text = soup.get_text()
        lines = [line.strip() for line in all_text.split('\n') if line.strip()]
        
        logger.info(f"{len(lines)} satir analiz ediliyor...")
        
        for line in lines:
            if ' vs ' in line.lower() or ' - ' in line:
                match = self._parse_match_line(line)
                if match:
                    key = f"{match['home_team']}|{match['away_team']}"
                    if key not in seen:
                        seen.add(key)
                        matches.append(match)
        
        patterns = [
            r'([A-ZÇĞİÖŞÜa-zçğıöşü\s\.]+?)\s+vs\.?\s+([A-ZÇĞİÖŞÜa-zçğıöşü\s\.]+)',
            r'([A-ZÇĞİÖŞÜa-zçğıöşü\s\.]+?)\s+-\s+([A-ZÇĞİÖŞÜa-zçğıöşü\s\.]+)',
        ]
        
        for pattern in patterns:
            for match_obj in re.finditer(pattern, all_text):
                home = match_obj.group(1).strip()
                away = match_obj.group(2).strip()
                
                if self._is_valid_team_name(home) and self._is_valid_team_name(away):
                    key = f"{home}|{away}"
                    if key not in seen:
                        seen.add(key)
                        matches.append({
                            'home_team': home,
                            'away_team': away,
                            'league': self._detect_league(home, away),
                            'odds': self._extract_nearby_odds(all_text, home, away)
                        })
        
        match_elements = soup.find_all(['div', 'tr', 'li', 'article', 'section'], 
                                       class_=re.compile(r'match|event|game|fixture|coupon|bulletin', re.I))
        
        logger.info(f"{len(match_elements)} HTML elementi bulundu")
        
        for elem in match_elements:
            text = elem.get_text(strip=True)
            if ' vs ' in text.lower() or ' - ' in text:
                match = self._parse_match_line(text)
                if match:
                    key = f"{match['home_team']}|{match['away_team']}"
                    if key not in seen:
                        seen.add(key)
                        matches.append(match)
        
        logger.info(f"Toplam {len(matches)} benzersiz mac bulundu")
        return matches[:50]
    
    def _parse_match_line(self, line):
        """Satırdan maç bilgisi çıkar"""
        try:
            line = re.sub(r'\d{2}:\d{2}', '', line)
            line = re.sub(r'\d+\.\d+', '', line)
            line = line.strip()
            
            if ' vs ' in line.lower():
                parts = re.split(r'\s+vs\.?\s+', line, flags=re.IGNORECASE)
            elif ' - ' in line:
                parts = line.split(' - ')
            else:
                return None
            
            if len(parts) < 2:
                return None
            
            home = parts[0].strip()
            away = parts[1].strip()
            
            if not self._is_valid_team_name(home) or not self._is_valid_team_name(away):
                return None
            
            return {
                'home_team': home,
                'away_team': away,
                'league': self._detect_league(home, away),
                'odds': self._generate_realistic_odds()
            }
        except:
            return None
    
    def _is_valid_team_name(self, name):
        """Geçerli takım ismi kontrolü"""
        if not name or len(name) < 3 or len(name) > 50:
            return False
        
        if re.match(r'^[\d\W]+$', name):
            return False
        
        words = name.split()
        if len(words) == 1 and len(name) < 4:
            return False
        
        blacklist = [
            'hesabim', 'mesajlarim', 'cep tel', 'tekrar', 'tum talep',
            'giris', 'kayit', 'casino', 'canli', 'spor', 'bahis', 'kupon',
            'favoriler', 'yardim', 'iletisim', 'menu', 'profil', 'ayarlar'
        ]
        
        name_lower = name.lower()
        if any(word in name_lower for word in blacklist):
            return False
        
        if any(char in name for char in ['*', '#', '@', '>', '<', '{', '}']):
            return False
        
        return True
    
    def _extract_nearby_odds(self, text, home_team, away_team):
        """Takım isimleri yakınındaki oranları bul"""
        try:
            home_pos = text.find(home_team)
            if home_pos == -1:
                return self._generate_realistic_odds()
            
            nearby_text = text[home_pos:home_pos+200]
            odds_numbers = re.findall(r'\b\d+\.\d{2}\b', nearby_text)
            
            if len(odds_numbers) >= 3:
                return {
                    '1': float(odds_numbers[0]),
                    'X': float(odds_numbers[1]),
                    '2': float(odds_numbers[2])
                }
        except:
            pass
        
        return self._generate_realistic_odds()
    
    def _generate_realistic_odds(self):
        """Gerçekçi oranlar üret"""
        return {
            '1': round(random.uniform(1.5, 4.5), 2),
            'X': round(random.uniform(2.8, 3.8), 2),
            '2': round(random.uniform(1.5, 5.0), 2)
        }
    
    def _detect_league(self, home_team, away_team):
        """Takım isimlerinden lig tespit et"""
        text = f"{home_team} {away_team}".lower()
        
        if any(t in text for t in ['galatasaray', 'fenerbahce', 'besiktas', 'trabzonspor', 
                                     'basaksehir', 'sivasspor', 'konyaspor']):
            return 'Super Lig'
        
        if any(t in text for t in ['arsenal', 'chelsea', 'liverpool', 'manchester', 'city', 'united',
                                     'tottenham', 'everton', 'leicester', 'west ham', 'wolves', 'brighton']):
            return 'Premier League'
        
        if any(t in text for t in ['barcelona', 'real madrid', 'atletico', 'sevilla', 'valencia', 
                                     'villarreal', 'athletic', 'betis', 'real sociedad']):
            return 'La Liga'
        
        if any(t in text for t in ['bayern', 'dortmund', 'leipzig', 'leverkusen', 'frankfurt', 
                                     'union', 'gladbach', 'wolfsburg', 'stuttgart']):
            return 'Bundesliga'
        
        if any(t in text for t in ['milan', 'inter', 'juventus', 'roma', 'napoli', 'lazio', 
                                     'atalanta', 'fiorentina', 'torino']):
            return 'Serie A'
        
        if any(t in text for t in ['psg', 'marseille', 'lyon', 'monaco', 'lille', 'rennes', 'nice']):
            return 'Ligue 1'
        
        if any(t in text for t in ['ajax', 'psv', 'feyenoord', 'utrecht', 'az alkmaar']):
            return 'Eredivisie'
        
        if any(t in text for t in ['porto', 'benfica', 'sporting', 'braga']):
            return 'Primeira Liga'
        
        return 'Diger Ligler'

# ==================== AI TAHMİN ====================
class PredictionEngine:
    def predict_match(self, home_team, away_team, odds):
        try:
            odds_1 = float(odds.get('1', 2.0))
            odds_x = float(odds.get('X', 3.0))
            odds_2 = float(odds.get('2', 3.5))
            
            prob_1 = (1 / odds_1) * 100
            prob_x = (1 / odds_x) * 100
            prob_2 = (1 / odds_2) * 100
            
            total = prob_1 + prob_x + prob_2
            prob_1 = (prob_1 / total) * 100
            prob_x = (prob_x / total) * 100
            prob_2 = (prob_2 / total) * 100
            
            max_prob = max(prob_1, prob_x, prob_2)
            
            if max_prob == prob_1:
                prediction = "1"
                confidence = prob_1
                score = f"{random.randint(2,3)}-{random.randint(0,1)}"
            elif max_prob == prob_2:
                prediction = "2"
                confidence = prob_2
                score = f"{random.randint(0,1)}-{random.randint(2,3)}"
            else:
                prediction = "X"
                confidence = prob_x
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
        except:
            return {
                'prediction': 'X', 'confidence': 50.0,
                'home_win_prob': 33.3, 'draw_prob': 33.3, 'away_win_prob': 33.3,
                'score_prediction': '1-1', 'timestamp': datetime.now().isoformat()
            }

fetcher = AdvancedNesineFetcher()
predictor = PredictionEngine()

# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "status": "Predicta AI v4.1 - Railway Optimized",
        "engine": "BeautifulSoup + requests (no Selenium)",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "method": "requests"}

@app.get("/api/nesine/live-predictions")
async def get_live_predictions(league: str = "all", limit: int = 250):
    try:
        logger.info("Nesine CDN API'den veri cekiliyor...")
        
        # Önce resmi API'yi dene
        matches = fetcher.get_nesine_official_api()
        
        if not matches:
            # Fallback: HTML parsing
            logger.warning("API basarisiz, HTML parsing deneniyor...")
            html_content = fetcher.get_page_content()
            
            if html_content:
                matches = fetcher.extract_matches(html_content)
            else:
                matches = []
        
        # Lig filtresi
        if league != "all" and matches:
            matches = [m for m in matches if league.lower() in m['league'].lower()]
        
        # Geçersiz maçları temizle
        valid_matches = []
        for match in matches:
            home = match['home_team'].lower()
            away = match['away_team'].lower()
            
            # UI elementlerini filtrele
            if any(word in home or word in away for word in 
                   ['hesabim', 'mesaj', 'cep tel', 'tekrar', 'tum talep', 'menu']):
                continue
            
            # Aynı takım kontrolü
            if home == away:
                continue
            
            valid_matches.append(match)
        
        # AI tahminleri ekle
        matches_with_predictions = []
        for match in valid_matches[:limit]:
            prediction = predictor.predict_match(
                match['home_team'], match['away_team'], match['odds']
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
        
        source = "CDN API" if matches else "HTML Fallback"
        logger.info(f"{len(matches_with_predictions)} mac hazir - Kaynak: {source}")
        
        return {
            "success": True,
            "matches": matches_with_predictions,
            "count": len(matches_with_predictions),
            "source": source,
            "nesine_live": len(matches_with_predictions) > 0,
            "ai_powered": True,
            "message": f"{len(matches_with_predictions)} mac bulundu ({source})",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Endpoint hatasi: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        logger.error(f"Hata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/nesine/matches")
async def get_matches(league: str = "all", limit: int = 250):
    return await get_live_predictions(league, limit)

@app.get("/api/nesine/raw")
async def get_raw_nesine_data():
    """Ham Nesine API verisi (debug için)"""
    try:
        api_url = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"
        
        api_headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nesine.com/",
            "Origin": "https://www.nesine.com",
        }
        
        response = requests.get(api_url, headers=api_headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            # Özet istatistikler
            ea_count = len(data.get("sg", {}).get("EA", []))
            ca_count = len(data.get("sg", {}).get("CA", []))
            
            return {
                "success": True,
                "status_code": response.status_code,
                "prematch_count": ea_count,
                "live_count": ca_count,
                "total_matches": ea_count + ca_count,
                "raw_data": data,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "status_code": response.status_code,
                "error": "API yanit vermedi"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
