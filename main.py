#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predicta AI v6.0 - Enhanced with Historical Data
Geçmiş maç verilerini kullanarak gelişmiş tahmin sistemi
"""
import os
import json
import time
import logging
import threading
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict
import requests

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# ----------------------------------------------------
# Logging
# ----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("predicta-v6.0")

# ----------------------------------------------------
# Global Ayarlar
# ----------------------------------------------------
DATA_DIR = "./data"
RAW_DATA_DIR = "./raw"
DATA_FILE = os.path.join(DATA_DIR, "future_matches.json")
HISTORICAL_DATA_FILE = os.path.join(DATA_DIR, "historical_cache.json")
DAYS_AHEAD = 14
REFRESH_INTERVAL_SECONDS = 6 * 3600  # 6 saat
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
API_ENDPOINT = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"

# ----------------------------------------------------
# Historical Data Parser
# ----------------------------------------------------
class HistoricalDataParser:
    """Raw klasöründeki txt dosyalarını parse eder"""
    
    def __init__(self, raw_data_dir: str = RAW_DATA_DIR):
        self.raw_data_dir = raw_data_dir
        self.team_stats = defaultdict(lambda: {
            'matches': [],
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_for': 0,
            'goals_against': 0
        })
        self.h2h_stats = defaultdict(list)
    
    def parse_football_data_file(self, filepath: str) -> List[Dict]:
        """Football-Data.org formatındaki dosyaları parse et"""
        matches = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Maç pattern: tarih, saat, ev sahibi v deplasman skor (yarı skor)
            pattern = r'(\w{3}\s+\w{3}/\d+(?:/\d+)?)\s+([\d:]+)\s+(.*?)\s+v\s+(.*?)\s+([\d-]+)\s*\(([\d-]+)\)'
            
            for match in re.findall(pattern, content):
                date_str, time_str, home_team, away_team, final_score, ht_score = match
                
                try:
                    home_goals, away_goals = map(int, final_score.split('-'))
                    
                    result = '1' if home_goals > away_goals else ('X' if home_goals == away_goals else '2')
                    
                    match_data = {
                        'date': date_str.strip(),
                        'time': time_str.strip(),
                        'home_team': home_team.strip(),
                        'away_team': away_team.strip(),
                        'home_goals': home_goals,
                        'away_goals': away_goals,
                        'result': result,
                        'source_file': os.path.basename(filepath)
                    }
                    
                    matches.append(match_data)
                except:
                    continue
            
            logger.info(f"Parsed {len(matches)} matches from {os.path.basename(filepath)}")
            
        except Exception as e:
            logger.error(f"Error parsing {filepath}: {e}")
        
        return matches
    
    def load_all_historical_data(self) -> Dict[str, Any]:
        """Tüm raw verilerini yükle ve istatistikleri hesapla"""
        all_matches = []
        
        if not os.path.exists(self.raw_data_dir):
            logger.warning(f"Raw data directory not found: {self.raw_data_dir}")
            return {'matches': [], 'team_stats': {}, 'h2h_stats': {}}
        
        # Tüm klasörleri tara
        for root, dirs, files in os.walk(self.raw_data_dir):
            for filename in files:
                if filename.endswith('.txt'):
                    filepath = os.path.join(root, filename)
                    matches = self.parse_football_data_file(filepath)
                    all_matches.extend(matches)
        
        # İstatistikleri hesapla
        for match in all_matches:
            self._update_team_stats(match['home_team'], match, is_home=True)
            self._update_team_stats(match['away_team'], match, is_home=False)
            self._update_h2h_stats(match)
        
        logger.info(f"Loaded {len(all_matches)} historical matches, {len(self.team_stats)} teams")
        
        return {
            'matches': all_matches,
            'team_stats': dict(self.team_stats),
            'h2h_stats': dict(self.h2h_stats)
        }
    
    def _update_team_stats(self, team: str, match: Dict, is_home: bool):
        """Takım istatistiklerini güncelle"""
        stats = self.team_stats[team]
        
        if is_home:
            goals_for = match['home_goals']
            goals_against = match['away_goals']
            result = match['result']
        else:
            goals_for = match['away_goals']
            goals_against = match['home_goals']
            result = '2' if match['result'] == '1' else ('1' if match['result'] == '2' else 'X')
        
        stats['matches'].append({
            'goals_for': goals_for,
            'goals_against': goals_against,
            'result': result,
            'venue': 'home' if is_home else 'away'
        })
        
        stats['goals_for'] += goals_for
        stats['goals_against'] += goals_against
        
        if (result == '1' and is_home) or (result == '2' and not is_home):
            stats['wins'] += 1
        elif result == 'X':
            stats['draws'] += 1
        else:
            stats['losses'] += 1
    
    def _update_h2h_stats(self, match: Dict):
        """H2H istatistiklerini güncelle"""
        key = f"{match['home_team']}_vs_{match['away_team']}"
        self.h2h_stats[key].append({
            'home_goals': match['home_goals'],
            'away_goals': match['away_goals'],
            'result': match['result']
        })

# ----------------------------------------------------
# Enhanced Prediction Engine
# ----------------------------------------------------
class EnhancedPredictionEngine:
    """Geçmiş verilerle geliştirilmiş tahmin motoru"""
    
    def __init__(self, historical_data: Dict):
        self.team_stats = historical_data.get('team_stats', {})
        self.h2h_stats = historical_data.get('h2h_stats', {})
        
        # Takım güçleri (varsayılan)
        self.team_power = {
            'galatasaray': 88, 'fenerbahce': 86, 'besiktas': 82, 'trabzonspor': 78,
            'manchester city': 94, 'liverpool': 91, 'arsenal': 89, 'chelsea': 86,
            'real madrid': 93, 'barcelona': 90, 'bayern munich': 95, 'inter': 88
        }
    
    def get_team_form(self, team: str, n_matches: int = 5) -> float:
        """Takımın son form durumu (0-1)"""
        stats = self.team_stats.get(team)
        if not stats or not stats['matches']:
            return 0.5
        
        recent = stats['matches'][-n_matches:]
        points = sum(3 if m['result'] == 'W' else (1 if m['result'] == 'D' else 0) 
                    for m in recent)
        
        return points / (len(recent) * 3)
    
    def get_h2h_advantage(self, home: str, away: str) -> float:
        """H2H avantajı (-1 ile 1 arası, pozitif ev sahibi lehine)"""
        key = f"{home}_vs_{away}"
        h2h = self.h2h_stats.get(key, [])
        
        if not h2h:
            return 0.0
        
        home_wins = sum(1 for m in h2h if m['result'] == '1')
        away_wins = sum(1 for m in h2h if m['result'] == '2')
        
        return (home_wins - away_wins) / len(h2h)
    
    def get_avg_goals(self, team: str) -> Dict[str, float]:
        """Ortalama gol istatistikleri"""
        stats = self.team_stats.get(team)
        if not stats or not stats['matches']:
            return {'scored': 1.5, 'conceded': 1.2}
        
        total_matches = len(stats['matches'])
        
        return {
            'scored': stats['goals_for'] / total_matches,
            'conceded': stats['goals_against'] / total_matches
        }
    
    def predict_match(self, home_team: str, away_team: str, odds: Dict, league: str) -> Dict:
        """Ana tahmin fonksiyonu"""
        
        # 1. Oranlardan base probabilities
        odds_1 = float(odds.get('1', 2.0))
        odds_x = float(odds.get('X', 3.0))
        odds_2 = float(odds.get('2', 3.5))
        
        imp_1 = (1 / odds_1) * 100
        imp_x = (1 / odds_x) * 100
        imp_2 = (1 / odds_2) * 100
        total_imp = imp_1 + imp_x + imp_2
        
        prob_1_base = (imp_1 / total_imp) * 100
        prob_x_base = (imp_x / total_imp) * 100
        prob_2_base = (imp_2 / total_imp) * 100
        
        # 2. Form analizi
        home_form = self.get_team_form(home_team)
        away_form = self.get_team_form(away_team)
        
        # 3. H2H analizi
        h2h_advantage = self.get_h2h_advantage(home_team, away_team)
        
        # 4. Gol ortalamaları
        home_goals_avg = self.get_avg_goals(home_team)
        away_goals_avg = self.get_avg_goals(away_team)
        
        # 5. Hibrit tahmin
        # %30 odds, %40 form, %30 H2H
        home_win_prob = (
            prob_1_base * 0.30 +
            home_form * 100 * 0.40 +
            max(0, h2h_advantage * 100 + 33.3) * 0.30
        )
        
        away_win_prob = (
            prob_2_base * 0.30 +
            away_form * 100 * 0.40 +
            max(0, -h2h_advantage * 100 + 33.3) * 0.30
        )
        
        draw_prob = 100 - home_win_prob - away_win_prob
        
        # Normalizasyon
        total = home_win_prob + draw_prob + away_win_prob
        home_win_prob = (home_win_prob / total) * 100
        draw_prob = (draw_prob / total) * 100
        away_win_prob = (away_win_prob / total) * 100
        
        # En yüksek olasılık
        probs = {'1': home_win_prob, 'X': draw_prob, '2': away_win_prob}
        prediction = max(probs, key=probs.get)
        confidence = probs[prediction]
        
        # Skor tahmini
        exp_home_goals = (home_goals_avg['scored'] + away_goals_avg['conceded']) / 2
        exp_away_goals = (away_goals_avg['scored'] + home_goals_avg['conceded']) / 2
        
        home_goals = round(exp_home_goals)
        away_goals = round(exp_away_goals)
        
        # Güven seviyesi
        if confidence >= 65:
            risk = "LOW"
            recommendation = "RECOMMENDED"
        elif confidence >= 50:
            risk = "MEDIUM"
            recommendation = "CONSIDER"
        else:
            risk = "HIGH"
            recommendation = "SKIP"
        
        return {
            'predicted_result': prediction,
            'confidence': round(confidence, 1),
            'probabilities': {
                'home_win': round(home_win_prob, 1),
                'draw': round(draw_prob, 1),
                'away_win': round(away_win_prob, 1)
            },
            'score_prediction': f"{home_goals}-{away_goals}",
            'analysis': {
                'home_form': round(home_form, 2),
                'away_form': round(away_form, 2),
                'h2h_advantage': round(h2h_advantage, 2),
                'home_avg_goals': round(home_goals_avg['scored'], 2),
                'away_avg_goals': round(away_goals_avg['scored'], 2)
            },
            'risk_level': risk,
            'recommendation': recommendation,
            'features_used': True,
            'timestamp': datetime.now().isoformat()
        }

# ----------------------------------------------------
# Nesine Fetcher
# ----------------------------------------------------
class NesineFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': USER_AGENT,
            'Accept': 'application/json'
        }
    
    def fetch_matches(self, days_ahead: int = DAYS_AHEAD) -> List[Dict]:
        """Nesine API'den maçları çek"""
        all_matches = []
        
        for i in range(days_ahead):
            date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            url = f"{API_ENDPOINT}?date={date}"
            
            try:
                r = self.session.get(url, headers=self.headers, timeout=15)
                
                if r.status_code == 200:
                    data = r.json()
                    parsed = self._parse_matches(data)
                    all_matches.extend(parsed)
                else:
                    logger.warning(f"API error {r.status_code} for {date}")
                    
            except Exception as e:
                logger.error(f"Fetch error for {date}: {e}")
                continue
        
        # Duplicate filtrele
        seen = set()
        unique = []
        for m in all_matches:
            key = f"{m['home_team']}|{m['away_team']}|{m['date']}"
            if key not in seen:
                seen.add(key)
                unique.append(m)
        
        unique.sort(key=lambda x: (x['date'], x['time']))
        logger.info(f"Fetched {len(unique)} unique matches")
        
        return unique
    
    def _parse_matches(self, data: Dict) -> List[Dict]:
        """API yanıtını parse et"""
        matches = []
        
        ea_matches = data.get("sg", {}).get("EA", [])
        
        for m in ea_matches:
            if m.get("GT") != 1:  # Sadece futbol
                continue
            
            home = m.get("HN", "").strip()
            away = m.get("AN", "").strip()
            
            if not home or not away:
                continue
            
            # Oranları bul
            odds = {'1': 2.0, 'X': 3.0, '2': 3.5}
            
            for bahis in m.get("MA", []):
                if bahis.get("MTID") == 1:  # Maç Sonucu
                    oranlar = bahis.get("OCA", [])
                    if len(oranlar) >= 3:
                        try:
                            odds['1'] = float(oranlar[0].get("O", 2.0))
                            odds['X'] = float(oranlar[1].get("O", 3.0))
                            odds['2'] = float(oranlar[2].get("O", 3.5))
                        except:
                            pass
            
            matches.append({
                'home_team': home,
                'away_team': away,
                'league': m.get("LC", "Unknown"),
                'match_id': m.get("C", ""),
                'date': m.get("D", datetime.now().strftime('%Y-%m-%d')),
                'time': m.get("T", "20:00"),
                'odds': odds,
                'is_live': m.get("S") == 1
            })
        
        return matches

# ----------------------------------------------------
# FastAPI App
# ----------------------------------------------------
app = FastAPI(title="Predicta AI v6.0 Enhanced", version="6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global değişkenler
_cached_matches: List[Dict] = []
_cache_lock = threading.Lock()
_historical_data: Dict = {}
_predictor: Optional[EnhancedPredictionEngine] = None
_fetcher = NesineFetcher()

# ----------------------------------------------------
# Startup & Background Tasks
# ----------------------------------------------------
def load_historical_data():
    """Geçmiş verileri yükle"""
    global _historical_data, _predictor
    
    parser = HistoricalDataParser()
    _historical_data = parser.load_all_historical_data()
    _predictor = EnhancedPredictionEngine(_historical_data)
    
    logger.info("Historical data loaded and predictor initialized")

def ensure_data_dir():
    """Veri dizinlerini oluştur"""
    os.makedirs(DATA_DIR, exist_ok=True)

def save_matches_to_disk(matches: List[Dict]):
    """Maçları diske kaydet"""
    ensure_data_dir()
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)

def load_matches_from_disk() -> List[Dict]:
    """Diskten maçları yükle"""
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def background_refresh():
    """Arka planda sürekli güncelleme"""
    global _cached_matches
    
    while True:
        try:
            matches = _fetcher.fetch_matches(DAYS_AHEAD)
            
            with _cache_lock:
                _cached_matches = matches
            
            save_matches_to_disk(matches)
            logger.info("Background refresh completed")
            
        except Exception as e:
            logger.error(f"Background refresh error: {e}")
        
        time.sleep(REFRESH_INTERVAL_SECONDS)

@app.on_event("startup")
def on_startup():
    """Uygulama başlangıcı"""
    global _cached_matches
    
    # Geçmiş verileri yükle
    load_historical_data()
    
    # Diskten cache yükle
    _cached_matches = load_matches_from_disk()
    
    # İlk fetch
    if not _cached_matches:
        try:
            _cached_matches = _fetcher.fetch_matches(DAYS_AHEAD)
            save_matches_to_disk(_cached_matches)
        except Exception as e:
            logger.error(f"Initial fetch failed: {e}")
    
    # Background thread başlat
    t = threading.Thread(target=background_refresh, daemon=True)
    t.start()
    
    logger.info("Predicta AI v6.0 Enhanced started successfully")

# ----------------------------------------------------
# API Endpoints
# ----------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "Predicta AI v6.0 Enhanced",
        "features": [
            "Historical data analysis",
            "Team form tracking",
            "Head-to-head statistics",
            "Advanced prediction algorithm"
        ],
        "status": "active",
        "historical_teams": len(_historical_data.get('team_stats', {})),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "6.0",
        "predictor_active": _predictor is not None,
        "cached_matches": len(_cached_matches)
    }

@app.get("/api/matches/upcoming")
def get_upcoming(limit: Optional[int] = 100):
    """Yaklaşan maçları getir"""
    with _cache_lock:
        matches = list(_cached_matches)
    
    if limit:
        matches = matches[:limit]
    
    return {
        "success": True,
        "count": len(matches),
        "matches": matches
    }

@app.post("/api/matches/refresh")
def manual_refresh():
    """Manuel veri güncellemesi"""
    global _cached_matches
    
    try:
        matches = _fetcher.fetch_matches(DAYS_AHEAD)
        
        with _cache_lock:
            _cached_matches = matches
        
        save_matches_to_disk(matches)
        
        return {
            "success": True,
            "count": len(matches),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions/upcoming")
def get_predictions(
    limit: Optional[int] = 50,
    min_confidence: Optional[float] = 0.0
):
    """Tahminli maçları getir"""
    
    if not _predictor:
        raise HTTPException(
            status_code=503,
            detail="Prediction engine not initialized"
        )
    
    with _cache_lock:
        matches = list(_cached_matches)
    
    if not matches:
        raise HTTPException(
            status_code=503,
            detail="No matches available"
        )
    
    # Tahminleri ekle
    predictions = []
    for m in matches:
        try:
            pred = _predictor.predict_match(
                m['home_team'],
                m['away_team'],
                m['odds'],
                m['league']
            )
            
            # Min confidence filtresi
            if pred['confidence'] >= min_confidence:
                m2 = m.copy()
                m2['ai_prediction'] = pred
                predictions.append(m2)
                
        except Exception as e:
            logger.error(f"Prediction error for {m.get('home_team')} vs {m.get('away_team')}: {e}")
            continue
    
    # Confidence'a göre sırala
    predictions.sort(key=lambda x: x['ai_prediction']['confidence'], reverse=True)
    
    if limit:
        predictions = predictions[:limit]
    
    return {
        "success": True,
        "count": len(predictions),
        "matches": predictions,
        "engine": "Enhanced with Historical Data v6.0",
        "features_enabled": True,
        "min_confidence": min_confidence,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/team/stats/{team_name}")
def get_team_stats(team_name: str):
    """Takım istatistiklerini getir"""
    
    stats = _historical_data.get('team_stats', {}).get(team_name)
    
    if not stats:
        raise HTTPException(
            status_code=404,
            detail=f"Team '{team_name}' not found in historical data"
        )
    
    total_matches = len(stats['matches'])
    
    return {
        "team": team_name,
        "total_matches": total_matches,
        "wins": stats['wins'],
        "draws": stats['draws'],
        "losses": stats['losses'],
        "goals_scored": stats['goals_for'],
        "goals_conceded": stats['goals_against'],
        "avg_goals_scored": round(stats['goals_for'] / total_matches, 2) if total_matches > 0 else 0,
        "avg_goals_conceded": round(stats['goals_against'] / total_matches, 2) if total_matches > 0 else 0,
        "win_rate": round(stats['wins'] / total_matches * 100, 1) if total_matches > 0 else 0
    }

@app.get("/api/stats/overview")
def get_stats_overview():
    """Sistem istatistikleri"""
    
    return {
        "historical_data": {
            "total_teams": len(_historical_data.get('team_stats', {})),
            "total_h2h_pairs": len(_historical_data.get('h2h_stats', {})),
            "total_historical_matches": len(_historical_data.get('matches', []))
        },
        "live_data": {
            "cached_matches": len(_cached_matches),
            "last_update": datetime.now().isoformat()
        },
        "predictor_status": "active" if _predictor else "inactive"
    }

# ----------------------------------------------------
# Run
# ----------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
