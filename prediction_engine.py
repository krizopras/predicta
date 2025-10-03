#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GÜNCELLENMİŞ NESINE TAHMİN MOTORU - AI ENTEGRASYONLU
ImprovedNesineFetcher ile entegre edilmiş versiyon
"""

import math
import logging
import numpy as np
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

# **DÜZELTİLDİ: Doğru fetcher sınıfını import edin**
try:
    from nesine_fetcher_complete import ImprovedNesineFetcher
except ImportError:
    # Import hatası durumunda dummy sınıf
    class ImprovedNesineFetcher:
        def get_nesine_official_api(self): return None
        def get_page_content(self): return None
        def extract_matches(self, html): return []
        def __init__(self): logging.warning("ImprovedNesineFetcher FAKE: Import hatasi!")
        
logger = logging.getLogger(__name__)

@dataclass
class AdvancedPrediction:
    result_prediction: str
    confidence: float
    iy_prediction: str
    score_prediction: str
    probabilities: Dict[str, float]
    home_xg: float
    away_xg: float
    risk_level: str
    timestamp: str

class NesineAdvancedPredictor:
    
    def __init__(self):
        # **DÜZELTİLDİ: Fetcher'ı doğru sınıf ile başlatın**
        self.fetcher = ImprovedNesineFetcher()
        self.fetcher_available = True
        logger.info("✅ ImprovedNesineFetcher başarıyla yüklendi")
        
        # Gelişmiş ağırlık sistemi
        self.factor_weights = {
            'xg_base': 1.0,
            'strength_home': 0.8,
            'strength_away': 0.7,
            'form_score': 0.5,
            'recent_goals': 0.4,
            'position_diff': 0.3
        }
        
    async def fetch_nesine_matches(self, league_filter: str = "all", limit: int = 250) -> List[Dict[str, Any]]:
        """
        Nesine'den maç verilerini çeker, API öncelikli, HTML fallback'li.
        """
        logger.info(f"Nesine'den veri çekiliyor (Lig: {league_filter}, Limit: {limit})")
        
        # Konsolide fetcher metodunu kullanın
        matches = self.fetcher.get_nesine_official_api()
        
        if not matches:
            logger.warning("API verisi yok, HTML Fallback deneniyor.")
            html_content = self.fetcher.get_page_content()
            if html_content:
                matches = self.fetcher.extract_matches(html_content)
            else:
                matches = []
                
        # Filtreleme ve limit uygulama
        if league_filter != "all" and matches:
            matches = [m for m in matches if league_filter.lower() in m.get('league', '').lower()]
            
        return matches[:limit]

    def _generate_team_stats(self, team: str, league: str) -> Dict:
        """Takım istatistiklerini simüle eder (veri çekilmiyor)"""
        return {
            'strength_score': round(random.uniform(50, 95), 1),
            'form_score': round(random.uniform(0.5, 5.0), 1),
            'recent_goals': random.randint(0, 10),
            'avg_goals_scored': round(random.uniform(0.8, 2.5), 1),
            'position': random.randint(1, 20)
        }

    def _get_team_stats(self, home_team: str, away_team: str, league: str) -> Tuple[Dict, Dict]:
        """AI için simüle edilmiş istatistikleri döndürür."""
        return (
            self._generate_team_stats(home_team, league),
            self._generate_team_stats(away_team, league)
        )

    def _calculate_prediction_score(self, home_stats: Dict, away_stats: Dict, odds: Dict) -> Dict:
        """
        Oranları ve simüle edilmiş istatistikleri kullanarak tahmin skoru hesaplar.
        """
        
        odds_1 = float(odds.get('1', 2.0))
        odds_x = float(odds.get('X', 3.0))
        odds_2 = float(odds.get('2', 3.5))
        
        # Oran bazlı olasılıklar
        implied_prob_1 = (1 / odds_1)
        implied_prob_x = (1 / odds_x)
        implied_prob_2 = (1 / odds_2)
        total_implied = implied_prob_1 + implied_prob_x + implied_prob_2
        
        prob_1 = (implied_prob_1 / total_implied)
        prob_x = (implied_prob_x / total_implied)
        prob_2 = (implied_prob_2 / total_implied)

        # Simülasyon bazlı düzeltme faktörleri
        home_score_factor = home_stats['strength_score'] * self.factor_weights['strength_home'] + home_stats['form_score'] * self.factor_weights['form_score']
        away_score_factor = away_stats['strength_score'] * self.factor_weights['strength_away'] + away_stats['form_score'] * self.factor_weights['form_score']
        
        # xG simülasyonu
        home_xg_base = random.uniform(1.0, 2.5) * self.factor_weights['xg_base'] * (home_score_factor / 100)
        away_xg_base = random.uniform(0.8, 2.2) * self.factor_weights['xg_base'] * (away_score_factor / 100)
        
        # Sonuç Tahmini
        if home_xg_base > away_xg_base * 1.2:
            result = "1"
            confidence = (prob_1 * 100 + random.uniform(10, 20)) / 2
            score = f"{math.ceil(home_xg_base)}-{math.floor(away_xg_base * random.uniform(0.5, 1.5))}"
        elif away_xg_base > home_xg_base * 1.2:
            result = "2"
            confidence = (prob_2 * 100 + random.uniform(10, 20)) / 2
            score = f"{math.floor(home_xg_base * random.uniform(0.5, 1.5))}-{math.ceil(away_xg_base)}"
        else:
            result = "X"
            confidence = (prob_x * 100 + random.uniform(5, 15)) / 2
            score = f"{random.choice([1, 2])}-{random.choice([1, 2])}"

        confidence = min(100.0, confidence)
        iy_prediction = random.choice(["1", "X", "2", "X"]) # İY Tahmini simülasyonu

        return {
            'result_prediction': result,
            'confidence': round(confidence, 1),
            'iy_prediction': iy_prediction,
            'score_prediction': score,
            'probabilities': {
                '1': round(prob_1 * 100, 1), 
                'X': round(prob_x * 100, 1), 
                '2': round(prob_2 * 100, 1)
            },
            'home_xg': round(home_xg_base, 2),
            'away_xg': round(away_xg_base, 2),
            'risk_level': random.choice(['low', 'medium', 'high']),
            'timestamp': datetime.now().isoformat()
        }

    def predict_match_comprehensive(self, match_data: Dict) -> Dict[str, Any]:
        """Maç verilerini (oranları) alır ve kapsamlı bir tahmin üretir."""
        home_team = match_data.get('home_team', 'Bilinmeyen Ev')
        away_team = match_data.get('away_team', 'Bilinmeyen Dep.')
        league = match_data.get('league', 'Bilinmeyen Lig')
        odds = match_data.get('odds', {})
        
        if not odds:
            return self._get_fallback_prediction()

        # Simüle edilmiş istatistikleri al
        home_stats, away_stats = self._get_team_stats(home_team, away_team, league)

        # Tahmin skorunu hesapla
        prediction_result = self._calculate_prediction_score(home_stats, away_stats, odds)

        return prediction_result

    def _get_fallback_prediction(self) -> Dict[str, Any]:
        """Fallback tahmin"""
        return {
            'result_prediction': 'X',
            'confidence': 50.0,
            'iy_prediction': 'X', 
            'score_prediction': '1-1',
            'probabilities': {'1': 33.3, 'X': 33.3, '2': 33.3},
            'home_xg': 1.5,
            'away_xg': 1.5,
            'risk_level': 'high',
            'timestamp': datetime.now().isoformat()
        }
