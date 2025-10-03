#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gelişmiş Tahmin Motoru - Düzeltilmiş Versiyon
Oran bazlı + istatistiksel hybrid model
"""

import math
import logging
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class ImprovedPredictionEngine:
    """
    Hybrid tahmin motoru:
    1. Bahis oranlarından gerçek olasılıkları çıkarır (overround düzeltmesi)
    2. İstatistiksel model varsa weighted average yapar
    3. Güven skorunu matematiksel olarak hesaplar
    """
    
    def __init__(self):
        self.league_avg_goals = {
            'Premier League': 2.8,
            'La Liga': 2.6,
            'Bundesliga': 3.1,
            'Serie A': 2.7,
            'Ligue 1': 2.5,
            'Süper Lig': 2.8,
            'Super Lig': 2.8,
            'Eredivisie': 3.2,
            'Primeira Liga': 2.6,
            'default': 2.7
        }
        
        # Güven seviyesi eşikleri
        self.confidence_levels = {
            'very_high': 80,
            'high': 65,
            'medium': 50,
            'low': 35
        }
    
    def predict_match(self, home_team: str, away_team: str, odds: Dict, 
                     home_stats: Optional[Dict] = None, 
                     away_stats: Optional[Dict] = None,
                     league: str = 'default') -> Dict[str, Any]:
        """
        Ana tahmin fonksiyonu
        """
        try:
            # 1. Oranlardan gerçek olasılıkları çıkar
            odds_probs = self.calculate_true_probabilities_from_odds(odds)
            
            # 2. İstatistik varsa xG hesapla
            if home_stats and away_stats:
                xg_probs = self.calculate_xg_probabilities(
                    home_stats, away_stats, league
                )
                
                # Weighted average: %60 oran, %40 istatistik
                final_probs = {
                    '1': odds_probs['1'] * 0.6 + xg_probs['1'] * 0.4,
                    'X': odds_probs['X'] * 0.6 + xg_probs['X'] * 0.4,
                    '2': odds_probs['2'] * 0.6 + xg_probs['2'] * 0.4
                }
                has_stats = True
            else:
                # Sadece oranlar
                final_probs = odds_probs
                has_stats = False
            
            # 3. Tahmin ve güven hesapla
            prediction = self.determine_prediction(final_probs)
            confidence = self.calculate_confidence(final_probs, odds, has_stats)
            
            # 4. Skor tahmini (güven bazlı)
            score = self.predict_score(final_probs, prediction, confidence, league)
            
            # 5. Risk seviyesi
            risk_level = self.assess_risk(confidence, final_probs)
            
            return {
                'prediction': prediction,
                'confidence': round(confidence, 1),
                'home_win_prob': round(final_probs['1'], 1),
                'draw_prob': round(final_probs['X'], 1),
                'away_win_prob': round(final_probs['2'], 1),
                'score_prediction': score,
                'risk_level': risk_level,
                'confidence_label': self.get_confidence_label(confidence),
                'has_team_stats': has_stats,
                'timestamp': datetime.now().isoformat(),
                'warning': self.get_betting_warning(confidence, risk_level)
            }
            
        except Exception as e:
            logger.error(f"Tahmin hatası: {e}")
            return self.get_fallback_prediction()
    
    def calculate_true_probabilities_from_odds(self, odds: Dict) -> Dict[str, float]:
        """
        Bahis oranlarından gerçek olasılıkları çıkarır
        Overround (bookmaker marjı) düzeltmesi yapar
        """
        try:
            odds_1 = float(odds.get('1', 2.5))
            odds_x = float(odds.get('X', 3.0))
            odds_2 = float(odds.get('2', 3.0))
            
            # Implied probabilities
            implied_1 = 1 / odds_1
            implied_x = 1 / odds_x
            implied_2 = 1 / odds_2
            
            # Overround (genelde %105-115 arası)
            total = implied_1 + implied_x + implied_2
            
            # Normalize et (gerçek olasılıklara dönüştür)
            true_prob_1 = (implied_1 / total) * 100
            true_prob_x = (implied_x / total) * 100
            true_prob_2 = (implied_2 / total) * 100
            
            return {
                '1': true_prob_1,
                'X': true_prob_x,
                '2': true_prob_2
            }
            
        except Exception as e:
            logger.error(f"Oran hesaplama hatası: {e}")
            return {'1': 33.3, 'X': 33.3, '2': 33.3}
    
    def calculate_xg_probabilities(self, home_stats: Dict, away_stats: Dict, 
                                   league: str) -> Dict[str, float]:
        """
        İstatistiklerden xG hesaplar ve olasılıklara dönüştürür
        """
        try:
            # xG hesapla
            home_xg, away_xg = self.calculate_xg(home_stats, away_stats, league)
            
            # Poisson ile olasılıkları hesapla
            prob_1 = 0.0
            prob_x = 0.0
            prob_2 = 0.0
            
            for home_goals in range(6):
                for away_goals in range(6):
                    prob = (self.poisson(home_xg, home_goals) * 
                           self.poisson(away_xg, away_goals))
                    
                    if home_goals > away_goals:
                        prob_1 += prob
                    elif home_goals < away_goals:
                        prob_2 += prob
                    else:
                        prob_x += prob
            
            total = prob_1 + prob_x + prob_2
            
            return {
                '1': (prob_1 / total) * 100,
                'X': (prob_x / total) * 100,
                '2': (prob_2 / total) * 100
            }
            
        except Exception as e:
            logger.error(f"xG hesaplama hatası: {e}")
            return {'1': 33.3, 'X': 33.3, '2': 33.3}
    
    def calculate_xg(self, home_stats: Dict, away_stats: Dict, 
                    league: str) -> Tuple[float, float]:
        """
        Beklenen gol hesaplama
        """
        league_avg = self.league_avg_goals.get(league, 
                                               self.league_avg_goals['default'])
        
        # Takım güçleri
        home_matches = home_stats.get('matches_played', 1)
        away_matches = away_stats.get('matches_played', 1)
        
        if home_matches == 0: home_matches = 1
        if away_matches == 0: away_matches = 1
        
        home_attack = home_stats.get('goals_for', 0) / home_matches
        home_defense = home_stats.get('goals_against', 0) / home_matches
        
        away_attack = away_stats.get('goals_for', 0) / away_matches
        away_defense = away_stats.get('goals_against', 0) / away_matches
        
        # Ev sahibi avantajı
        home_xg = (home_attack / league_avg) * (away_defense / league_avg) * league_avg * 1.15
        away_xg = (away_attack / league_avg) * (home_defense / league_avg) * league_avg * 0.95
        
        # Sınırla
        home_xg = max(0.3, min(4.0, home_xg))
        away_xg = max(0.3, min(4.0, away_xg))
        
        return home_xg, away_xg
    
    def poisson(self, mean: float, k: int) -> float:
        """Poisson olasılık fonksiyonu"""
        try:
            return (mean ** k * math.exp(-mean)) / math.factorial(k)
        except:
            return 0.0
    
    def determine_prediction(self, probabilities: Dict) -> str:
        """En yüksek olasılığı seç"""
        max_prob = max(probabilities.values())
        
        for key, prob in probabilities.items():
            if prob == max_prob:
                return key
        
        return 'X'
    
    def calculate_confidence(self, probabilities: Dict, odds: Dict, 
                           has_stats: bool) -> float:
        """
        Matematiksel güven hesabı
        """
        # 1. En yüksek olasılık
        max_prob = max(probabilities.values())
        
        # 2. Olasılık farkı (1. ve 2. arasında)
        sorted_probs = sorted(probabilities.values(), reverse=True)
        prob_gap = sorted_probs[0] - sorted_probs[1]
        
        # 3. Base confidence
        base = 25.0
        
        # 4. Olasılık bonusu
        prob_bonus = (max_prob - 33.3) * 1.2
        
        # 5. Gap bonusu
        gap_bonus = prob_gap * 0.8
        
        # 6. İstatistik bonusu
        stats_bonus = 10.0 if has_stats else 0.0
        
        # 7. Oran tutarlılığı bonusu
        consistency_bonus = self.check_odds_consistency(odds, probabilities)
        
        confidence = base + prob_bonus + gap_bonus + stats_bonus + consistency_bonus
        
        # Sınırla: min %15, max %92
        return max(15.0, min(92.0, confidence))
    
    def check_odds_consistency(self, odds: Dict, probabilities: Dict) -> float:
        """
        Oranlar ve hesaplanan olasılıklar tutarlı mı?
        """
        try:
            odds_1 = float(odds.get('1', 2.5))
            odds_x = float(odds.get('X', 3.0))
            odds_2 = float(odds.get('2', 3.0))
            
            # En düşük oran = en yüksek olasılık olmalı
            min_odds = min(odds_1, odds_x, odds_2)
            max_prob = max(probabilities.values())
            
            if (min_odds == odds_1 and probabilities['1'] == max_prob) or \
               (min_odds == odds_x and probabilities['X'] == max_prob) or \
               (min_odds == odds_2 and probabilities['2'] == max_prob):
                return 8.0  # Tutarlı
            
            return 0.0  # Tutarsız
            
        except:
            return 0.0
    
    def predict_score(self, probabilities: Dict, prediction: str, 
                     confidence: float, league: str) -> str:
        """
        Güven skoruna göre muhafazakar skor tahmini
        """
        league_avg = self.league_avg_goals.get(league, 2.7)
        
        # Güven bazlı gol farkı limiti
        if confidence >= 80:
            max_diff = 3
        elif confidence >= 65:
            max_diff = 2
        elif confidence >= 50:
            max_diff = 1
        else:
            max_diff = 1  # Düşük güven = muhafazakar
        
        # Tahmine göre skor
        if prediction == '1':
            # Ev sahibi kazanır
            home_goals = random.randint(1, min(3, int(league_avg) + 1))
            away_goals = max(0, home_goals - random.randint(1, max_diff))
        elif prediction == '2':
            # Deplasman kazanır
            away_goals = random.randint(1, min(3, int(league_avg) + 1))
            home_goals = max(0, away_goals - random.randint(1, max_diff))
        else:
            # Beraberlik
            goals = random.randint(0, min(2, int(league_avg)))
            home_goals = away_goals = goals
        
        return f"{home_goals}-{away_goals}"
    
    def assess_risk(self, confidence: float, probabilities: Dict) -> str:
        """Risk seviyesi değerlendirmesi"""
        max_prob = max(probabilities.values())
        
        if confidence >= 75 and max_prob >= 50:
            return 'low'
        elif confidence >= 55 and max_prob >= 40:
            return 'medium'
        elif confidence >= 40:
            return 'high'
        else:
            return 'very_high'
    
    def get_confidence_label(self, confidence: float) -> str:
        """Güven etiketi"""
        if confidence >= self.confidence_levels['very_high']:
            return 'Çok Yüksek Güven'
        elif confidence >= self.confidence_levels['high']:
            return 'Yüksek Güven'
        elif confidence >= self.confidence_levels['medium']:
            return 'Orta Güven'
        else:
            return 'Düşük Güven'
    
    def get_betting_warning(self, confidence: float, risk_level: str) -> str:
        """Bahis uyarısı"""
        if confidence < 50:
            return "⚠️ Düşük güven - Bu tahmin spekülatiftir"
        elif risk_level in ['high', 'very_high']:
            return "⚠️ Yüksek risk - Dikkatli değerlendirin"
        elif confidence < 65:
            return "ℹ️ Orta güven - Dikkatli bahis önerilir"
        else:
            return "✓ Güvenilir tahmin"
    
    def get_fallback_prediction(self) -> Dict[str, Any]:
        """Hata durumunda varsayılan tahmin"""
        return {
            'prediction': 'X',
            'confidence': 33.3,
            'home_win_prob': 33.3,
            'draw_prob': 33.3,
            'away_win_prob': 33.3,
            'score_prediction': '1-1',
            'risk_level': 'very_high',
            'confidence_label': 'Bilinmiyor',
            'has_team_stats': False,
            'timestamp': datetime.now().isoformat(),
            'warning': '⚠️ Tahmin hesaplanamadı'
        }


# Test fonksiyonu
if __name__ == "__main__":
    predictor = ImprovedPredictionEngine()
    
    # Test 1: Sadece oranlar
    print("Test 1: Sadece oranlar")
    result = predictor.predict_match(
        home_team="Arsenal",
        away_team="Chelsea",
        odds={'1': 2.10, 'X': 3.40, '2': 3.50},
        league='Premier League'
    )
    print(f"Tahmin: {result['prediction']}")
    print(f"Güven: {result['confidence']}%")
    print(f"Skor: {result['score_prediction']}")
    print(f"Uyarı: {result['warning']}\n")
    
    # Test 2: İstatistiklerle
    print("Test 2: İstatistiklerle")
    result = predictor.predict_match(
        home_team="Arsenal",
        away_team="Chelsea",
        odds={'1': 2.10, 'X': 3.40, '2': 3.50},
        home_stats={
            'matches_played': 10,
            'goals_for': 25,
            'goals_against': 8
        },
        away_stats={
            'matches_played': 10,
            'goals_for': 18,
            'goals_against': 12
        },
        league='Premier League'
    )
    print(f"Tahmin: {result['prediction']}")
    print(f"Güven: {result['confidence']}%")
    print(f"Skor: {result['score_prediction']}")
    print(f"Stats var: {result['has_team_stats']}")
    print(f"Uyarı: {result['warning']}")
