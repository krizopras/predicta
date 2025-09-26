#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NESINE.COM KAPSAMLI FUTBOL TAHMIN SISTEMI
Nesine'den maç, takım ve oyuncu istatistiklerini çekip detaylı tahmin yapan sistem
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
import requests
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import time
import re
import math
from bs4 import BeautifulSoup
# from data_scraper import NesineDataScraper # Kodun tam çalışması için gerçek dosya adını kullanın
# from database_manager import DatabaseManager # Kodun tam çalışması için gerçek dosya adını kullanın

# Logging kurulumu
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Örnek Sınıflar (Modül importları düzgün çalışmazsa bu sınıflara ihtiyacınız olabilir)
@dataclass
class PlayerStats:
    name: str
    position: str
    appearances: int
    goals: int
    assists: int
    yellow_cards: int
    red_cards: int
    rating: float
    is_injured: bool = False
    is_suspended: bool = False

@dataclass
class TeamStats:
    name: str
    league: str
    position: int
    points: int
    matches_played: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    home_form: str
    away_form: str
    recent_form: List[str]

class NesineAdvancedPredictor:
    
    def __init__(self):
        # Gelişmiş ağırlık sistemi
        self.factor_weights = {
            'xg_base': 1.0,
            'strength_home': 0.8,
            'strength_away': 0.7,
            'form_score': 0.5,
            'recent_goals': 0.4,
            'position_diff': 0.3
        }
        # Diğer başlangıç ayarları (varsa)
        
    def fetch_nesine_matches(self) -> List[Dict]:
        """
        Bu fonksiyon, main.py'deki periodic_data_update tarafından çağrılacaktır.
        Gerçek veriyi çekme/işleme mantığınız buraya gelir.
        Şu an sadece boş bir liste döndürüyor.
        """
        # Burada veri çekme ve işleme kodu olmalıdır.
        return []

    # Yerel gol/gol yeme gücü hesaplama
    def calculate_strength(self, team_stats: Dict, is_home: bool) -> Tuple[float, float]:
        """Gol atma ve gol yeme gücünü hesaplar."""
        
        # Basitleştirilmiş güç hesaplama (daha karmaşık olabilir)
        if team_stats['matches_played'] == 0:
            return 1.0, 1.0
            
        avg_goals_for = team_stats['goals_for'] / team_stats['matches_played']
        avg_goals_against = team_stats['goals_against'] / team_stats['matches_played']
        
        # Pozisyon faktörü
        total_teams = team_stats.get('total_teams', 20)
        position_factor = (total_teams - team_stats['position'] + 1) / total_teams
        
        attack_strength = avg_goals_for * (1 + position_factor * 0.2)
        defense_strength = 1 / (avg_goals_against * (1 + position_factor * 0.1))
        
        # Basitleştirilmiş güç çarpanları:
        attack_factor = (attack_strength + team_stats['wins'] * 0.1) / 1.5
        defense_factor = (defense_strength + team_stats['losses'] * 0.1) / 1.5
        
        # Home/Away formu ile ayarlama
        if is_home:
            attack_factor *= 1.15
            defense_factor *= 0.95
        else:
            attack_factor *= 0.95
            defense_factor *= 1.15
            
        return attack_factor, defense_factor

    def estimate_recent_goals(self, recent_form: List[str], avg_goals: float) -> float:
        """Son maçlardaki gol tahmini"""
        if not recent_form:
            return avg_goals
        
        form_score = self.calculate_form_score_advanced(recent_form)
        
        return avg_goals * (0.8 + form_score * 0.4)
    
    def estimate_recent_conceded_goals(self, recent_form: List[str], avg_conceded: float) -> float:
        """Son maçlardaki yenilen gol tahmini"""
        if not recent_form:
            return avg_conceded
        
        form_score = self.calculate_form_score_advanced(recent_form)
        
        return avg_conceded * (1.2 - form_score * 0.4)

    def calculate_form_score_advanced(self, recent_form: List[str]) -> float:
        """Gelişmiş form skoru"""
        if not recent_form:
            return 0.5
        
        score = 0
        total_weight = 0
        
        # Son 5 maça ağırlık verme (en son maça en yüksek ağırlık)
        for i, result in enumerate(recent_form[:5]):
            weight = 5 - i
            total_weight += weight
            
            if result == 'G': # Galibiyet
                score += 1.0 * weight
            elif result == 'B': # Beraberlik
                score += 0.5 * weight
            # Mağlubiyet 0.0 * weight
        
        # Skor 0.0 ile 1.0 arasında normalize edilir.
        return score / total_weight if total_weight > 0 else 0.5

    # Beklenen Gol (xG) hesaplama
    def calculate_xg(self, home_stats: Dict, away_stats: Dict, match_info: Dict) -> Tuple[float, float]:
        """Beklenen Gol (xG) değerlerini hesaplar."""
        
        # Basit ortalama goller
        league_avg_goals = 2.5 # Varsayım
        
        # Home ve Away gol/gol yeme güçleri
        home_attack, home_defense = self.calculate_strength(home_stats, is_home=True)
        away_attack, away_defense = self.calculate_strength(away_stats, is_home=False)
        
        # xG hesaplama
        home_xg_base = home_attack * away_defense * league_avg_goals
        away_xg_base = away_attack * home_defense * league_avg_goals
        
        # xG'yi son form ile ayarlama
        home_xg = self.estimate_recent_goals(home_stats['recent_form'], home_xg_base)
        away_xg = self.estimate_recent_goals(away_stats['recent_form'], away_xg_base)
        
        # Home/Away yenen gol formunu da hesaba katma (Savunma ayarlaması)
        home_conceded_factor = self.estimate_recent_conceded_goals(home_stats['recent_form'], 1.0)
        away_conceded_factor = self.estimate_recent_conceded_goals(away_stats['recent_form'], 1.0)
        
        # Rakibin yiyeceği gol beklentisine savunma formunu ters etki ile uygulama
        home_xg *= (2.0 - away_conceded_factor) 
        away_xg *= (2.0 - home_conceded_factor)
        
        # Lig önem, hava durumu gibi faktörler
        home_xg *= match_info.get('importance', 1.0) * match_info.get('weather_impact', 1.0)
        away_xg *= match_info.get('importance', 1.0) * match_info.get('weather_impact', 1.0)
        
        # Makul sınırlar içinde tutma
        home_xg = max(0.5, min(3.5, home_xg))
        away_xg = max(0.5, min(3.5, away_xg))
        
        return home_xg, away_xg

    # Poisson olasılık hesaplama
    def poisson_probability(self, mean: float, k: int) -> float:
        """Belirli bir ortalama (lambda) için k olayı olasılığını hesaplar."""
        # math.factorial(k) kullanılırken k'nın negatif olmadığından emin olunur (range 0'dan başlıyor)
        if k < 0: return 0.0
        return (mean ** k * math.exp(-mean)) / math.factorial(k)

    # Maç sonucu olasılıkları
    def calculate_match_probabilities(self, home_xg: float, away_xg: float) -> Dict[str, float]:
        """Maç sonucu olasılıklarını (1, X, 2) hesaplar."""
        
        prob_1 = 0.0
        prob_x = 0.0
        prob_2 = 0.0
        max_goals = 6
        
        for home_goals in range(max_goals):
            for away_goals in range(max_goals):
                prob = self.poisson_probability(home_xg, home_goals) * self.poisson_probability(away_xg, away_goals)
                
                if home_goals > away_goals:
                    prob_1 += prob
                elif home_goals < away_goals:
                    prob_2 += prob
                else:
                    prob_x += prob
        
        total_prob = prob_1 + prob_x + prob_2
        if total_prob == 0:
            return {'1': 33.3, 'X': 33.3, '2': 33.3}
            
        # Olasılıkları normalize et ve yuvarla
        return {
            '1': round(prob_1 / total_prob * 100, 2),
            'X': round(prob_x / total_prob * 100, 2),
            '2': round(prob_2 / total_prob * 100, 2)
        }

    def get_score_ms_prediction(self, home_goals: int, away_goals: int) -> str:
        """Skordan MS tahmini"""
        if home_goals > away_goals:
            return '1'
        elif home_goals < away_goals:
            return '2'
        else:
            return 'X'
    
    def determine_match_result(self, probabilities: Dict, home_xg: float, away_xg: float) -> str:
        """Tutarlı MS sonucu belirleme"""
        
        # 1. Olasılık farkına göre
        prob_1 = probabilities.get('1', 0)
        prob_x = probabilities.get('X', 0) 
        prob_2 = probabilities.get('2', 0)
        
        # 2. xG farkına göre
        xg_diff = home_xg - away_xg
        
        # 3. Kombine karar
        # Net fark ve XG desteği varsa
        if prob_1 > prob_2 + 10 and prob_1 > prob_x + 10 and xg_diff > 0.3:
            return '1'
        elif prob_2 > prob_1 + 10 and prob_2 > prob_x + 10 and xg_diff < -0.3:
            return '2'
        # Beraberlik baskınsa veya olasılıklar yakınsa
        elif prob_x > max(prob_1, prob_2) or abs(prob_1 - prob_2) < 8:
            return 'X'
        # Sadece xG farkı ile
        elif xg_diff > 0.2:
            return '1'
        elif xg_diff < -0.2:
            return '2'
        else:
            return 'X'

    def predict_most_likely_scores(self, home_xg: float, away_xg: float, ms_prediction: str) -> List[Dict]:
        """
        MS tahmini ile uyumlu skor tahmini.
        Ana MS tahminine öncelik vermek için olasılıkları ayarlama yapılır.
        """
        
        score_probs = []
        max_goals = 6
        
        for home_goals in range(max_goals):
            for away_goals in range(max_goals):
                # Poisson olasılık hesaplama
                home_prob = self.poisson_probability(home_xg, home_goals)
                away_prob = self.poisson_probability(away_xg, away_goals)
                combined_prob = home_prob * away_prob
                
                # Skorun MS karşılığını bulma
                score_ms = self.get_score_ms_prediction(home_goals, away_goals)
                
                # 🟢 DÜZELTİLEN MANTIK: MS tahminiyle uyumlu olanı zorla, uyumsuz olanı zayıflat.
                if score_ms == ms_prediction:
                    # MS tahminiyle uyumlu skorların olasılığını %60 artır
                    combined_prob *= 1.6 
                else:
                    # MS tahminiyle uyumsuz skorların olasılığını %60 düşür
                    combined_prob *= 0.4
                
                score_probs.append({
                    'score': f"{home_goals}-{away_goals}",
                    'probability': round(combined_prob * 100, 2),
                    'ms_match': score_ms == ms_prediction
                })
        
        # Olasılığa göre sırala
        score_probs.sort(key=lambda x: x['probability'], reverse=True)
        
        return score_probs[:5]
    
    def predict_first_half(self, home_xg: float, away_xg: float, probabilities: Dict) -> str:
        """İlk yarı tahmini (daha tutarlı)"""
        
        # İlk yarı gol beklentisi (daha düşük)
        iy_home_xg = home_xg * 0.38
        iy_away_xg = away_xg * 0.38
        
        iy_probs = self.calculate_match_probabilities(iy_home_xg, iy_away_xg)
        
        # İlk yarıda beraberlik daha olasıdır, buna öncelik verilir.
        iy_xg_diff = iy_home_xg - iy_away_xg
        
        if abs(iy_xg_diff) < 0.15:
            return 'X'
        elif iy_probs['X'] > 40:
            return 'X'
        elif iy_xg_diff > 0.1:
            return '1'
        elif iy_xg_diff < -0.1:
            return '2'
        else:
            return 'X'
    
    def calculate_prediction_confidence(self, home_strength: float, away_strength: float,
                                    probabilities: Dict, home_xg: float, away_xg: float) -> float:
        """Tahmin güven seviyesi"""
        
        max_prob = max(probabilities.values())
        strength_diff = abs(home_strength - away_strength)
        xg_diff = abs(home_xg - away_xg)
        
        base_confidence = 45.0
        prob_bonus = (max_prob - 33.33) * 0.6
        strength_bonus = min(25, strength_diff * 0.5)
        xg_bonus = min(20, xg_diff * 10)
        
        confidence = base_confidence + prob_bonus + strength_bonus + xg_bonus
        
        return min(92, max(15, confidence))
    
    def assess_prediction_risk(self, home_stats: Dict, away_stats: Dict, probabilities: Dict) -> str:
        """Tahmin risk seviyesi"""
        max_prob = max(probabilities.values())
        
        if max_prob > 60:
            return 'low'
        elif max_prob > 45:
            return 'medium'
        else:
            return 'high'
    
    # Ana tahmin fonksiyonu (tüm mantığı birleştirir)
    def predict_match_advanced(self, home_stats: Dict, away_stats: Dict, match_info: Dict) -> Dict[str, Any]:
        """Tüm faktörleri kullanarak gelişmiş maç tahmini yapar."""
        
        # 1. xG Hesaplama
        home_xg, away_xg = self.calculate_xg(home_stats, away_stats, match_info)
        
        # 2. Olasılık Hesaplama
        probabilities = self.calculate_match_probabilities(home_xg, away_xg)
        
        # 3. MS Sonucu Belirleme
        ms_prediction = self.determine_match_result(probabilities, home_xg, away_xg)
        
        # 4. İY Sonucu Belirleme
        iy_prediction = self.predict_first_half(home_xg, away_xg, probabilities)
        
        # 5. Skor Tahmini (MS sonucuyla uyumlu)
        score_probs = self.predict_most_likely_scores(home_xg, away_xg, ms_prediction)
        
        # 6. Güven Seviyesi
        home_attack, home_defense = self.calculate_strength(home_stats, True)
        away_attack, away_defense = self.calculate_strength(away_stats, False)
        
        confidence = self.calculate_prediction_confidence(home_attack, away_attack,
                                                        probabilities, home_xg, away_xg)
        
        # 7. Risk Değerlendirmesi
        risk = self.assess_prediction_risk(home_stats, away_stats, probabilities)
        
        return {
            'ms_prediction': ms_prediction,
            'iy_prediction': iy_prediction,
            'predicted_score': score_probs[0]['score'] if score_probs else 'N/A',
            'probabilities': probabilities,
            'score_probabilities': score_probs,
            'confidence': round(confidence, 2),
            'risk_level': risk,
            'home_xg': round(home_xg, 2),
            'away_xg': round(away_xg, 2),
            'last_update': datetime.now().isoformat()
        }

# Ana test bloğu (hata vermemesi için koruyucu kod)
if __name__ == '__main__':
    try:
        # Örnek Kullanım
        home_stats_example = {
            'matches_played': 20,
            'wins': 10,
            'draws': 5,
            'losses': 5,
            'goals_for': 35,
            'goals_against': 20,
            'position': 3,
            'total_teams': 18,
            'recent_form': ['G', 'G', 'B', 'M', 'G'],
            'home_wins': 7,
            'home_draws': 2,
            'home_losses': 1,
            'home_matches_played': 10
        }
        
        away_stats_example = {
            'matches_played': 20,
            'wins': 8,
            'draws': 6,
            'losses': 6,
            'goals_for': 28,
            'goals_against': 25,
            'position': 7,
            'total_teams': 18,
            'recent_form': ['B', 'M', 'G', 'B', 'M'],
            'away_wins': 3,
            'away_draws': 4,
            'away_losses': 3,
            'away_matches_played': 10
        }
        
        match_info_example = {
            'league': 'Super Lig',
            'importance': 1.1,
            'attendance_factor': 1.05,
            'weather_impact': 1.0 
        }
        
        engine = NesineAdvancedPredictor()
        prediction = engine.predict_match_advanced(home_stats_example, away_stats_example, match_info_example)
        
        print("\n=== GELİŞMİŞ MAÇ TAHMİNİ ===")
        print(f"MS Tahmini: {prediction['ms_prediction']}")
        print(f"İY Tahmini: {prediction['iy_prediction']}")
        print(f"Tahmin Edilen Skor: {prediction['predicted_score']}")
        print(f"Güven Seviyesi: %{prediction['confidence']:.2f}")
        print(f"Risk Seviyesi: {prediction['risk_level']}")
        print(f"Olasılıklar (1/X/2): {prediction['probabilities']}")
        print(f"Top Skor Olasılıkları: {prediction['score_probabilities']}")

    except Exception as e:
        print(f"Test çalışması sırasında hata oluştu: {e}")
