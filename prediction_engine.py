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
from typing import List, Dict, Optional, Tuple
import numpy as np
import time
import re
from bs4 import BeautifulSoup
from data_scraper import NesineDataScraper
from database_manager import DatabaseManager # Import'un doğru olduğundan emin olun

# Logging kurulumu
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    # 🚨 DÜZELTME BURADA: db_manager'ı zorunlu olmayan bir argüman olarak alıyoruz
    def __init__(self):
        # Gelişmiş ağırlık sistemi
        self.factor_weights = {
            'team_form': 0.25,           # Takım formu
            'attacking_strength': 0.20,  # Atak gücü
            'defensive_strength': 0.20,  # Savunma gücü
            'home_advantage': 0.15,      # Ev sahibi avantajı
            'head_to_head': 0.10,        # Karşılıklı geçmiş
            'player_quality': 0.05,      # Oyuncu kalitesi
            'motivation': 0.05           # Motivasyon faktörü
        }
        
        # Poisson dağılımı için lambda hesaplama faktörleri
        self.poisson_factors = {
            'league_avg_goals': 2.5,     # Lig ortalaması
            'regression_factor': 0.3     # Regresyon katsayısı
        }
        
        # Lig katsayıları
        self.league_coefficients = {
            'super': 1.2,
            'premier': 1.3,
            'la liga': 1.3,
            'bundesliga': 1.2,
            'serie a': 1.2,
            'ligue 1': 1.1,
            'default': 1.0
        }
    
    def fetch_nesine_matches(self):
        """Nesine.com'dan güncel maç programını çek"""
        try:
            matches = self.scraper.fetch_matches()
            
            if matches:
                # Maçları veritabanına kaydet
                for match in matches:
                    self.db_manager.save_match(match)
                
                logger.info(f"{len(matches)} maç verisi çekildi ve kaydedildi")
                return matches
            else:
                # Eğer canlı veri çekilemiyorsa veritabanından en son maçları getir
                return self.db_manager.get_recent_matches()
                
        except Exception as e:
            logger.error(f"Nesine veri çekme hatası: {e}")
            return self.db_manager.get_recent_matches()
    
    def predict_match_advanced(self, home_stats: Dict, away_stats: Dict, match_info: Dict) -> Dict:
        """Gelişmiş maç tahmini sistemi"""
        
        # Veri kalitesi kontrolü
        if not self.validate_data(home_stats, away_stats):
            return self.get_default_prediction()
        
        # 1. Temel güç değerlendirmesi
        home_strength = self.calculate_team_strength_detailed(home_stats)
        away_strength = self.calculate_team_strength_detailed(away_stats)
        
        # 2. Atak ve savunma gücü analizi
        home_attack = self.calculate_attacking_strength(home_stats)
        home_defense = self.calculate_defensive_strength(home_stats)
        away_attack = self.calculate_attacking_strength(away_stats)
        away_defense = self.calculate_defensive_strength(away_stats)
        
        # 3. Ev sahibi avantajı hesaplama
        home_advantage = self.calculate_home_advantage_detailed(home_stats, match_info)
        
        # 4. Motivasyon faktörü
        motivation_factor = self.calculate_motivation_factor(home_stats, away_stats, match_info)
        
        # 5. Karşılıklı geçmiş analizi
        h2h_factor = self.analyze_head_to_head_detailed(home_stats, away_stats, match_info)
        
        # 6. Expected Goals (xG) hesaplama
        home_xg = self.calculate_expected_goals(
            home_attack, away_defense, home_advantage, motivation_factor['home'],
            home_stats, away_stats, True
        )
        away_xg = self.calculate_expected_goals(
            away_attack, home_defense, -home_advantage * 0.5, motivation_factor['away'],
            away_stats, home_stats, False
        )
        
        # 7. Poisson dağılımı ile olasılık hesaplama
        probabilities = self.calculate_match_probabilities(home_xg, away_xg)
        
        # 8. Skor tahmini (en olası skorlar)
        most_likely_scores = self.predict_most_likely_scores(home_xg, away_xg)
        
        # 9. MS ve İY tahminleri
        ms_prediction = self.determine_match_result(probabilities)
        iy_prediction = self.predict_first_half(home_xg, away_xg, probabilities)
        
        # 10. Güven seviyesi hesaplama
        confidence = self.calculate_prediction_confidence(
            home_strength, away_strength, probabilities, home_xg, away_xg
        )
        
        # 11. Risk analizi
        risk_assessment = self.assess_prediction_risk(home_stats, away_stats, probabilities)
        
        return {
            'home_xg': round(home_xg, 2),
            'away_xg': round(away_xg, 2),
            'probabilities': probabilities,
            'most_likely_scores': most_likely_scores,
            'ms_prediction': ms_prediction,
            'iy_prediction': iy_prediction,
            'confidence': round(confidence, 1),
            'risk_level': risk_assessment,
            'prediction_timestamp': datetime.now().isoformat(),
            'detailed_analysis': {
                'home_strength': round(home_strength, 1),
                'away_strength': round(away_strength, 1),
                'home_attack': round(home_attack, 3),
                'away_attack': round(away_attack, 3),
                'home_defense': round(home_defense, 3),
                'away_defense': round(away_defense, 3),
                'home_advantage': round(home_advantage, 3),
                'motivation': motivation_factor,
                'h2h_factor': round(h2h_factor, 3)
            }
        }
    
    def validate_data(self, home_stats: Dict, away_stats: Dict) -> bool:
        """Veri kalitesi kontrolü"""
        required_fields = ['matches_played', 'goals_for', 'goals_against']
        
        for stats in [home_stats, away_stats]:
            if not stats:
                return False
            for field in required_fields:
                if field not in stats or stats[field] < 0:
                    return False
        
        return True
    def get_default_prediction(self) -> Dict:
        """Varsayılan tahmin (veri yoksa)"""
        return {
            'home_xg': 1.5,
            'away_xg': 1.5,
            'probabilities': {'1': 33.3, 'X': 33.3, '2': 33.3},
            'most_likely_scores': [{'score': '1-1', 'probability': 15.0}],
            'ms_prediction': 'X',
            'iy_prediction': 'X',
            'confidence': 20.0,
            'risk_level': 'high',
            'prediction_timestamp': datetime.now().isoformat(),
            'detailed_analysis': {}
        }
    
        
    def calculate_team_strength_detailed(self, team_stats: Dict) -> float:
        """Detaylı takım gücü hesaplama"""
        if not team_stats:
            return 50.0
        
        # Temel metrikler
        matches_played = max(team_stats.get('matches_played', 1), 1)
        wins = team_stats.get('wins', 0)
        draws = team_stats.get('draws', 0)
        losses = team_stats.get('losses', 0)
        goals_for = team_stats.get('goals_for', 0)
        goals_against = team_stats.get('goals_against', 0)
        
        # Puan ortalaması
        points = wins * 3 + draws * 1
        points_per_game = points / matches_played
        
        # Gol farkı
        goal_difference = goals_for - goals_against
        goal_diff_per_game = goal_difference / matches_played
        
        # Form faktörü (son 5 maç)
        recent_form = team_stats.get('recent_form', [])
        form_score = self.calculate_form_score_advanced(recent_form)
        
        # Performans istikrarı
        consistency_score = self.calculate_consistency_score(team_stats)
        
        # Ağırlıklı hesaplama
        strength = (
            (points_per_game / 3) * 35 +  # Maksimum 35 puan
            (max(-2, min(2, goal_diff_per_game)) + 2) / 4 * 30 +  # Maksimum 30 puan
            form_score * 25 +  # Maksimum 25 puan
            consistency_score * 10  # Maksimum 10 puan
        )
        
        return min(100, max(0, strength))
    
    def calculate_consistency_score(self, team_stats: Dict) -> float:
        """Performans istikrarı skoru"""
        recent_results = team_stats.get('recent_form', [])
        if len(recent_results) < 3:
            return 0.5
        
        # Sonuçların standart sapması (G=3, B=1, M=0)
        points = []
        for result in recent_results[:5]:
            if result == 'G':
                points.append(3)
            elif result == 'B':
                points.append(1)
            else:
                points.append(0)
        
        if len(points) > 1:
            std_dev = np.std(points)
            # Düşük standart sapma = yüksek istikrar
            consistency = max(0, 1 - std_dev / 2)
            return consistency
        return 0.5
    
    def calculate_attacking_strength(self, team_stats: Dict) -> float:
        """Atak gücü hesaplama"""
        if not team_stats:
            return 1.0
        
        matches_played = max(team_stats.get('matches_played', 1), 1)
        goals_for = team_stats.get('goals_for', 0)
        
        # Temel gol ortalaması
        goals_per_game = goals_for / matches_played
        
        # Lig ortalamasına göre normalize et
        league_avg = self.poisson_factors['league_avg_goals'] * 0.5
        attacking_strength = goals_per_game / league_avg if league_avg > 0 else 1.0
        
        # Son form etkisi
        recent_form = team_stats.get('recent_form', [])
        recent_goals = self.estimate_recent_goals(recent_form, goals_per_game)
        form_multiplier = min(1.5, max(0.5, recent_goals / goals_per_game)) if goals_per_game > 0 else 1.0
        
        # Şut verimliliği (varsa)
        shooting_efficiency = team_stats.get('shooting_efficiency', 1.0)
        
        return attacking_strength * form_multiplier * shooting_efficiency
    
    def calculate_defensive_strength(self, team_stats: Dict) -> float:
        """Savunma gücü hesaplama"""
        if not team_stats:
            return 1.0
        
        matches_played = max(team_stats.get('matches_played', 1), 1)
        goals_against = team_stats.get('goals_against', 0)
        
        # Temel gol yeme ortalaması
        goals_conceded_per_game = goals_against / matches_played
        
        # Lig ortalamasına göre normalize et
        league_avg = self.poisson_factors['league_avg_goals'] * 0.5
        defensive_strength = goals_conceded_per_game / league_avg if league_avg > 0 else 1.0
        
        # Son form etkisi
        recent_form = team_stats.get('recent_form', [])
        recent_conceded = self.estimate_recent_conceded_goals(recent_form, goals_conceded_per_game)
        form_multiplier = min(1.5, max(0.5, recent_conceded / goals_conceded_per_game)) if goals_conceded_per_game > 0 else 1.0
        
        # Kart disiplini (varsa)
        discipline_factor = team_stats.get('discipline_factor', 1.0)
        
        return defensive_strength * form_multiplier * discipline_factor
    
    def calculate_home_advantage_detailed(self, home_stats: Dict, match_info: Dict) -> float:
        """Detaylı ev sahibi avantajı"""
        base_advantage = 0.3
        
        if not home_stats:
            return base_advantage
        
        # Ev performansı
        home_matches = home_stats.get('home_matches_played', 0)
        if home_matches > 5:
            home_win_rate = home_stats.get('home_wins', 0) / home_matches
            home_performance_bonus = (home_win_rate - 0.4) * 0.3
        else:
            home_performance_bonus = 0
        
        # Seyirci faktörü
        attendance_factor = match_info.get('attendance_factor', 1.0)
        
        # Lig seviyesi etkisi
        league_multiplier = self.get_league_multiplier(match_info.get('league', ''))
        
        # Hava durumu etkisi (basitleştirilmiş)
        weather_impact = match_info.get('weather_impact', 0.0)
        
        advantage = (base_advantage + home_performance_bonus) * league_multiplier * attendance_factor
        advantage *= (1 + weather_impact)
        
        return max(0.1, min(0.6, advantage))
    
    def get_league_multiplier(self, league_name: str) -> float:
        """Lig katsayısını belirleme"""
        league_lower = league_name.lower()
        for key, value in self.league_coefficients.items():
            if key in league_lower:
                return value
        return self.league_coefficients['default']
    
    def calculate_motivation_factor(self, home_stats: Dict, away_stats: Dict, match_info: Dict) -> Dict:
        """Motivasyon faktörü hesaplama"""
        base_motivation = 1.0
        
        # Lig pozisyonu
        home_pos = home_stats.get('position', 10)
        away_pos = away_stats.get('position', 10)
        total_teams = home_stats.get('total_teams', 20)
        
        home_motivation = base_motivation
        away_motivation = base_motivation
        
        # Küme düşme bölgesi
        relegation_zone = total_teams - 3
        if home_pos > relegation_zone:
            home_motivation *= 1.15
        if away_pos > relegation_zone:
            away_motivation *= 1.15
        
        # Şampiyonluk yarışı
        if home_pos <= 3:
            home_motivation *= 1.1
        if away_pos <= 3:
            away_motivation *= 1.1
        
        # Önemli maç faktörü
        match_importance = match_info.get('importance', 1.0)
        home_motivation *= match_importance
        away_motivation *= match_importance
        
        return {
            'home': min(1.3, max(0.7, home_motivation)),
            'away': min(1.3, max(0.7, away_motivation))
        }
    
    def analyze_head_to_head_detailed(self, home_stats: Dict, away_stats: Dict, match_info: Dict) -> float:
        """Detaylı karşılıklı geçmiş analizi"""
        h2h_data = match_info.get('head_to_head', [])
        
        if h2h_data and len(h2h_data) > 0:
            # Gerçek H2H verisi varsa
            home_wins = 0
            away_wins = 0
            draws = 0
            
            for match in h2h_data[:10]:  # Son 10 maç
                if match['home_goals'] > match['away_goals']:
                    home_wins += 1
                elif match['home_goals'] < match['away_goals']:
                    away_wins += 1
                else:
                    draws += 1
            
            total_matches = home_wins + away_wins + draws
            if total_matches > 0:
                home_win_rate = home_wins / total_matches
                away_win_rate = away_wins / total_matches
                
                # Ev sahibi avantajını H2H'ye yansıt
                h2h_factor = (home_win_rate - away_win_rate) * 0.4
                return max(-0.3, min(0.3, h2h_factor))
        
        # H2H verisi yoksa takım güçlerine göre tahmin
        home_strength = self.calculate_team_strength_detailed(home_stats)
        away_strength = self.calculate_team_strength_detailed(away_stats)
        
        strength_diff = (home_strength - away_strength) / 100
        return max(-0.2, min(0.2, strength_diff * 0.3))
    
    def calculate_expected_goals(self, attack_strength: float, defense_strength: float, 
                               advantage: float, motivation: float, team_stats: Dict,
                               opponent_stats: Dict, is_home: bool) -> float:
        """Expected Goals hesaplama"""
        
        # Temel xG hesaplama
        base_xg = self.poisson_factors['league_avg_goals'] * 0.5
        
        # Faktörleri uygula
        xg = base_xg * attack_strength / defense_strength
        
        # Ev sahibi avantajı ve motivasyon
        xg *= (1 + advantage) * motivation
        
        # Takım özellikleri
        pressure_factor = self.calculate_pressure_factor(team_stats, opponent_stats, is_home)
        xg *= pressure_factor
        
        # Realistik sınırlar
        return max(0.1, min(4.5, xg))
    
    def calculate_pressure_factor(self, team_stats: Dict, opponent_stats: Dict, is_home: bool) -> float:
        """Basınç faktörü (büyük takımlara karşı)"""
        team_strength = self.calculate_team_strength_detailed(team_stats)
        opponent_strength = self.calculate_team_strength_detailed(opponent_stats)
        
        strength_ratio = opponent_strength / max(team_strength, 1)
        
        if strength_ratio > 1.2:  # Daha güçlü rakip
            return 0.9 if is_home else 0.85
        elif strength_ratio < 0.8:  # Daha zayıf rakip
            return 1.1 if is_home else 1.05
        
        return 1.0
    
    def calculate_match_probabilities(self, home_xg: float, away_xg: float) -> Dict:
        """Poisson dağılımı ile maç olasılıkları"""
        
        prob_1 = 0.0
        prob_x = 0.0  
        prob_2 = 0.0
        
        # Daha geniş gol aralığı
        max_goals = 8
        
        for home_goals in range(max_goals):
            for away_goals in range(max_goals):
                home_prob = (home_xg ** home_goals * math.exp(-home_xg)) / math.factorial(home_goals)
                away_prob = (away_xg ** away_goals * math.exp(-away_xg)) / math.factorial(away_goals)
                
                combined_prob = home_prob * away_prob
                
                if home_goals > away_goals:
                    prob_1 += combined_prob
                elif home_goals == away_goals:
                    prob_x += combined_prob
                else:
                    prob_2 += combined_prob
        
        # Normalize et
        total = prob_1 + prob_x + prob_2
        if total > 0:
            prob_1 /= total
            prob_x /= total  
            prob_2 /= total
        
        return {
            '1': round(prob_1 * 100, 1),
            'X': round(prob_x * 100, 1),
            '2': round(prob_2 * 100, 1)
        }
    
    def predict_most_likely_scores(self, home_xg: float, away_xg: float) -> List[Dict]:
        """En olası skorlar"""
        
        score_probs = []
        max_goals = 6
        
        for home_goals in range(max_goals):
            for away_goals in range(max_goals):
                home_prob = (home_xg ** home_goals * math.exp(-home_xg)) / math.factorial(home_goals)
                away_prob = (away_xg ** away_goals * math.exp(-away_xg)) / math.factorial(away_goals)
                
                combined_prob = home_prob * away_prob
                
                score_probs.append({
                    'score': f"{home_goals}-{away_goals}",
                    'probability': round(combined_prob * 100, 2)
                })
        
        # Olasılığa göre sırala ve ilk 5'ini al
        score_probs.sort(key=lambda x: x['probability'], reverse=True)
        
        return score_probs[:5]
    
    def determine_match_result(self, probabilities: Dict) -> str:
        """MS sonucu belirleme"""
        probs = [(k, v) for k, v in probabilities.items()]
        probs.sort(key=lambda x: x[1], reverse=True)
        
        # %5'ten az fark varsa beraberliği değerlendir
        if len(probs) > 1 and probs[0][1] - probs[1][1] < 5:
            if probs[1][0] == 'X':
                return 'X'
        
        return probs[0][0]
    
    def predict_first_half(self, home_xg: float, away_xg: float, match_probs: Dict) -> str:
        """İlk yarı tahmini"""
        # İlk yarı gol beklentisi
        iy_home_xg = home_xg * 0.42  # İstatistiklere göre ayarlandı
        iy_away_xg = away_xg * 0.42
        
        iy_probs = self.calculate_match_probabilities(iy_home_xg, iy_away_xg)
        
        # İlk yarıda beraberlik daha olası
        if iy_probs['X'] > 35 and abs(iy_probs['1'] - iy_probs['2']) < 15:
            return 'X'
        
        return self.determine_match_result(iy_probs)
    
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
    
    def calculate_form_score_advanced(self, recent_form: List[str]) -> float:
        """Gelişmiş form skoru"""
        if not recent_form:
            return 0.5
        
        score = 0
        total_weight = 0
        
        for i, result in enumerate(recent_form[:5]):
            weight = 5 - i
            total_weight += weight
            
            if result == 'G':
                score += 1.0 * weight
            elif result == 'B':
                score += 0.5 * weight
        
        return score / total_weight if total_weight > 0 else 0.5
    
    def estimate_recent_goals(self, recent_form: List[str], avg_goals: float) -> float:
        """Son maçlardaki gol tahmini"""
        if not recent_form:
            return avg_goals
        
        form_score = self.calculate_form_score_advanced(recent_form)
        
        # Forma göre gol tahmini ayarla
        return avg_goals * (0.8 + form_score * 0.4)
    
    def estimate_recent_conceded_goals(self, recent_form: List[str], avg_conceded: float) -> float:
        """Son maçlardaki yenilen gol tahmini"""
        if not recent_form:
            return avg_conceded
        
        form_score = self.calculate_form_score_advanced(recent_form)
        
        # Forma göre gol yeme tahmini ayarla
        return avg_conceded * (1.2 - form_score * 0.4)

# Kullanım örneği
if __name__ == "__main__":
    # Örnek takım istatistikleri
    home_stats_example = {
        'matches_played': 20,
        'wins': 12,
        'draws': 4,
        'losses': 4,
        'goals_for': 35,
        'goals_against': 18,
        'position': 2,
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
        'weather_impact': -0.05
    }
    
    engine = AdvancedPredictionEngine()
    prediction = engine.predict_match_advanced(home_stats_example, away_stats_example, match_info_example)
    
    print("=== GELİŞMİŞ MAÇ TAHMİNİ ===")
    print(f"MS Tahmini: {prediction['ms_prediction']}")
    print(f"İY Tahmini: {prediction['iy_prediction']}")
    print(f"Güven Seviyesi: %{prediction['confidence']}")
    print(f"Risk Seviyesi: {prediction['risk_level']}")
    print(f"Ev xG: {prediction['home_xg']}, Deplasman xG: {prediction['away_xg']}")
    print(f"Olasılıklar: Ev %{prediction['probabilities']['1']}, Beraberlik %{prediction['probabilities']['X']}, Deplasman %{prediction['probabilities']['2']}")
    print("\nEn Olası Skorlar:")
    for score in prediction['most_likely_scores']:
        print(f"  {score['score']}: %{score['probability']}")
