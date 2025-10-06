import json
import os
from typing import Dict, Any, List
import numpy as np
from datetime import datetime

class EnhancedPredictionEngine:
    def __init__(self, team_stats_path: str = "data/team_stats/team_statistics.json"):
        self.team_stats = self._load_team_stats(team_stats_path)
        
    def _load_team_stats(self, path: str) -> Dict:
        """Takım istatistiklerini yükle"""
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def get_team_form(self, team: str) -> Dict[str, float]:
        """Takım formunu getir"""
        if team not in self.team_stats:
            return self._get_default_stats()
            
        stats = self.team_stats[team]
        
        # Form hesapla
        win_rate = stats['wins'] / stats['matches_played'] if stats['matches_played'] > 0 else 0.3
        home_win_rate = stats['home_wins'] / stats['home_matches'] if stats['home_matches'] > 0 else 0.4
        
        # Son 5 maç formu
        last_5_results = stats.get('last_5_results', [])
        last_5_points = sum(3 if r == 'W' else 1 if r == 'D' else 0 for r in last_5_results)
        last_5_form = last_5_points / (len(last_5_results) * 3) if last_5_results else 0.5
        
        # Gol ortalamaları
        goals_scored_avg = np.mean(stats.get('last_5_goals_scored', [1.2]))
        goals_conceded_avg = np.mean(stats.get('last_5_goals_conceded', [1.3]))
        
        return {
            'win_rate': win_rate,
            'home_win_rate': home_win_rate,
            'last_5_form': last_5_form,
            'goals_scored_avg': goals_scored_avg,
            'goals_conceded_avg': goals_conceded_avg,
            'attack_strength': goals_scored_avg / max(goals_conceded_avg, 0.1),
            'defense_strength': 1 / max(goals_conceded_avg, 0.1)
        }
    
    def _get_default_stats(self) -> Dict[str, float]:
        """Varsayılan istatistikler"""
        return {
            'win_rate': 0.3,
            'home_win_rate': 0.4,
            'last_5_form': 0.5,
            'goals_scored_avg': 1.2,
            'goals_conceded_avg': 1.3,
            'attack_strength': 0.9,
            'defense_strength': 0.8
        }
    
    def predict_match(self, home_team: str, away_team: str, odds: Dict, league: str = "Unknown") -> Dict[str, Any]:
        """Geçmiş verilere dayalı tahmin yap"""
        # Takım formları
        home_form = self.get_team_form(home_team)
        away_form = self.get_team_form(away_team)
        
        # Temel olasılıklar
        home_advantage = 1.1  # Ev sahibi avantajı
        
        # Güç skorları
        home_power = (
            home_form['win_rate'] * 0.3 +
            home_form['last_5_form'] * 0.4 + 
            home_form['attack_strength'] * 0.2 +
            home_form['defense_strength'] * 0.1
        ) * home_advantage
        
        away_power = (
            away_form['win_rate'] * 0.3 +
            away_form['last_5_form'] * 0.4 +
            away_form['attack_strength'] * 0.2 +
            away_form['defense_strength'] * 0.1
        )
        
        # Oranlardan implied probabilities
        odds_1 = float(odds.get('1', 2.0))
        odds_x = float(odds.get('X', 3.2))
        odds_2 = float(odds.get('2', 3.5))
        
        imp_1 = 1 / odds_1
        imp_x = 1 / odds_x
        imp_2 = 1 / odds_2
        
        total_imp = imp_1 + imp_x + imp_2
        prob_1_base = imp_1 / total_imp
        prob_x_base = imp_x / total_imp
        prob_2_base = imp_2 / total_imp
        
        # Hibrit model (%40 geçmiş veri, %30 güç analizi, %30 oranlar)
        total_power = home_power + away_power
        
        prob_1 = (
            prob_1_base * 0.3 +
            (home_power / total_power) * 0.4 +
            home_form['home_win_rate'] * 0.3
        ) * 100
        
        prob_2 = (
            prob_2_base * 0.3 +
            (away_power / total_power) * 0.4 +
            away_form['win_rate'] * 0.3
        ) * 100
        
        prob_x = 100 - prob_1 - prob_2
        
        # Normalizasyon
        total = prob_1 + prob_x + prob_2
        prob_1 = (prob_1 / total) * 100
        prob_x = (prob_x / total) * 100  
        prob_2 = (prob_2 / total) * 100
        
        # Tahmin
        probs = {'1': prob_1, 'X': prob_x, '2': prob_2}
        prediction = max(probs, key=probs.get)
        confidence = probs[prediction]
        
        # Value bet analizi
        value_index = (confidence / 100) * odds_1 - 1 if prediction == '1' else \
                     (confidence / 100) * odds_x - 1 if prediction == 'X' else \
                     (confidence / 100) * odds_2 - 1
        
        if value_index > 0.15:
            value_rating = "EXCELLENT"
            recommendation = "🔥 STRONG BET"
        elif value_index > 0.08:
            value_rating = "GOOD" 
            recommendation = "✅ RECOMMENDED"
        elif value_index > 0.03:
            value_rating = "FAIR"
            recommendation = "⚠️ CONSIDER"
        else:
            value_rating = "POOR"
            recommendation = "❌ SKIP"
        
        return {
            'prediction': prediction,
            'confidence': round(confidence, 1),
            'probabilities': {
                'home_win': round(prob_1, 1),
                'draw': round(prob_x, 1),
                'away_win': round(prob_2, 1)
            },
            'value_bet': {
                'value_index': round(value_index, 3),
                'rating': value_rating
            },
            'recommendation': recommendation,
            'analysis': {
                'home_form': home_form,
                'away_form': away_form,
                'home_power': round(home_power, 2),
                'away_power': round(away_power, 2)
            },
            'timestamp': datetime.now().isoformat()
        }
