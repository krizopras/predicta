#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GÜNCELLENMİŞ NESINE TAHMİN MOTORU - AI ENTEGRASYONLU
NesineCompleteFetcher ile entegre edilmiş versiyon
"""

import math
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

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
        # NesineCompleteFetcher import
        try:
            from nesine_fetcher_complete import NesineCompleteFetcher
            self.fetcher = NesineCompleteFetcher
            self.fetcher_available = True
        except ImportError:
            self.fetcher_available = False
            logger.warning("NesineCompleteFetcher bulunamadı, fallback modunda çalışıyor")
        
        # Gelişmiş ağırlık sistemi
        self.factor_weights = {
            'xg_base': 1.0,
            'strength_home': 0.8,
            'strength_away': 0.7,
            'form_score': 0.5,
            'recent_goals': 0.4,
            'position_diff': 0.3
        }
    
    async def fetch_nesine_matches(self, league_filter: str = "all") -> List[Dict]:
        """NesineCompleteFetcher ile maç verilerini çek"""
        
        if not self.fetcher_available:
            logger.warning("Fetcher kullanılamıyor, fallback veriler kullanılıyor")
            return self._get_fallback_matches()
        
        try:
            async with self.fetcher() as fetcher:
                matches = await fetcher.fetch_matches_with_odds_and_stats(league_filter=league_filter)
                logger.info(f"✅ {len(matches)} maç başarıyla çekildi")
                return matches
                
        except Exception as e:
            logger.error(f"❌ Nesine veri çekme hatası: {e}")
            return self._get_fallback_matches()
    
    def predict_match_comprehensive(self, match_data: Dict) -> Dict[str, Any]:
        """Bir maç için kapsamlı tahmin yapar"""
        try:
            home_stats = match_data.get('home_stats', {})
            away_stats = match_data.get('away_stats', {})
            
            # Eğer istatistik yoksa, generate et
            if not home_stats:
                home_stats = self._generate_team_stats(match_data['home_team'], match_data.get('league', 'Süper Lig'))
            if not away_stats:
                away_stats = self._generate_team_stats(match_data['away_team'], match_data.get('league', 'Süper Lig'))
            
            match_info = {
                'importance': match_data.get('importance', 1.0),
                'weather_impact': match_data.get('weather_impact', 1.0)
            }
            
            # Gelişmiş tahmin
            prediction = self.predict_match_advanced(home_stats, away_stats, match_info)
            
            return {
                'result_prediction': prediction.get('ms_prediction', 'X'),
                'confidence': prediction.get('confidence', 50.0),
                'iy_prediction': prediction.get('iy_prediction', 'X'),
                'score_prediction': prediction.get('predicted_score', '1-1'),
                'probabilities': prediction.get('probabilities', {}),
                'home_xg': prediction.get('home_xg', 1.5),
                'away_xg': prediction.get('away_xg', 1.5),
                'risk_level': prediction.get('risk_level', 'medium'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Tahmin hesaplama hatası: {e}")
            return self._get_fallback_prediction()

    def calculate_strength(self, team_stats: Dict, is_home: bool) -> Tuple[float, float]:
        """Gol atma ve gol yeme gücünü hesaplar."""
        
        matches_played = team_stats.get('matches_played', 1)
        if matches_played == 0:
            return 1.0, 1.0
            
        avg_goals_for = team_stats.get('goals_for', 0) / matches_played
        avg_goals_against = team_stats.get('goals_against', 0) / matches_played
        
        total_teams = team_stats.get('total_teams', 20)
        position = team_stats.get('position', 10)
        position_factor = (total_teams - position + 1) / total_teams
        
        attack_strength = avg_goals_for * (1 + position_factor * 0.2)
        defense_strength = 1 / (avg_goals_against * (1 + position_factor * 0.1) + 0.1)  # Sıfıra bölme hatasını önle
        
        attack_factor = (attack_strength + team_stats.get('wins', 0) * 0.1) / 1.5
        defense_factor = (defense_strength + team_stats.get('losses', 0) * 0.1) / 1.5
        
        if is_home:
            attack_factor *= 1.15
            defense_factor *= 0.95
        else:
            attack_factor *= 0.95
            defense_factor *= 1.15
            
        return max(0.1, attack_factor), max(0.1, defense_factor)

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
        
        for i, result in enumerate(recent_form[:5]):
            weight = 5 - i
            total_weight += weight
            
            if result == 'G': 
                score += 1.0 * weight
            elif result == 'B': 
                score += 0.5 * weight
        
        return score / total_weight if total_weight > 0 else 0.5

    def calculate_xg(self, home_stats: Dict, away_stats: Dict, match_info: Dict) -> Tuple[float, float]:
        """Beklenen Gol (xG) değerlerini hesaplar."""
        
        league_avg_goals = 2.5 
        
        home_attack, home_defense = self.calculate_strength(home_stats, is_home=True)
        away_attack, away_defense = self.calculate_strength(away_stats, is_home=False)
        
        home_xg_base = home_attack * away_defense * league_avg_goals
        away_xg_base = away_attack * home_defense * league_avg_goals
        
        home_recent_form = home_stats.get('recent_form', [])
        away_recent_form = away_stats.get('recent_form', [])
        
        home_xg = self.estimate_recent_goals(home_recent_form, home_xg_base)
        away_xg = self.estimate_recent_goals(away_recent_form, away_xg_base)
        
        home_conceded_factor = self.estimate_recent_conceded_goals(home_recent_form, 1.0)
        away_conceded_factor = self.estimate_recent_conceded_goals(away_recent_form, 1.0)
        
        home_xg *= (2.0 - away_conceded_factor) 
        away_xg *= (2.0 - home_conceded_factor)
        
        home_xg *= match_info.get('importance', 1.0) * match_info.get('weather_impact', 1.0)
        away_xg *= match_info.get('importance', 1.0) * match_info.get('weather_impact', 1.0)
        
        home_xg = max(0.5, min(3.5, home_xg))
        away_xg = max(0.5, min(3.5, away_xg))
        
        return home_xg, away_xg

    def poisson_probability(self, mean: float, k: int) -> float:
        """Belirli bir ortalama (lambda) için k olayı olasılığını hesaplar."""
        if k < 0: 
            return 0.0
        try:
            return (mean ** k * math.exp(-mean)) / math.factorial(k)
        except:
            return 0.0

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
        
        prob_1 = probabilities.get('1', 0)
        prob_x = probabilities.get('X', 0) 
        prob_2 = probabilities.get('2', 0)
        xg_diff = home_xg - away_xg
        
        if prob_1 > prob_2 + 10 and prob_1 > prob_x + 10 and xg_diff > 0.3:
            return '1'
        elif prob_2 > prob_1 + 10 and prob_2 > prob_x + 10 and xg_diff < -0.3:
            return '2'
        elif prob_x > max(prob_1, prob_2) or abs(prob_1 - prob_2) < 8:
            return 'X'
        elif xg_diff > 0.2:
            return '1'
        elif xg_diff < -0.2:
            return '2'
        else:
            return 'X'

    def predict_most_likely_scores(self, home_xg: float, away_xg: float, ms_prediction: str) -> List[Dict]:
        """MS tahmini ile uyumlu skor tahmini"""
        
        score_probs = []
        max_goals = 6
        
        for home_goals in range(max_goals):
            for away_goals in range(max_goals):
                home_prob = self.poisson_probability(home_xg, home_goals)
                away_prob = self.poisson_probability(away_xg, away_goals)
                combined_prob = home_prob * away_prob
                
                score_ms = self.get_score_ms_prediction(home_goals, away_goals)
                
                # Uyumlu olanı %60 artır, uyumsuz olanı %60 düşür
                if score_ms == ms_prediction:
                    combined_prob *= 1.6 
                else:
                    combined_prob *= 0.4
                
                score_probs.append({
                    'score': f"{home_goals}-{away_goals}",
                    'probability': round(combined_prob * 100, 2),
                    'ms_match': score_ms == ms_prediction
                })
        
        score_probs.sort(key=lambda x: x['probability'], reverse=True)
        
        return score_probs[:5]
    
    def predict_first_half(self, home_xg: float, away_xg: float, probabilities: Dict) -> str:
        """İlk yarı tahmini"""
        
        iy_home_xg = home_xg * 0.38
        iy_away_xg = away_xg * 0.38
        
        iy_probs = self.calculate_match_probabilities(iy_home_xg, iy_away_xg)
        
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
    
    def predict_match_advanced(self, home_stats: Dict, away_stats: Dict, match_info: Dict) -> Dict[str, Any]:
        """Tüm faktörleri kullanarak gelişmiş maç tahmini yapar."""
        
        home_xg, away_xg = self.calculate_xg(home_stats, away_stats, match_info)
        probabilities = self.calculate_match_probabilities(home_xg, away_xg)
        ms_prediction = self.determine_match_result(probabilities, home_xg, away_xg)
        iy_prediction = self.predict_first_half(home_xg, away_xg, probabilities)
        score_probs = self.predict_most_likely_scores(home_xg, away_xg, ms_prediction)
        
        home_attack, home_defense = self.calculate_strength(home_stats, True)
        away_attack, away_defense = self.calculate_strength(away_stats, False)
        
        confidence = self.calculate_prediction_confidence(home_attack, away_attack,
                                                        probabilities, home_xg, away_xg)
        risk = self.assess_prediction_risk(home_stats, away_stats, probabilities)
        
        return {
            'ms_prediction': ms_prediction,
            'iy_prediction': iy_prediction,
            'predicted_score': score_probs[0]['score'] if score_probs else '1-1',
            'probabilities': probabilities,
            'score_probabilities': score_probs,
            'confidence': round(confidence, 2),
            'risk_level': risk,
            'home_xg': round(home_xg, 2),
            'away_xg': round(away_xg, 2),
            'last_update': datetime.now().isoformat()
        }

    def _generate_team_stats(self, team_name: str, league: str) -> Dict:
        """Takım için varsayılan istatistikler üretir"""
        
        strong_teams = ['Galatasaray', 'Fenerbahçe', 'Beşiktaş', 'Trabzonspor', 
                       'Real Madrid', 'Barcelona', 'Manchester City', 'Liverpool']
        
        is_strong = any(strong in team_name for strong in strong_teams)
        
        if is_strong:
            base_stats = {
                'position': np.random.randint(1, 5),
                'points': np.random.randint(20, 35),
                'wins': np.random.randint(7, 12),
                'draws': np.random.randint(1, 4),
                'losses': np.random.randint(0, 3),
                'goals_for': np.random.randint(18, 35),
                'goals_against': np.random.randint(5, 15)
            }
        else:
            base_stats = {
                'position': np.random.randint(8, 18),
                'points': np.random.randint(8, 25),
                'wins': np.random.randint(2, 8),
                'draws': np.random.randint(2, 6),
                'losses': np.random.randint(3, 8),
                'goals_for': np.random.randint(8, 22),
                'goals_against': np.random.randint(12, 28)
            }
        
        matches_played = base_stats['wins'] + base_stats['draws'] + base_stats['losses']
        
        # Form bilgisi
        form_options = [['G', 'G', 'B', 'G', 'M'], ['B', 'G', 'G', 'B', 'G'], 
                       ['M', 'B', 'G', 'M', 'B'], ['G', 'M', 'G', 'G', 'B']]
        recent_form = form_options[np.random.randint(0, len(form_options))]
        
        return {
            'name': team_name,
            'league': league,
            'position': base_stats['position'],
            'points': base_stats['points'],
            'matches_played': matches_played,
            'wins': base_stats['wins'],
            'draws': base_stats['draws'],
            'losses': base_stats['losses'],
            'goals_for': base_stats['goals_for'],
            'goals_against': base_stats['goals_against'],
            'recent_form': recent_form,
            'total_teams': 20
        }

    def _get_fallback_matches(self) -> List[Dict]:
        """Fallback maç verileri"""
        return [
            {
                'home_team': 'Galatasaray',
                'away_team': 'Fenerbahçe',
                'league': 'Süper Lig',
                'date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'time': '19:00',
                'odds': {'1': 2.10, 'X': 3.20, '2': 3.40},
                'home_stats': self._generate_team_stats('Galatasaray', 'Süper Lig'),
                'away_stats': self._generate_team_stats('Fenerbahçe', 'Süper Lig'),
                'importance': 1.2,
                'weather_impact': 1.0
            }
        ]

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

# Kullanım örneği
async def main():
    predictor = NesineAdvancedPredictor()
    matches = await predictor.fetch_nesine_matches("super-lig")
    
    if matches:
        prediction = predictor.predict_match_comprehensive(matches[0])
        print("Tahmin:", prediction)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
