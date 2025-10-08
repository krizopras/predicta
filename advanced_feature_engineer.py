# ==============================================================
#  advanced_feature_engineer.py
#  Predicta Europe ML v2 - Güvenli Feature Oluşturma (FIXED)
# ==============================================================

import math
import json
import logging
from typing import Dict, Any, Optional, List
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="[FeatureEngineer] %(message)s")

FEATURE_NAMES = [
    "odds_1", "odds_x", "odds_2",
    "prob_1", "prob_x", "prob_2",
    "goal_diff", "total_goals",
    "home_len", "away_len",
    "is_derby", "league_country_score",
    # Form features
    "home_form", "away_form",
    "home_goals_avg", "away_goals_avg",
    "h2h_home_wins", "h2h_draws", "h2h_away_wins"
]

def safe_lower(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value).lower()
    except Exception:
        return ""

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def ratio(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return round(a / b, 3)

def goal_diff(home_goals: int, away_goals: int) -> int:
    return home_goals - away_goals

class AdvancedFeatureEngineer:
    """ML pipeline'ları için feature üretici (TÜM METOTLAR EKSİKSİZ)"""
    
    def __init__(self):
        self.team_history = defaultdict(list)
        self.h2h_history = defaultdict(list)
        self.league_stats = defaultdict(lambda: {"1": 0, "X": 0, "2": 0})
        self.league_results = defaultdict(lambda: {"1": 0, "X": 0, "2": 0})
    
    def extract_features(self, match_data: Dict) -> np.ndarray:
        """Maç verisinden feature vektörü üretir"""
        try:
            home = safe_lower(match_data.get("home_team"))
            away = safe_lower(match_data.get("away_team"))
            league = safe_lower(match_data.get("league"))
            
            # Oranlar
            odds = match_data.get("odds", {})
            odds_1 = safe_float(odds.get("1", 2.0))
            odds_x = safe_float(odds.get("X", 3.0))
            odds_2 = safe_float(odds.get("2", 3.5))
            
            # Geçersiz oran kontrolü
            if any(v <= 1.01 or math.isnan(v) for v in [odds_1, odds_x, odds_2]):
                odds_1, odds_x, odds_2 = 2.0, 3.0, 3.5
            
            # Olasılıklar
            odds_sum = odds_1 + odds_x + odds_2
            prob_1 = ratio(1 / odds_1, 1 / odds_sum) if odds_sum > 0 else 0.33
            prob_x = ratio(1 / odds_x, 1 / odds_sum) if odds_sum > 0 else 0.33
            prob_2 = ratio(1 / odds_2, 1 / odds_sum) if odds_sum > 0 else 0.33
            
            # Basit özellikler
            home_len = len(home)
            away_len = len(away)
            is_derby = self._is_derby(home, away)
            league_country_score = self._get_league_score(league)
            
            # Form özellikleri
            home_form_data = self._calculate_form(home, is_home=True)
            away_form_data = self._calculate_form(away, is_home=False)
            
            # H2H özellikleri
            h2h_data = self._calculate_h2h(home, away)
            
            # Feature vector
            features = [
                odds_1, odds_x, odds_2,
                prob_1, prob_x, prob_2,
                0, 0,  # goal_diff, total_goals (geçmiş maç için)
                home_len, away_len,
                is_derby, league_country_score,
                home_form_data["form_score"],
                away_form_data["form_score"],
                home_form_data["goals_per_match"],
                away_form_data["goals_per_match"],
                h2h_data["home_wins"],
                h2h_data["draws"],
                h2h_data["away_wins"]
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logging.error(f"Feature extraction hatası: {e}")
            return np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    
    def _calculate_form(self, team: str, is_home: bool = True) -> Dict:
        """
        EKSİK METOT - Takım formunu hesaplar
        """
        team_lower = safe_lower(team)
        history = self.team_history.get(team_lower, [])
        
        if not history:
            return {
                "form_score": 0.5,
                "goals_per_match": 1.5,
                "wins": 0,
                "draws": 0,
                "losses": 0
            }
        
        # Son 5 maç
        recent = history[-5:]
        
        wins = sum(1 for m in recent if m.get("result") == "W")
        draws = sum(1 for m in recent if m.get("result") == "D")
        losses = sum(1 for m in recent if m.get("result") == "L")
        
        # Form skoru (0-1 arası)
        points = wins * 3 + draws * 1
        max_points = len(recent) * 3
        form_score = points / max_points if max_points > 0 else 0.5
        
        # Ortalama gol
        total_goals = sum(safe_int(m.get("goals_for", 0)) for m in recent)
        goals_per_match = total_goals / len(recent) if recent else 1.5
        
        return {
            "form_score": round(form_score, 3),
            "goals_per_match": round(goals_per_match, 2),
            "wins": wins,
            "draws": draws,
            "losses": losses
        }
    
    def _calculate_h2h(self, home: str, away: str) -> Dict:
        """Karşılıklı maç geçmişini hesaplar"""
        key = f"{safe_lower(home)}_vs_{safe_lower(away)}"
        h2h = self.h2h_history.get(key, [])
        
        if not h2h:
            return {"home_wins": 0, "draws": 0, "away_wins": 0}
        
        home_wins = sum(1 for m in h2h if m.get("result") == "1")
        draws = sum(1 for m in h2h if m.get("result") == "X")
        away_wins = sum(1 for m in h2h if m.get("result") == "2")
        
        return {
            "home_wins": home_wins,
            "draws": draws,
            "away_wins": away_wins
        }
    
    def _is_derby(self, home: str, away: str) -> int:
        """Derbi maçı kontrolü"""
        derby_teams = ["galatasaray", "fenerbahçe", "beşiktaş", "trabzonspor"]
        home_lower = safe_lower(home)
        away_lower = safe_lower(away)
        
        home_is_big = any(t in home_lower for t in derby_teams)
        away_is_big = any(t in away_lower for t in derby_teams)
        
        return 1 if (home_is_big and away_is_big) else 0
    
    def _get_league_score(self, league: str) -> float:
        """Lig kalite skoru"""
        league_lower = safe_lower(league)
        
        if "turk" in league_lower or "süper" in league_lower:
            return 1.0
        elif "eng" in league_lower or "prem" in league_lower:
            return 2.0
        elif "esp" in league_lower or "lalig" in league_lower:
            return 2.5
        elif "ital" in league_lower or "serie" in league_lower:
            return 2.2
        elif "germ" in league_lower or "bund" in league_lower:
            return 2.3
        elif "fran" in league_lower or "ligue" in league_lower:
            return 2.0
        
        return 1.0
    
    def update_team_history(self, team: str, match_data: Dict):
        """Takım geçmişini günceller"""
        team_lower = safe_lower(team)
        self.team_history[team_lower].append(match_data)
        
        # Son 20 maçı tut
        if len(self.team_history[team_lower]) > 20:
            self.team_history[team_lower] = self.team_history[team_lower][-20:]
    
    def update_h2h_history(self, home_team: str, away_team: str, match_data: Dict):
        """Karşılıklı maç geçmişini günceller"""
        key = f"{safe_lower(home_team)}_vs_{safe_lower(away_team)}"
        self.h2h_history[key].append(match_data)
        
        if len(self.h2h_history[key]) > 10:
            self.h2h_history[key] = self.h2h_history[key][-10:]
    
    def update_league_results(self, league: str, result: str):
        """Lig istatistiklerini günceller"""
        league_lower = safe_lower(league)
        if result in ["1", "X", "2"]:
            self.league_results[league_lower][result] += 1

# Test
if __name__ == "__main__":
    engineer = AdvancedFeatureEngineer()
    
    test_match = {
        "home_team": "Galatasaray",
        "away_team": "Fenerbahçe",
        "league": "Turkey Super Lig",
        "odds": {"1": 2.05, "X": 3.30, "2": 3.40}
    }
    
    features = engineer.extract_features(test_match)
    print(f"✅ Feature vector shape: {features.shape}")
    print(f"✅ Feature names: {len(FEATURE_NAMES)}")
    print(f"✅ Sample features: {features[:5]}")
