# ==============================================================
#  advanced_feature_engineer.py
#  Predicta Europe ML v2.1 - Güvenli ve Gelişmiş Feature Oluşturma
# ==============================================================

import math
import json
import logging
import pickle
import os
from typing import Dict, Any, Optional, List
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="[FeatureEngineer] %(message)s")
logger = logging.getLogger(__name__)

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


# ==============================================================
#  Helper Fonksiyonlar
# ==============================================================

def safe_lower(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value).lower().strip()
    except Exception:
        return ""


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def ratio(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return round(a / b, 3)


# ==============================================================
#  Ana Sınıf
# ==============================================================

class AdvancedFeatureEngineer:
    """ML pipeline'ları için gelişmiş ve güvenli feature üretici"""
    
    def __init__(self, model_path: str = "data/ai_models_v2"):
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)
        
        self.team_history = defaultdict(list)
        self.h2h_history = defaultdict(list)
        self.league_stats = defaultdict(lambda: {"1": 0, "X": 0, "2": 0})
        self.league_results = defaultdict(lambda: {"1": 0, "X": 0, "2": 0})
        
        self._load_data()
    
    # ----------------------------------------------------------
    #  Veri Kaydetme/Yükleme
    # ----------------------------------------------------------
    def _load_data(self):
        data_file = os.path.join(self.model_path, "feature_data.pkl")
        if not os.path.exists(data_file):
            logger.info("ℹ️ Yeni feature data oluşturulacak")
            return False
        
        try:
            with open(data_file, "rb") as f:
                data = pickle.load(f)
            self.team_history = data.get("team_history", defaultdict(list))
            self.h2h_history = data.get("h2h_history", defaultdict(list))
            self.league_stats = data.get("league_stats", defaultdict(lambda: {"1": 0, "X": 0, "2": 0}))
            self.league_results = data.get("league_results", defaultdict(lambda: {"1": 0, "X": 0, "2": 0}))
            logger.info("✅ Feature data yüklendi")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Feature data yükleme hatası: {e}")
            return False
    
    def _save_data(self):
        data_file = os.path.join(self.model_path, "feature_data.pkl")
        temp_file = data_file + ".tmp"
        try:
            data = {
                "team_history": dict(self.team_history),
                "h2h_history": dict(self.h2h_history),
                "league_stats": dict(self.league_stats),
                "league_results": dict(self.league_results)
            }
            with open(temp_file, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_file, data_file)
            logger.info("💾 Feature data kaydedildi")
            return True
        except Exception as e:
            logger.error(f"❌ Feature data kaydetme hatası: {e}")
            return False
    
    # ----------------------------------------------------------
    #  Ana Feature Fonksiyonu (NaN/None Güvenli)
    # ----------------------------------------------------------
    def extract_features(self, match_data: Dict) -> Optional[np.ndarray]:
        try:
            if not match_data:
                logger.warning("extract_features: match_data is None or empty")
                return None
            
            home = safe_lower(match_data.get("home_team"))
            away = safe_lower(match_data.get("away_team"))
            league = safe_lower(match_data.get("league", "unknown"))
            if not home or not away:
                logger.warning(f"extract_features: Invalid team names - {home}, {away}")
                return None
            
            # Oranlar
            odds = match_data.get("odds", {})
            if not isinstance(odds, dict):
                odds = {"1": 2.0, "X": 3.0, "2": 3.5}
            odds_1 = safe_float(odds.get("1", 2.0))
            odds_x = safe_float(odds.get("X", 3.0))
            odds_2 = safe_float(odds.get("2", 3.5))
            if any(v <= 1.01 or v > 100 for v in [odds_1, odds_x, odds_2]):
                odds_1, odds_x, odds_2 = 2.0, 3.0, 3.5
            
            # Olasılıklar
            try:
                total = (1/odds_1) + (1/odds_x) + (1/odds_2)
                prob_1 = (1/odds_1) / total
                prob_x = (1/odds_x) / total
                prob_2 = (1/odds_2) / total
            except:
                prob_1, prob_x, prob_2 = 0.33, 0.33, 0.34
            
            # Basit istatistikler
            home_len = len(home)
            away_len = len(away)
            is_derby = self._is_derby(home, away)
            league_country_score = self._get_league_score(league)
            
            # Form ve H2H
            home_form_data = self._calculate_form_v2(home, is_home=True)
            away_form_data = self._calculate_form_v2(away, is_home=False)
            h2h_data = self._calculate_h2h(home, away)
            
            features = [
                odds_1, odds_x, odds_2,
                prob_1, prob_x, prob_2,
                0.0, 0.0,
                float(home_len), float(away_len),
                float(is_derby), float(league_country_score),
                float(home_form_data["form_score"]),
                float(away_form_data["form_score"]),
                float(home_form_data["goals_per_match"]),
                float(away_form_data["goals_per_match"]),
                float(h2h_data["home_wins"]),
                float(h2h_data["draws"]),
                float(h2h_data["away_wins"])
            ]
            
            feature_array = np.array(features, dtype=np.float32)
            if np.any(np.isnan(feature_array)) or np.any(np.isinf(feature_array)):
                logger.error(f"extract_features: NaN/Inf detected: {features}")
                return None
            if len(feature_array) != len(FEATURE_NAMES):
                logger.error(f"extract_features: Size mismatch ({len(feature_array)} != {len(FEATURE_NAMES)})")
                return None
            
            return feature_array
        except Exception as e:
            logger.error(f"extract_features: Unexpected error - {e}", exc_info=True)
            return None
    
    # ----------------------------------------------------------
    #  Form Hesaplamaları (v2)
    # ----------------------------------------------------------
    def _calculate_form_v2(self, team: str, is_home: bool = True) -> Dict:
        team_lower = safe_lower(team)
        history = self.team_history.get(team_lower, [])
        if not history:
            return {"form_score": 0.5, "goals_per_match": 1.5}
        
        recent = history[-7:]  # son 7 maç
        wins = sum(1 for m in recent if m.get("result") == "W")
        draws = sum(1 for m in recent if m.get("result") == "D")
        losses = sum(1 for m in recent if m.get("result") == "L")
        
        points = wins * 3 + draws * 1
        form_score = points / (len(recent) * 3)
        
        goals_for = [safe_int(m.get("goals_for", 0)) for m in recent]
        goals_against = [safe_int(m.get("goals_against", 0)) for m in recent]
        avg_goals_for = np.mean(goals_for) if goals_for else 1.2
        avg_goals_against = np.mean(goals_against) if goals_against else 1.2
        
        # Trend analizi: son 3 maçta artış varsa bonus
        last3 = goals_for[-3:] if len(goals_for) >= 3 else goals_for
        if len(last3) >= 2 and last3[-1] > last3[0]:
            form_score += 0.05
        
        form_score = min(1.0, max(0.0, form_score))
        goals_per_match = round(avg_goals_for, 2)
        
        return {"form_score": round(form_score, 3), "goals_per_match": goals_per_match}
    
    # ----------------------------------------------------------
    #  H2H / Derbi / Lig Fonksiyonları
    # ----------------------------------------------------------
    def _calculate_h2h(self, home: str, away: str) -> Dict:
        key = f"{safe_lower(home)}_vs_{safe_lower(away)}"
        h2h = self.h2h_history.get(key, [])
        if not h2h:
            return {"home_wins": 0, "draws": 0, "away_wins": 0}
        return {
            "home_wins": sum(1 for m in h2h if m.get("result") == "1"),
            "draws": sum(1 for m in h2h if m.get("result") == "X"),
            "away_wins": sum(1 for m in h2h if m.get("result") == "2")
        }
    
    def _is_derby(self, home: str, away: str) -> int:
        derby_teams = ["galatasaray", "fenerbahçe", "beşiktaş", "trabzonspor"]
        home_is_big = any(t in safe_lower(home) for t in derby_teams)
        away_is_big = any(t in safe_lower(away) for t in derby_teams)
        return 1 if home_is_big and away_is_big else 0
    
    def _get_league_score(self, league: str) -> float:
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
    
    # ----------------------------------------------------------
    #  Güncelleme Fonksiyonları
    # ----------------------------------------------------------
    def update_team_history(self, team: str, match_data: Dict):
        team_lower = safe_lower(team)
        self.team_history[team_lower].append(match_data)
        if len(self.team_history[team_lower]) > 20:
            self.team_history[team_lower] = self.team_history[team_lower][-20:]
    
    def update_h2h_history(self, home_team: str, away_team: str, match_data: Dict):
        key = f"{safe_lower(home_team)}_vs_{safe_lower(away_team)}"
        self.h2h_history[key].append(match_data)
        if len(self.h2h_history[key]) > 10:
            self.h2h_history[key] = self.h2h_history[key][-10:]
    
    def update_league_results(self, league: str, result: str):
        league_lower = safe_lower(league)
        if result in ["1", "X", "2"]:
            self.league_results[league_lower][result] += 1


# ==============================================================
#  Test Çalıştırma
# ==============================================================

if __name__ == "__main__":
    eng = AdvancedFeatureEngineer()
    test = {
        "home_team": "Barcelona",
        "away_team": "Real Madrid",
        "league": "La Liga",
        "odds": {"1": 2.1, "X": 3.3, "2": 3.2}
    }
    feats = eng.extract_features(test)
    print(f"✅ Feature vector: {feats.shape if feats is not None else None}")
