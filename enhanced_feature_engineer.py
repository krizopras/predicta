"""
==============================================================
 Enhanced Feature Engineer v3.5 - IMPROVED
--------------------------------------------------------------
 45 → 70 özelliğe çıkarıldı
 Yeni özellikler:
   ✅ Ev/Deplasman split performansı (ÇOK ÖNEMLİ!)
   ✅ Ağırlıklı form (son maçlar daha önemli)
   ✅ Savunma metrikleri
   ✅ Momentum analizi
   ✅ Rest days (dinlenme avantajı)
   ✅ Motivasyon faktörleri
==============================================================
"""

import math
import logging
import pickle
import os
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="[EnhancedFeatureEngineer] %(message)s")
logger = logging.getLogger(__name__)

# Yeni feature isimleri (70 özellik)
ENHANCED_FEATURE_NAMES_V35 = [
    # Temel oranlar (6)
    "odds_1", "odds_x", "odds_2",
    "prob_1", "prob_x", "prob_2",
    
    # Oran türevleri (8)
    "odds_diff_home_draw", "odds_diff_draw_away", "odds_margin",
    "log_odds_1", "log_odds_x", "log_odds_2",
    "odds_entropy", "odds_variance",
    
    # Takım özellikleri (4)
    "home_len", "away_len", "is_derby", "league_country_score",
    
    # Gelişmiş form ve performans (20) - ARTIRILDI
    "home_form", "away_form", "form_diff",
    "home_goals_avg", "away_goals_avg", "goals_diff",
    "home_win_streak", "away_win_streak",
    "home_last3_trend", "away_last3_trend",
    "home_weighted_form", "away_weighted_form",  # YENİ
    "home_at_home_win_rate", "away_at_away_win_rate",  # YENİ - ÇOK ÖNEMLİ!
    "home_defense_avg", "away_defense_avg",  # YENİ
    "home_clean_sheet_rate", "away_clean_sheet_rate",  # YENİ
    "home_ppg", "away_ppg",  # YENİ - Points per game
    
    # H2H (5)
    "h2h_home_wins", "h2h_draws", "h2h_away_wins",
    "h2h_total_goals_avg", "h2h_home_dominance",
    
    # Zaman özellikleri (13) - ARTIRILDI
    "month", "day_of_week", "is_weekend", "season_phase",
    "days_since_last_match_home", "days_since_last_match_away",
    "rest_advantage",  # YENİ
    "home_fatigued", "away_fatigued",  # YENİ
    "home_optimal_rest", "away_optimal_rest",  # YENİ
    "home_momentum", "away_momentum",  # YENİ
    
    # Lig ve sıralama (14) - ARTIRILDI
    "home_league_position", "away_league_position", "position_diff",
    "home_points", "away_points", "points_diff",
    "home_title_race", "away_title_race",  # YENİ
    "home_relegation_battle", "away_relegation_battle",  # YENİ
    "home_mid_table", "away_mid_table",  # YENİ
    "home_form_last_5", "away_form_last_5"  # YENİ
]

TOTAL_FEATURES_V35 = 70

assert len(ENHANCED_FEATURE_NAMES_V35) == TOTAL_FEATURES_V35, \
    f"Feature sayısı uyuşmuyor: {len(ENHANCED_FEATURE_NAMES_V35)} != {TOTAL_FEATURES_V35}"


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


class EnhancedFeatureEngineer:
    """Geliştirilmiş Feature Engineering v3.5 - 70 özellik"""
    
    def __init__(self, model_path: str = "data/ai_models_v3", flush_every: int = 1000):
        self.model_path = model_path
        self.flush_every = flush_every
        self.update_count = 0
        os.makedirs(model_path, exist_ok=True)
        
        self.team_history = defaultdict(list)
        self.h2h_history = defaultdict(list)
        self.league_standings = defaultdict(dict)
        self.team_last_match_date = defaultdict(lambda: None)
        self.league_stats = defaultdict(dict)
        
        self._load_data()
        
        logger.info(f"🚀 EnhancedFeatureEngineer v3.5 initialized")
        logger.info(f"📊 Total features: {TOTAL_FEATURES_V35}")
    
    def _load_data(self):
        data_file = os.path.join(self.model_path, "enhanced_feature_data.pkl")
        if not os.path.exists(data_file):
            logger.info("ℹ️ Yeni feature data oluşturulacak")
            return False
        
        try:
            with open(data_file, "rb") as f:
                data = pickle.load(f)
            self.team_history = defaultdict(list, data.get("team_history", {}))
            self.h2h_history = defaultdict(list, data.get("h2h_history", {}))
            self.league_standings = defaultdict(dict, data.get("league_standings", {}))
            
            team_dates = {}
            for team, date_str in data.get("team_last_match_date", {}).items():
                if date_str and isinstance(date_str, str):
                    try:
                        team_dates[team] = datetime.fromisoformat(date_str)
                    except ValueError:
                        team_dates[team] = None
            self.team_last_match_date = defaultdict(lambda: None, team_dates)
            
            self.league_stats = defaultdict(dict, data.get("league_stats", {}))
            
            logger.info("✅ Feature data yüklendi")
            logger.info(f"📈 Loaded: {len(self.team_history)} teams, {len(self.h2h_history)} H2H pairs")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Feature data yükleme hatası: {e}")
            return False
    
    def _save_data(self):
        data_file = os.path.join(self.model_path, "enhanced_feature_data.pkl")
        temp_file = data_file + ".tmp"
        try:
            team_dates_str = {}
            for team, date_obj in self.team_last_match_date.items():
                if date_obj and isinstance(date_obj, datetime):
                    team_dates_str[team] = date_obj.isoformat()
                else:
                    team_dates_str[team] = None
            
            data = {
                "team_history": dict(self.team_history),
                "h2h_history": dict(self.h2h_history),
                "league_standings": dict(self.league_standings),
                "team_last_match_date": team_dates_str,
                "league_stats": dict(self.league_stats)
            }
            with open(temp_file, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_file, data_file)
            logger.info("💾 Feature data kaydedildi")
            return True
        except Exception as e:
            logger.error(f"❌ Feature data kaydetme hatası: {e}")
            return False
    
    def extract_features(self, match_data: Dict) -> Optional[np.ndarray]:
        """70 özellikli feature vector üretir"""
        try:
            if not match_data:
                return None
            
            home = safe_lower(match_data.get("home_team"))
            away = safe_lower(match_data.get("away_team"))
            league = safe_lower(match_data.get("league", "unknown"))
            
            if not home or not away:
                return None
            
            # Tüm feature gruplarını çıkar
            feature_groups = [
                self._extract_odds_features(match_data),  # 14
                self._extract_team_features(home, away, league),  # 4
                self._extract_advanced_form_features(home, away),  # 20
                self._extract_h2h_features(home, away),  # 5
                self._extract_time_features(match_data, home, away),  # 13
                self._extract_standings_features(home, away, league)  # 14
            ]
            
            # Birleştir
            features = np.concatenate(feature_groups)
            
            # Güvenlik
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Kontrol
            if len(features) != TOTAL_FEATURES_V35:
                logger.error(f"Feature sayısı hatalı: {len(features)} != {TOTAL_FEATURES_V35}")
                return None
            
            return features.astype(np.float32)
            
        except Exception as e:
            logger.error(f"extract_features hatası: {e}", exc_info=True)
            return None
    
    def _extract_odds_features(self, match_data: Dict) -> np.ndarray:
        """Oran özellikleri (14)"""
        odds = match_data.get("odds", {})
        if not isinstance(odds, dict):
            odds = {"1": 2.0, "X": 3.0, "2": 3.5}
        
        odds_1 = safe_float(odds.get("1", 2.0))
        odds_x = safe_float(odds.get("X", 3.0))
        odds_2 = safe_float(odds.get("2", 3.5))
        
        if any(v <= 1.01 or v > 100 for v in [odds_1, odds_x, odds_2]):
            odds_1, odds_x, odds_2 = 2.0, 3.0, 3.5
        
        # Olasılıklar
        total = (1/odds_1) + (1/odds_x) + (1/odds_2)
        prob_1 = (1/odds_1) / total
        prob_x = (1/odds_x) / total
        prob_2 = (1/odds_2) / total
        
        # Türevler
        odds_diff_home_draw = odds_1 - odds_x
        odds_diff_draw_away = odds_x - odds_2
        odds_margin = total - 1.0
        
        log_odds_1 = np.log(odds_1) if odds_1 > 0 else 0.0
        log_odds_x = np.log(odds_x) if odds_x > 0 else 0.0
        log_odds_2 = np.log(odds_2) if odds_2 > 0 else 0.0
        
        probs = np.array([prob_1, prob_x, prob_2])
        probs = probs[probs > 0]
        odds_entropy = -np.sum(probs * np.log(probs)) if len(probs) > 0 else 0.0
        
        odds_variance = np.var([odds_1, odds_x, odds_2])
        
        return np.array([
            odds_1, odds_x, odds_2,
            prob_1, prob_x, prob_2,
            odds_diff_home_draw, odds_diff_draw_away, odds_margin,
            log_odds_1, log_odds_x, log_odds_2,
            odds_entropy, odds_variance
        ])
    
    def _extract_team_features(self, home: str, away: str, league: str) -> np.ndarray:
        """Takım özellikleri (4)"""
        home_len = len(home)
        away_len = len(away)
        is_derby = float(self._is_derby(home, away))
        league_score = self._get_league_score(league)
        
        return np.array([home_len, away_len, is_derby, league_score])
    
    def _extract_advanced_form_features(self, home: str, away: str) -> np.ndarray:
        """Gelişmiş form özellikleri (20) - YENİ!"""
        home_data = self._calculate_comprehensive_form(home)
        away_data = self._calculate_comprehensive_form(away)
        
        form_diff = home_data["form_score"] - away_data["form_score"]
        goals_diff = home_data["goals_avg"] - away_data["goals_avg"]
        
        return np.array([
            # Temel form
            home_data["form_score"],
            away_data["form_score"],
            form_diff,
            home_data["goals_avg"],
            away_data["goals_avg"],
            goals_diff,
            home_data["win_streak"],
            away_data["win_streak"],
            home_data["last3_trend"],
            away_data["last3_trend"],
            # YENİ: Ağırlıklı form
            home_data["weighted_form"],
            away_data["weighted_form"],
            # YENİ: Ev/Deplasman split (ÇOK ÖNEMLİ!)
            home_data["home_win_rate"],
            away_data["away_win_rate"],
            # YENİ: Savunma
            home_data["defense_avg"],
            away_data["defense_avg"],
            # YENİ: Clean sheets
            home_data["clean_sheet_rate"],
            away_data["clean_sheet_rate"],
            # YENİ: Points per game
            home_data["ppg"],
            away_data["ppg"]
        ])
    
    def _calculate_comprehensive_form(self, team: str) -> Dict:
        """Kapsamlı form hesaplama"""
        history = self.team_history.get(team, [])
        if not history:
            return {
                "form_score": 0.5,
                "goals_avg": 1.5,
                "win_streak": 0,
                "last3_trend": 0.0,
                "weighted_form": 0.5,
                "home_win_rate": 0.45,
                "away_win_rate": 0.30,
                "defense_avg": 1.2,
                "clean_sheet_rate": 0.3,
                "ppg": 1.5
            }
        
        recent = history[-10:]
        
        # Temel form
        wins = sum(1 for m in recent if m.get("result") == "W")
        draws = sum(1 for m in recent if m.get("result") == "D")
        points = wins * 3 + draws * 1
        form_score = points / (len(recent) * 3) if recent else 0.5
        
        # Gol ortalaması
        goals_for = [safe_int(m.get("goals_for", 0)) for m in recent]
        goals_avg = np.mean(goals_for) if goals_for else 1.5
        
        # Kazanma serisi
        win_streak = 0
        for m in reversed(recent):
            if m.get("result") == "W":
                win_streak += 1
            else:
                break
        
        # Son 3 maç trendi
        last3 = goals_for[-3:] if len(goals_for) >= 3 else goals_for
        if len(last3) >= 2:
            last3_trend = (last3[-1] - last3[0]) / max(1, len(last3))
        else:
            last3_trend = 0.0
        
        # YENİ: Ağırlıklı form (son maçlar daha önemli)
        weights = [0.1, 0.15, 0.2, 0.25, 0.3]  # Son 5 maç için
        recent_5 = recent[-5:]
        if len(recent_5) > 0:
            weighted_points = sum(
                w * (3 if m['result'] == 'W' else 1 if m['result'] == 'D' else 0)
                for w, m in zip(weights[:len(recent_5)], recent_5)
            )
            weighted_form = weighted_points / (sum(weights[:len(recent_5)]) * 3)
        else:
            weighted_form = 0.5
        
        # YENİ: Ev/Deplasman split (ÇOK ÖNEMLİ!)
        home_matches = [m for m in history if m.get('venue') == 'home'][-10:]
        away_matches = [m for m in history if m.get('venue') == 'away'][-10:]
        
        home_wins = sum(1 for m in home_matches if m['result'] == 'W')
        home_win_rate = home_wins / len(home_matches) if home_matches else 0.45
        
        away_wins = sum(1 for m in away_matches if m['result'] == 'W')
        away_win_rate = away_wins / len(away_matches) if away_matches else 0.30
        
        # YENİ: Savunma metrikleri
        goals_against = [safe_int(m.get("goals_against", 0)) for m in recent]
        defense_avg = np.mean(goals_against) if goals_against else 1.2
        
        # YENİ: Clean sheet rate
        clean_sheets = sum(1 for m in recent if m.get("goals_against", 1) == 0)
        clean_sheet_rate = clean_sheets / len(recent) if recent else 0.3
        
        # YENİ: Points per game
        ppg = points / len(recent) if recent else 1.5
        
        return {
            "form_score": round(form_score, 3),
            "goals_avg": round(goals_avg, 2),
            "win_streak": win_streak,
            "last3_trend": round(last3_trend, 2),
            "weighted_form": round(weighted_form, 3),
            "home_win_rate": round(home_win_rate, 3),
            "away_win_rate": round(away_win_rate, 3),
            "defense_avg": round(defense_avg, 2),
            "clean_sheet_rate": round(clean_sheet_rate, 3),
            "ppg": round(ppg, 2)
        }
    
    def _extract_h2h_features(self, home: str, away: str) -> np.ndarray:
        """H2H özellikleri (5)"""
        key = self._get_h2h_key(home, away)
        h2h = self.h2h_history.get(key, [])
        
        if not h2h:
            return np.array([0, 0, 0, 2.5, 0.5])
        
        home_wins = sum(1 for m in h2h if m.get("result") == "1")
        draws = sum(1 for m in h2h if m.get("result") == "X")
        away_wins = sum(1 for m in h2h if m.get("result") == "2")
        
        total_goals = [safe_int(m.get("total_goals", 0)) for m in h2h]
        h2h_total_goals_avg = np.mean(total_goals) if total_goals else 2.5
        
        total_matches = len(h2h)
        h2h_home_dominance = home_wins / total_matches if total_matches > 0 else 0.5
        
        return np.array([
            home_wins, draws, away_wins,
            h2h_total_goals_avg, h2h_home_dominance
        ])
    
    def _get_h2h_key(self, home: str, away: str) -> str:
        return "_vs_".join(sorted([home, away]))
    
    def _extract_time_features(self, match_data: Dict, home: str, away: str) -> np.ndarray:
        """Zaman özellikleri (13) - Geliştirilmiş"""
        date_str = match_data.get("date", "")
        
        try:
            if date_str:
                match_date = datetime.strptime(date_str, "%Y-%m-%d")
            else:
                match_date = datetime.now()
        except:
            match_date = datetime.now()
        
        month = match_date.month
        day_of_week = match_date.weekday()
        is_weekend = 1.0 if day_of_week >= 5 else 0.0
        
        # Sezon dönemi
        if month in [8, 9]:
            season_phase = 1
        elif month in [10, 11, 12, 1]:
            season_phase = 2
        elif month in [2, 3, 4]:
            season_phase = 3
        else:
            season_phase = 4
        
        # Rest days
        days_since_last_home = self._days_since_last_match(home, match_date)
        days_since_last_away = self._days_since_last_match(away, match_date)
        
        # YENİ: Rest advantage
        rest_advantage = days_since_last_home - days_since_last_away
        
        # YENİ: Fatigue indicators
        home_fatigued = 1.0 if days_since_last_home < 3 else 0.0
        away_fatigued = 1.0 if days_since_last_away < 3 else 0.0
        
        # YENİ: Optimal rest
        home_optimal_rest = 1.0 if 3 <= days_since_last_home <= 5 else 0.0
        away_optimal_rest = 1.0 if 3 <= days_since_last_away <= 5 else 0.0
        
        # YENİ: Momentum
        home_momentum = self._calculate_momentum(home)
        away_momentum = self._calculate_momentum(away)
        
        return np.array([
            month, day_of_week, is_weekend, season_phase,
            days_since_last_home, days_since_last_away,
            rest_advantage,
            home_fatigued, away_fatigued,
            home_optimal_rest, away_optimal_rest,
            home_momentum, away_momentum
        ])
    
    def _days_since_last_match(self, team: str, current_date: datetime) -> float:
        """Son maçtan beri geçen gün"""
        last_date = self.team_last_match_date.get(team)
        if last_date is None:
            return 7.0
        
        try:
            days = (current_date - last_date).days
            return float(max(0, min(days, 30)))
        except:
            return 7.0
    
    def _calculate_momentum(self, team: str) -> float:
        """Momentum hesapla (son 3 vs önceki 3 maç)"""
        history = self.team_history.get(team, [])
        if len(history) < 3:
            return 0.0
        
        recent_3 = history[-3:]
        earlier_3 = history[-6:-3] if len(history) >= 6 else history[:3]
        
        recent_points = sum(
            3 if m['result'] == 'W' else 1 if m['result'] == 'D' else 0
            for m in recent_3
        ) / 3
        
        earlier_points = sum(
            3 if m['result'] == 'W' else 1 if m['result'] == 'D' else 0
            for m in earlier_3
        ) / 3
        
        momentum = recent_points - earlier_points
        return round(momentum, 2)
    
    def _extract_standings_features(self, home: str, away: str, league: str) -> np.ndarray:
        """Lig sıralaması özellikleri (14) - Geliştirilmiş"""
        standings = self.league_standings.get(league, {})
        league_size = len(standings) if standings else 20
        
        home_data = standings.get(home, {"position": 10, "points": 30})
        away_data = standings.get(away, {"position": 10, "points": 30})
        
        home_pos = safe_float(home_data.get("position", 10))
        away_pos = safe_float(away_data.get("position", 10))
        pos_diff = home_pos - away_pos
        
        home_points = safe_float(home_data.get("points", 30))
        away_points = safe_float(away_data.get("points", 30))
        points_diff = home_points - away_points
        
        # YENİ: Motivasyon faktörleri
        # Şampiyonluk yarışı
        home_title_race = 1.0 if home_pos <= 4 else 0.0
        away_title_race = 1.0 if away_pos <= 4 else 0.0
        
        # Düşme mücadelesi
        home_relegation = 1.0 if home_pos >= (league_size - 3) else 0.0
        away_relegation = 1.0 if away_pos >= (league_size - 3) else 0.0
        
        # Orta sıra (düşük motivasyon)
        home_mid_table = 1.0 if 5 <= home_pos <= (league_size - 4) else 0.0
        away_mid_table = 1.0 if 5 <= away_pos <= (league_size - 4) else 0.0
        
        # Son 5 maç formu (standings'ten)
        home_form_5 = safe_float(home_data.get("form_last_5", 1.5))
        away_form_5 = safe_float(away_data.get("form_last_5", 1.5))
        
        return np.array([
            home_pos, away_pos, pos_diff,
            home_points, away_points, points_diff,
            home_title_race, away_title_race,
            home_relegation, away_relegation,
            home_mid_table, away_mid_table,
            home_form_5, away_form_5
        ])
    
    def _is_derby(self, home: str, away: str) -> bool:
        derby_teams = ["galatasaray", "fenerbahçe", "beşiktaş", "trabzonspor"]
        home_is_big = any(t in home for t in derby_teams)
        away_is_big = any(t in away for t in derby_teams)
        return home_is_big and away_is_big
    
    def _get_league_score(self, league: str) -> float:
        league_scores = {
            "premier league": 2.5, "la liga": 2.5, "serie a": 2.3,
            "bundesliga": 2.4, "ligue 1": 2.2, "süper lig": 1.8,
            "eredivisie": 1.5, "liga nos": 1.4
        }
        
        for key, score in league_scores.items():
            if key in league:
                return score
        
        if league in self.league_stats and "avg_goals" in self.league_stats[league]:
            avg_goals = self.league_stats[league]["avg_goals"]
            league_score = avg_goals / 1.2
            return float(np.clip(league_score, 1.0, 2.8))
        
        return 1.0
    
    def update_match_result(self, home: str, away: str, league: str, 
                           result: str, home_goals: int, away_goals: int,
                           match_date: str, venue_home: str = "home", venue_away: str = "away"):
        """Maç sonrası güncelleme - venue bilgisi eklendi"""
        home = safe_lower(home)
        away = safe_lower(away)
        league = safe_lower(league)
        
        try:
            match_date_obj = datetime.strptime(match_date, "%Y-%m-%d")
        except:
            match_date_obj = datetime.now()
        
        # Takım geçmişi güncelle (venue ile)
        home_match_data = {
            "result": "W" if result == "1" else ("D" if result == "X" else "L"),
            "goals_for": home_goals,
            "goals_against": away_goals,
            "date": match_date,
            "venue": venue_home  # YENİ
        }
        self.team_history[home].append(home_match_data)
        
        away_match_data = {
            "result": "W" if result == "2" else ("D" if result == "X" else "L"),
            "goals_for": away_goals,
            "goals_against": home_goals,
            "date": match_date,
            "venue": venue_away  # YENİ
        }
        self.team_history[away].append(away_match_data)
        
        # H2H güncelle
        key = self._get_h2h_key(home, away)
        self.h2h_history[key].append({
            "result": result,
            "total_goals": home_goals + away_goals,
            "date": match_date
        })
        
        # Son maç tarihleri
        self.team_last_match_date[home] = match_date_obj
        self.team_last_match_date[away] = match_date_obj
        
        # Lig istatistikleri
        self._update_league_stats(league, home_goals + away_goals)
        
        # Auto-save
        self.update_count += 1
        if self.update_count % self.flush_every == 0:
            self._save_data()
    
    def _update_league_stats(self, league: str, total_goals: int):
        if league not in self.league_stats:
            self.league_stats[league] = {"total_goals": 0, "match_count": 0}
        
        self.league_stats[league]["total_goals"] += total_goals
        self.league_stats[league]["match_count"] += 1
        self.league_stats[league]["avg_goals"] = (
            self.league_stats[league]["total_goals"] / 
            self.league_stats[league]["match_count"]
        )
    
    def update_standings(self, league: str, standings: Dict[str, Dict]):
        league = safe_lower(league)
        self.league_standings[league] = {
            safe_lower(team): data for team, data in standings.items()
        }
    
    def manual_save(self):
        return self._save_data()
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_teams": len(self.team_history),
            "total_h2h_pairs": len(self.h2h_history),
            "total_leagues": len(self.league_standings),
            "update_count": self.update_count,
            "total_features": TOTAL_FEATURES_V35
        }


# TEST
if __name__ == "__main__":
    eng = EnhancedFeatureEngineer()
    
    test_match = {
        "home_team": "Barcelona",
        "away_team": "Real Madrid", 
        "league": "La Liga",
        "odds": {"1": 2.1, "X": 3.3, "2": 3.2},
        "date": "2025-10-15"
    }
    
    features = eng.extract_features(test_match)
    if features is not None:
        print(f"✅ Feature vector: {features.shape}")
        print(f"✅ Expected: {TOTAL_FEATURES_V35}")
        print(f"✅ Match: {len(features) == TOTAL_FEATURES_V35}")
        print(f"✅ Sample: {features[:10]}")
    else:
        print("❌ Feature extraction failed")
