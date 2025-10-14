"""
==============================================================
 Enhanced Feature Engineer v4.0 - MAXIMUM PERFORMANCE (FINAL)
--------------------------------------------------------------
 70 → 100+ özelliğe çıkarıldı
 Yeni özellikler:
   ✅ Gelişmiş oran metrikleri (favorite_gap, betting_heat)
   ✅ ELO rating sistemi
   ✅ Çok daha detaylı form analizi
   ✅ Maç önem seviyesi (kritik maç tespiti)
   ✅ Lig kalite endeksi
   ✅ Gelişmiş zaman özellikleri
   ✅ Güvenli veri kaydetme (hata korumalı)
==============================================================
"""

import math
import logging
import pickle
import os
from typing import Dict, Any, Optional
import numpy as np
from collections import defaultdict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="[EnhancedFeatureEngineer] %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================
# FEATURE ADLARI
# ==============================================================

ENHANCED_FEATURE_NAMES_V4 = [
    # Temel oranlar (6)
    "odds_1", "odds_x", "odds_2",
    "prob_1", "prob_x", "prob_2",

    # Oran türevleri (18)
    "odds_diff_home_draw", "odds_diff_draw_away", "odds_margin",
    "log_odds_1", "log_odds_x", "log_odds_2",
    "odds_entropy", "odds_variance",
    "favorite_gap", "underdog_value", "draw_premium",
    "odds_imbalance", "min_odds", "max_odds",
    "odds_ratio_home_away", "prob_diff_home_away", "odds_confidence",
    "betting_heat",

    # Takım özellikleri (8)
    "home_len", "away_len", "is_derby", "league_country_score",
    "home_elo", "away_elo", "elo_diff", "league_quality_index",

    # Form ve performans (28)
    "home_form", "away_form", "form_diff",
    "home_goals_avg", "away_goals_avg", "goals_diff",
    "home_win_streak", "away_win_streak",
    "home_last3_trend", "away_last3_trend",
    "home_weighted_form", "away_weighted_form",
    "home_at_home_win_rate", "away_at_away_win_rate",
    "home_defense_avg", "away_defense_avg",
    "home_clean_sheet_rate", "away_clean_sheet_rate",
    "home_ppg", "away_ppg",
    "home_attack_strength", "away_attack_strength",
    "home_defense_strength", "away_defense_strength",
    "home_recent_goals_avg", "away_recent_goals_avg",
    "home_consistency", "away_consistency",
    "form_momentum_diff",

    # H2H (7)
    "h2h_home_wins", "h2h_draws", "h2h_away_wins",
    "h2h_total_goals_avg", "h2h_home_dominance",
    "h2h_recent_trend", "h2h_goal_variance",

    # Zaman özellikleri (15)
    "month", "day_of_week", "is_weekend", "season_phase",
    "days_since_last_match_home", "days_since_last_match_away",
    "rest_advantage",
    "home_fatigued", "away_fatigued",
    "home_optimal_rest", "away_optimal_rest",
    "home_momentum", "away_momentum",
    "is_midweek", "season_progress",

    # Lig ve sıralama (18)
    "home_league_position", "away_league_position", "position_diff",
    "home_points", "away_points", "points_diff",
    "home_title_race", "away_title_race",
    "home_relegation_battle", "away_relegation_battle",
    "home_mid_table", "away_mid_table",
    "home_form_last_5", "away_form_last_5",
    "match_importance_home", "match_importance_away",
    "is_critical_match", "motivation_imbalance"
]

TOTAL_FEATURES_V4 = 101
assert len(ENHANCED_FEATURE_NAMES_V4) == TOTAL_FEATURES_V4


# ==============================================================
# YARDIMCI FONKSİYONLAR
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


# ==============================================================
# ANA SINIF
# ==============================================================

class EnhancedFeatureEngineer:
    """Geliştirilmiş Feature Engineering v4.0 - 100+ özellik"""

    def __init__(self, model_path: str = "data/ai_models_v3", flush_every: int = 1000):
        # ✅ 4 boşluk girinti
        self.model_path = os.path.abspath(model_path)
        os.makedirs(self.model_path, exist_ok=True)

        # ✅ Alt klasör oluşturma
        self.data_dir = os.path.join(self.model_path, "feature_engineer_data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.flush_every = flush_every
        self.update_count = 0

        # ✅ Diğer değişkenlerin tanımları
        self.team_history = defaultdict(list)
        self.h2h_history = defaultdict(list)
        self.league_standings = defaultdict(dict)
        self.team_last_match_date = defaultdict(lambda: None)
        self.league_stats = defaultdict(dict)
        self.team_elo = defaultdict(lambda: 1500.0)

        # ✅ Kayıtlı verileri yükle
        self._load_history()
        self._load_elo_system()
        self._load_standings()

        logger.info("🚀 EnhancedFeatureEngineer v4.0 initialized")
        logger.info(f"📊 Total features: {TOTAL_FEATURES_V4}")


    # ==========================================================
    # VERİ YÜKLEME & KAYDETME
    # ==========================================================

    def _load_history(self):
        try:
            fpath = os.path.join(self.data_dir, "history.pkl")
            if os.path.exists(fpath):
                with open(fpath, "rb") as f:
                    data = pickle.load(f)
                    self.team_history = data.get("team_history", defaultdict(list))
                    self.h2h_history = data.get("h2h_history", defaultdict(list))
                logger.info("✅ History data loaded")
        except Exception as e:
            logger.warning(f"⚠️ Could not load history: {e}")

    def _load_elo_system(self):
        try:
            fpath = os.path.join(self.data_dir, "elo_ratings.pkl")
            if os.path.exists(fpath):
                with open(fpath, "rb") as f:
                    self.team_elo = pickle.load(f)
                logger.info("✅ ELO ratings loaded")
        except Exception as e:
            logger.warning(f"⚠️ Could not load ELO: {e}")

    def _load_standings(self):
        try:
            fpath = os.path.join(self.data_dir, "standings.pkl")
            if os.path.exists(fpath):
                with open(fpath, "rb") as f:
                    data = pickle.load(f)
                    self.league_standings = data.get("league_standings", defaultdict(dict))
                    self.league_stats = data.get("league_stats", defaultdict(dict))
                logger.info("✅ Standings data loaded")
        except Exception as e:
            logger.warning(f"⚠️ Could not load standings: {e}")

    def _save_data(self):
        """Feature verilerini güvenli şekilde kaydeder"""
        data_file = os.path.join(self.model_path, "enhanced_feature_data_v4.pkl")
        temp_file = data_file + ".tmp"
        try:
            os.makedirs(self.model_path, exist_ok=True)
            team_dates_str = {
                t: d.isoformat() if isinstance(d, datetime) else None
                for t, d in self.team_last_match_date.items()
            }

            data = {
                "team_history": dict(self.team_history),
                "h2h_history": dict(self.h2h_history),
                "league_standings": dict(self.league_standings),
                "team_last_match_date": team_dates_str,
                "league_stats": dict(self.league_stats),
                "team_elo": dict(self.team_elo)
            }

            if os.path.exists(temp_file):
                os.remove(temp_file)

            with open(temp_file, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            os.replace(temp_file, data_file)
            logger.info("💾 Feature data güvenli şekilde kaydedildi")
            return True

        except Exception as e:
            logger.error(f"❌ Feature data kaydetme hatası: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return False

    # ==========================================================
    # ANA FONKSİYONLAR
    # ==========================================================

    def extract_features(self, match_data: Dict) -> Optional[np.ndarray]:
        """100 özellikli feature vector üretir"""
        try:
            if not match_data:
                return None

            home = safe_lower(match_data.get("home_team"))
            away = safe_lower(match_data.get("away_team"))
            league = safe_lower(match_data.get("league", "unknown"))
            if not home or not away:
                return None

            feature_groups = [
                self._extract_advanced_odds_features(match_data),
                self._extract_team_features_v4(home, away, league),
                self._extract_comprehensive_form_features(home, away),
                self._extract_h2h_features_v4(home, away),
                self._extract_time_features_v4(match_data, home, away),
                self._extract_standings_features_v4(home, away, league)
            ]

            features = np.concatenate(feature_groups)
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

            if len(features) != TOTAL_FEATURES_V4:
                logger.error(f"Feature sayısı hatalı: {len(features)} != {TOTAL_FEATURES_V4}")
                return None

            return features.astype(np.float32)

        except Exception as e:
            logger.error(f"extract_features hatası: {e}", exc_info=True)
            return None
    
    def _extract_advanced_odds_features(self, match_data: Dict) -> np.ndarray:
        """Gelişmiş oran özellikleri (6 temel + 18 türev = 24 toplam)"""
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
        
        # Mevcut türevler
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
        
        # YENİ: Gelişmiş oran metrikleri
        favorite_gap = abs(min(odds_1, odds_2) - max(odds_1, odds_2))  # Favori-underdog farkı
        underdog_value = max(odds_1, odds_2) / min(odds_1, odds_2)  # Underdog değeri
        draw_premium = odds_x - np.mean([odds_1, odds_2])  # Beraberlik primi
        odds_imbalance = abs(prob_1 - prob_2)  # Oran dengesizliği
        min_odds = min(odds_1, odds_x, odds_2)
        max_odds = max(odds_1, odds_x, odds_2)
        odds_ratio_home_away = odds_1 / odds_2 if odds_2 > 0 else 1.0
        prob_diff_home_away = prob_1 - prob_2
        
        # Oran güveni (düşük entropy = yüksek güven)
        odds_confidence = 1.0 - (odds_entropy / np.log(3))  # Normalize edilmiş
        
        # Betting heat (düşük margin = yüksek likidite/güven)
        betting_heat = 1.0 / (1.0 + odds_margin)
        
        return np.array([
            # Temel (6)
            odds_1, odds_x, odds_2,
            prob_1, prob_x, prob_2,
            # Mevcut türevler (8)
            odds_diff_home_draw, odds_diff_draw_away, odds_margin,
            log_odds_1, log_odds_x, log_odds_2,
            odds_entropy, odds_variance,
            # YENİ türevler (10)
            favorite_gap, underdog_value, draw_premium,
            odds_imbalance, min_odds, max_odds,
            odds_ratio_home_away, prob_diff_home_away, odds_confidence,
            betting_heat
        ])
    
    def _extract_team_features_v4(self, home: str, away: str, league: str) -> np.ndarray:
        """Takım özellikleri v4 (8)"""
        home_len = len(home)
        away_len = len(away)
        is_derby = float(self._is_derby(home, away))
        league_score = self._get_league_score(league)
        
        # YENİ: ELO ratings
        home_elo = self.team_elo[home]
        away_elo = self.team_elo[away]
        elo_diff = home_elo - away_elo
        
        # YENİ: Lig kalite endeksi
        league_quality = self._get_league_quality_index(league)
        
        return np.array([
            home_len, away_len, is_derby, league_score,
            home_elo, away_elo, elo_diff, league_quality
        ])
    
    def _extract_comprehensive_form_features(self, home: str, away: str) -> np.ndarray:
        """Kapsamlı form özellikleri (28)"""
        home_data = self._calculate_comprehensive_form(home)
        away_data = self._calculate_comprehensive_form(away)
        
        form_diff = home_data["form_score"] - away_data["form_score"]
        goals_diff = home_data["goals_avg"] - away_data["goals_avg"]
        
        # YENİ: Form momentum farkı
        form_momentum_diff = home_data["momentum"] - away_data["momentum"]
        
        return np.array([
            # Temel form (20)
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
            home_data["weighted_form"],
            away_data["weighted_form"],
            home_data["home_win_rate"],
            away_data["away_win_rate"],
            home_data["defense_avg"],
            away_data["defense_avg"],
            home_data["clean_sheet_rate"],
            away_data["clean_sheet_rate"],
            home_data["ppg"],
            away_data["ppg"],
            # YENİ: Detaylı performans (8)
            home_data["attack_strength"],
            away_data["attack_strength"],
            home_data["defense_strength"],
            away_data["defense_strength"],
            home_data["recent_goals_avg"],
            away_data["recent_goals_avg"],
            home_data["consistency"],
            away_data["consistency"],
            form_momentum_diff
        ])
    
    def _calculate_comprehensive_form(self, team: str) -> Dict:
        """Kapsamlı form hesaplama"""
        history = self.team_history.get(team, [])
        if not history:
            return {
                "form_score": 0.5, "goals_avg": 1.5, "win_streak": 0,
                "last3_trend": 0.0, "weighted_form": 0.5,
                "home_win_rate": 0.45, "away_win_rate": 0.30,
                "defense_avg": 1.2, "clean_sheet_rate": 0.3, "ppg": 1.5,
                "attack_strength": 1.0, "defense_strength": 1.0,
                "recent_goals_avg": 1.5, "consistency": 0.5, "momentum": 0.0
            }
        
        recent = history[-10:]
        
        # Temel metriker
        wins = sum(1 for m in recent if m.get("result") == "W")
        draws = sum(1 for m in recent if m.get("result") == "D")
        points = wins * 3 + draws * 1
        form_score = points / (len(recent) * 3) if recent else 0.5
        
        goals_for = [safe_int(m.get("goals_for", 0)) for m in recent]
        goals_avg = np.mean(goals_for) if goals_for else 1.5
        
        win_streak = 0
        for m in reversed(recent):
            if m.get("result") == "W":
                win_streak += 1
            else:
                break
        
        last3 = goals_for[-3:] if len(goals_for) >= 3 else goals_for
        if len(last3) >= 2:
            last3_trend = (last3[-1] - last3[0]) / max(1, len(last3))
        else:
            last3_trend = 0.0
        
        # Ağırlıklı form
        weights = [0.1, 0.15, 0.2, 0.25, 0.3]
        recent_5 = recent[-5:]
        if len(recent_5) > 0:
            weighted_points = sum(
                w * (3 if m['result'] == 'W' else 1 if m['result'] == 'D' else 0)
                for w, m in zip(weights[:len(recent_5)], recent_5)
            )
            weighted_form = weighted_points / (sum(weights[:len(recent_5)]) * 3)
        else:
            weighted_form = 0.5
        
        # Ev/Deplasman split
        home_matches = [m for m in history if m.get('venue') == 'home'][-10:]
        away_matches = [m for m in history if m.get('venue') == 'away'][-10:]
        
        home_wins = sum(1 for m in home_matches if m['result'] == 'W')
        home_win_rate = home_wins / len(home_matches) if home_matches else 0.45
        
        away_wins = sum(1 for m in away_matches if m['result'] == 'W')
        away_win_rate = away_wins / len(away_matches) if away_matches else 0.30
        
        # Savunma
        goals_against = [safe_int(m.get("goals_against", 0)) for m in recent]
        defense_avg = np.mean(goals_against) if goals_against else 1.2
        
        clean_sheets = sum(1 for m in recent if m.get("goals_against", 1) == 0)
        clean_sheet_rate = clean_sheets / len(recent) if recent else 0.3
        
        ppg = points / len(recent) if recent else 1.5
        
        # YENİ: Attack/Defense strength (lig ortalamasına göre)
        league_avg_goals = 1.5  # Default
        attack_strength = goals_avg / league_avg_goals if league_avg_goals > 0 else 1.0
        defense_strength = league_avg_goals / defense_avg if defense_avg > 0 else 1.0
        
        # YENİ: Son 3 maç gol ortalaması
        recent_3_goals = goals_for[-3:] if len(goals_for) >= 3 else goals_for
        recent_goals_avg = np.mean(recent_3_goals) if recent_3_goals else 1.5
        
        # YENİ: Tutarlılık (sonuçların standart sapması)
        recent_points = [
            3 if m['result'] == 'W' else 1 if m['result'] == 'D' else 0
            for m in recent
        ]
        consistency = 1.0 - (np.std(recent_points) / 3.0) if len(recent_points) > 1 else 0.5
        
        # YENİ: Momentum (trend)
        momentum = self._calculate_momentum_v2(team)
        
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
            "ppg": round(ppg, 2),
            "attack_strength": round(attack_strength, 2),
            "defense_strength": round(defense_strength, 2),
            "recent_goals_avg": round(recent_goals_avg, 2),
            "consistency": round(consistency, 3),
            "momentum": round(momentum, 2)
        }
    
    def _calculate_momentum_v2(self, team: str) -> float:
        """Gelişmiş momentum hesabı"""
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
    
    def _extract_h2h_features_v4(self, home: str, away: str) -> np.ndarray:
        """H2H özellikleri v4 (7)"""
        key = self._get_h2h_key(home, away)
        h2h = self.h2h_history.get(key, [])
        
        if not h2h:
            return np.array([0, 0, 0, 2.5, 0.5, 0.0, 0.0])
        
        home_wins = sum(1 for m in h2h if m.get("result") == "1")
        draws = sum(1 for m in h2h if m.get("result") == "X")
        away_wins = sum(1 for m in h2h if m.get("result") == "2")
        
        total_goals = [safe_int(m.get("total_goals", 0)) for m in h2h]
        h2h_total_goals_avg = np.mean(total_goals) if total_goals else 2.5
        
        total_matches = len(h2h)
        h2h_home_dominance = home_wins / total_matches if total_matches > 0 else 0.5
        
        # YENİ: Son 3 H2H trendi
        recent_h2h = h2h[-3:]
        recent_home_wins = sum(1 for m in recent_h2h if m.get("result") == "1")
        h2h_recent_trend = recent_home_wins / len(recent_h2h) if recent_h2h else 0.5
        
        # YENİ: H2H gol varyansı
        h2h_goal_variance = np.var(total_goals) if len(total_goals) > 1 else 0.0
        
        return np.array([
            home_wins, draws, away_wins,
            h2h_total_goals_avg, h2h_home_dominance,
            h2h_recent_trend, h2h_goal_variance
        ])
    
    def _get_h2h_key(self, home: str, away: str) -> str:
        return "_vs_".join(sorted([home, away]))
    
    def _extract_time_features_v4(self, match_data: Dict, home: str, away: str) -> np.ndarray:
        """Zaman özellikleri v4 (15)"""
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
        rest_advantage = days_since_last_home - days_since_last_away
        
        home_fatigued = 1.0 if days_since_last_home < 3 else 0.0
        away_fatigued = 1.0 if days_since_last_away < 3 else 0.0
        home_optimal_rest = 1.0 if 3 <= days_since_last_home <= 5 else 0.0
        away_optimal_rest = 1.0 if 3 <= days_since_last_away <= 5 else 0.0
        
        home_momentum = self._calculate_momentum_v2(home)
        away_momentum = self._calculate_momentum_v2(away)
        
        # YENİ: Hafta içi maç mı?
        is_midweek = 1.0 if 1 <= day_of_week <= 3 else 0.0
        
        # YENİ: Sezon ilerlemesi (0-1 arası)
        if month >= 8:
            season_start_month = 8
            months_elapsed = (month - season_start_month) if month >= 8 else (month + 12 - season_start_month)
        else:
            season_start_month = 8
            months_elapsed = month + (12 - season_start_month)
        season_progress = min(months_elapsed / 10.0, 1.0)  # 10 aylık sezon varsayımı
        
        return np.array([
            month, day_of_week, is_weekend, season_phase,
            days_since_last_home, days_since_last_away,
            rest_advantage,
            home_fatigued, away_fatigued,
            home_optimal_rest, away_optimal_rest,
            home_momentum, away_momentum,
            is_midweek, season_progress
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
    
    def _extract_standings_features_v4(self, home: str, away: str, league: str) -> np.ndarray:
        """Lig sıralaması özellikleri v4 (18)"""
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
        
        # Motivasyon faktörleri
        home_title_race = 1.0 if home_pos <= 4 else 0.0
        away_title_race = 1.0 if away_pos <= 4 else 0.0
        home_relegation = 1.0 if home_pos >= (league_size - 3) else 0.0
        away_relegation = 1.0 if away_pos >= (league_size - 3) else 0.0
        home_mid_table = 1.0 if 5 <= home_pos <= (league_size - 4) else 0.0
        away_mid_table = 1.0 if 5 <= away_pos <= (league_size - 4) else 0.0
        
        home_form_5 = safe_float(home_data.get("form_last_5", 1.5))
        away_form_5 = safe_float(away_data.get("form_last_5", 1.5))
        
        # YENİ: Maç önem seviyesi
        match_importance_home = self._calculate_match_importance(home_pos, home_title_race, home_relegation, league_size)
        match_importance_away = self._calculate_match_importance(away_pos, away_title_race, away_relegation, league_size)
        
        # YENİ: Kritik maç mı?
        is_critical_match = 1.0 if (home_title_race + away_title_race + home_relegation + away_relegation) >= 2 else 0.0
        
        # YENİ: Motivasyon dengesizliği
        motivation_imbalance = abs(match_importance_home - match_importance_away)
        
        return np.array([
            home_pos, away_pos, pos_diff,
            home_points, away_points, points_diff,
            home_title_race, away_title_race,
            home_relegation, away_relegation,
            home_mid_table, away_mid_table,
            home_form_5, away_form_5,
            match_importance_home, match_importance_away,
            is_critical_match, motivation_imbalance
        ])
    
    def _calculate_match_importance(self, position: float, title_race: float, relegation: float, league_size: int) -> float:
        """Maç önem seviyesi hesapla (0-1)"""
        if title_race == 1.0:
            return 0.9  # Şampiyonluk yarışı çok önemli
        elif relegation == 1.0:
            return 0.85  # Düşme mücadelesi çok önemli
        elif position <= league_size / 4:
            return 0.7  # Üst sıralarda önemli
        elif position >= 3 * league_size / 4:
            return 0.6  # Alt sıralarda önemli
        else:
            return 0.4  # Orta sıra, düşük önem
    
    def _is_derby(self, home: str, away: str) -> bool:
        """Derby tespiti"""
        derby_teams = ["galatasaray", "fenerbahçe", "beşiktaş", "trabzonspor"]
        home_is_big = any(t in home for t in derby_teams)
        away_is_big = any(t in away for t in derby_teams)
        return home_is_big and away_is_big
    
    def _get_league_score(self, league: str) -> float:
        """Lig skoru (gol ortalaması bazlı)"""
        league_scores = {
            "premier league": 2.7, "la liga": 2.6, "serie a": 2.4,
            "bundesliga": 2.9, "ligue 1": 2.5, "süper lig": 2.6,
            "eredivisie": 3.0, "liga nos": 2.3, "championship": 2.6
        }
        
        for key, score in league_scores.items():
            if key in league:
                return score
        
        if league in self.league_stats and "avg_goals" in self.league_stats[league]:
            return self.league_stats[league]["avg_goals"]
        
        return 2.5  # Default
    
    def _get_league_quality_index(self, league: str) -> float:
        """Lig kalite endeksi (1-10 arası)"""
        quality_map = {
            "premier league": 10.0, "la liga": 9.5, "bundesliga": 9.0,
            "serie a": 8.5, "ligue 1": 8.0, "süper lig": 6.5,
            "eredivisie": 7.0, "liga nos": 6.0, "championship": 7.5
        }
        
        for key, quality in quality_map.items():
            if key in league:
                return quality
        
        return 5.0  # Default orta kalite
    
    def update_match_result(self, home: str, away: str, league: str, 
                           result: str, home_goals: int, away_goals: int,
                           match_date: str, venue_home: str = "home", venue_away: str = "away"):
        """Maç sonrası güncelleme + ELO güncelleme"""
        home = safe_lower(home)
        away = safe_lower(away)
        league = safe_lower(league)
        
        try:
            match_date_obj = datetime.strptime(match_date, "%Y-%m-%d")
        except:
            match_date_obj = datetime.now()
        
        # Takım geçmişi güncelle
        home_match_data = {
            "result": "W" if result == "1" else ("D" if result == "X" else "L"),
            "goals_for": home_goals,
            "goals_against": away_goals,
            "date": match_date,
            "venue": venue_home
        }
        self.team_history[home].append(home_match_data)
        
        away_match_data = {
            "result": "W" if result == "2" else ("D" if result == "X" else "L"),
            "goals_for": away_goals,
            "goals_against": home_goals,
            "date": match_date,
            "venue": venue_away
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
        
        # YENİ: ELO güncelleme
        self._update_elo_ratings(home, away, result)
        
        # Lig istatistikleri
        self._update_league_stats(league, home_goals + away_goals)
        
        # Auto-save
        self.update_count += 1
        if self.update_count % self.flush_every == 0:
            self._save_data()
    
    def _update_elo_ratings(self, home: str, away: str, result: str):
        """ELO rating güncelleme (Chess ELO sistemine benzer)"""
        K = 32  # K-factor (değişim hızı)
        
        home_elo = self.team_elo[home]
        away_elo = self.team_elo[away]
        
        # Beklenen skorlar
        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        expected_away = 1 - expected_home
        
        # Gerçek skorlar
        if result == "1":
            actual_home, actual_away = 1.0, 0.0
        elif result == "X":
            actual_home, actual_away = 0.5, 0.5
        else:  # "2"
            actual_home, actual_away = 0.0, 1.0
        
        # Yeni ELO'lar
        new_home_elo = home_elo + K * (actual_home - expected_home)
        new_away_elo = away_elo + K * (actual_away - expected_away)
        
        self.team_elo[home] = round(new_home_elo, 1)
        self.team_elo[away] = round(new_away_elo, 1)
    
    def _update_league_stats(self, league: str, total_goals: int):
        """Lig istatistikleri güncelle"""
        if league not in self.league_stats:
            self.league_stats[league] = {"total_goals": 0, "match_count": 0}
        
        self.league_stats[league]["total_goals"] += total_goals
        self.league_stats[league]["match_count"] += 1
        self.league_stats[league]["avg_goals"] = (
            self.league_stats[league]["total_goals"] / 
            self.league_stats[league]["match_count"]
        )
    
    def update_standings(self, league: str, standings: Dict[str, Dict]):
        """Lig sıralaması güncelle"""
        league = safe_lower(league)
        self.league_standings[league] = {
            safe_lower(team): data for team, data in standings.items()
        }
    
    def manual_save(self):
        """Manuel kaydetme"""
        return self._save_data()
    
    def get_stats(self) -> Dict[str, Any]:
        """İstatistikler"""
        return {
            "total_teams": len(self.team_history),
            "total_h2h_pairs": len(self.h2h_history),
            "total_leagues": len(self.league_standings),
            "update_count": self.update_count,
            "total_features": TOTAL_FEATURES_V4,
            "avg_elo": round(np.mean(list(self.team_elo.values())), 1) if self.team_elo else 1500.0
        }



    # ==========================================================
    # İSTATİSTİKLER
    # ==========================================================
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_teams": len(self.team_history),
            "total_h2h_pairs": len(self.h2h_history),
            "total_leagues": len(self.league_standings),
            "update_count": self.update_count,
            "total_features": TOTAL_FEATURES_V4,
            "avg_elo": round(np.mean(list(self.team_elo.values())), 1)
            if self.team_elo else 1500.0,
        }


# ==============================================================
# TEST
# ==============================================================

if __name__ == "__main__":
    eng = EnhancedFeatureEngineer()

    test_match = {
        "home_team": "Barcelona",
        "away_team": "Real Madrid",
        "league": "La Liga",
        "odds": {"1": 2.1, "X": 3.3, "2": 3.2},
        "date": "2025-10-15",
    }

    features = eng.extract_features(test_match)
    if features is not None:
        print(f"✅ Feature vector shape: {features.shape}")
        print(f"✅ Match: {len(features) == TOTAL_FEATURES_V4}")
        print(f"📊 Stats: {eng.get_stats()}")
        print(f"🔹 First 10 features: {features[:10]}")
    else:
        print("❌ Feature extraction failed")
