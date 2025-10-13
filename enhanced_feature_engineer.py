"""
==============================================================
 Enhanced Feature Engineer - Predicta Europe ML v3.0
--------------------------------------------------------------
 Yazar   : Olcay Arslan
 Sürüm   : 3.0 (2025-10-12)
 Amaç    : Takım formu, oran, H2H, lig, zaman ve momentum 
           gibi faktörlerden 45 boyutlu özellik vektörü üretir.
 Çıktı   : ML modeline doğrudan beslenebilen np.ndarray (float32)
--------------------------------------------------------------
 Özellik Grupları:
   ⚙️ odds_basic      → 6 temel oran (1X2 + implied probs)
   📈 odds_derived    → 8 türev oran (log, entropy, margin, variance)
   🧠 team            → 4 temel takım özelliği (isim uzunluğu, derbi, lig skoru)
   🔥 form            → 10 performans metriği (form puanı, trend, win-streak)
   ⚔️ h2h             → 5 kafa kafaya metrik (kazanç oranı, gol ort.)
   🕓 time            → 6 zaman ve sezon fazı özelliği
   🏆 standings       → 6 lig sıralaması ve puan farkı
==============================================================
"""
# ==============================================================
#  enhanced_feature_engineer.py
#  Predicta Europe ML v3.0 - Gelişmiş Feature Engineering
# ==============================================================

import math
import logging
import pickle
import os
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from collections import defaultdict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="[EnhancedFeatureEngineer] %(message)s")
logger = logging.getLogger(__name__)

ENHANCED_FEATURE_NAMES = [
    # Temel oranlar (6)
    "odds_1", "odds_x", "odds_2",
    "prob_1", "prob_x", "prob_2",
    
    # Oran türevleri (8)
    "odds_diff_home_draw", "odds_diff_draw_away", "odds_margin",
    "log_odds_1", "log_odds_x", "log_odds_2",
    "odds_entropy", "odds_variance",
    
    # Takım özellikleri (4)
    "home_len", "away_len", "is_derby", "league_country_score",
    
    # Form ve performans (10)
    "home_form", "away_form", "form_diff",
    "home_goals_avg", "away_goals_avg", "goals_diff",
    "home_win_streak", "away_win_streak",
    "home_last3_trend", "away_last3_trend",
    
    # H2H (5)
    "h2h_home_wins", "h2h_draws", "h2h_away_wins",
    "h2h_total_goals_avg", "h2h_home_dominance",
    
    # Zaman özellikleri (6)
    "month", "day_of_week", "is_weekend", "season_phase",
    "days_since_last_match_home", "days_since_last_match_away",
    
    # Lig ve sıralama (6)
    "home_league_position", "away_league_position", "position_diff",
    "home_points", "away_points", "points_diff"
]

# Feature grupları için doğrulama
FEATURE_GROUP_SIZES = {
    "odds_basic": 6,
    "odds_derived": 8,
    "team": 4,
    "form": 10,
    "h2h": 5,
    "time": 6,
    "standings": 6
}

TOTAL_EXPECTED_FEATURES = sum(FEATURE_GROUP_SIZES.values())

# Feature sayılarını kontrol et
assert len(ENHANCED_FEATURE_NAMES) == TOTAL_EXPECTED_FEATURES, \
    f"Feature name count mismatch: {len(ENHANCED_FEATURE_NAMES)} != {TOTAL_EXPECTED_FEATURES}"


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
    """Gelişmiş Feature Engineering - Zaman, Trend, Oran Türevleri"""
    
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
        
        logger.info(f"🚀 EnhancedFeatureEngineer initialized (flush_every={flush_every})")
        logger.info(f"📊 Expected features: {TOTAL_EXPECTED_FEATURES}")
    
    def _load_data(self):
        data_file = os.path.join(self.model_path, "enhanced_feature_data.pkl")
        if not os.path.exists(data_file):
            logger.info("ℹ️ Yeni enhanced feature data oluşturulacak")
            return False
        
        try:
            with open(data_file, "rb") as f:
                data = pickle.load(f)
            self.team_history = defaultdict(list, data.get("team_history", {}))
            self.h2h_history = defaultdict(list, data.get("h2h_history", {}))
            self.league_standings = defaultdict(dict, data.get("league_standings", {}))
            
            # Datetime nesnelerini geri yükle
            team_dates = {}
            for team, date_str in data.get("team_last_match_date", {}).items():
                if date_str and isinstance(date_str, str):
                    try:
                        team_dates[team] = datetime.fromisoformat(date_str)
                    except ValueError:
                        team_dates[team] = None
            self.team_last_match_date = defaultdict(lambda: None, team_dates)
            
            self.league_stats = defaultdict(dict, data.get("league_stats", {}))
            
            logger.info("✅ Enhanced feature data yüklendi")
            logger.info(f"📈 Loaded: {len(self.team_history)} teams, {len(self.h2h_history)} H2H pairs")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Enhanced feature data yükleme hatası: {e}")
            return False
    
    def _save_data(self):
        data_file = os.path.join(self.model_path, "enhanced_feature_data.pkl")
        temp_file = data_file + ".tmp"
        try:
            # Datetime nesnelerini string'e çevir
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
            logger.info("💾 Enhanced feature data kaydedildi")
            return True
        except Exception as e:
            logger.error(f"❌ Enhanced feature data kaydetme hatası: {e}")
            return False
    
    def extract_features(self, match_data: Dict) -> Optional[np.ndarray]:
        """Enhanced feature vector üretir"""
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
            
            # Tüm feature gruplarını çıkar
            feature_groups = [
                self._extract_odds_features(match_data),
                self._extract_team_features(home, away, league),
                self._extract_form_features(home, away),
                self._extract_h2h_features(home, away),
                self._extract_time_features(match_data, home, away),
                self._extract_standings_features(home, away, league)
            ]
            
            # Tüm özellikleri birleştir
            features = np.concatenate(feature_groups)
            
            # Güvenlik kontrolleri
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                logger.warning("extract_features: NaN/Inf detected, applying correction")
                features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Feature sayısı kontrolü
            if len(features) != TOTAL_EXPECTED_FEATURES:
                logger.error(f"extract_features: Feature count mismatch ({len(features)} != {TOTAL_EXPECTED_FEATURES})")
                self._log_feature_groups(feature_groups)
                return None
            
            return features.astype(np.float32)
            
        except Exception as e:
            logger.error(f"extract_features: Unexpected error - {e}", exc_info=True)
            return None
    
    def _log_feature_groups(self, feature_groups: List[np.ndarray]):
        """Feature gruplarını debug için loglar"""
        logger.debug("🔍 Feature Group Breakdown:")
        group_names = list(FEATURE_GROUP_SIZES.keys())
        for i, (name, group) in enumerate(zip(group_names, feature_groups)):
            logger.debug(f"  {name:<15}: {len(group):2d} features (expected: {FEATURE_GROUP_SIZES[name]})")
    
    def _extract_odds_features(self, match_data: Dict) -> np.ndarray:
        """Oran türevleri: log-odds, margin, entropy, variance"""
        odds = match_data.get("odds", {})
        if not isinstance(odds, dict):
            odds = {"1": 2.0, "X": 3.0, "2": 3.5}
        
        odds_1 = safe_float(odds.get("1", 2.0))
        odds_x = safe_float(odds.get("X", 3.0))
        odds_2 = safe_float(odds.get("2", 3.5))
        
        # Güvenlik
        if any(v <= 1.01 or v > 100 for v in [odds_1, odds_x, odds_2]):
            odds_1, odds_x, odds_2 = 2.0, 3.0, 3.5
        
        # Olasılıklar
        total = (1/odds_1) + (1/odds_x) + (1/odds_2)
        prob_1 = (1/odds_1) / total
        prob_x = (1/odds_x) / total
        prob_2 = (1/odds_2) / total
        
        # Oran farkları
        odds_diff_home_draw = odds_1 - odds_x
        odds_diff_draw_away = odds_x - odds_2
        odds_margin = total - 1.0
        
        # Log-odds
        log_odds_1 = np.log(odds_1) if odds_1 > 0 else 0.0
        log_odds_x = np.log(odds_x) if odds_x > 0 else 0.0
        log_odds_2 = np.log(odds_2) if odds_2 > 0 else 0.0
        
        # Entropy (belirsizlik)
        probs = np.array([prob_1, prob_x, prob_2])
        probs = probs[probs > 0]
        odds_entropy = -np.sum(probs * np.log(probs)) if len(probs) > 0 else 0.0
        
        # Variance
        odds_variance = np.var([odds_1, odds_x, odds_2])
        
        return np.array([
            odds_1, odds_x, odds_2,
            prob_1, prob_x, prob_2,
            odds_diff_home_draw, odds_diff_draw_away, odds_margin,
            log_odds_1, log_odds_x, log_odds_2,
            odds_entropy, odds_variance
        ])
    
    def _extract_team_features(self, home: str, away: str, league: str) -> np.ndarray:
        """Basit takım özellikleri"""
        home_len = len(home)
        away_len = len(away)
        is_derby = float(self._is_derby(home, away))
        league_score = self._get_league_score(league)
        
        return np.array([home_len, away_len, is_derby, league_score])
    
    def _extract_form_features(self, home: str, away: str) -> np.ndarray:
        """Form, gol ortalaması, kazanma serisi, trend"""
        home_data = self._calculate_advanced_form(home)
        away_data = self._calculate_advanced_form(away)
        
        form_diff = home_data["form_score"] - away_data["form_score"]
        goals_diff = home_data["goals_avg"] - away_data["goals_avg"]
        
        return np.array([
            home_data["form_score"],
            away_data["form_score"],
            form_diff,
            home_data["goals_avg"],
            away_data["goals_avg"],
            goals_diff,
            home_data["win_streak"],
            away_data["win_streak"],
            home_data["last3_trend"],
            away_data["last3_trend"]
        ])
    
    def _calculate_advanced_form(self, team: str) -> Dict:
        """Gelişmiş form hesaplama"""
        history = self.team_history.get(team, [])
        if not history:
            return {
                "form_score": 0.5,
                "goals_avg": 1.5,
                "win_streak": 0,
                "last3_trend": 0.0
            }
        
        recent = history[-10:]
        wins = sum(1 for m in recent if m.get("result") == "W")
        draws = sum(1 for m in recent if m.get("result") == "D")
        
        points = wins * 3 + draws * 1
        form_score = points / (len(recent) * 3) if recent else 0.5
        
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
        
        return {
            "form_score": round(form_score, 3),
            "goals_avg": round(goals_avg, 2),
            "win_streak": win_streak,
            "last3_trend": round(last3_trend, 2)
        }
    
    def _extract_h2h_features(self, home: str, away: str) -> np.ndarray:
        """Kafa kafaya geçmiş"""
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
        """H2H anahtarını sıralı olarak oluştur"""
        return "_vs_".join(sorted([home, away]))
    
    def _extract_time_features(self, match_data: Dict, home: str, away: str) -> np.ndarray:
        """Tarih, ay, hafta sonu, sezon dönemi, son maçtan beri geçen gün"""
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
        
        # Sezon dönemi (1-4: erken, orta, son, playoff)
        if month in [8, 9]:
            season_phase = 1
        elif month in [10, 11, 12, 1]:
            season_phase = 2
        elif month in [2, 3, 4]:
            season_phase = 3
        else:
            season_phase = 4
        
        # Son maçtan beri geçen gün sayısı
        days_since_last_home = self._days_since_last_match(home, match_date)
        days_since_last_away = self._days_since_last_match(away, match_date)
        
        return np.array([
            month, day_of_week, is_weekend, season_phase,
            days_since_last_home, days_since_last_away
        ])
    
    def _days_since_last_match(self, team: str, current_date: datetime) -> float:
        """Son maçtan beri geçen gün sayısı"""
        last_date = self.team_last_match_date.get(team)
        if last_date is None:
            return 7.0
        
        try:
            days = (current_date - last_date).days
            return float(max(0, min(days, 30)))
        except:
            return 7.0
    
    def _extract_standings_features(self, home: str, away: str, league: str) -> np.ndarray:
        """Lig sıralaması ve puan farkı"""
        standings = self.league_standings.get(league, {})
        
        home_data = standings.get(home, {"position": 10, "points": 30})
        away_data = standings.get(away, {"position": 10, "points": 30})
        
        home_pos = safe_float(home_data.get("position", 10))
        away_pos = safe_float(away_data.get("position", 10))
        pos_diff = home_pos - away_pos
        
        home_points = safe_float(home_data.get("points", 30))
        away_points = safe_float(away_data.get("points", 30))
        points_diff = home_points - away_points
        
        return np.array([
            home_pos, away_pos, pos_diff,
            home_points, away_points, points_diff
        ])
    
    def _is_derby(self, home: str, away: str) -> bool:
        derby_teams = ["galatasaray", "fenerbahçe", "beşiktaş", "trabzonspor"]
        home_is_big = any(t in home for t in derby_teams)
        away_is_big = any(t in away for t in derby_teams)
        return home_is_big and away_is_big
    
    def _get_league_score(self, league: str) -> float:
        """Dinamik lig skoru - lig ortalamasına göre"""
        league_scores = {
            "premier league": 2.5, "la liga": 2.5, "serie a": 2.3,
            "bundesliga": 2.4, "ligue 1": 2.2, "süper lig": 1.8,
            "eredivisie": 1.5, "liga nos": 1.4
        }
        
        for key, score in league_scores.items():
            if key in league:
                return score
        
        # Dinamik skor mevcutsa kullan
        if league in self.league_stats and "avg_goals" in self.league_stats[league]:
            avg_goals = self.league_stats[league]["avg_goals"]
            league_score = avg_goals / 1.2
            return float(np.clip(league_score, 1.0, 2.8))
        
        return 1.0
    
    def update_match_result(self, home: str, away: str, league: str, 
                           result: str, home_goals: int, away_goals: int,
                           match_date: str):
        """Maç sonrası güncelleme"""
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
            "date": match_date
        }
        self.team_history[home].append(home_match_data)
        
        away_match_data = {
            "result": "W" if result == "2" else ("D" if result == "X" else "L"),
            "goals_for": away_goals,
            "goals_against": home_goals,
            "date": match_date
        }
        self.team_history[away].append(away_match_data)
        
        # H2H güncelle
        key = self._get_h2h_key(home, away)
        self.h2h_history[key].append({
            "result": result,
            "total_goals": home_goals + away_goals,
            "date": match_date
        })
        
        # Son maç tarihleri güncelle
        self.team_last_match_date[home] = match_date_obj
        self.team_last_match_date[away] = match_date_obj
        
        # Lig istatistiklerini güncelle
        self._update_league_stats(league, home_goals + away_goals)
        
        # Performans optimizasyonu
        self.update_count += 1
        if self.update_count % self.flush_every == 0:
            self._save_data()
            logger.info(f"💾 Auto-saved after {self.update_count} updates")
    
    def _update_league_stats(self, league: str, total_goals: int):
        """Lig istatistiklerini güncelle"""
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
    
    def get_feature_importance_analysis(self, model) -> List[Tuple[str, float]]:
        """Feature importance analizi"""
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return []
        
        importances = model.feature_importances_
        if len(importances) != len(ENHANCED_FEATURE_NAMES):
            logger.warning(f"Feature importance count mismatch: {len(importances)} != {len(ENHANCED_FEATURE_NAMES)}")
            return []
        
        feature_importance = list(zip(ENHANCED_FEATURE_NAMES, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("🏆 TOP 10 FEATURE IMPORTANCE:")
        for i, (name, importance) in enumerate(feature_importance[:10]):
            logger.info(f"  {i+1:2d}. {name:<25} {importance:.4f}")
        
        return feature_importance
    
    def manual_save(self):
        """Manuel kaydetme"""
        success = self._save_data()
        if success:
            logger.info("💾 Manual save completed successfully")
        return success
    
    def get_feature_group_info(self) -> Dict[str, int]:
        """Feature grupları hakkında bilgi döndürür"""
        return FEATURE_GROUP_SIZES.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Sistem istatistiklerini döndürür"""
        return {
            "total_teams": len(self.team_history),
            "total_h2h_pairs": len(self.h2h_history),
            "total_leagues": len(self.league_standings),
            "update_count": self.update_count,
            "expected_features": TOTAL_EXPECTED_FEATURES,
            "feature_groups": FEATURE_GROUP_SIZES
        }


# ==============================================================
#  TEST
# ==============================================================
if __name__ == "__main__":
    # Test için daha küçük flush_every değeri
    eng = EnhancedFeatureEngineer(flush_every=100)
    
    test_match = {
        "home_team": "Barcelona",
        "away_team": "Real Madrid", 
        "league": "La Liga",
        "odds": {"1": 2.1, "X": 3.3, "2": 3.2},
        "date": "2025-10-15"
    }
    
    features = eng.extract_features(test_match)
    if features is not None:
        print(f"✅ Feature vector shape: {features.shape}")
        print(f"✅ Expected features: {TOTAL_EXPECTED_FEATURES}")
        print(f"✅ Actual features: {len(features)}")
        print(f"✅ Sample features: {features[:8]}")
        
        # Feature gruplarını kontrol et
        print(f"\n🔍 Feature Group Breakdown:")
        start_idx = 0
        for group_name, size in FEATURE_GROUP_SIZES.items():
            end_idx = start_idx + size
            group_features = features[start_idx:end_idx]
            print(f"  {group_name:<15}: indices {start_idx:2d}-{end_idx-1:2d} ({size:2d} features) - Sample: {group_features[:2]}")
            start_idx = end_idx
        
        # Feature sayısı doğrulama
        if len(features) == TOTAL_EXPECTED_FEATURES:
            print(f"\n🎉 Feature count validation: PASSED ({TOTAL_EXPECTED_FEATURES} features)")
        else:
            print(f"\n❌ Feature count validation: FAILED (expected {TOTAL_EXPECTED_FEATURES}, got {len(features)})")
        
        # İstatistikleri göster
        stats = eng.get_stats()
        print(f"\n📊 System Statistics:")
        for key, value in stats.items():
            if key != "feature_groups":
                print(f"  {key}: {value}")
            
    else:
        print("❌ Feature extraction failed")
    
    # Test güncelleme
    print(f"\n🧪 Testing update functionality...")
    eng.update_match_result(
        home="Barcelona",
        away="Real Madrid", 
        league="La Liga",
        result="1",
        home_goals=3,
        away_goals=1,
        match_date="2025-10-15"
    )
    print("✅ Update test completed")
    
    # Manuel kaydetme testi
    if eng.manual_save():
        print("✅ Manual save test completed")
