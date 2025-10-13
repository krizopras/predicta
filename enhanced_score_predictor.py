import os
import pickle
import numpy as np
import logging
import random
from typing import Dict, Tuple, List, Optional, Any
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd
from enhanced_feature_engineer import EnhancedFeatureEngineer
from auto_league_mapper import AutoLeagueMapper

logger = logging.getLogger(__name__)

class TemporalLeagueStats:
    """
    Zamansal ağırlıklı lig istatistikleri
    - Son 1 yıl ağırlıklı ortalamalar
    - Son 500 maç sliding window
    - Trend analizi
    """
    
    def __init__(self, data_csv_path: str, max_matches: int = 500, days_back: int = 365):
        self.data_csv_path = data_csv_path
        self.max_matches = max_matches
        self.days_back = days_back
        
    def calculate_temporal_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Zamansal ağırlıklı istatistikleri hesapla
        """
        try:
            # Tarih sütununu bul ve temizle
            date_col = self._find_date_column(df)
            if date_col is None:
                return self._calculate_simple_stats(df)
                
            df_clean = df.dropna(subset=[date_col])
            df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
            df_clean = df_clean.dropna(subset=[date_col])
            
            # Son 1 yıl ve son 500 maç filtreleri
            recent_cutoff = datetime.now() - timedelta(days=self.days_back)
            df_recent = df_clean[df_clean[date_col] >= recent_cutoff]
            df_recent_sorted = df_recent.sort_values(date_col, ascending=False)
            df_last_500 = df_recent_sorted.head(self.max_matches)
            
            # Ağırlık hesaplama (yeni maçlar daha ağırlıklı)
            stats = {}
            
            if len(df_recent) > 0:
                stats.update(self._calculate_weighted_stats(df_recent, date_col, "recent"))
                
            if len(df_last_500) > 0:
                stats.update(self._calculate_weighted_stats(df_last_500, date_col, "last_500"))
                
            if len(df_clean) > 0:
                stats.update(self._calculate_simple_stats(df_clean, "all_time"))
                
            return self._combine_temporal_stats(stats)
            
        except Exception as e:
            logger.error(f"Zamansal istatistik hatası: {e}")
            return self._calculate_simple_stats(df)
    
    def _find_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Tarih sütununu bul"""
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'tarih' in col.lower()]
        return date_columns[0] if date_columns else None
    
    def _calculate_weighted_stats(self, df: pd.DataFrame, date_col: str, period: str) -> Dict[str, Any]:
        """Ağırlıklı istatistikleri hesapla"""
        try:
            # Zaman ağırlıkları (yeni maçlar daha yüksek ağırlık)
            df_sorted = df.sort_values(date_col, ascending=True)
            dates = df_sorted[date_col]
            max_date = dates.max()
            
            # Exponential decay ağırlıkları (yarı ömür = 90 gün)
            weights = []
            for date in dates:
                days_diff = (max_date - date).days
                weight = np.exp(-days_diff / 90)  # Yarım ömür 90 gün
                weights.append(weight)
            
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            
            # Gol sütunlarını bul
            home_goals_col = next((c for c in df.columns if "home" in c.lower() and "goal" in c.lower()), None)
            away_goals_col = next((c for c in df.columns if "away" in c.lower() and "goal" in c.lower()), None)
            
            if not all([home_goals_col, away_goals_col]):
                return {}
                
            home_goals = df_sorted[home_goals_col].astype(float)
            away_goals = df_sorted[away_goals_col].astype(float)
            
            # Ağırlıklı ortalamalar
            weighted_home_avg = np.average(home_goals, weights=weights)
            weighted_away_avg = np.average(away_goals, weights=weights)
            weighted_total_avg = weighted_home_avg + weighted_away_avg
            weighted_home_advantage = weighted_home_avg - weighted_away_avg
            
            # Galibiyet oranları
            home_wins = ((home_goals > away_goals) * weights).sum()
            draws = ((home_goals == away_goals) * weights).sum()
            away_wins = ((home_goals < away_goals) * weights).sum()
            
            return {
                f"{period}_weighted_home_avg": weighted_home_avg,
                f"{period}_weighted_away_avg": weighted_away_avg,
                f"{period}_weighted_total_avg": weighted_total_avg,
                f"{period}_weighted_home_advantage": weighted_home_advantage,
                f"{period}_home_win_rate": home_wins,
                f"{period}_draw_rate": draws,
                f"{period}_away_win_rate": away_wins,
                f"{period}_match_count": len(df)
            }
            
        except Exception as e:
            logger.error(f"Ağırlıklı istatistik hatası: {e}")
            return {}
    
    def _calculate_simple_stats(self, df: pd.DataFrame, period: str = "all_time") -> Dict[str, Any]:
        """Basit istatistikleri hesapla"""
        try:
            home_goals_col = next((c for c in df.columns if "home" in c.lower() and "goal" in c.lower()), None)
            away_goals_col = next((c for c in df.columns if "away" in c.lower() and "goal" in c.lower()), None)
            
            if not all([home_goals_col, away_goals_col]):
                return {}
                
            home_goals = df[home_goals_col].astype(float)
            away_goals = df[away_goals_col].astype(float)
            
            home_avg = home_goals.mean()
            away_avg = away_goals.mean()
            total_avg = home_avg + away_avg
            home_advantage = home_avg - away_avg
            
            # Galibiyet oranları
            home_win_rate = (home_goals > away_goals).mean()
            draw_rate = (home_goals == away_goals).mean()
            away_win_rate = (home_goals < away_goals).mean()
            
            return {
                f"{period}_home_avg": home_avg,
                f"{period}_away_avg": away_avg,
                f"{period}_total_avg": total_avg,
                f"{period}_home_advantage": home_advantage,
                f"{period}_home_win_rate": home_win_rate,
                f"{period}_draw_rate": draw_rate,
                f"{period}_away_win_rate": away_win_rate,
                f"{period}_match_count": len(df)
            }
            
        except Exception as e:
            logger.error(f"Basit istatistik hatası: {e}")
            return {}
    
    def _combine_temporal_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Zamansal istatistikleri birleştir"""
        combined = {}
        
        # Öncelik sırası: recent -> last_500 -> all_time
        for period in ["recent", "last_500", "all_time"]:
            period_key = f"{period}_weighted_home_avg" if period != "all_time" else f"{period}_home_avg"
            
            if period_key in stats:
                combined["home_avg"] = stats.get(f"{period}_weighted_home_avg", stats.get(f"{period}_home_avg"))
                combined["away_avg"] = stats.get(f"{period}_weighted_away_avg", stats.get(f"{period}_away_avg"))
                combined["total_avg"] = stats.get(f"{period}_weighted_total_avg", stats.get(f"{period}_total_avg"))
                combined["home_advantage"] = stats.get(f"{period}_weighted_home_advantage", stats.get(f"{period}_home_advantage"))
                combined["home_win_rate"] = stats.get(f"{period}_home_win_rate", 0.45)
                combined["draw_rate"] = stats.get(f"{period}_draw_rate", 0.25)
                combined["away_win_rate"] = stats.get(f"{period}_away_win_rate", 0.30)
                combined["match_count"] = stats.get(f"{period}_match_count", 0)
                break
        
        # Trend analizi
        if "recent_home_avg" in stats and "all_time_home_avg" in stats:
            recent_home = stats.get("recent_weighted_home_avg", stats.get("recent_home_avg"))
            all_time_home = stats.get("all_time_home_avg")
            combined["home_trend"] = recent_home - all_time_home
            
            recent_away = stats.get("recent_weighted_away_avg", stats.get("recent_away_avg"))
            all_time_away = stats.get("all_time_away_avg")
            combined["away_trend"] = recent_away - all_time_away
        else:
            combined["home_trend"] = 0.0
            combined["away_trend"] = 0.0
            
        return combined


class AdvancedLeagueMetrics:
    """
    Gelişmiş lig metrikleri
    - Lig tempo katsayısı
    - Favori galibiyet oranı
    - İç saha puan yüzdesi
    - Gol verimliliği
    """
    
    def calculate_advanced_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Gelişmiş lig metriklerini hesapla
        """
        try:
            home_goals_col = next((c for c in df.columns if "home" in c.lower() and "goal" in c.lower()), None)
            away_goals_col = next((c for c in df.columns if "away" in c.lower() and "goal" in c.lower()), None)
            odds_1_col = next((c for c in df.columns if "odds" in c.lower() and "1" in c.lower()), None)
            
            if not all([home_goals_col, away_goals_col]):
                return {}
                
            home_goals = df[home_goals_col].astype(float)
            away_goals = df[away_goals_col].astype(float)
            
            metrics = {}
            
            # 1. Lig Tempo Katsayısı (Gol üretim hızı)
            total_goals = home_goals + away_goals
            high_scoring_matches = (total_goals > 2.5).mean()
            very_high_scoring = (total_goals > 3.5).mean()
            metrics["tempo_coefficient"] = min(1.0, (high_scoring_matches * 0.7 + very_high_scoring * 0.3) * 2)
            
            # 2. Favori Galibiyet Oranı
            if odds_1_col and odds_1_col in df.columns:
                try:
                    odds_1 = pd.to_numeric(df[odds_1_col], errors='coerce')
                    favorite_wins = ((home_goals > away_goals) & (odds_1 < 2.0)).mean()
                    metrics["favorite_win_rate"] = favorite_wins
                except:
                    metrics["favorite_win_rate"] = 0.55
            else:
                metrics["favorite_win_rate"] = 0.55
                
            # 3. İç Saha Puan Yüzdesi
            home_points = (home_goals > away_goals) * 3 + (home_goals == away_goals) * 1
            total_points = len(df) * 3  # Maksimum puan
            metrics["home_point_percentage"] = home_points.sum() / total_points if total_points > 0 else 0.45
            
            # 4. Gol Verimliliği (Şut başına gol)
            metrics["scoring_efficiency"] = min(1.0, total_goals.mean() / 3.0)
            
            # 5. Savunma Stabilitesi (Clean sheet oranları)
            home_clean_sheets = (away_goals == 0).mean()
            away_clean_sheets = (home_goals == 0).mean()
            metrics["defensive_stability"] = (home_clean_sheets + away_clean_sheets) / 2
            
            # 6. Maç Canlılığı (Her iki takımın gol attığı maç oranı)
            both_scored = ((home_goals > 0) & (away_goals > 0)).mean()
            metrics["match_liveliness"] = both_scored
            
            return metrics
            
        except Exception as e:
            logger.error(f"Gelişmiş metrik hesaplama hatası: {e}")
            return {
                "tempo_coefficient": 0.5,
                "favorite_win_rate": 0.55,
                "home_point_percentage": 0.45,
                "scoring_efficiency": 0.5,
                "defensive_stability": 0.3,
                "match_liveliness": 0.45
            }


class RealisticScorePredictor:
    """
    Temel Skor Tahmin Sistemi - Eksik metodların tanımlandığı base class
    """

    def __init__(
        self,
        models_dir: str = "data/ai_models_v2",
        data_csv_path: str = "data/merged_all.csv",
        rng_seed: Optional[int] = None,
        max_goals: int = 6,
        simulation_iterations: int = 1000
    ):
        self.models_dir = models_dir
        self.data_csv_path = data_csv_path
        self.engineer = EnhancedFeatureEngineer(model_path=models_dir)
        self.league_mapper = AutoLeagueMapper(data_csv_path)
        self.model = None
        self.space = None
        self.max_goals = max_goals
        self.simulation_iterations = simulation_iterations
        self.rng = np.random.default_rng(rng_seed if rng_seed is not None else 0)
        self.feature_length = self._get_feature_length()
        self.league_stats = {}
        self.eps = 1e-9
        self._load_league_stats()
        self._load()

    def _get_feature_length(self) -> int:
        """EnhancedFeatureEngineer'dan feature boyutunu dinamik al"""
        try:
            probe = {
                "home_team": "probe_home",
                "away_team": "probe_away", 
                "league": "probe_league",
                "odds": {"1": 2.4, "X": 3.2, "2": 2.9},
                "date": "2025-10-09",
            }
            feats = self.engineer.extract_features(probe)
            if feats is not None:
                length = len(feats)
                logger.info(f"📐 Feature boyutu dinamik tespit edildi: {length}")
                return length
        except Exception as e:
            logger.warning(f"⚠️ Feature boyutu tespit edilemedi: {e}")
        return 46

    def _load_league_stats(self):
        """Temel lig istatistiklerini yükle"""
        pickle_path = os.path.join(self.models_dir, "league_stats.pkl")
        if os.path.exists(pickle_path):
            try:
                with open(pickle_path, "rb") as f:
                    self.league_stats = pickle.load(f)
                logger.info(f"✅ Lig istatistikleri yüklendi ({len(self.league_stats)} lig)")
                return
            except Exception as e:
                logger.warning(f"⚠️ Pickle yükleme hatası: {e}")
        self._build_basic_league_stats()

    def _build_basic_league_stats(self):
        """Temel lig istatistiklerini oluştur"""
        if not os.path.exists(self.data_csv_path):
            logger.warning(f"⚠️ CSV bulunamadı: {self.data_csv_path}")
            return
        try:
            df = pd.read_csv(self.data_csv_path)
            df.columns = [c.lower().strip() for c in df.columns]
            league_col = next((c for c in df.columns if "lig" in c or "league" in c), None)
            home_goals_col = next((c for c in df.columns if "home" in c and "goal" in c), None)
            away_goals_col = next((c for c in df.columns if "away" in c and "goal" in c), None)
            
            if not all([league_col, home_goals_col, away_goals_col]):
                return
                
            for league in df[league_col].unique():
                if not league or str(league).strip() == "":
                    continue
                league_df = df[df[league_col] == league]
                if len(league_df) < 10:
                    continue
                    
                home_goals = league_df[home_goals_col].astype(float)
                away_goals = league_df[away_goals_col].astype(float)
                
                self.league_stats[str(league).strip()] = {
                    "home_avg": home_goals.mean(),
                    "away_avg": away_goals.mean(),
                    "total_avg": home_goals.mean() + away_goals.mean(),
                    "home_advantage": home_goals.mean() - away_goals.mean(),
                    "match_count": len(league_df)
                }
                
            logger.info(f"✅ Temel lig istatistikleri oluşturuldu: {len(self.league_stats)} lig")
        except Exception as e:
            logger.error(f"❌ Temel istatistik oluşturma hatası: {e}")

    def _load(self):
        """Model dosyasını güvenli yükle"""
        path = os.path.join(self.models_dir, "score_model.pkl")
        if not os.path.exists(path):
            logger.warning(f"⚠️ Skor modeli bulunamadı: {path}")
            return
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model = data.get("model")
            self.space = data.get("score_space")
            if self.model is None or self.space is None:
                logger.error("❌ Model veya score_space eksik!")
                self.model = None
                self.space = None
                return
            logger.info(f"✅ Skor modeli yüklendi ({len(self.space)} sınıf)")
        except Exception as e:
            logger.error(f"❌ Skor model yükleme hatası: {e}")
            self.model = None
            self.space = None

    def _safe_odds(self, odds: Dict, key: str, default: float) -> float:
        """Güvenli oran dönüşümü"""
        try:
            val = odds.get(key, default)
            if val is None:
                return default
            val = float(val)
            return val if val > 0.0 else default
        except:
            return default

    def _normalize_lambda_diff(self, lambda_h: float, lambda_a: float) -> Tuple[float, float]:
        """Lambda farkını normalize et"""
        diff = lambda_h - lambda_a
        max_diff = 2.5
        if abs(diff) > max_diff:
            sign = 1 if diff > 0 else -1
            diff = sign * max_diff
            total = lambda_h + lambda_a
            lambda_h = (total + diff) / 2.0
            lambda_a = (total - diff) / 2.0
        return lambda_h, lambda_a

    def _soft_cap_lambda(self, lam: float, odds: float) -> float:
        """Yumuşak blowout baskılama"""
        if odds < 1.20:
            max_lam = 3.8
        elif odds < 1.50:
            max_lam = 3.2
        else:
            max_lam = 2.8
        return lam if lam <= max_lam else max_lam + (lam - max_lam) * 0.3

    def _auto_detect_league(self, match_data: Dict) -> str:
        """Otomatik lig tespiti"""
        existing_league = match_data.get("league", "")
        if existing_league and existing_league.strip() and existing_league != "Bilinmeyen Lig":
            return existing_league.strip()
        home_team = match_data.get("home_team", "")
        away_team = match_data.get("away_team", "")
        if not home_team and not away_team:
            return "Bilinmeyen Lig"
        detected_league = self.league_mapper.detect_league(home_team, away_team)
        if detected_league != "Bilinmeyen Lig":
            logger.info(f"🔍 Lig otomatik tespit edildi: '{home_team}' vs '{away_team}' → '{detected_league}'")
        else:
            logger.warning(f"⚠️ Lig tespit edilemedi: '{home_team}' vs '{away_team}'")
        return detected_league

    def _safe_feature_extraction(self, match_data: Dict) -> Optional[np.ndarray]:
        """Güvenli feature extraction"""
        try:
            feats = self.engineer.extract_features(match_data)
            if feats is None:
                return None
            return feats.astype(np.float32)
        except Exception as e:
            logger.error(f"Feature extraction hatası: {e}")
            return None

    def _ml_prediction(self, match_data: Dict, ms_pred: str) -> Tuple[str, List[Dict]]:
        """ML tabanlı tahmin"""
        if self.model is None or self.space is None:
            return self._intelligent_fallback(match_data, ms_pred)
        try:
            feats = self._safe_feature_extraction(match_data)
            if feats is None:
                return self._intelligent_fallback(match_data, ms_pred)
            feats = feats.reshape(1, -1)
            probs = self.model.predict_proba(feats)[0]
            adj = probs.copy()
            for i, lab in enumerate(self.space):
                if lab != "OTHER" and not self._ms_ok(ms_pred, lab):
                    adj[i] *= 0.12
            if adj.sum() > 0:
                adj = adj / adj.sum()
            order = np.argsort(adj)[::-1]
            top = next((self.space[i] for i in order if self.space[i] != "OTHER"), "1-1")
            alts = []
            for i in order:
                lab = self.space[i]
                if lab == "OTHER":
                    continue
                alts.append({"score": lab, "prob": float(adj[i])})
                if len(alts) == 3:
                    break
            return top, alts
        except Exception as e:
            logger.error(f"ML tahmin hatası: {e}")
            return self._intelligent_fallback(match_data, ms_pred)

    def _ms_ok(self, ms: str, score_label: str) -> bool:
        """MS tutarlılık kontrolü"""
        try:
            a, b = map(int, score_label.split("-"))
            if ms == "1":
                return a > b
            elif ms == "X":
                return a == b
            elif ms == "2":
                return a < b
            return True
        except:
            return True

    def _intelligent_fallback(self, match_data: Dict, ms_pred: str) -> Tuple[str, List[Dict]]:
        """Akıllı fallback"""
        league = self._auto_detect_league(match_data)
        league_stats = self.league_stats.get(league, {"home_avg": 1.52, "away_avg": 1.15, "home_advantage": 0.37})
        odds = match_data.get("odds", {})
        o1 = float(self._safe_odds(odds, "1", 2.5) or 2.5)
        o2 = float(self._safe_odds(odds, "2", 2.8) or 2.8)
        bias = np.clip((1/o1 - 1/o2) * 4.0, -1.2, 1.2)
        base_h = np.clip(league_stats["home_avg"] + league_stats["home_advantage"] + bias * 0.25, 0.3, 3.8)
        base_a = np.clip(league_stats["away_avg"] - league_stats["home_advantage"] - bias * 0.10, 0.3, 3.5)
        h = int(round(base_h))
        a = int(round(base_a))
        cand = [f"{h}-{a}", f"{h}-{max(0, a-1)}", f"{max(0, h-1)}-{a}"]
        probs = self._softmax([0.55, 0.25, 0.20])
        alts = [{"score": s, "prob": float(p)} for s, p in zip(cand, probs)]
        return cand[0], alts

    def _softmax(self, x):
        """Softmax normalizasyonu"""
        x = np.asarray(x, dtype=float)
        x = x - np.max(x)
        ex = np.exp(x)
        s = ex.sum()
        return ex / s if s > 0 else np.ones_like(x) / len(x)

    def _generate_realistic_alternatives(self, final_h: int, final_a: int, ms_pred: str, lam_h: float, lam_a: float) -> List[Dict]:
        """Alternatif skorlar üret"""
        cands = [
            (final_h, final_a),
            (max(0, final_h - 1), final_a),
            (final_h, max(0, final_a - 1)),
            (final_h + 1, final_a),
            (final_h, final_a + 1),
        ]
        seen = set()
        out = []
        for h, a in cands:
            lab = f"{h}-{a}"
            if lab in seen:
                continue
            seen.add(lab)
            out.append({"score": lab, "prob": 0.0})
            if len(out) == 3:
                break
        return out

    def _get_draw_calibration(self, odds_x: float) -> float:
        """Beraberlik oranından toplam gol kalibrasyonu"""
        if odds_x < 2.5:
            return 0.85
        elif odds_x < 3.0:
            return 0.92
        elif odds_x < 3.5:
            return 1.00
        elif odds_x < 4.0:
            return 1.08
        else:
            return 1.15

    def _calculate_expected_goals(self, match_data: Dict) -> Tuple[float, float]:
        """Temel beklenen gol hesaplama"""
        try:
            feats = self._safe_feature_extraction(match_data)
            league = self._auto_detect_league(match_data)
            league_stats = self.league_stats.get(league, {"home_avg": 1.52, "away_avg": 1.15, "home_advantage": 0.37})
            
            if feats is None:
                return league_stats["home_avg"], league_stats["away_avg"]

            home_form = feats[12] if len(feats) > 12 else 0.5
            away_form = feats[13] if len(feats) > 13 else 0.5
            home_avg_feat = feats[14] if len(feats) > 14 else league_stats["home_avg"]
            away_avg_feat = feats[15] if len(feats) > 15 else league_stats["away_avg"]
            
            odds = match_data.get("odds", {})
            odds_1 = self._safe_odds(odds, "1", 2.5)
            odds_x = self._safe_odds(odds, "X", 3.3)
            odds_2 = self._safe_odds(odds, "2", 2.8)
            
            base_strength = 2.5 / (league_stats.get("total_avg", 2.67) / 2.67)
            home_strength = min(2.0, max(0.3, base_strength / (odds_1 + self.eps)))
            away_strength = min(2.0, max(0.3, base_strength / (odds_2 + self.eps)))
            
            goal_calibration = self._get_draw_calibration(odds_x)
            
            lambda_h = (
                league_stats["home_avg"] * 0.30 +
                home_avg_feat * 0.25 +
                home_form * home_strength * 1.3 * 0.45
            ) * goal_calibration + league_stats["home_advantage"]

            lambda_a = (
                league_stats["away_avg"] * 0.30 +
                away_avg_feat * 0.25 +
                away_form * away_strength * 1.1 * 0.45
            ) * goal_calibration

            lambda_h, lambda_a = self._normalize_lambda_diff(lambda_h, lambda_a)
            lambda_h = self._soft_cap_lambda(lambda_h, odds_1)
            lambda_a = self._soft_cap_lambda(lambda_a, odds_2)
            
            max_home_goals = min(4.5, league_stats["home_avg"] * 2.5)
            max_away_goals = min(4.0, league_stats["away_avg"] * 2.5)
            lambda_h = np.clip(lambda_h, 0.3, max_home_goals)
            lambda_a = np.clip(lambda_a, 0.3, max_away_goals)
            
            lambda_h *= (1 + self.rng.uniform(-0.15, 0.15))
            lambda_a *= (1 + self.rng.uniform(-0.15, 0.15))
            lambda_h = np.clip(lambda_h, 0.3, max_home_goals)
            lambda_a = np.clip(lambda_a, 0.3, max_away_goals)

            return lambda_h, lambda_a

        except Exception as e:
            logger.error(f"Beklenen gol hesaplama hatası: {e}")
            league = self._auto_detect_league(match_data)
            league_stats = self.league_stats.get(league, {"home_avg": 1.52, "away_avg": 1.15})
            return league_stats["home_avg"], league_stats["away_avg"]

    def _ensemble_prediction(self, lambda_h: float, lambda_a: float, ml_pred: Tuple[str, List[Dict]], match_data: Dict) -> Tuple[float, float]:
        """Ensemble tahmin"""
        try:
            ml_h = ml_a = None
            if ml_pred and isinstance(ml_pred, tuple) and isinstance(ml_pred[0], str) and "-" in ml_pred[0]:
                try:
                    ml_h, ml_a = map(int, ml_pred[0].split("-"))
                except:
                    ml_h = ml_a = None
            
            ps_h = int(np.clip(round(lambda_h), 0, self.max_goals))
            ps_a = int(np.clip(round(lambda_a), 0, self.max_goals))

            league = self._auto_detect_league(match_data)
            league_stats = self.league_stats.get(league, {"home_avg": 1.52, "away_avg": 1.15})

            if ml_h is not None:
                w_ml, w_ps, w_lg = 0.60, 0.30, 0.10
                ens_h = w_ml * ml_h + w_ps * ps_h + w_lg * league_stats["home_avg"]
                ens_a = w_ml * ml_a + w_ps * ps_a + w_lg * league_stats["away_avg"]
            else:
                w_ps, w_lg = 0.85, 0.15
                ens_h = w_ps * ps_h + w_lg * league_stats["home_avg"]
                ens_a = w_ps * ps_a + w_lg * league_stats["away_avg"]

            return float(ens_h), float(ens_a)
        except Exception as e:
            logger.warning(f"Ensemble hatası: {e}")
            return lambda_h, lambda_a

    def _calculate_confidence(self, lam_h: float, lam_a: float, match_data: Dict) -> float:
        """Güven skoru hesaplama"""
        try:
            total = lam_h + lam_a
            balance = 1.0 - min(0.9, abs(lam_h - lam_a) / 4.0)
            pace = min(1.0, total / 3.2)
            base = 0.55 * balance + 0.45 * pace
            
            odds = match_data.get("odds", {})
            ox = float(self._safe_odds(odds, "X", 3.2) or 3.2)
            draw_hint = 1.0 - min(0.6, max(0.0, (ox - 3.0)) / 2.0)
            
            conf = np.clip(0.2 + 0.8 * base * draw_hint, 0.1, 0.95)
            return float(conf)
        except:
            return 0.5

    def predict_with_details(self, match_data: Dict, ms_pred: str = "1") -> Dict[str, Any]:
        """Temel tahmin fonksiyonu"""
        original_league = match_data.get("league", "Belirtilmemiş")
        detected_league = self._auto_detect_league(match_data)
        
        ml_prediction = self._ml_prediction(match_data, ms_pred)
        lambda_h, lambda_a = self._calculate_expected_goals(match_data)
        ensemble_h, ensemble_a = self._ensemble_prediction(lambda_h, lambda_a, ml_prediction, match_data)
        confidence = self._calculate_confidence(ensemble_h, ensemble_a, match_data)
        
        final_h, final_a = int(round(ensemble_h)), int(round(ensemble_a))
        final_score = f"{final_h}-{final_a}"
        
        alternatives = self._generate_realistic_alternatives(final_h, final_a, ms_pred, ensemble_h, ensemble_a)
        
        result = {
            "predicted_score": final_score,
            "expected_goals": {"home": round(ensemble_h, 2), "away": round(ensemble_a, 2)},
            "confidence": round(confidence, 3),
            "alternatives": alternatives,
            "league_info": {
                "original": original_league,
                "detected": detected_league,
                "auto_detected": detected_league != original_league
            }
        }
        
        home = match_data.get("home_team", "?")
        away = match_data.get("away_team", "?")
        
        logger.info(
            f"[{home}–{away}] "
            f"λ=({ensemble_h:.2f},{ensemble_a:.2f}) | "
            f"MS={ms_pred} → {final_score} | "
            f"Conf={confidence:.2f}"
        )
        
        return result

    def predict(self, match_data: Dict, ms_pred: str = "1") -> Tuple[str, List[Dict]]:
        """Eski formatı desteklemek için wrapper"""
        detailed_result = self.predict_with_details(match_data, ms_pred)
        return detailed_result["predicted_score"], detailed_result["alternatives"]


class EnhancedRealisticScorePredictor(RealisticScorePredictor):
    """
    Geliştirilmiş Skor Tahmin Sistemi - Zamansal İstatistikler & Gelişmiş Metrikler
    RealisticScorePredictor'dan miras alır, tüm temel metodları içerir
    """

    LEAGUE_STATS_VERSION = "v4.0"

    def __init__(
        self,
        models_dir: str = "data/ai_models_v2",
        data_csv_path: str = "data/merged_all.csv",
        rng_seed: Optional[int] = None,
        max_goals: int = 6,
        simulation_iterations: int = 1000,
        enable_temporal_stats: bool = True,
        enable_advanced_metrics: bool = True
    ):
        # Temel sınıfı başlat
        super().__init__(models_dir, data_csv_path, rng_seed, max_goals, simulation_iterations)
        
        # Yeni bileşenler
        self.temporal_analyzer = TemporalLeagueStats(data_csv_path) if enable_temporal_stats else None
        self.metrics_calculator = AdvancedLeagueMetrics() if enable_advanced_metrics else None
        
        # Gelişmiş lig istatistikleri
        self.enhanced_league_stats = {}
        self.advanced_metrics = {}
        
        # Varsayılan değerler
        self.default_stats = {
            "home_avg": 1.52, "away_avg": 1.15, "total_avg": 2.67, "home_advantage": 0.37,
            "home_win_rate": 0.45, "draw_rate": 0.25, "away_win_rate": 0.30,
            "home_trend": 0.0, "away_trend": 0.0, "match_count": 1000
        }
        
        self.default_metrics = {
            "tempo_coefficient": 0.5, "favorite_win_rate": 0.55, "home_point_percentage": 0.45,
            "scoring_efficiency": 0.5, "defensive_stability": 0.3, "match_liveliness": 0.45
        }
        
        # Gelişmiş istatistikleri yükle
        self._load_enhanced_league_stats()

    def _load_enhanced_league_stats(self):
        """Gelişmiş lig istatistiklerini yükle"""
        pickle_path = self._league_stats_pickle_path()
        
        if os.path.exists(pickle_path):
            try:
                with open(pickle_path, "rb") as f:
                    data = pickle.load(f)
                    self.enhanced_league_stats = data.get("basic_stats", {})
                    self.advanced_metrics = data.get("advanced_metrics", {})
                logger.info(f"✅ Gelişmiş lig istatistikleri yüklendi ({len(self.enhanced_league_stats)} lig)")
                return
            except Exception as e:
                logger.warning(f"⚠️ Gelişmiş pickle yükleme hatası: {e}")
        
        # CSV'den yeniden oluştur
        self._build_enhanced_league_stats()
        self._save_enhanced_league_stats()

    def _build_enhanced_league_stats(self):
        """CSV'den gelişmiş lig istatistiklerini oluştur"""
        if not os.path.exists(self.data_csv_path):
            logger.warning(f"⚠️ CSV bulunamadı: {self.data_csv_path}")
            return

        try:
            df = pd.read_csv(self.data_csv_path)
            df.columns = [c.lower().strip() for c in df.columns]
            
            # Lig sütununu bul
            league_col = next((c for c in df.columns if "lig" in c or "league" in c), None)
            if not league_col:
                logger.error("❌ Lig sütunu bulunamadı")
                return
            
            # Her lig için istatistikleri hesapla
            unique_leagues = df[league_col].dropna().unique()
            
            for league in unique_leagues:
                if not league or str(league).strip() == "":
                    continue
                    
                league_str = str(league).strip()
                league_df = df[df[league_col] == league]
                
                if len(league_df) < 10:  # Minimum maç sayısı
                    continue
                
                # Temel istatistikler
                if self.temporal_analyzer:
                    basic_stats = self.temporal_analyzer.calculate_temporal_stats(league_df)
                else:
                    basic_stats = self.temporal_analyzer._calculate_simple_stats(league_df, "all_time") if self.temporal_analyzer else {}
                
                # Gelişmiş metrikler
                if self.metrics_calculator:
                    advanced_metrics = self.metrics_calculator.calculate_advanced_metrics(league_df)
                else:
                    advanced_metrics = {}
                
                if basic_stats:
                    self.enhanced_league_stats[league_str] = basic_stats
                    self.advanced_metrics[league_str] = advanced_metrics
            
            logger.info(f"✅ Gelişmiş lig istatistikleri oluşturuldu: {len(self.enhanced_league_stats)} lig")
            
        except Exception as e:
            logger.error(f"❌ Gelişmiş istatistik oluşturma hatası: {e}")

    def _save_enhanced_league_stats(self):
        """Gelişmiş lig istatistiklerini kaydet"""
        try:
            os.makedirs(self.models_dir, exist_ok=True)
            pickle_path = self._league_stats_pickle_path()
            
            data = {
                "basic_stats": self.enhanced_league_stats,
                "advanced_metrics": self.advanced_metrics,
                "version": self.LEAGUE_STATS_VERSION,
                "created_at": datetime.now().isoformat()
            }
            
            with open(pickle_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            logger.info(f"💾 Gelişmiş lig istatistikleri kaydedildi: {pickle_path}")
            
        except Exception as e:
            logger.error(f"❌ Gelişmiş istatistik kaydetme hatası: {e}")

    def _league_stats_pickle_path(self):
        return os.path.join(self.models_dir, f"enhanced_league_stats_{self.LEAGUE_STATS_VERSION}.pkl")

    def _get_enhanced_league_stats(self, league: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Gelişmiş lig istatistiklerini döndür"""
        if not league:
            return self.default_stats, self.default_metrics
            
        league_lower = league.lower()
        
        # Tam eşleşme
        for lig_name, stats in self.enhanced_league_stats.items():
            if lig_name.lower() == league_lower:
                metrics = self.advanced_metrics.get(lig_name, self.default_metrics)
                return stats, metrics
        
        # Kısmi eşleşme
        for lig_name, stats in self.enhanced_league_stats.items():
            if lig_name.lower() in league_lower or league_lower in lig_name.lower():
                metrics = self.advanced_metrics.get(lig_name, self.default_metrics)
                return stats, metrics
        
        return self.default_stats, self.default_metrics

    def _calculate_expected_goals_enhanced(self, match_data: Dict) -> Tuple[float, float]:
        """
        Geliştirilmiş beklenen gol hesaplama - Zamansal istatistikler & metrikler ile
        """
        try:
            feats = self._safe_feature_extraction(match_data)
            league = self._auto_detect_league(match_data)
            
            if feats is None:
                stats, _ = self._get_enhanced_league_stats(league)
                return stats["home_avg"], stats["away_avg"]

            # Gelişmiş lig istatistikleri
            stats, metrics = self._get_enhanced_league_stats(league)
            
            # Feature'lar
            home_form = feats[12] if len(feats) > 12 else 0.5
            away_form = feats[13] if len(feats) > 13 else 0.5
            home_avg_feat = feats[14] if len(feats) > 14 else stats["home_avg"]
            away_avg_feat = feats[15] if len(feats) > 15 else stats["away_avg"]
            
            # Oranlar
            odds = match_data.get("odds", {})
            odds_1 = self._safe_odds(odds, "1", 2.5)
            odds_x = self._safe_odds(odds, "X", 3.3)
            odds_2 = self._safe_odds(odds, "2", 2.8)
            
            # Gelişmiş güç hesaplama
            base_strength = 2.5 / (stats["total_avg"] / 2.67)
            
            # Lig tempo katsayısı ile modifiye edilmiş güç
            tempo_factor = metrics.get("tempo_coefficient", 0.5)
            home_strength = min(2.0, max(0.3, base_strength / (odds_1 + self.eps))) * (0.8 + 0.4 * tempo_factor)
            away_strength = min(2.0, max(0.3, base_strength / (odds_2 + self.eps))) * (0.8 + 0.4 * tempo_factor)
            
            # Trend düzeltmesi - maç sayısına göre ağırlıklandırılmış
            match_count = stats.get("match_count", 100)
            trend_weight = min(1.0, match_count / 300)  # Daha fazla maç = daha güvenilir trend
            
            home_trend = stats.get("home_trend", 0.0)
            away_trend = stats.get("away_trend", 0.0)
            
            trend_boost_home = 1.0 + home_trend * 0.3 * trend_weight
            trend_boost_away = 1.0 + away_trend * 0.3 * trend_weight
            
            home_avg_adj = stats["home_avg"] * trend_boost_home
            away_avg_adj = stats["away_avg"] * trend_boost_away
            
            # Lambda hesaplama (gelişmiş)
            goal_calibration = self._get_draw_calibration(odds_x)
            
            lambda_h = (
                home_avg_adj * 0.35 +  # Trend-adjusted lig ortalaması
                home_avg_feat * 0.25 +  # Takım ortalaması
                home_form * home_strength * 1.3 * 0.40  # Form ve güç
            ) * goal_calibration + stats["home_advantage"]

            lambda_a = (
                away_avg_adj * 0.35 +  # Trend-adjusted lig ortalaması
                away_avg_feat * 0.25 +  # Takım ortalaması
                away_form * away_strength * 1.1 * 0.40  # Form ve güç
            ) * goal_calibration

            # Gol verimliliği düzeltmesi
            scoring_efficiency = metrics.get("scoring_efficiency", 0.5)
            lambda_h *= (0.8 + 0.4 * scoring_efficiency)
            lambda_a *= (0.8 + 0.4 * scoring_efficiency)

            # Optimizasyonlar
            lambda_h, lambda_a = self._normalize_lambda_diff(lambda_h, lambda_a)
            lambda_h = self._soft_cap_lambda(lambda_h, odds_1)
            lambda_a = self._soft_cap_lambda(lambda_a, odds_2)
            
            # Dinamik clipping (trend-aware)
            max_home_goals = min(4.5, stats["home_avg"] * (2.5 + tempo_factor))
            max_away_goals = min(4.0, stats["away_avg"] * (2.5 + tempo_factor))
            lambda_h = np.clip(lambda_h, 0.3, max_home_goals)
            lambda_a = np.clip(lambda_a, 0.3, max_away_goals)
            
            # Jitter
            lambda_h *= (1 + self.rng.uniform(-0.15, 0.15))
            lambda_a *= (1 + self.rng.uniform(-0.15, 0.15))
            lambda_h = np.clip(lambda_h, 0.3, max_home_goals)
            lambda_a = np.clip(lambda_a, 0.3, max_away_goals)

            # Debug log
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"⚽ Gelişmiş Lig '{league}': "
                           f"Tempo={tempo_factor:.2f}, "
                           f"Trend=({home_trend:+.2f},{away_trend:+.2f}) "
                           f"Weight={trend_weight:.2f}")

            return lambda_h, lambda_a

        except Exception as e:
            logger.error(f"Gelişmiş beklenen gol hesaplama hatası: {e}")
            league = self._auto_detect_league(match_data)
            stats, _ = self._get_enhanced_league_stats(league)
            return stats["home_avg"], stats["away_avg"]

    def _calculate_confidence_enhanced(self, lam_h: float, lam_a: float, match_data: Dict) -> float:
        """
        Geliştirilmiş güven skoru - İleri metrikler ile
        """
        try:
            league = self._auto_detect_league(match_data)
            stats, metrics = self._get_enhanced_league_stats(league)
            
            total = lam_h + lam_a
            balance = 1.0 - min(0.9, abs(lam_h - lam_a) / 4.0)
            pace = min(1.0, total / 3.2)
            
            # Temel güven
            base = 0.55 * balance + 0.45 * pace
            
            # Lig stabilite faktörü
            match_count = stats.get("match_count", 100)
            stability = min(1.0, match_count / 200)  # Daha fazla maç = daha stabil
            
            # Savunma stabilitesi
            defensive_stability = metrics.get("defensive_stability", 0.3)
            stability_factor = 0.7 + 0.3 * defensive_stability
            
            # Oran tabanlı düzeltme
            odds = match_data.get("odds", {})
            ox = float(self._safe_odds(odds, "X", 3.2) or 3.2)
            draw_hint = 1.0 - min(0.6, max(0.0, (ox - 3.0)) / 2.0)
            
            # Favori performansı
            favorite_consistency = metrics.get("favorite_win_rate", 0.55)
            favorite_factor = 0.5 + 0.5 * favorite_consistency
            
            # Tempo korelasyonu - düşük tempolu liglerde daha yüksek güven
            tempo_coefficient = metrics.get("tempo_coefficient", 0.5)
            tempo_confidence = 0.9 + 0.2 * tempo_coefficient  # Yüksek tempo = daha yüksek güven
            
            conf = np.clip(0.2 + 0.8 * base * draw_hint * stability * stability_factor * favorite_factor * tempo_confidence, 0.1, 0.95)
            return float(conf)
            
        except:
            return 0.5

    def predict_with_enhanced_details(self, match_data: Dict, ms_pred: str = "1") -> Dict[str, Any]:
        """
        Geliştirilmiş tahmin fonksiyonu - Zamansal istatistikler & metrikler ile
        """
        # Otomatik lig tespiti
        original_league = match_data.get("league", "Belirtilmemiş")
        detected_league = self._auto_detect_league(match_data)
        
        # ML tahmini
        ml_prediction = self._ml_prediction(match_data, ms_pred)
        
        # Gelişmiş beklenen goller
        lambda_h, lambda_a = self._calculate_expected_goals_enhanced(match_data)
        
        # Ensemble tahmin
        ensemble_h, ensemble_a = self._ensemble_prediction(lambda_h, lambda_a, ml_prediction, match_data)
        
        # Gelişmiş güven skoru
        confidence = self._calculate_confidence_enhanced(ensemble_h, ensemble_a, match_data)
        
        # Lig istatistikleri
        stats, metrics = self._get_enhanced_league_stats(detected_league)
        
        # Final skor
        final_h, final_a = int(round(ensemble_h)), int(round(ensemble_a))
        final_score = f"{final_h}-{final_a}"
        
        # Alternatifler
        alternatives = self._generate_realistic_alternatives(final_h, final_a, ms_pred, ensemble_h, ensemble_a)
        
        # Zengin çıktı
        result = {
            "predicted_score": final_score,
            "expected_goals": {"home": round(ensemble_h, 2), "away": round(ensemble_a, 2)},
            "confidence": round(confidence, 3),
            "alternatives": alternatives,
            "league_info": {
                "original": original_league,
                "detected": detected_league,
                "auto_detected": detected_league != original_league,
                "stats": {
                    "home_avg": round(stats.get("home_avg", 1.52), 2),
                    "away_avg": round(stats.get("away_avg", 1.15), 2),
                    "total_avg": round(stats.get("total_avg", 2.67), 2),
                    "home_advantage": round(stats.get("home_advantage", 0.37), 2),
                    "match_count": stats.get("match_count", 0),
                    "home_trend": round(stats.get("home_trend", 0.0), 3),
                    "away_trend": round(stats.get("away_trend", 0.0), 3)
                },
                "metrics": {
                    "tempo_coefficient": round(metrics.get("tempo_coefficient", 0.5), 2),
                    "favorite_win_rate": round(metrics.get("favorite_win_rate", 0.55), 2),
                    "scoring_efficiency": round(metrics.get("scoring_efficiency", 0.5), 2),
                    "defensive_stability": round(metrics.get("defensive_stability", 0.3), 2),
                    "match_liveliness": round(metrics.get("match_liveliness", 0.45), 2)
                }
            },
            "system_info": {
                "feature_length": self.feature_length,
                "version": self.LEAGUE_STATS_VERSION,
                "temporal_stats_enabled": self.temporal_analyzer is not None,
                "advanced_metrics_enabled": self.metrics_calculator is not None
            }
        }
        
        # Gelişmiş log
        home = match_data.get("home_team", "?")
        away = match_data.get("away_team", "?")
        
        league_info = ""
        if result["league_info"]["auto_detected"]:
            league_info = f" | 🔍 Lig: {original_league}→{detected_league}"
        
        tempo = metrics.get("tempo_coefficient", 0.5)
        home_trend = stats.get("home_trend", 0.0)
        away_trend = stats.get("away_trend", 0.0)
        
        logger.info(
            f"[{home}–{away}] "
            f"λ=({ensemble_h:.2f},{ensemble_a:.2f}) | "
            f"MS={ms_pred} → {final_score} | "
            f"Conf={confidence:.2f} | "
            f"Tempo={tempo:.2f} "
            f"Trend=({home_trend:+.2f},{away_trend:+.2f})"
            f"{league_info}"
        )
        
        return result

    def refresh_all_data(self):
        """Tüm verileri yenile (lig istatistikleri + takım haritası)"""
        logger.info("🔄 Tüm veriler yenileniyor...")
        
        # Temel lig istatistiklerini yenile
        self._build_basic_league_stats()
        
        # Gelişmiş lig istatistiklerini yenile
        self._build_enhanced_league_stats()
        self._save_enhanced_league_stats()
        
        # Takım-lig haritasını yenile
        self.league_mapper.refresh_map()
        
        # Feature boyutunu güncelle
        new_length = self._get_feature_length()
        if new_length != self.feature_length:
            logger.info(f"📐 Feature boyutu güncellendi: {self.feature_length} → {new_length}")
            self.feature_length = new_length
        
        logger.info(f"✅ Tüm veriler güncellendi: {len(self.enhanced_league_stats)} gelişmiş lig")

    def get_detailed_system_info(self) -> Dict[str, Any]:
        """Detaylı sistem bilgisi"""
        base_info = {
            "version": self.LEAGUE_STATS_VERSION,
            "feature_length": self.feature_length,
            "model_loaded": self.model is not None,
            "temporal_stats_enabled": self.temporal_analyzer is not None,
            "advanced_metrics_enabled": self.metrics_calculator is not None,
            "basic_league_count": len(self.league_stats),
            "enhanced_league_count": len(self.enhanced_league_stats),
            "team_mapping_count": len(self.league_mapper.team_to_league)
        }
        
        return base_info

    # Backward compatibility
    def predict_with_details(self, match_data: Dict, ms_pred: str = "1") -> Dict[str, Any]:
        return self.predict_with_enhanced_details(match_data, ms_pred)


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("🚀 GELİŞTİRİLMİŞ SKOR TAHMİN SİSTEMİ - ZAMANSAL İSTATİSTİKLER & METRİKLER")
    print("=" * 80)
    
    # Gelişmiş predictor'u test et
    enhanced_predictor = EnhancedRealisticScorePredictor(
        rng_seed=42,
        enable_temporal_stats=True,
        enable_advanced_metrics=True
    )
    
    # Sistem bilgisi
    sys_info = enhanced_predictor.get_detailed_system_info()
    print(f"✅ Gelişmiş sistem hazır - v{sys_info['version']}")
    print(f"📊 {sys_info['basic_league_count']} temel lig, {sys_info['enhanced_league_count']} gelişmiş lig")
    print(f"📐 Feature boyutu: {sys_info['feature_length']}")
    print(f"🕐 Zamansal istatistikler: {'Aktif' if sys_info['temporal_stats_enabled'] else 'Pasif'}")
    print(f"📈 Gelişmiş metrikler: {'Aktif' if sys_info['advanced_metrics_enabled'] else 'Pasif'}")
    print(f"🎯 Model durumu: {'Yüklü' if sys_info['model_loaded'] else 'Fallback'}")
    
    # Test maçları
    test_matches = [
        {
            "home_team": "Manchester City",
            "away_team": "Liverpool", 
            "league": "Premier League",
            "odds": {"1": 2.1, "X": 3.4, "2": 3.2},
            "date": "2025-10-09"
        },
        {
            "home_team": "Grorud IL", 
            "away_team": "Follo",
            # Lig bilgisi YOK - otomatik tespit edilecek
            "odds": {"1": 1.8, "X": 3.6, "2": 4.0},
            "date": "2025-10-09"
        }
    ]
    
    for i, match in enumerate(test_matches, 1):
        print(f"\n🎯 GELİŞMİŞ TEST {i}: {match['home_team']} vs {match['away_team']}")
        print("-" * 60)
        
        result = enhanced_predictor.predict_with_enhanced_details(match, "1")
        
        print(f"📊 Tahmin: {result['predicted_score']} (Güven: {result['confidence']:.1%})")
        print(f"📈 Beklenen Goller: {result['expected_goals']['home']:.2f}-{result['expected_goals']['away']:.2f}")
        
        league_info = result['league_info']
        stats = league_info['stats']
        metrics = league_info['metrics']
        
        if league_info['auto_detected']:
            print(f"🔍 LİG TESPİTİ: '{league_info['original']}' → '{league_info['detected']}'")
        else:
            print(f"🏆 Lig: {league_info['detected']}")
        
        print(f"📊 Lig İstatistikleri:")
        print(f"   • Ortalama: {stats['home_avg']:.2f}-{stats['away_avg']:.2f} (Toplam: {stats['total_avg']:.2f})")
        print(f"   • Ev Avantajı: {stats['home_advantage']:.2f}")
        print(f"   • Trend: Ev {stats['home_trend']:+.3f}, Deplasman {stats['away_trend']:+.3f}")
        print(f"   • Maç Sayısı: {stats['match_count']}")
        
        print(f"📈 Gelişmiş Metrikler:")
        print(f"   • Tempo Katsayısı: {metrics['tempo_coefficient']:.2f}")
        print(f"   • Favori Galibiyet: {metrics['favorite_win_rate']:.2f}")
        print(f"   • Gol Verimliliği: {metrics['scoring_efficiency']:.2f}")
        print(f"   • Savunma Stabilitesi: {metrics['defensive_stability']:.2f}")
        print(f"   • Maç Canlılığı: {metrics['match_liveliness']:.2f}")
        
        print("🔄 Alternatifler:")
        for alt in result['alternatives']:
            print(f"   • {alt['score']} ({alt['prob']:.1%})")
