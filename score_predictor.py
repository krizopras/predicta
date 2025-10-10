import os
import pickle
import numpy as np
import logging
from typing import Dict, Tuple, List, Optional
from advanced_feature_engineer import AdvancedFeatureEngineer

logger = logging.getLogger(__name__)

class RealisticScorePredictor:
    """
    Gerçeğe en yakın skor tahmini - Production-ready güvenlik katmanları
    
    Özellikler:
    - Güvenli oran dönüşümleri (string/0.0 koruması)
    - Deterministik RNG (seed olmasa bile)
    - Lambda normalizasyonu (aşırı farkları önler)
    - Yumuşak blowout baskılama (istatistiksel)
    - Garantili 3 farklı alternatif skor
    - Tek satır debug log formatı
    """

    def __init__(
        self,
        models_dir: str = "data/ai_models_v2",
        rng_seed: Optional[int] = None,
        max_goals: int = 6
    ):
        self.models_dir = models_dir
        self.engineer = AdvancedFeatureEngineer(model_path=models_dir)
        self.model = None
        self.space = None
        self.max_goals = int(max_goals)
        
        # Her zaman deterministik RNG (seed yoksa 0 kullan)
        self.rng = np.random.default_rng(rng_seed if rng_seed is not None else 0)
        
        # Gerçekçi futbol istatistikleri (5 büyük Avrupa ligi ortalaması)
        self.avg_home_goals = 1.52
        self.avg_away_goals = 1.15
        self.base_home_advantage = 0.37
        
        # Lig bazlı ev sahibi avantajı çarpanları
        self.league_home_factors = {
            "Premier League": 1.05,
            "La Liga": 0.98,
            "Serie A": 0.95,
            "Bundesliga": 1.08,
            "Ligue 1": 1.02,
            "Eredivisie": 1.12,
            "Championship": 1.10,
            "Turkish Super Lig": 1.15,
            "Süper Lig": 1.15,
            "Primeira Liga": 1.10,
            "Scottish Premiership": 1.12,
        }
        
        # Epsilon (sıfır bölme koruması)
        self.eps = 1e-9
        
        self._load()

    def _load(self):
        """Model dosyasını güvenli yükle"""
        path = os.path.join(self.models_dir, "score_model.pkl")

        if not os.path.exists(path):
            logger.warning(f"⚠️ Skor modeli bulunamadı: {path}")
            return

        file_size = os.path.getsize(path)
        if file_size < 100:
            logger.error(f"❌ Bozuk skor modeli ({file_size} bytes)")
            try:
                os.remove(path)
            except Exception:
                pass
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

            logger.info(f"✅ Skor modeli yüklendi ({file_size} bytes, {len(self.space)} sınıf)")

        except Exception as e:
            logger.error(f"❌ Skor model yükleme hatası: {e}")
            self.model = None
            self.space = None

    @staticmethod
    def _ms_ok(ms: str, score_label: str) -> bool:
        """MS tutarlılık kontrolü"""
        try:
            a, b = score_label.split("-")
            a, b = int(a), int(b)
        except Exception:
            return True

        if ms == "1":
            return a > b
        elif ms == "X":
            return a == b
        elif ms == "2":
            return a < b
        return True

    def _safe_odds(self, odds: Dict, key: str, default: float) -> float:
        """
        Güvenli oran dönüşümü
        - String → float
        - 0.0 veya negatif → default
        - None → default
        """
        try:
            val = odds.get(key, default)
            if val is None:
                return default
            val = float(val)
            if val <= 0.0:
                return default
            return val
        except (ValueError, TypeError):
            return default

    def _get_draw_calibration(self, odds_x: float) -> float:
        """
        Beraberlik oranından toplam gol kalibrasyonu
        X oranı düşükse (2.80-3.20) → düşük skor beklenir
        X oranı yüksekse (4.0+) → yüksek skor beklenir
        """
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

    def _get_league_home_advantage(self, league: str) -> float:
        """
        Lig bazlı ev sahibi avantajı çarpanı
        Eşleşme yoksa 1.0 (ortalama)
        """
        league_lower = league.lower()
        for key, factor in self.league_home_factors.items():
            if key.lower() in league_lower:
                return factor
        return 1.0  # Varsayılan (bilinmeyen ligler)

    def _soft_cap_lambda(self, lam: float, odds: float) -> float:
        """
        Yumuşak blowout baskılama (istatistiksel)
        Sert kesim yerine sigmoid benzeri yumuşatma
        
        odds < 1.20 → çok büyük favori, λ ≤ 3.8
        odds < 1.50 → büyük favori, λ ≤ 3.2
        odds ≥ 1.50 → normal, λ ≤ 2.8
        """
        if odds < 1.20:
            max_lam = 3.8
        elif odds < 1.50:
            max_lam = 3.2
        else:
            max_lam = 2.8
        
        # Sigmoid yumuşatma (sert kesim yerine)
        if lam <= max_lam:
            return lam
        else:
            # Aşan kısmı ezici şekilde bastır
            excess = lam - max_lam
            return max_lam + excess * 0.3  # %30'unu korur

    def _normalize_lambda_diff(self, lambda_h: float, lambda_a: float) -> Tuple[float, float]:
        """
        Lambda farkını normalize et (aşırı farkları önle)
        Zayıf liglerde λ_h=4.5, λ_a=0.3 gibi aşırı farklar olabilir
        → Toplam gol korunarak farkı daralt
        """
        total = lambda_h + lambda_a
        diff = lambda_h - lambda_a
        
        # Fark çok büyükse (> 2.5) daralt
        max_diff = 2.5
        if abs(diff) > max_diff:
            sign = 1 if diff > 0 else -1
            diff = sign * max_diff
            
            # Yeniden dağıt (toplam korunsun)
            lambda_h = (total + diff) / 2.0
            lambda_a = (total - diff) / 2.0
        
        return lambda_h, lambda_a

    def _calculate_expected_goals(self, match_data: Dict) -> Tuple[float, float]:
        """
        Gerçekçi beklenen gol hesaplama (Poisson λ parametreleri)
        - Takım formu, ortalama goller, oranlar
        - Lig bazlı ev sahibi avantajı
        - Beraberlik oranıyla toplam gol kalibrasyonu
        - Lambda normalizasyonu
        """
        try:
            feats = self.engineer.extract_features(match_data)
            if feats is None:
                return self.avg_home_goals, self.avg_away_goals

            # Feature'lardan takım güçleri
            home_form = feats[12] if len(feats) > 12 else 0.5
            away_form = feats[13] if len(feats) > 13 else 0.5
            home_avg = feats[14] if len(feats) > 14 else 1.5
            away_avg = feats[15] if len(feats) > 15 else 1.2

            # Güvenli oran dönüşümü
            odds = match_data.get("odds", {})
            odds_1 = self._safe_odds(odds, "1", 2.5)
            odds_x = self._safe_odds(odds, "X", 3.3)
            odds_2 = self._safe_odds(odds, "2", 2.8)

            # Oran → güç (düşük oran = güçlü takım)
            home_strength = min(2.0, max(0.3, 2.5 / (odds_1 + self.eps)))
            away_strength = min(2.0, max(0.3, 2.5 / (odds_2 + self.eps)))

            # Lig bazlı ev sahibi avantajı
            league = match_data.get("league", "")
            league_factor = self._get_league_home_advantage(league)
            home_advantage = self.base_home_advantage * league_factor

            # Beraberlik oranından toplam gol kalibrasyonu
            goal_calibration = self._get_draw_calibration(odds_x)

            # Lambda hesaplama (beklenen gol sayısı)
            lambda_h = (
                self.avg_home_goals * 0.25 +
                home_avg * 0.30 +
                home_form * home_strength * 1.3 * 0.45
            ) * goal_calibration + home_advantage

            lambda_a = (
                self.avg_away_goals * 0.25 +
                away_avg * 0.30 +
                away_form * away_strength * 1.1 * 0.45
            ) * goal_calibration

            # Lambda farkını normalize et (aşırı farkları önle)
            lambda_h, lambda_a = self._normalize_lambda_diff(lambda_h, lambda_a)

            # Yumuşak blowout baskılama
            lambda_h = self._soft_cap_lambda(lambda_h, odds_1)
            lambda_a = self._soft_cap_lambda(lambda_a, odds_2)

            # Alt sınır (çok düşük skorlar nadir)
            lambda_h = max(0.6, lambda_h)
            lambda_a = max(0.5, lambda_a)

            return lambda_h, lambda_a

        except Exception as e:
            logger.error(f"Beklenen gol hesaplama hatası: {e}")
            return self.avg_home_goals, self.avg_away_goals

    def _poisson_mode(self, lam: float) -> int:
        """Poisson dağılımının modu (en olası değer)"""
        return max(0, int(np.floor(lam)))

    def _poisson_score(self, lambda_h: float, lambda_a: float, ms: str) -> Tuple[int, int]:
        """
        Poisson dağılımı ile gerçekçi skor üret + MS düzeltmesi
        
        MS düzeltmesi gerçekçi:
        - MS="X" → geniş beraberlik dağılımı (0-0, 1-1, 2-2, 3-3)
        - MS="1" → ev sahibi önde, küçük farklar tercih
        - MS="2" → deplasman önde, küçük farklar tercih
        """
        # İlk Poisson çekimi
        home_goals = int(self.rng.poisson(lambda_h))
        away_goals = int(self.rng.poisson(lambda_a))
        
        # MS ile uyumlu hale getir (gerçekçi düzeltme)
        if ms == "X":
            # Beraberlik: geniş dağılım (0-0'a saplanma yok)
            mode_h = self._poisson_mode(lambda_h)
            mode_a = self._poisson_mode(lambda_a)
            
            # Ortalama lambda'ya yakın beraberlik seç
            avg_lam = (lambda_h + lambda_a) / 2.0
            
            if avg_lam < 1.0:
                # Düşük gol beklentisi → 0-0, 1-1 ağırlıklı
                target = self.rng.choice([0, 1], p=[0.55, 0.45])
            elif avg_lam < 2.0:
                # Orta gol → 1-1, 2-2 ağırlıklı
                target = self.rng.choice([1, 2], p=[0.60, 0.40])
            else:
                # Yüksek gol → 2-2, 3-3 ağırlıklı
                target = self.rng.choice([2, 3], p=[0.55, 0.45])
            
            home_goals = away_goals = target
            
        elif ms == "1":
            # Ev sahibi kazanmalı
            if home_goals <= away_goals:
                # Küçük fark tercih et (1-2 gol)
                diff = self.rng.choice([1, 2], p=[0.75, 0.25])
                home_goals = away_goals + diff
                
        elif ms == "2":
            # Deplasman kazanmalı
            if away_goals <= home_goals:
                diff = self.rng.choice([1, 2], p=[0.75, 0.25])
                away_goals = home_goals + diff
        
        # Sınırla
        home_goals = max(0, min(self.max_goals, home_goals))
        away_goals = max(0, min(self.max_goals, away_goals))
        
        return home_goals, away_goals

    def _poisson_prob(self, k: int, lam: float) -> float:
        """Poisson PMF: P(X=k|λ)"""
        if lam <= 0:
            return 0.0
        try:
            return (lam ** k) * np.exp(-lam) / np.math.factorial(k)
        except (OverflowError, ValueError):
            return 0.0

    def _generate_realistic_alternatives(
        self, h: int, a: int, ms: str, lambda_h: float, lambda_a: float
    ) -> List[Dict]:
        """
        Joint Poisson olasılıklarıyla gerçekçi alternatif skorlar
        - 0-6 ızgarasında tüm MS-uyumlu skorları değerlendir
        - Garantili 3 FARKLI skor döndür
        - Epsilon koruması (sıfır olasılık senaryoları)
        """
        candidates = []
        
        for gh in range(self.max_goals + 1):
            for ga in range(self.max_goals + 1):
                if self._ms_ok(ms, f"{gh}-{ga}"):
                    # Joint probability (bağımsız Poisson varsayımı)
                    prob = self._poisson_prob(gh, lambda_h) * self._poisson_prob(ga, lambda_a)
                    candidates.append({
                        "score": f"{gh}-{ga}",
                        "prob": prob
                    })
        
        # Olasılığa göre sırala
        candidates = sorted(candidates, key=lambda x: x["prob"], reverse=True)
        
        # Top 3 al (FARKLI skorlar garantisi)
        top3 = []
        seen_scores = set()
        for cand in candidates:
            if cand["score"] not in seen_scores:
                top3.append(cand)
                seen_scores.add(cand["score"])
            if len(top3) == 3:
                break
        
        # 3'ten az bulunduysa fallback skorlar ekle
        if len(top3) < 3:
            fallback_scores = self._get_fallback_scores(ms, h, a)
            for fb_score in fallback_scores:
                if fb_score not in seen_scores and len(top3) < 3:
                    top3.append({"score": fb_score, "prob": 0.01})
                    seen_scores.add(fb_score)
        
        # Normalize et (epsilon koruması)
        total = sum(x["prob"] for x in top3)
        if total > self.eps:
            for x in top3:
                x["prob"] = round(x["prob"] / total, 3)
        else:
            # Çok düşük olasılıklar → eşit dağıt
            for x in top3:
                x["prob"] = round(1.0 / len(top3), 3)
        
        return top3

    def _get_fallback_scores(self, ms: str, h: int, a: int) -> List[str]:
        """Fallback skorlar (alternatif üretemezse)"""
        if ms == "1":
            return [f"{h}-{a}", "2-1", "1-0", "2-0", "3-1"]
        elif ms == "X":
            return ["1-1", "0-0", "2-2", "3-3"]
        else:
            return [f"{h}-{a}", "1-2", "0-1", "0-2", "1-3"]

    def _intelligent_fallback(self, match_data: Dict, ms_pred: str) -> Tuple[str, List[Dict]]:
        """
        Gerçekçi fallback - Poisson + tüm kalibrasyonlar
        """
        try:
            lambda_h, lambda_a = self._calculate_expected_goals(match_data)
            
            # Poisson ile skor üret
            h, a = self._poisson_score(lambda_h, lambda_a, ms_pred)
            score = f"{h}-{a}"
            
            # Alternatifler (joint Poisson)
            top3 = self._generate_realistic_alternatives(h, a, ms_pred, lambda_h, lambda_a)
            
            # Tek satır debug log
            home = match_data.get("home_team", "?")
            away = match_data.get("away_team", "?")
            logger.info(f"🎯 {home} vs {away} | λ=({lambda_h:.2f},{lambda_a:.2f}) | MS={ms_pred} → {score}")
            
            return score, top3

        except Exception as e:
            logger.error(f"Fallback hatası: {e}")
            return self._simple_fallback(ms_pred)

    def _simple_fallback(self, ms_pred: str) -> Tuple[str, List[Dict]]:
        """Son çare fallback - genel futbol istatistikleri"""
        if ms_pred == "1":
            return "2-1", [
                {"score": "2-1", "prob": 0.38},
                {"score": "1-0", "prob": 0.32},
                {"score": "2-0", "prob": 0.18}
            ]
        elif ms_pred == "X":
            return "1-1", [
                {"score": "1-1", "prob": 0.48},
                {"score": "0-0", "prob": 0.22},
                {"score": "2-2", "prob": 0.18}
            ]
        else:
            return "1-2", [
                {"score": "1-2", "prob": 0.38},
                {"score": "0-1", "prob": 0.30},
                {"score": "0-2", "prob": 0.20}
            ]

    def predict(self, match_data: Dict, ms_pred: str = "1") -> Tuple[str, List[Dict]]:
        """
        Ana tahmin fonksiyonu
        Args:
            match_data: {"home_team", "away_team", "league", "odds", "date"}
            ms_pred: Maç sonucu tahmini ("1", "X", "2")
        Returns:
            (predicted_score, top3_alternatives)
        """
        if self.model is None or self.space is None:
            logger.warning("⚠️ Model yüklü değil, gerçekçi fallback kullanılıyor")
            return self._intelligent_fallback(match_data, ms_pred)

        try:
            feats = self.engineer.extract_features(match_data)
            if feats is None:
                return self._intelligent_fallback(match_data, ms_pred)

            feats = feats.reshape(1, -1)
            probs = self.model.predict_proba(feats)[0]

            # MS tutarsız sınıfları cezalandır
            adj = probs.copy()
            for i, lab in enumerate(self.space):
                if lab != "OTHER" and not self._ms_ok(ms_pred, lab):
                    adj[i] *= 0.12

            s = adj.sum()
            if s > 0:
                adj = adj / s

            top_idx = int(np.argmax(adj))
            top_label = self.space[top_idx]

            # Top 3 seç (OTHER hariç)
            order = np.argsort(adj)[::-1]
            top3 = []
            for i in order:
                if self.space[i] == "OTHER":
                    continue
                top3.append({"score": self.space[i], "prob": float(adj[i])})
                if len(top3) == 3:
                    break

            # Model belirsizse fallback
            if top_label == "OTHER" or (len(top3) > 0 and top3[0]["prob"] < 0.15):
                logger.info(f"Model belirsiz (prob={adj[top_idx]:.3f}), fallback kullanılıyor")
                return self._intelligent_fallback(match_data, ms_pred)

            return top_label, top3

        except Exception as e:
            logger.error(f"❌ Tahmin hatası: {e}")
            return self._intelligent_fallback(match_data, ms_pred)


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    predictor = RealisticScorePredictor(rng_seed=42)

    test_matches = [
        {
            "home_team": "Manchester City",
            "away_team": "Liverpool",
            "league": "Premier League",
            "odds": {"1": 2.10, "X": 3.40, "2": 3.20},
            "date": "2025-10-09"
        },
        {
            "home_team": "Barcelona",
            "away_team": "Getafe",
            "league": "La Liga",
            "odds": {"1": 1.25, "X": 6.50, "2": 11.0},
            "date": "2025-10-09"
        },
        {
            "home_team": "Bayern Munich",
            "away_team": "Augsburg",
            "league": "Bundesliga",
            "odds": {"1": "1.15", "X": "8.00", "2": "15.0"},  # String test
            "date": "2025-10-09"
        },
        {
            "home_team": "Real Madrid",
            "away_team": "Atletico Madrid",
            "league": "La Liga",
            "odds": {"1": 1.95, "X": 3.20, "2": 3.80},
            "date": "2025-10-09"
        },
        {
            "home_team": "Unknown Team",
            "away_team": "Another Team",
            "league": "Mystery League",
            "odds": {"1": None, "X": 0.0, "2": "invalid"},  # Edge case test
            "date": "2025-10-09"
        }
    ]

    for match in test_matches:
        print(f"\n{'='*75}")
        print(f"🏟️  {match['home_team']} vs {match['away_team']} ({match['league']})")
        print(f"📊 Oranlar: 1={match['odds']['1']}, X={match['odds']['X']}, 2={match['odds']['2']}")
        
        for ms in ["1", "X", "2"]:
            score, top3 = predictor.predict(match, ms_pred=ms)
            ms_label = {"1": "Ev Sahibi", "X": "Beraberlik", "2": "Deplasman"}[ms]
            print(f"\n   {ms_label} (MS={ms}) → {score}")
            for i, s in enumerate(top3, 1):
                print(f"      {i}. {s['score']}: {s['prob']*100:.1f}%")
