import math
import random
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)

class ImprovedPredictionEngine:
    def __init__(self, seed: Optional[int] = None):
        # Rastgelelik deterministik olsun istersek seed alabiliyoruz
        self.rng = random.Random(seed)

    def predict_match(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """ Ana tahmin fonksiyonu """
        try:
            odds = match_data.get("odds", {})
            stats = match_data.get("stats", {})

            # Oranları çıkar
            home_odds, draw_odds, away_odds = self._extract_odds(odds)

            # Olasılıklar
            home_win_prob = 1 / home_odds
            draw_prob = 1 / draw_odds
            away_win_prob = 1 / away_odds

            # Normalize
            total = home_win_prob + draw_prob + away_win_prob
            if total > 0:
                home_win_prob /= total
                draw_prob /= total
                away_win_prob /= total
            else:
                raise ValueError("Invalid odds sum (0)")

            # % çevir
            home_win_prob *= 100
            draw_prob *= 100
            away_win_prob *= 100

            # xG hesapla
            has_team_stats = bool(stats)
            if has_team_stats:
                home_xg, away_xg = self.calculate_xg(stats)
                home_win_prob, draw_prob, away_win_prob = \
                    self.adjust_probabilities_with_stats(
                        home_win_prob, draw_prob, away_win_prob, home_xg, away_xg
                    )

            # Normalize tekrar
            total = home_win_prob + draw_prob + away_win_prob
            if total > 0:
                home_win_prob = max(0, min(100, home_win_prob / total * 100))
                draw_prob = max(0, min(100, draw_prob / total * 100))
                away_win_prob = max(0, min(100, away_win_prob / total * 100))

            # Tahmin
            prediction = self.determine_prediction(home_win_prob, draw_prob, away_win_prob)

            # Güven
            confidence = self.calculate_confidence(home_win_prob, draw_prob, away_win_prob)

            # Skor
            score_pred = self.predict_score(prediction, confidence, home_win_prob, draw_prob, away_win_prob)

            # Oran-tutarlılık
            self.check_probability_consistency(home_odds, draw_odds, away_odds,
                                               home_win_prob, draw_prob, away_win_prob)

            # Sonuç
            return {
                "prediction": prediction,
                "confidence": round(confidence, 2),
                "home_win_prob": round(home_win_prob, 1),
                "draw_prob": round(draw_prob, 1),
                "away_win_prob": round(away_win_prob, 1),
                "score_prediction": score_pred,
                "has_team_stats": has_team_stats,
                "warning": self.generate_warning(confidence, has_team_stats)
            }

        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return {
                "prediction": "N/A",
                "confidence": 0,
                "home_win_prob": 0,
                "draw_prob": 0,
                "away_win_prob": 0,
                "score_prediction": "N/A",
                "warning": f"Hata: {str(e)}"
            }

    # --- Yardımcı fonksiyonlar ---

    def _extract_odds(self, odds: Dict[str, Any]):
        """Oranları farklı formatlardan güvenli şekilde çıkarır."""
        def get_val(keys, default):
            for k in keys:
                if k in odds:
                    try:
                        val = float(odds[k])
                        if val > 0:
                            return val
                    except:
                        continue
            return default

        home = get_val(["1", "home", "h"], 2.5)
        draw = get_val(["X", "draw", "d"], 3.2)
        away = get_val(["2", "away", "a"], 2.8)
        return home, draw, away

    def calculate_xg(self, stats: Dict[str, Any]):
        """Basit xG tahmini (şut ve isabet oranı üzerinden)."""
        hs = stats.get("home_shots", 0)
        hst = stats.get("home_shots_on_target", 0)
        as_ = stats.get("away_shots", 0)
        ast = stats.get("away_shots_on_target", 0)

        # epsilon ile sıfıra bölünme engellendi
        home_xg = (hs * 0.1 + hst * 0.15) * (0.8 + self.rng.random() * 0.4)
        away_xg = (as_ * 0.1 + ast * 0.15) * (0.8 + self.rng.random() * 0.4)

        return min(max(home_xg, 0.2), 3.5), min(max(away_xg, 0.2), 3.5)

    def adjust_probabilities_with_stats(self, h, d, a, hxg, axg):
        total = h + d + a
        if total == 0:
            return h, d, a
        h /= total
        d /= total
        a /= total

        diff = hxg - axg
        h += diff * 0.08
        a -= diff * 0.08

        h = max(0.01, min(0.95, h))
        a = max(0.01, min(0.95, a))
        d = 1 - (h + a)
        if d < 0:
            d = 0.05
            norm = h + a + d
            h /= norm
            a /= norm
            d /= norm

        return h * 100, d * 100, a * 100

    def determine_prediction(self, h, d, a):
        arr = [h, d, a]
        idx = max(range(3), key=lambda i: arr[i])
        return ["1", "X", "2"][idx]

    def calculate_confidence(self, h, d, a):
        arr = [h, d, a]
        top = max(arr)
        second = sorted(arr, reverse=True)[1]
        return min(100, max(0, top - second + 50))

    def predict_score(self, prediction, confidence, h, d, a):
        """Basit skor tahmini (xG yerine olasılıklara göre)."""
        if prediction == "1":
            g1 = self.rng.randint(1, 3) + (1 if confidence > 65 else 0)
            g2 = self.rng.randint(0, 2)
        elif prediction == "2":
            g1 = self.rng.randint(0, 2)
            g2 = self.rng.randint(1, 3) + (1 if confidence > 65 else 0)
        else:  # "X"
            g1 = g2 = self.rng.randint(0, 2)
        return f"{g1}-{g2}"

    def check_probability_consistency(self, ho, do, ao, hp, dp, ap):
        implied = [1/ho, 1/do, 1/ao]
        total = sum(implied)
        if total <= 0:
            return
        implied = [x / total for x in implied]
        actual = [hp/100, dp/100, ap/100]
        diffs = [abs(i-a) for i,a in zip(implied, actual)]
        if any(d > 0.15 for d in diffs):
            logging.warning("⚠️ Odds-based vs Adjusted probabilities mismatch")

    def generate_warning(self, confidence, has_stats):
        if confidence < 55:
            return "⚠️ Düşük güven - dikkatli olun"
        elif confidence < 65:
            return "ℹ️ Orta güven - Dikkatli bahis önerilir"
        else:
            return "✓ Güvenilir tahmin" if has_stats else "ℹ️ Yüksek güven ama istatistik verisi eksik"

    
    def get_fallback_prediction(self) -> Dict[str, Any]:
        """Hata durumunda varsayılan tahmin"""
        return {
            'prediction': 'X',
            'confidence': 33.3,
            'home_win_prob': 33.3,
            'draw_prob': 33.3,
            'away_win_prob': 33.3,
            'score_prediction': '1-1',
            'risk_level': 'very_high',
            'confidence_label': 'Bilinmiyor',
            'has_team_stats': False,
            'timestamp': datetime.now().isoformat(),
            'warning': '⚠️ Tahmin hesaplanamadı'
        }


# Test fonksiyonu
if __name__ == "__main__":
    predictor = ImprovedPredictionEngine()
    
    # Test 1: Sadece oranlar
    print("Test 1: Sadece oranlar")
    result = predictor.predict_match(
        home_team="Arsenal",
        away_team="Chelsea",
        odds={'1': 2.10, 'X': 3.40, '2': 3.50},
        league='Premier League'
    )
    print(f"Tahmin: {result['prediction']}")
    print(f"Güven: {result['confidence']}%")
    print(f"Skor: {result['score_prediction']}")
    print(f"Uyarı: {result['warning']}\n")
    
    # Test 2: İstatistiklerle
    print("Test 2: İstatistiklerle")
    result = predictor.predict_match(
        home_team="Arsenal",
        away_team="Chelsea",
        odds={'1': 2.10, 'X': 3.40, '2': 3.50},
        home_stats={
            'matches_played': 10,
            'goals_for': 25,
            'goals_against': 8
        },
        away_stats={
            'matches_played': 10,
            'goals_for': 18,
            'goals_against': 12
        },
        league='Premier League'
    )
    print(f"Tahmin: {result['prediction']}")
    print(f"Güven: {result['confidence']}%")
    print(f"Skor: {result['score_prediction']}")
    print(f"Stats var: {result['has_team_stats']}")
    print(f"Uyarı: {result['warning']}")
