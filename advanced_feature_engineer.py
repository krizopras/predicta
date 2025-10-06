# ==============================================================
#  advanced_feature_engineer.py
#  Predicta Europe ML v2 - Güvenli Feature Oluşturma
#  Olcay Arslan
# ==============================================================
#  Bu modül, ham maç verilerinden (raw data) ML eğitimine uygun
#  feature setleri oluşturur. String / int hatalarına karşı
#  korumalıdır. Özellikle "int object has no attribute 'lower'"
#  hatası bu sürümde tamamen engellenmiştir.
# ==============================================================

import math
import json
import logging
from typing import Dict, Any, Optional, List
import numpy as np

# Log ayarı
logging.basicConfig(level=logging.INFO, format="[FeatureEngineer] %(message)s")

# Feature isimleri - ML modellerinde kullanılacak
FEATURE_NAMES = [
    "odds_1", "odds_x", "odds_2",
    "prob_1", "prob_x", "prob_2",
    "goal_diff", "total_goals",
    "home_len", "away_len",
    "is_derby", "league_country_score"
]


# ==============================================================
#  Yardımcı Fonksiyonlar
# ==============================================================

def safe_lower(value: Any) -> str:
    """
    String olmayan bir değeri güvenli biçimde küçük harfe çevirir.
    Örnek: 123 -> "123", None -> "", "GALATASARAY" -> "galatasaray"
    """
    if value is None:
        return ""
    try:
        return str(value).lower()
    except Exception:
        return ""


def safe_float(value: Any, default: float = 0.0) -> float:
    """Her türlü girdiyi float'a dönüştürür; hata durumunda varsayılan döner."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Her türlü girdiyi int'e dönüştürür; hata durumunda varsayılan döner."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def ratio(a: float, b: float) -> float:
    """0 bölü hatasına karşı güvenli oran hesaplama."""
    if b == 0:
        return 0.0
    return round(a / b, 3)


def goal_diff(home_goals: int, away_goals: int) -> int:
    """Gol farkı hesaplar."""
    return home_goals - away_goals


# ==============================================================
#  Ana Feature Oluşturucu Fonksiyon
# ==============================================================

def generate_features(match_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Bir maç satırından modelin anlayacağı numerik özellikleri üretir.
    Hatalı veya eksik veri varsa None döndürür.
    """
    try:
        # Takım adlarını güvenli al
        home = safe_lower(match_row.get("home_team"))
        away = safe_lower(match_row.get("away_team"))
        league = safe_lower(match_row.get("league"))

        # Zorunlu alanlar kontrol
        if not home or not away:
            logging.warning(f"Eksik takım bilgisi: {match_row}")
            return None

        # Oranlar
        odds_1 = safe_float(match_row.get("odds_1"))
        odds_x = safe_float(match_row.get("odds_x"))
        odds_2 = safe_float(match_row.get("odds_2"))

        # ⚙️ ORANLARIN GEÇERLİLİĞİNİ KONTROL ET
        # Eğer oranlar eksik, 0.0, NaN veya çok düşükse varsayılan değerleri ata
        if any(v <= 1.01 or math.isnan(v) for v in [odds_1, odds_x, odds_2]):
            logging.warning(f"Geçersiz oran tespit edildi, varsayılan atanıyor: {match_row}")
            odds_1, odds_x, odds_2 = 2.0, 3.0, 3.5

        # Basit oran metrikleri
        try:
            odds_sum = odds_1 + odds_x + odds_2
            prob_1 = ratio(1 / odds_1, 1 / odds_sum)
            prob_x = ratio(1 / odds_x, 1 / odds_sum)
            prob_2 = ratio(1 / odds_2, 1 / odds_sum)
        except ZeroDivisionError:
            # Bölme hatasına karşı koruma
            logging.warning(f"Oran bölme hatası, varsayılan 1/3 olasılık kullanılıyor: {match_row}")
            prob_1 = prob_x = prob_2 = 1 / 3

        # Gol verileri
        home_goals = safe_int(match_row.get("home_goals"))
        away_goals = safe_int(match_row.get("away_goals"))
        goal_diff_val = goal_diff(home_goals, away_goals)
        total_goals = home_goals + away_goals

        # Takım isimlerinden nitelik çıkarma (örnek: derbi, "spor" içeren takım vs.)
        home_len = len(home)
        away_len = len(away)
        is_derby = 1 if any(keyword in home for keyword in ["galatasaray", "fenerbahçe", "beşiktaş"]) and \
                        any(keyword in away for keyword in ["galatasaray", "fenerbahçe", "beşiktaş"]) else 0

        # Lig özellikleri (ülke bazlı genellemeler)
        league_country_score = 0
        if "turk" in league:
            league_country_score = 1.0
        elif "eng" in league or "prem" in league:
            league_country_score = 2.0
        elif "esp" in league or "lalig" in league:
            league_country_score = 2.5
        elif "ital" in league:
            league_country_score = 2.2
        elif "germ" in league or "bund" in league:
            league_country_score = 2.3
        elif "fran" in league or "ligue" in league:
            league_country_score = 2.0

        # Sonuç etiketleri (model eğitimi için)
        result = None
        if home_goals > away_goals:
            result = "1"
        elif home_goals < away_goals:
            result = "2"
        else:
            result = "X"

        # Özellik seti
        features = {
            "home_team": home,
            "away_team": away,
            "league": league,
            "odds_1": odds_1,
            "odds_x": odds_x,
            "odds_2": odds_2,
            "prob_1": prob_1,
            "prob_x": prob_x,
            "prob_2": prob_2,
            "goal_diff": goal_diff_val,
            "total_goals": total_goals,
            "home_len": home_len,
            "away_len": away_len,
            "is_derby": is_derby,
            "league_country_score": league_country_score,
            "target": result
        }

        return features

    except Exception as e:
        logging.error(f"Feature hatası: {e} | Satır: {match_row}")
        return None


# ==============================================================
#  Sınıf Wrapper - ML Pipeline için
# ==============================================================

class AdvancedFeatureEngineer:
    """
    ML pipeline'larında kullanılmak üzere sınıf wrapper.
    Takım geçmişi, form, momentum gibi ileri düzey özellikleri
    de yönetebilir (ileride genişletilebilir).
    """
    
    def __init__(self):
        self.team_history = {}
        self.h2h_history = {}
        self.league_stats = {}
    
    def extract_features(self, match_data: Dict) -> np.ndarray:
        """
        Maç verisinden feature vektörü üretir.
        Returns: numpy array [len(FEATURE_NAMES)]
        """
        feats = generate_features(match_data)
        if feats is None:
            # Hata durumunda sıfır vektör döndür
            return np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        
        # Sadece numerik feature'ları sırayla al
        return np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float32)
    
    def update_team_history(self, team: str, match_data: Dict):
        """Takım geçmişini günceller (form, son maçlar vs.)"""
        if team not in self.team_history:
            self.team_history[team] = []
        self.team_history[team].append(match_data)
        # Son 10 maçı tut
        if len(self.team_history[team]) > 10:
            self.team_history[team] = self.team_history[team][-10:]
    
    def update_h2h_history(self, home_team: str, away_team: str, match_data: Dict):
        """Karşılıklı maç geçmişini günceller"""
        key = f"{home_team}_vs_{away_team}"
        if key not in self.h2h_history:
            self.h2h_history[key] = []
        self.h2h_history[key].append(match_data)
        if len(self.h2h_history[key]) > 5:
            self.h2h_history[key] = self.h2h_history[key][-5:]
    
    def update_league_results(self, league: str, result: str):
        """Lig istatistiklerini günceller"""
        if league not in self.league_stats:
            self.league_stats[league] = {"1": 0, "X": 0, "2": 0}
        self.league_stats[league][result] = self.league_stats[league].get(result, 0) + 1


# ==============================================================
#  Örnek Kullanım (Test)
# ==============================================================

if __name__ == "__main__":
    test_row = {
        "home_team": "Galatasaray",
        "away_team": 12345,  # int değer -> artık güvenli
        "league": "TR1",
        "odds_1": "2.05",
        "odds_x": "3.30",
        "odds_2": "3.40",
        "home_goals": 3,
        "away_goals": 1
    }

    # Fonksiyon testi
    f = generate_features(test_row)
    print("=== Fonksiyon Çıktısı ===")
    print(json.dumps(f, indent=2, ensure_ascii=False))
    
    # Sınıf testi
    engineer = AdvancedFeatureEngineer()
    feat_vector = engineer.extract_features(test_row)
    print("\n=== Sınıf Çıktısı (Feature Vector) ===")
    print(f"Shape: {feat_vector.shape}")
    print(f"Values: {feat_vector}")
    print(f"Feature Names: {FEATURE_NAMES}")
