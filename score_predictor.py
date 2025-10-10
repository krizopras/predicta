# score_predictor.py - İYİLEŞTİRİLMİŞ VERSİYON v2
import os
import pickle
import numpy as np
import logging
from typing import Dict, Tuple, List
from advanced_feature_engineer import AdvancedFeatureEngineer

logger = logging.getLogger(__name__)

class ScorePredictor:
    def __init__(self, models_dir: str = "data/ai_models_v2"):
        self.models_dir = models_dir
        self.engineer = AdvancedFeatureEngineer(model_path=models_dir)
        self.model = None
        self.space = None
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
            os.remove(path)
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
            
        except (EOFError, pickle.UnpicklingError) as e:
            logger.error(f"❌ Bozuk pickle dosyası: {e}")
            os.remove(path)
            self.model = None
            self.space = None
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
        except:
            return True
        
        if ms == "1":
            return a > b
        elif ms == "X":
            return a == b
        elif ms == "2":
            return a < b
        return True
    
    def _intelligent_fallback(self, match_data: Dict, ms_pred: str) -> Tuple[str, List[Dict]]:
        """
        Feature'lardan akıllı skor tahmini
        - Takım formu
        - Ortalama gol sayıları
        - Lig seviyesi
        - Odds analizi (GÜÇ FARKI)
        """
        try:
            # Feature çıkar
            feats = self.engineer.extract_features(match_data)
            if feats is None:
                return self._simple_fallback(ms_pred)
            
            # Takım formları (index 12, 13)
            home_form = feats[12] if len(feats) > 12 else 0.5
            away_form = feats[13] if len(feats) > 13 else 0.5
            
            # Ortalama gol sayıları (index 14, 15)
            home_avg = feats[14] if len(feats) > 14 else 1.5
            away_avg = feats[15] if len(feats) > 15 else 1.5
            
            # Odds'lardan beklenen gol ve GÜÇ FARKI
            odds = match_data.get("odds", {})
            odds_1 = float(odds.get("1", 2.0))
            odds_x = float(odds.get("X", 3.0))
            odds_2 = float(odds.get("2", 3.5))
            
            # Oran ne kadar düşükse, o takım o kadar güçlü
            home_strength = 1.0 / odds_1 if odds_1 > 1.01 else 0.5
            away_strength = 1.0 / odds_2 if odds_2 > 1.01 else 0.5
            
            # GÜÇ FARKI HESAPLANSğIN (kritik!)
            strength_diff = home_strength - away_strength
            
            # Form + ortalama gol + oran gücü
            home_attack = (home_form * 2 + home_avg + home_strength * 2) / 5
            away_attack = (away_form * 2 + away_avg + away_strength * 2) / 5
            
            # MS tahminine göre ayarla
            if ms_pred == "1":  # Ev sahibi galip
                # Güç farkı çok büyükse (ör: Fransa vs Azerbaycan)
                if strength_diff > 0.3:  # Büyük fark
                    home_goals = max(2, round(home_attack * 2.5))
                    away_goals = max(0, round(away_attack * 0.8))
                else:
                    home_goals = max(1, round(home_attack * 2.0))
                    away_goals = max(0, round(away_attack * 1.2))
                
                # Ev sahibi galip olmalı
                if home_goals <= away_goals:
                    home_goals = away_goals + np.random.choice([1, 2])
                
            elif ms_pred == "X":  # Beraberlik
                avg_goals = (home_attack + away_attack) / 2
                
                # GÜÇ FARKI BÜYÜKSE BERABERLİK OLASI DEĞİL!
                if abs(strength_diff) > 0.25:
                    # Favoriye göre hafif avantaj ver ama beraberlik yap
                    if strength_diff > 0:  # Ev sahibi güçlü
                        base = max(1, round(home_attack * 1.3))
                    else:  # Deplasman güçlü
                        base = max(1, round(away_attack * 1.3))
                    home_goals = away_goals = min(3, max(1, base))
                else:
                    # Normal beraberlik senaryosu
                    possible_draws = [(0, 0), (1, 1), (2, 2), (3, 3)]
                    
                    # Ortalama gol sayısına göre ağırlıklı seçim
                    if avg_goals < 1.0:
                        weights = [0.5, 0.3, 0.15, 0.05]
                    elif avg_goals < 1.5:
                        weights = [0.3, 0.4, 0.25, 0.05]
                    elif avg_goals < 2.5:
                        weights = [0.1, 0.35, 0.45, 0.1]
                    else:
                        weights = [0.05, 0.25, 0.45, 0.25]
                    
                    idx = np.random.choice(len(possible_draws), p=weights)
                    home_goals, away_goals = possible_draws[idx]
                
            else:  # Deplasman galip
                # Güç farkı çok büyükse
                if strength_diff < -0.3:  # Deplasman çok güçlü
                    home_goals = max(0, round(home_attack * 0.8))
                    away_goals = max(2, round(away_attack * 2.5))
                else:
                    home_goals = max(0, round(home_attack * 1.2))
                    away_goals = max(1, round(away_attack * 2.0))
                
                # Deplasman galip olmalı
                if away_goals <= home_goals:
                    away_goals = home_goals + np.random.choice([1, 2])
            
            # Makul sınırlar
            home_goals = min(6, max(0, home_goals))
            away_goals = min(6, max(0, away_goals))
            
            score = f"{home_goals}-{away_goals}"
            
            # Top 3 alternatif
            top3 = self._generate_alternatives(home_goals, away_goals, ms_pred, strength_diff)
            
            logger.info(f"🎯 Akıllı fallback: {score} (H_form={home_form:.2f}, A_form={away_form:.2f}, diff={strength_diff:.2f})")
            return score, top3
            
        except Exception as e:
            logger.error(f"Akıllı fallback hatası: {e}")
            return self._simple_fallback(ms_pred)
    
    def _generate_alternatives(self, h: int, a: int, ms: str, strength_diff: float = 0.0) -> List[Dict]:
        """Verilen skora benzer alternatif skorlar üret (güç farkını dikkate alarak)"""
        alternatives = []
        
        # Ana tahmin
        alternatives.append({
            "score": f"{h}-{a}",
            "prob": 0.40
        })
        
        # Varyasyonlar (güç farkına göre)
        if ms == "1":  # Ev sahibi galip
            if strength_diff > 0.3:  # Büyük fark
                variations = [
                    (h+1, a), (h, max(0, a-1)), (h+2, a)
                ]
            else:
                variations = [
                    (h+1, a), (h, a+1) if h > a+1 else (h+1, a), 
                    (h, a-1) if a > 0 else (h+2, a)
                ]
        elif ms == "X":  # Beraberlik
            variations = [
                (h+1, a+1), (h-1, a-1) if h > 0 else (h+1, a+1),
                (h, a)
            ]
        else:  # Deplasman galip
            if strength_diff < -0.3:  # Deplasman çok güçlü
                variations = [
                    (h, a+1), (max(0, h-1), a), (h, a+2)
                ]
            else:
                variations = [
                    (h, a+1), (h+1, a) if a > h+1 else (h, a+1),
                    (h-1, a) if h > 0 else (h, a+2)
                ]
        
        for var_h, var_a in variations[:2]:
            var_h = min(5, max(0, var_h))
            var_a = min(5, max(0, var_a))
            
            # MS tutarlılığı kontrol et
            var_score = f"{var_h}-{var_a}"
            if self._ms_ok(ms, var_score):
                alternatives.append({
                    "score": var_score,
                    "prob": np.random.uniform(0.15, 0.30)
                })
        
        # Olasılıkları normalize et
        total_prob = sum(alt["prob"] for alt in alternatives)
        for alt in alternatives:
            alt["prob"] = round(alt["prob"] / total_prob, 3)
        
        return alternatives[:3]
    
    def _simple_fallback(self, ms_pred: str) -> Tuple[str, List[Dict]]:
        """Basit fallback (son çare)"""
        if ms_pred == "1":
            return "2-1", [{"score": "2-1", "prob": 0.33}, {"score": "3-1", "prob": 0.25}, {"score": "1-0", "prob": 0.22}]
        elif ms_pred == "X":
            return "1-1", [{"score": "1-1", "prob": 0.35}, {"score": "2-2", "prob": 0.25}, {"score": "0-0", "prob": 0.20}]
        else:
            return "1-2", [{"score": "1-2", "prob": 0.33}, {"score": "0-2", "prob": 0.25}, {"score": "1-3", "prob": 0.22}]
    
    def predict(self, match_data: Dict, ms_pred: str = "1") -> Tuple[str, List[Dict]]:
        """
        Skor tahmini yap
        
        Args:
            match_data: {"home_team","away_team","league","odds","date"}
            ms_pred: Maç sonucu tahmini ("1", "X", "2")
        
        Returns:
            (predicted_score, top3_scores)
        """
        # Model yüklü değilse akıllı fallback
        if self.model is None or self.space is None:
            logger.warning("⚠️ Skor modeli yüklü değil, akıllı fallback kullanılıyor")
            return self._intelligent_fallback(match_data, ms_pred)
        
        try:
            # Feature çıkar
            feats = self.engineer.extract_features(match_data)
            if feats is None:
                return self._intelligent_fallback(match_data, ms_pred)
            
            feats = feats.reshape(1, -1)
            probs = self.model.predict_proba(feats)[0]
            
            # MS tutarlılık düzeltmesi
            adj = probs.copy()
            for i, lab in enumerate(self.space):
                if lab != "OTHER" and not self._ms_ok(ms_pred, lab):
                    adj[i] *= 0.25  # Tutarsız skorları daha fazla cezalandır
            
            # Normalize
            s = adj.sum()
            if s > 0:
                adj = adj / s
            
            # En yüksek olasılıklı skor
            top_idx = int(np.argmax(adj))
            top_label = self.space[top_idx]
            
            # Top 3
            top3_idx = np.argsort(adj)[-3:][::-1]
            top3 = [
                {
                    "score": self.space[i],
                    "prob": float(adj[i])
                }
                for i in top3_idx
                if self.space[i] != "OTHER"  # OTHER'ı gösterme
            ]
            
            # OTHER ise veya olasılık çok düşükse fallback
            if top_label == "OTHER" or adj[top_idx] < 0.10:
                logger.info(f"Model belirsiz (prob={adj[top_idx]:.3f}), fallback kullanılıyor")
                return self._intelligent_fallback(match_data, ms_pred)
            
            return top_label, top3
            
        except Exception as e:
            logger.error(f"❌ Skor tahmin hatası: {e}")
            return self._intelligent_fallback(match_data, ms_pred)


# Test
if __name__ == "__main__":
    predictor = ScorePredictor()
    
    test_matches = [
        {
            "home_team": "Fransa",
            "away_team": "Azerbaycan",
            "league": "UEFA Qualifiers",
            "odds": {"1": 1.05, "X": 10.0, "2": 25.0},  # Çok büyük fark
            "date": "2025-10-09"
        },
        {
            "home_team": "Barcelona",
            "away_team": "Real Madrid",
            "league": "La Liga",
            "odds": {"1": 2.10, "X": 3.40, "2": 3.20},  # Dengeli
            "date": "2025-10-09"
        },
        {
            "home_team": "Manchester City",
            "away_team": "Liverpool",
            "league": "Premier League",
            "odds": {"1": 1.85, "X": 3.60, "2": 4.20},
            "date": "2025-10-09"
        }
    ]
    
    for match in test_matches:
        print(f"\n{'='*60}")
        print(f"{match['home_team']} vs {match['away_team']}")
        print(f"Odds: 1={match['odds']['1']}, X={match['odds']['X']}, 2={match['odds']['2']}")
        
        for ms in ["1", "X", "2"]:
            score, top3 = predictor.predict(match, ms_pred=ms)
            print(f"\nMS Tahmini: {ms}")
            print(f"  → Skor: {score}")
            print(f"  → Alternatifler:")
            for i, s in enumerate(top3, 1):
                print(f"      {i}. {s['score']}: {s['prob']*100:.1f}%")
