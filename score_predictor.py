# score_predictor.py
# Skor modelini bağımsız çağırmak için küçük yardımcı (FIXED)
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

    def predict(self, match_data: Dict, ms_pred: str = "1") -> Tuple[str, List[Dict]]:
        """
        Skor tahmini yap
        
        Args:
            match_data: {"home_team","away_team","league","odds","date"}
            ms_pred: Maç sonucu tahmini ("1", "X", "2")
        
        Returns:
            (predicted_score, top3_scores)
        """
        # Model yüklü değilse fallback
        if self.model is None or self.space is None:
            logger.warning("⚠️ Skor modeli yüklü değil, fallback kullanılıyor")
            default = "2-1" if ms_pred == "1" else ("1-1" if ms_pred == "X" else "1-2")
            return default, []

        try:
            # Feature çıkar
            feats = self.engineer.extract_features(match_data).reshape(1, -1)
            probs = self.model.predict_proba(feats)[0]

            # MS tutarlılık düzeltmesi
            adj = probs.copy()
            for i, lab in enumerate(self.space):
                if lab != "OTHER" and not self._ms_ok(ms_pred, lab):
                    adj[i] *= 0.35  # Tutarsız skorları cezalandır
            
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
            ]

            # OTHER ise fallback
            if top_label == "OTHER":
                top_label = "2-1" if ms_pred == "1" else ("1-1" if ms_pred == "X" else "1-2")
            
            return top_label, top3
            
        except Exception as e:
            logger.error(f"❌ Skor tahmin hatası: {e}")
            default = "2-1" if ms_pred == "1" else ("1-1" if ms_pred == "X" else "1-2")
            return default, []


# Test
if __name__ == "__main__":
    predictor = ScorePredictor()
    
    test_match = {
        "home_team": "Barcelona",
        "away_team": "Real Madrid",
        "league": "La Liga",
        "odds": {"1": 2.10, "X": 3.40, "2": 3.20},
        "date": "2025-10-09"
    }
    
    score, top3 = predictor.predict(test_match, ms_pred="1")
    print(f"\n✅ Tahmin edilen skor: {score}")
    print(f"✅ Top 3:")
    for i, s in enumerate(top3, 1):
        print(f"   {i}. {s['score']}: {s['prob']*100:.1f}%")
