# score_predictor.py
# Skor modelini bağımsız çağırmak için küçük yardımcı
import os
import pickle
import numpy as np
from typing import Dict, Tuple, List
from advanced_feature_engineer import AdvancedFeatureEngineer

class ScorePredictor:
    def __init__(self, models_dir: str = "data/ai_models_v2"):
        self.models_dir = models_dir
        self.engineer = AdvancedFeatureEngineer()
        self.model, self.space = self._load()

    def _load(self):
        path = os.path.join(self.models_dir, "score_model.pkl")
        if not os.path.exists(path):
            return None, None
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["model"], data["score_space"]

    @staticmethod
    def _ms_ok(ms: str, score_label: str) -> bool:
        try:
            a, b = score_label.split("-"); a, b = int(a), int(b)
        except: return True
        return (a>b) if ms=="1" else ((a==b) if ms=="X" else (a<b))

    def predict(self, match_data: Dict, ms_pred: str = "1") -> Tuple[str, List[Dict]]:
        """
        match_data: {"home_team","away_team","league","odds","date"}
        """
        if self.model is None or self.space is None:
            default = "2-1" if ms_pred=="1" else ("1-1" if ms_pred=="X" else "1-2")
            return default, []

        feats = self.engineer.extract_features(match_data).reshape(1, -1)
        probs = self.model.predict_proba(feats)[0]

        # MS tutarlılık düzeltmesi
        adj = probs.copy()
        for i, lab in enumerate(self.space):
            if lab != "OTHER" and not self._ms_ok(ms_pred, lab):
                adj[i] *= 0.35
        s = adj.sum()
        if s > 0: adj = adj / s

        top_idx = int(np.argmax(adj))
        top_label = self.space[top_idx]
        top3_idx = np.argsort(adj)[-3:][::-1]
        top3 = [{"score": self.space[i], "prob": float(adj[i])} for i in top3_idx]

        if top_label == "OTHER":
            top_label = "2-1" if ms_pred=="1" else ("1-1" if ms_pred=="X" else "1-2")
        return top_label, top3
