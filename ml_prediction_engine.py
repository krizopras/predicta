#!/usr/bin/env python3
"""
ML Prediction Engine with Sklearn Compatibility Fix
FULLY FIXED VERSION - Indentation + Feature Engineer + Model Loader
"""

import os
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("MLEngine")

# =====================================================
# 🔹 Sklearn Compatibility Loader
# =====================================================
class SklearnCompatLoader:
    """Sklearn version uyumsuzluklarını handle eder"""
    @staticmethod
    def safe_load_pickle(file_path: Path):
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except (ValueError, AttributeError) as e:
            if "node array" in str(e) or "dtype" in str(e):
                logger.warning(f"Sklearn uyumsuzluğu tespit edildi: {e}")
                logger.warning("Model dosyası eski sklearn versiyonuyla kaydedilmiş.")
                logger.warning("Çözüm: Modelleri yeniden eğitin (POST /api/training/start)")
                return None
            raise

# =====================================================
# 🔹 Ana Prediction Engine
# =====================================================
class MLPredictionEngine:
    """Ana Prediction Engine (v4.x destekli)"""

    def __init__(self, model_path: str = "data/ai_models_v3"):
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)

        self.models = {"xgboost": None, "gradient_boost": None, "random_forest": None}
        self.scaler = StandardScaler()
        self.score_predictor = None
        self.is_trained = False

        # Feature engineer yükle
        self.feature_engineer = None
        self._load_feature_engineer()

        # Modelleri yükle
        self.load_models()

    # =====================================================
    # 🔹 Feature Engineer Yükleme
    # =====================================================
    def _load_feature_engineer(self):
        """Feature Engineer'ı güvenli şekilde yükler"""
        logger.info("🔄 Feature Engineer yükleniyor...")
        try:
            from enhanced_feature_engineer_v4 import EnhancedFeatureEngineer
            self.feature_engineer = EnhancedFeatureEngineer(model_path=str(self.model_path))
            logger.info("✅ Feature Engineer v4.0 loaded (100+ features)")
        except ImportError:
            try:
                from enhanced_feature_engineer import EnhancedFeatureEngineer
                self.feature_engineer = EnhancedFeatureEngineer(model_path=str(self.model_path))
                logger.warning("⚠️ Feature Engineer v3.5 loaded (fallback)")
            except ImportError:
                logger.error("❌ Feature Engineer yüklenemedi, dummy mode aktif!")
                self.feature_engineer = self._dummy_feature_engineer()

    # =====================================================
    # 🔹 Dummy Feature Engineer
    # =====================================================
    def _dummy_feature_engineer(self):
        class DummyFeatureEngineer:
            def extract_features(self, match_data: Dict) -> np.ndarray:
                try:
                    odds = match_data.get("odds", {"1": 2.0, "X": 3.0, "2": 3.5})
                    o1, ox, o2 = map(float, [odds.get("1", 2.0), odds.get("X", 3.0), odds.get("2", 3.5)])
                    total = 1 / o1 + 1 / ox + 1 / o2
                    return np.array([o1, ox, o2, (1/o1)/total, (1/ox)/total, (1/o2)/total], dtype=np.float32)
                except Exception:
                    return np.array([2.0, 3.0, 3.5, 0.33, 0.33, 0.33], dtype=np.float32)

            def update_match_result(self, match_data: Dict, actual_result: str):
                pass

        return DummyFeatureEngineer()

    # =====================================================
    # 🔹 Model Yükleme (FIXED)
    # =====================================================
    def load_models(self) -> bool:
        """Load models with compatibility handling (supports pkl, json, model)"""
        logger.info("📄 Loading ML models...")

        try:
            import xgboost as xgb
            ms_path = self.model_path / "ensemble_models.pkl"

            if not ms_path.exists():
                logger.warning(f"⚠️ Model file not found: {ms_path}")
                logger.warning("💡 Please train models first: POST /api/training/start")
                return False

            logger.info(f"📂 Loading from: {ms_path}")

            ms_data = None
            try:
                ms_data = SklearnCompatLoader.safe_load_pickle(ms_path)
            except Exception as e:
                logger.warning(f"⚠️ Pickle load failed ({e}), trying XGBoost loader...")

            if ms_data is None:
                try:
                    model = xgb.XGBClassifier()
                    model.load_model(str(ms_path))
                    ms_data = {"models": {"xgb": model}, "is_trained": True}
                    logger.info("✅ XGBoost model loaded (JSON/MODEL compatible format)")
                except Exception as e:
                    logger.error(f"❌ Model load failed: {e}")
                    logger.error("🔧 Retrain with current sklearn/XGBoost version")
                    return False

            if isinstance(ms_data, dict):
                self.models = ms_data.get("models", {})
                self.scaler = ms_data.get("scaler", StandardScaler())
                self.is_trained = ms_data.get("is_trained", False)
                loaded_models = [n for n, m in self.models.items() if m is not None]
                logger.info(f"✅ Loaded MS models: {', '.join(loaded_models)}")

            # Score model
            score_path = self.model_path / "score_model.pkl"
            if score_path.exists():
                try:
                    score_data = SklearnCompatLoader.safe_load_pickle(score_path)
                    if score_data:
                        from score_predictor import ScorePredictor
                        self.score_predictor = ScorePredictor()
                        self.score_predictor.model = score_data.get("model")
                        self.score_predictor.space = score_data.get("score_space", [])
                        logger.info("✅ Score model loaded")
                except Exception as e:
                    logger.warning(f"⚠️ Score model yükleme hatası: {e}")

            # Scaler
            scaler_path = self.model_path / "scaler.pkl"
            if scaler_path.exists():
                scaler_data = SklearnCompatLoader.safe_load_pickle(scaler_path)
                if scaler_data:
                    self.scaler = scaler_data
                    logger.info("✅ Scaler loaded")

            self.is_trained = len([m for m in self.models.values() if m is not None]) > 0
            return self.is_trained

        except Exception as e:
            logger.error(f"❌ Model loading error: {e}", exc_info=True)
            return False

    # =====================================================
    # 🔹 Tahmin
    # =====================================================
    def predict_match(self, home_team: str, away_team: str, odds: Dict[str, float], league: str = "Unknown") -> Dict[str, Any]:
        """Maç sonucu tahmini"""
        if self.feature_engineer is None:
            return self._error_response("Feature Engineer not loaded")

        if not self.is_trained:
            return self._error_response("Models not trained")

        try:
            match_data = {"home_team": home_team, "away_team": away_team, "odds": odds, "league": league}
            features = self.feature_engineer.extract_features(match_data)
            if features is None:
                raise ValueError("Feature extraction failed")

            features = np.nan_to_num(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features) if hasattr(self.scaler, "mean_") else features

            preds, confs = [], []
            for name, model in self.models.items():
                if model:
                    proba = model.predict_proba(features_scaled)[0]
                    preds.append(proba)
                    confs.append(max(proba))

            if not preds:
                raise ValueError("No models available")

            avg = np.mean(preds, axis=0)
            res_map = {0: "1", 1: "X", 2: "2"}
            result = res_map[int(np.argmax(avg))]
            conf = float(np.max(avg) * 100)

            probs = {"1": float(avg[0] * 100), "X": float(avg[1] * 100), "2": float(avg[2] * 100)}
            value_bet = self._calculate_value_bet(probs, odds, conf)

            return {"prediction": result, "confidence": round(conf, 2), "probabilities": probs, "value_bet": value_bet}
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return self._error_response(str(e))

    def _error_response(self, msg: str):
        return {"error": msg, "prediction": "X", "confidence": 0, "probabilities": {"1": 33, "X": 33, "2": 33}}

    def _calculate_value_bet(self, probs: Dict[str, float], odds: Dict[str, float], conf: float):
        try:
            values = {}
            for o in ["1", "X", "2"]:
                p, odd = probs[o] / 100, float(odds[o])
                values[o] = (p * odd) - 1 if odd > 1 else 0
            best = max(values, key=values.get)
            return {"best_outcome": best, "value_index": round(values[best], 3), "confidence": round(conf, 2)}
        except Exception as e:
            return {"value_index": 0, "error": str(e)}

# =====================================================
# 🔹 Local Test
# =====================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = MLPredictionEngine()
    print(engine.get_system_info() if hasattr(engine, "get_system_info") else "✅ Engine loaded successfully.")
