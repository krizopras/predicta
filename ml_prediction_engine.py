# ml_prediction_engine.py
"""
Predicta Europe ML v2
- Ensemble MS (1/X/2): XGBoost + RandomForest + GradientBoost
- ML Tabanlı Tam Skor Modeli (RandomForest - top-K exact scores + OTHER)
- Value bet & risk değerlendirme
"""
import os
import json
import pickle
import logging
from typing import Dict, Any, List
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False
    logger.warning("ML kütüphaneleri eksik. pip install scikit-learn xgboost")

from advanced_feature_engineer import AdvancedFeatureEngineer, FEATURE_NAMES


class MLPredictionEngine:
    def __init__(self, model_path: str = "data/ai_models_v2"):
        self.model_path = model_path
        os.makedirs(self.model_path, exist_ok=True)

        self.feature_engineer = AdvancedFeatureEngineer()

        # Ensemble modelleri (MS 1/X/2)
        self.models = {
            'xgboost': None,
            'random_forest': None,
            'gradient_boost': None
        }
        self.weights = {'xgboost': 0.45, 'random_forest': 0.30, 'gradient_boost': 0.25}
        self.scaler = StandardScaler()
        self.is_trained = False

        # Skor modeli
        self.score_model = None
        self.score_space = None  # ["1-0","2-1",...,"OTHER"]

        self._load_models()

    # ----- Model yükleme/kaydetme ------------------------------------------
    def _initialize_models(self):
        if not ML_AVAILABLE:
            logger.error("ML kütüphaneleri yüklenemedi.")
            return False
        try:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                objective='multi:softprob', num_class=3, random_state=42
            )
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=150, max_depth=12, min_samples_split=5, min_samples_leaf=2, random_state=42
            )
            self.models['gradient_boost'] = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            )
            logger.info("✅ MS modelleri oluşturuldu")
            return True
        except Exception as e:
            logger.error(f"Model oluşturma hatası: {e}")
            return False

    def _load_models(self):
        # MS modelleri
        ens_path = os.path.join(self.model_path, "ensemble_models.pkl")
        if os.path.exists(ens_path):
            try:
                with open(ens_path, "rb") as f:
                    data = pickle.load(f)
                self.models = data['models']
                self.scaler = data['scaler']
                self.is_trained = data['is_trained']
                logger.info("✅ MS ensemble modelleri yüklendi")
            except Exception as e:
                logger.warning(f"MS modelleri yüklenemedi: {e}")
                self._initialize_models()
        else:
            self._initialize_models()

        # Skor modeli
        score_path = os.path.join(self.model_path, "score_model.pkl")
        if os.path.exists(score_path):
            try:
                with open(score_path, "rb") as f:
                    sdata = pickle.load(f)
                self.score_model = sdata["model"]
                self.score_space = sdata["score_space"]
                logger.info(f"✅ Skor modeli yüklendi ({len(self.score_space)} sınıf)")
            except Exception as e:
                logger.warning(f"Skor modeli yüklenemedi: {e}")
                self.score_model, self.score_space = None, None

    def _save_models(self):
        try:
            ens_path = os.path.join(self.model_path, "ensemble_models.pkl")
            with open(ens_path, "wb") as f:
                pickle.dump({
                    "models": self.models,
                    "scaler": self.scaler,
                    "is_trained": self.is_trained
                }, f)
            logger.info("💾 MS modelleri kaydedildi")
            return True
        except Exception as e:
            logger.error(f"Model kaydetme hatası: {e}")
            return False

    # ----- Eğitim -----------------------------------------------------------
    def train(self, historical_matches: List[Dict]) -> Dict[str, Any]:
        """ MS modellerini eğitir (skor modeli, model_trainer.py tarafından eğitilir) """
        if not ML_AVAILABLE:
            return {"success": False, "error": "ML kütüphaneleri yüklü değil"}

        if len(historical_matches) < 100:
            return {"success": False, "error": f"Yetersiz veri: {len(historical_matches)}"}

        try:
            X, y = [], []

            for m in historical_matches:
                feats = self.feature_engineer.extract_features({
                    "home_team": m["home_team"],
                    "away_team": m["away_team"],
                    "league": m.get("league", "Unknown"),
                    "odds": m.get("odds", {"1": 2.0, "X": 3.0, "2": 3.5}),
                    "date": m.get("date", datetime.now().isoformat())
                })
                X.append(feats)
                y.append({"1": 0, "X": 1, "2": 2}[m.get("result", "X")])

                # geçmişi güncelle
                self._update_feature_history(m)

            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.int64)

            X_scaled = self.scaler.fit_transform(X)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )

            accuracies = {}
            for name, model in self.models.items():
                if model is None:
                    continue
                model.fit(X_tr, y_tr)
                acc = model.score(X_te, y_te)
                accuracies[name] = acc
                logger.info(f"✅ {name}: %{acc*100:.2f}")

            self.is_trained = True
            self._save_models()

            ensemble_acc = self._ensemble_accuracy(X_te, y_te)

            return {
                "success": True,
                "training_samples": len(X_tr),
                "test_samples": len(X_te),
                "accuracies": accuracies,
                "ensemble_accuracy": ensemble_acc,
                "features_used": len(FEATURE_NAMES)
            }

        except Exception as e:
            logger.error(f"Eğitim hatası: {e}")
            return {"success": False, "error": str(e)}

    def _update_feature_history(self, m: Dict):
        home_res = 'W' if m['result'] == '1' else ('D' if m['result'] == 'X' else 'L')
        away_res = 'L' if m['result'] == '1' else ('D' if m['result'] == 'X' else 'W')
        self.feature_engineer.update_team_history(m['home_team'], {
            "result": home_res, "goals_for": m.get("home_goals", 0),
            "goals_against": m.get("away_goals", 0), "date": m.get("date", ""), "venue": "home"
        })
        self.feature_engineer.update_team_history(m['away_team'], {
            "result": away_res, "goals_for": m.get("away_goals", 0),
            "goals_against": m.get("home_goals", 0), "date": m.get("date", ""), "venue": "away"
        })
        self.feature_engineer.update_h2h_history(m['home_team'], m['away_team'], {
            "result": m["result"], "home_goals": m.get("home_goals", 0), "away_goals": m.get("away_goals", 0)
        })
        self.feature_engineer.update_league_results(m.get("league", "Unknown"), m["result"])

    # ----- Tahmin -----------------------------------------------------------
    def predict_match(self, home_team: str, away_team: str, odds: Dict,
                      league: str = "Unknown") -> Dict[str, Any]:
        """
        1) Ensemble ile MS (1/X/2)
        2) Skor modeli ile "tam skor" (MS ile tutarlı sınıfları önceliklendirerek)
        """
        match_data = {
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
            "odds": odds,
            "date": datetime.now().isoformat()
        }
        try:
            feats = self.feature_engineer.extract_features(match_data)
        except Exception as e:
            logger.error(f"Feature hatası: {e}")
            return self._basic_prediction(home_team, away_team, odds)

        # MS tahmini
        if not self.is_trained:
            ms_out = self._basic_prediction(home_team, away_team, odds)
        else:
            X = feats.reshape(1, -1)
            Xs = self.scaler.transform(X)
            probs = self._ensemble_predict_proba(Xs)[0]
            idx = int(np.argmax(probs))
            pred = ['1', 'X', '2'][idx]
            conf = float(probs[idx]) * 100.0
            ms_out = {
                "prediction": pred,
                "confidence": round(conf, 1),
                "probabilities": {
                    "home_win": round(probs[0] * 100, 1),
                    "draw": round(probs[1] * 100, 1),
                    "away_win": round(probs[2] * 100, 1)
                }
            }

        # Skor tahmini (varsa)
        score_pred, score_top = self._predict_exact_score(feats, ms_out["prediction"])

        # Value & risk
        value_index = self._calculate_value_bet(
            np.array([ms_out["probabilities"]["home_win"],
                      ms_out["probabilities"]["draw"],
                      ms_out["probabilities"]["away_win"]]) / 100.0,
            odds
        )
        risk = "LOW" if ms_out["confidence"] >= 70 else ("MEDIUM" if ms_out["confidence"] >= 55 else "HIGH")

        return {
            "prediction": ms_out["prediction"],
            "confidence": ms_out["confidence"],
            "probabilities": ms_out["probabilities"],
            "score_prediction": score_pred,
            "score_alternatives": score_top,  # ilk 3 sınıf
            "value_bet": {"value_index": round(value_index, 3), "rating": self._get_value_rating(value_index)},
            "risk_level": risk,
            "recommendation": self._get_recommendation(ms_out["confidence"], value_index),
            "model": "MS: (XGB+RF+GB) + Score: RF(K-class)",
            "features_used": len(FEATURE_NAMES),
            "timestamp": datetime.now().isoformat()
        }

    # ----- Skor modeli tahmini ---------------------------------------------
    def _predict_exact_score(self, features: np.ndarray, ms_pred: str):
        """
        Skor modelinden top olasılıkları al; MS sonucu ile uyuşmayanları düşük ağırlıkla bırak.
        """
        if self.score_model is None or self.score_space is None:
            return "2-1" if ms_pred == "1" else ("1-1" if ms_pred == "X" else "1-2"), []

        X = features.reshape(1, -1)
        # Skor modeli scaler kullanmıyor; doğrudan ham feats ile eğitildi
        try:
            probs = self.score_model.predict_proba(X)[0]  # shape: [K]
        except Exception as e:
            logger.warning(f"Skor modeli tahmin hatası: {e}")
            return "2-1" if ms_pred == "1" else ("1-1" if ms_pred == "X" else "1-2"), []

        # MS tutarlılık maskesi
        masks = {
            "1": lambda s: self._parse_score(s)[0] > self._parse_score(s)[1],
            "X": lambda s: self._parse_score(s)[0] == self._parse_score(s)[1],
            "2": lambda s: self._parse_score(s)[0] < self._parse_score(s)[1],
        }

        adjusted = probs.copy()
        for i, lab in enumerate(self.score_space):
            if lab == "OTHER":
                continue
            ok = masks[ms_pred](lab)
            if not ok:
                adjusted[i] *= 0.35  # MS'e aykırıysa ceza

        # normalize
        s = adjusted.sum()
        if s > 0:
            adjusted = adjusted / s

        # En iyi sınıf ve ilk 3
        top_idx = int(np.argmax(adjusted))
        top_label = self.score_space[top_idx]
        top3_idx = np.argsort(adjusted)[-3:][::-1]
        top3 = [{"score": self.score_space[i], "prob": float(adjusted[i])} for i in top3_idx]

        if top_label == "OTHER":
            # MS'e uygun default skor ver
            return ("2-1" if ms_pred == "1" else ("1-1" if ms_pred == "X" else "1-2")), top3
        return top_label, top3

    @staticmethod
    def _parse_score(label: str):
        try:
            a, b = label.split("-")
            return int(a), int(b)
        except:
            return 0, 0

    # ----- Ensemble yardımcıları -------------------------------------------
    def _ensemble_predict_proba(self, X) -> np.ndarray:
        if not self.is_trained:
            return np.ones((len(X), 3), dtype=np.float32) / 3.0
        out = np.zeros((len(X), 3), dtype=np.float32)
        for name, model in self.models.items():
            if model is None:
                continue
            w = self.weights.get(name, 0.33)
            p = model.predict_proba(X)
            out += w * p
        # normalize
        out /= out.sum(axis=1, keepdims=True)
        return out

    def _ensemble_accuracy(self, X_te, y_te) -> float:
        probs = self._ensemble_predict_proba(X_te)
        y_pred = np.argmax(probs, axis=1)
        return float(np.mean(y_pred == y_te))

    # ----- Değer & öneri ----------------------------------------------------
    def _calculate_value_bet(self, probs: np.ndarray, odds: Dict) -> float:
        try:
            odds_vals = [float(odds.get('1', 2.0)), float(odds.get('X', 3.0)), float(odds.get('2', 3.5))]
            ev = probs * np.array(odds_vals, dtype=float)
            return float(np.max(ev) - 1.0)
        except Exception:
            return 0.0

    @staticmethod
    def _get_value_rating(value_index: float) -> str:
        return "EXCELLENT" if value_index > 0.15 else \
               "VERY GOOD" if value_index > 0.10 else \
               "GOOD" if value_index > 0.05 else \
               "FAIR" if value_index > 0.02 else "POOR"

    @staticmethod
    def _get_recommendation(confidence: float, value_index: float) -> str:
        if confidence >= 70 and value_index > 0.10:
            return "🔥 STRONG BET"
        elif confidence >= 65 and value_index > 0.05:
            return "✅ RECOMMENDED"
        elif confidence >= 55:
            return "⚠️ CONSIDER"
        else:
            return "❌ SKIP"

    # ----- Fallback (model yoksa) ------------------------------------------
    def _basic_prediction(self, home_team: str, away_team: str, odds: Dict) -> Dict:
        try:
            o1, ox, o2 = float(odds.get('1', 2.0)), float(odds.get('X', 3.0)), float(odds.get('2', 3.5))
            imp = np.array([1/o1, 1/ox, 1/o2], dtype=float)
            imp = imp / imp.sum()
            idx = int(np.argmax(imp))
            pred = ['1','X','2'][idx]
            conf = float(imp[idx] * 100.0)
            return {
                "prediction": pred,
                "confidence": round(conf, 1),
                "probabilities": {
                    "home_win": round(float(imp[0]*100), 1),
                    "draw": round(float(imp[1]*100), 1),
                    "away_win": round(float(imp[2]*100), 1)
                }
            }
        except Exception:
            return {
                "prediction": "1", "confidence": 60.0,
                "probabilities": {"home_win": 60.0, "draw": 20.0, "away_win": 20.0}
            }
