#!/usr/bin/env python3
"""
ML Prediction Engine — FULL TRAIN + PREDICT pipeline
Optimized for: speed, compatibility (sklearn 1.x/2.x & XGBoost 2.x), and Railway deploys.

Includes:
- SklearnCompatLoader (safe pickle loader)
- FeatureEngineer v4 integration (101 features)
- ModelTrainer (XGB/RF/GB/LogReg) with RandomizedSearchCV + early stopping
- AdvancedMLPredictor (weighted soft-voting ensemble)
- Safe model save/load (pickle + XGBoost JSON fallback)
- Predict API-ready class (MLPredictionEngine)

Tested with Python 3.11+.
"""
from __future__ import annotations

import os
import json
import pickle
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

# Optional imports guarded for environments without these libs
try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

logger = logging.getLogger("ml_prediction_engine")
logging.basicConfig(level=logging.INFO, format="[ML] %(levelname)s: %(message)s")

# =============================================================
# Utilities & Compatibility
# =============================================================
class SklearnCompatLoader:
    """Safely load pickle files across sklearn versions.
    Returns None on known incompatibilities so callers can fallback.
    """
    @staticmethod
    def safe_load_pickle(file_path: Path) -> Optional[Any]:
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except (ValueError, AttributeError, pickle.UnpicklingError) as e:
            # Common when loading models trained with different sklearn
            logger.warning(f"Sklearn/XGB pickle incompatibility: {e}")
            return None
        except Exception as e:
            logger.error(f"Pickle load error: {e}")
            return None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# =============================================================
# Feature Engineering (v4)
# =============================================================
class _DummyFeatureEngineer:
    """Fallback FE with minimal odds-derived features (6 dims)."""
    def extract_features(self, match: Dict[str, Any]) -> NDArray[np.float32]:
        odds = match.get("odds", {"1": 2.0, "X": 3.0, "2": 3.5})
        o1 = float(odds.get("1", 2.0)); ox = float(odds.get("X", 3.0)); o2 = float(odds.get("2", 3.5))
        total = (1/o1) + (1/ox) + (1/o2)
        p1 = (1/o1)/total; px = (1/ox)/total; p2 = (1/o2)/total
        return np.array([o1, ox, o2, p1, px, p2], dtype=np.float32)


def load_feature_engineer(models_dir: Path):
    try:
        from enhanced_feature_engineer_v4 import EnhancedFeatureEngineer
        return EnhancedFeatureEngineer(model_path=str(models_dir))
    except Exception as e:
        logger.warning(f"FeatureEngineer v4 not available ({e}); using Dummy FE")
        return _DummyFeatureEngineer()


# =============================================================
# Advanced Ensemble Predictor
# =============================================================
@dataclass
class AdvancedMLPredictor:
    models: Dict[str, Any] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        parts: List[NDArray[np.float64]] = []
        wsum = 0.0
        for name, model in self.models.items():
            if model is None:
                continue
            try:
                proba = model.predict_proba(X)
                weight = float(self.weights.get(name, 1.0))
                parts.append(proba * weight)
                wsum += weight
            except Exception as e:
                logger.warning(f"Model {name} predict_proba failed: {e}")
        if not parts:
            raise RuntimeError("No valid models in ensemble")
        summed = np.sum(parts, axis=0)
        if wsum > 0:
            summed = summed / wsum
        # Ensure numerical stability
        summed = np.clip(summed, 1e-9, 1.0)
        summed = summed / summed.sum(axis=1, keepdims=True)
        return summed


# =============================================================
# Model Trainer (train + search + save)
# =============================================================
@dataclass
class TrainingConfig:
    random_state: int = 42
    cv_folds: int = 5
    n_iter_search: int = 12
    verbose: bool = False
    use_gpu: bool = False


class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.cfg = config

    # ---------------------------- XGBoost ----------------------------
    def _make_xgb(self) -> Any:
        if xgb is None:
            raise ImportError("xgboost not installed")
        tree_method = "gpu_hist" if self.cfg.use_gpu else "hist"
        return xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            random_state=self.cfg.random_state,
            n_jobs=-1,
            tree_method=tree_method,
            learning_rate=0.08,
            n_estimators=300,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.2,
            min_child_weight=3,
            reg_lambda=1.0,
            reg_alpha=0.2,
            use_label_encoder=False,
            verbosity=0,
        )

    def _xgb_param_dist(self) -> Dict[str, List[Any]]:
        return {
            "max_depth": [4, 5, 6],
            "learning_rate": [0.05, 0.08, 0.1],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "n_estimators": [150, 200, 250, 300],
            "gamma": [0.0, 0.1, 0.2, 0.3],
            "min_child_weight": [1, 2, 3],
        }

    def train_xgb(self, X: NDArray, y: NDArray, X_val: NDArray, y_val: NDArray) -> Tuple[Any, float]:
        base = self._make_xgb()
        cv = StratifiedKFold(n_splits=self.cfg.cv_folds, shuffle=True, random_state=self.cfg.random_state)
        search = RandomizedSearchCV(
            base,
            self._xgb_param_dist(),
            n_iter=self.cfg.n_iter_search,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            random_state=self.cfg.random_state,
            verbose=1 if self.cfg.verbose else 0,
        )
        # Early stopping via fit params
        search.fit(
            X, y,
            **{"eval_set": [(X_val, y_val)], "early_stopping_rounds": 25, "verbose": False}
        )
        model: Any = search.best_estimator_
        pred = model.predict(X_val)
        f1 = f1_score(y_val, pred, average="macro")
        bal_acc = balanced_accuracy_score(y_val, pred)
        acc = accuracy_score(y_val, pred)
        logger.info(f"✅ XGBoost: Acc={acc:.4f}, BalAcc={bal_acc:.4f}, F1={f1:.4f}")
        logger.info(f"   Best params: {search.best_params_}")
        return model, float(f1)

    # ------------------------ Random Forest -------------------------
    def train_rf(self, X: NDArray, y: NDArray, X_val: NDArray, y_val: NDArray) -> Tuple[Any, float]:
        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=self.cfg.random_state,
        )
        rf.fit(X, y)
        pred = rf.predict(X_val)
        f1 = f1_score(y_val, pred, average="macro")
        bal_acc = balanced_accuracy_score(y_val, pred)
        acc = accuracy_score(y_val, pred)
        logger.info(f"✅ RF: Acc={acc:.4f}, BalAcc={bal_acc:.4f}, F1={f1:.4f}")
        return rf, float(f1)

    # --------------------- Gradient Boosting ------------------------
    def train_gb(self, X: NDArray, y: NDArray, X_val: NDArray, y_val: NDArray) -> Tuple[Any, float]:
        gb = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.08, max_depth=3, random_state=self.cfg.random_state
        )
        gb.fit(X, y)
        pred = gb.predict(X_val)
        f1 = f1_score(y_val, pred, average="macro")
        bal_acc = balanced_accuracy_score(y_val, pred)
        acc = accuracy_score(y_val, pred)
        logger.info(f"✅ GB: Acc={acc:.4f}, BalAcc={bal_acc:.4f}, F1={f1:.4f}")
        return gb, float(f1)

    # ---------------------- Logistic Regression ---------------------
    def train_lr(self, X: NDArray, y: NDArray, X_val: NDArray, y_val: NDArray) -> Tuple[Any, float]:
        lr = LogisticRegression(max_iter=500, n_jobs=-1, multi_class="multinomial", C=1.5, solver="saga")
        lr.fit(X, y)
        pred = lr.predict(X_val)
        f1 = f1_score(y_val, pred, average="macro")
        bal_acc = balanced_accuracy_score(y_val, pred)
        acc = accuracy_score(y_val, pred)
        logger.info(f"✅ LR: Acc={acc:.4f}, BalAcc={bal_acc:.4f}, F1={f1:.4f}")
        return lr, float(f1)

    # -------------------------- Train All ---------------------------
    def train_all(self, X: NDArray, y: NDArray, valid_size: float = 0.2) -> Dict[str, Any]:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=valid_size, random_state=self.cfg.random_state, stratify=y
        )

        scaler = StandardScaler()
        X_trs = scaler.fit_transform(X_tr)
        X_vals = scaler.transform(X_val)

        results: Dict[str, Tuple[Any, float]] = {}
        # Train models
        try:
            model_xgb, f1_xgb = self.train_xgb(X_trs, y_tr, X_vals, y_val)
            results["xgb"] = (model_xgb, f1_xgb)
        except Exception as e:
            logger.warning(f"XGB training skipped: {e}")

        try:
            model_rf, f1_rf = self.train_rf(X_trs, y_tr, X_vals, y_val)
            results["rf"] = (model_rf, f1_rf)
        except Exception as e:
            logger.warning(f"RF training skipped: {e}")

        try:
            model_gb, f1_gb = self.train_gb(X_trs, y_tr, X_vals, y_val)
            results["gb"] = (model_gb, f1_gb)
        except Exception as e:
            logger.warning(f"GB training skipped: {e}")

        try:
            model_lr, f1_lr = self.train_lr(X_trs, y_tr, X_vals, y_val)
            results["lr"] = (model_lr, f1_lr)
        except Exception as e:
            logger.warning(f"LR training skipped: {e}")

        if not results:
            raise RuntimeError("No models were trained")

        # Build ensemble weights proportional to F1
        total_f1 = sum(v[1] for v in results.values())
        weights = {name: (f1/total_f1 if total_f1 > 0 else 1.0/len(results)) for name, (_, f1) in results.items()}
        models = {name: m for name, (m, _) in results.items()}
        ensemble = AdvancedMLPredictor(models=models, weights=weights)

        # Evaluate ensemble on validation
        proba = ensemble.predict_proba(X_vals)
        pred = np.argmax(proba, axis=1)
        f1_ens = f1_score(y_val, pred, average="macro")
        logger.info(f"🌟 ENSEMBLE: F1={f1_ens:.4f} (weights={weights})")

        return {
            "scaler": scaler,
            "models": models,
            "weights": weights,
            "f1_ensemble": float(f1_ens),
        }


# =============================================================
# Save / Load models (safe for Railway)
# =============================================================

class ModelIO:
    ENSEMBLE_PKL = "ensemble_models.pkl"
    XGB_JSON = "xgb_model.json"           # ✅ XGB JSON format
    XGB_CONFIG = "xgb_model.config"       # ✅ XGB config backup
    SCALER_PKL = "scaler.pkl"
    META_JSON = "training_metadata.json"

    @staticmethod
    def save(models_dir: Path, bundle: Dict[str, Any]) -> None:
        ensure_dir(models_dir)
        models: Dict[str, Any] = bundle.get("models", {})
        scaler: StandardScaler = bundle.get("scaler")
        weights: Dict[str, float] = bundle.get("weights", {})
        f1 = float(bundle.get("f1_ensemble", 0.0))

        # ✅ Save XGB as JSON (cross-version compatible)
        if xgb is not None and "xgb" in models and isinstance(models["xgb"], xgb.XGBClassifier):
            try:
                xgb_model = models["xgb"]
                
                # Method 1: Save booster as JSON
                xgb_model.save_model(str(models_dir / ModelIO.XGB_JSON))
                logger.info("💾 XGB model saved as JSON (version-safe)")
                
                # Method 2: Save config separately
                config = {
                    "n_estimators": xgb_model.n_estimators,
                    "max_depth": xgb_model.max_depth,
                    "learning_rate": xgb_model.learning_rate,
                    "objective": xgb_model.objective,
                    "num_class": 3,
                    "_Booster": str(models_dir / ModelIO.XGB_JSON)
                }
                with open(models_dir / ModelIO.XGB_CONFIG, 'w') as f:
                    json.dump(config, f, indent=2)
                
            except Exception as e:
                logger.warning(f"XGB JSON save failed: {e}")

        # ✅ Save non-XGB models via pickle (safe)
        safe_models = {k: v for k, v in models.items() if k != "xgb"}
        bundle_pickle = {
            "models": safe_models,
            "weights": weights,
            "scaler": scaler,
            "is_trained": True,
            "meta": {"f1_ensemble": f1},
        }
        
        with open(models_dir / ModelIO.ENSEMBLE_PKL, "wb") as f:
            pickle.dump(bundle_pickle, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("💾 Ensemble bundle saved (pickle)")

        # Save scaler separately (redundancy)
        if scaler is not None:
            with open(models_dir / ModelIO.SCALER_PKL, "wb") as f:
                pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save metadata
        meta_path = models_dir / ModelIO.META_JSON
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "f1_ensemble": f1,
                    "models": list(models.keys()),
                    "xgb_version": xgb.__version__ if xgb else None,
                    "sklearn_version": __import__('sklearn').__version__,
                    "saved_at": datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Metadata save failed: {e}")

    @staticmethod
    def load(models_dir: Path) -> Dict[str, Any]:
        bundle_path = models_dir / ModelIO.ENSEMBLE_PKL
        
        if not bundle_path.exists():
            raise FileNotFoundError(f"Model bundle not found: {bundle_path}")

        # 1) Load pickle bundle (non-XGB models)
        bundle = SklearnCompatLoader.safe_load_pickle(bundle_path)
        
        if bundle is None:
            # ✅ Fallback: Try to load scaler separately
            logger.warning("⚠️ Main bundle failed, trying scaler fallback...")
            scaler_path = models_dir / ModelIO.SCALER_PKL
            
            if scaler_path.exists():
                scaler = SklearnCompatLoader.safe_load_pickle(scaler_path)
                bundle = {
                    "models": {},
                    "scaler": scaler,
                    "weights": {},
                    "is_trained": False
                }
                logger.info("✅ Scaler loaded from fallback")
            else:
                raise RuntimeError("Incompatible pickle bundle; retrain required")

        models: Dict[str, Any] = bundle.get("models", {})
        scaler = bundle.get("scaler")
        weights = bundle.get("weights", {})

        # 2) Load XGB from JSON (version-safe)
        if xgb is not None:
            xgb_json_path = models_dir / ModelIO.XGB_JSON
            xgb_config_path = models_dir / ModelIO.XGB_CONFIG
            
            if xgb_json_path.exists():
                try:
                    # ✅ Load XGB from JSON
                    xgb_model = xgb.XGBClassifier(
                        objective="multi:softprob",
                        num_class=3
                    )
                    xgb_model.load_model(str(xgb_json_path))
                    models["xgb"] = xgb_model
                    
                    # ✅ Restore config if available
                    if xgb_config_path.exists():
                        with open(xgb_config_path, 'r') as f:
                            config = json.load(f)
                        logger.info(f"✅ XGB loaded from JSON (v{xgb.__version__})")
                    else:
                        logger.info("✅ XGB loaded from JSON")
                    
                except Exception as e:
                    logger.warning(f"XGB JSON load failed: {e}")
            else:
                logger.warning("⚠️ XGB JSON file not found")

        # 3) Construct ensemble
        if not models:
            logger.warning("⚠️ No models available after load")
            return {
                "models": {},
                "weights": {},
                "scaler": scaler,
                "ensemble": None,
                "is_trained": False
            }
        
        ensemble = AdvancedMLPredictor(models=models, weights=weights)
        
        return {
            "models": models,
            "weights": weights,
            "scaler": scaler,
            "ensemble": ensemble,
            "is_trained": True,
        }


# ============================================
# USAGE EXAMPLE
# ============================================
"""
# Training sonrası kayıt:
bundle = trainer.train_all(X, y)
ModelIO.save(Path("data/ai_models_v3"), bundle)

# Load (cross-version safe):
try:
    loaded = ModelIO.load(Path("data/ai_models_v3"))
    engine.scaler = loaded["scaler"]
    engine.ensemble = loaded["ensemble"]
    engine.is_trained = loaded["is_trained"]
except Exception as e:
    print(f"❌ Model load failed: {e}")
    print("💡 Retrain required")
"""

# =============================================================
# Public Engine (train + predict)
# =============================================================
class MLPredictionEngine:
    def __init__(self, models_dir: str | Path = "data/ai_models_v3", use_gpu: bool = False):
        self.models_dir = Path(models_dir).absolute()
        ensure_dir(self.models_dir)

        self.config = TrainingConfig(use_gpu=use_gpu)
        self.trainer = ModelTrainer(self.config)
        self.feature_engineer = load_feature_engineer(self.models_dir)

        self.scaler: Optional[StandardScaler] = None
        self.ensemble: Optional[AdvancedMLPredictor] = None
        self.is_trained: bool = False

        # Attempt to load existing models
        try:
            loaded = ModelIO.load(self.models_dir)
            self.scaler = loaded.get("scaler")
            self.ensemble = loaded.get("ensemble")
            self.is_trained = bool(loaded.get("is_trained", False))
            logger.info("✅ Models loaded from disk")
        except Exception as e:
            logger.warning(f"Models not loaded at init ({e}) — training required before predict")

    # ---------------------------- TRAIN ----------------------------
    def fit(self, X: NDArray, y: NDArray) -> Dict[str, Any]:
        bundle = self.trainer.train_all(X, y)
        ModelIO.save(self.models_dir, bundle)
        # Reload to ensure compatibility path works
        loaded = ModelIO.load(self.models_dir)
        self.scaler = loaded.get("scaler")
        self.ensemble = loaded.get("ensemble")
        self.is_trained = True
        return {"status": "ok", "f1_ensemble": bundle.get("f1_ensemble", 0.0)}

    # --------------------------- PREDICT ---------------------------
    def predict_from_features(self, feats: NDArray[np.float32]) -> Dict[str, Any]:
        if not self.is_trained or self.ensemble is None:
            return {"error": "Models not trained"}
        try:
            X = np.nan_to_num(feats.astype(np.float32)).reshape(1, -1)
            Xs = self.scaler.transform(X) if self.scaler is not None and hasattr(self.scaler, "mean_") else X
            proba = self.ensemble.predict_proba(Xs)[0]
            idx = int(np.argmax(proba))
            mapping = {0: "1", 1: "X", 2: "2"}
            result = mapping.get(idx, "X")
            probs = {"1": float(proba[0]*100), "X": float(proba[1]*100), "2": float(proba[2]*100)}
            conf = float(np.max(proba) * 100)
            return {"prediction": result, "confidence": round(conf, 2), "probabilities": probs}
        except Exception as e:
            logger.error(f"predict_from_features error: {e}")
            return {"error": str(e)}

    def predict_match(self, home_team: str, away_team: str, league: str, odds: Dict[str, float], date: Optional[str] = None) -> Dict[str, Any]:
        match = {
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
            "odds": odds,
            "date": date or "",
        }
        feats = self.feature_engineer.extract_features(match)
        return self.predict_from_features(feats)

    # ------------------------- Diagnostics -------------------------
    def get_system_info(self) -> Dict[str, Any]:
        libs = {"numpy": np.__version__}
        try:
            import sklearn as sk
            libs["sklearn"] = sk.__version__
        except Exception:
            pass
        if xgb is not None:
            libs["xgboost"] = xgb.__version__
        return {
            "models_dir": str(self.models_dir),
            "is_trained": self.is_trained,
            "libs": libs,
        }


# =============================================================
# CLI test
# =============================================================
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    engine = MLPredictionEngine(models_dir=os.environ.get("MODELS_DIR", "data/ai_models_v3"))

    # Quick smoke test for predict (without training)
    sample = engine.predict_match(
        home_team="Barcelona",
        away_team="Real Madrid",
        league="La Liga",
        odds={"1": 2.1, "X": 3.3, "2": 3.2},
        date="2025-10-15",
    )
    print("Sample predict (may error if not trained):", sample)

    # Demonstration for training with synthetic data (comment out in production)
    # Generate a tiny toy dataset (features=32); in prod feed your real feature matrix
    rng = np.random.default_rng(42)
    X_demo = rng.normal(size=(1200, 32)).astype(np.float32)
    y_demo = rng.integers(0, 3, size=(1200,))
    try:
        out = engine.fit(X_demo, y_demo)
        print("Training done:", out)
        sample2 = engine.predict_from_features(rng.normal(size=(32,)).astype(np.float32))
        print("Sample predict after training:", sample2)
    except Exception as e:
        print("Training skipped (env missing libs?):", e)
