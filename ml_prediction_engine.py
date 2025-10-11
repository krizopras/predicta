#!/usr/bin/env python3
"""
ML Prediction Engine with Sklearn Compatibility Fix
"""

import os
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

# ML imports
try:
    import sklearn
    from sklearn.preprocessing import StandardScaler
    # Sklearn version check
    SKLEARN_VERSION = tuple(map(int, sklearn.__version__.split('.')[:2]))
    logging.info(f"Sklearn version: {sklearn.__version__}")
except ImportError:
    logging.error("Sklearn not available")
    SKLEARN_VERSION = (0, 0)

logger = logging.getLogger("MLEngine")


class SklearnCompatLoader:
    """Sklearn version uyumsuzluklarını handle eder"""
    
    @staticmethod
    def safe_load_pickle(file_path: Path):
        """
        Sklearn modelleri için güvenli pickle yükleme
        Versiyon uyumsuzluklarını handle eder
        """
        try:
            with open(file_path, 'rb') as f:
                # Normal yükleme dene
                return pickle.load(f)
        except (ValueError, AttributeError) as e:
            if "node array" in str(e) or "dtype" in str(e):
                logger.warning(f"Sklearn uyumsuzluğu tespit edildi: {e}")
                logger.warning("Model dosyası eski sklearn versiyonuyla kaydedilmiş")
                logger.warning("Çözüm: Modelleri yeniden eğitin (Training endpoint)")
                return None
            raise


class MLPredictionEngine:
    """ML Prediction Engine with version compatibility"""
    
    def __init__(self, model_path: str = "data/ai_models_v2"):
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Models
        self.models = {
            'xgboost': None,
            'gradient_boost': None,
            'random_forest': None
        }
        self.scaler = StandardScaler()
        self.score_predictor = None
        self.is_trained = False
        
        # Feature engineer
        try:
            from advanced_feature_engineer import AdvancedFeatureEngineer
            self.feature_engineer = AdvancedFeatureEngineer(model_path=str(self.model_path))
        except ImportError:
            logger.error("AdvancedFeatureEngineer not found")
            self.feature_engineer = None
        
        # Load models
        self.load_models()
    
    def load_models(self) -> bool:
        """Load models with compatibility handling"""
        logger.info("🔄 Loading ML models...")
        
        try:
            # MS Models
            ms_path = self.model_path / "ensemble_models.pkl"
            if not ms_path.exists():
                logger.warning(f"⚠️ Model file not found: {ms_path}")
                logger.warning("💡 Please train models first: POST /api/training/start")
                return False
            
            logger.info(f"📂 Loading from: {ms_path}")
            
            # Uyumlu yükleme
            ms_data = SklearnCompatLoader.safe_load_pickle(ms_path)
            
            if ms_data is None:
                logger.error("❌ Model loading failed due to sklearn incompatibility")
                logger.error("🔧 Solution: Retrain models with current sklearn version")
                logger.error("   Run: POST /api/training/start")
                return False
            
            if isinstance(ms_data, dict):
                self.models = ms_data.get('models', {})
                self.scaler = ms_data.get('scaler', StandardScaler())
                self.is_trained = ms_data.get('is_trained', False)
                
                # Modelleri kontrol et
                loaded_models = [name for name, model in self.models.items() if model is not None]
                logger.info(f"✅ Loaded MS models: {', '.join(loaded_models)}")
            
            # Score Model
            score_path = self.model_path / "score_model.pkl"
            if score_path.exists():
                score_data = SklearnCompatLoader.safe_load_pickle(score_path)
                
                if score_data and isinstance(score_data, dict):
                    try:
                        from score_predictor import ScorePredictor
                        self.score_predictor = ScorePredictor()
                        self.score_predictor.model = score_data.get('model')
                        self.score_predictor.space = score_data.get('score_space', [])
                        logger.info(f"✅ Score model loaded ({len(self.score_predictor.space)} classes)")
                    except ImportError:
                        logger.warning("⚠️ ScorePredictor not available")
            
            # Scaler
            scaler_path = self.model_path / "scaler.pkl"
            if scaler_path.exists():
                scaler_data = SklearnCompatLoader.safe_load_pickle(scaler_path)
                if scaler_data:
                    self.scaler = scaler_data
                    logger.info("✅ Scaler loaded")
            
            return len([m for m in self.models.values() if m is not None]) > 0
            
        except FileNotFoundError as e:
            logger.error(f"❌ Model file not found: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Model loading error: {e}", exc_info=True)
            return False
    
    def predict_match(
        self, 
        home_team: str, 
        away_team: str, 
        odds: Dict[str, float],
        league: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Predict match outcome
        
        Returns:
            {
                "prediction": "1" | "X" | "2",
                "confidence": float (0-100),
                "probabilities": {"1": float, "X": float, "2": float},
                "score_prediction": "2-1",
                "alternative_scores": [...],
                "value_bet": {...}
            }
        """
        if not self.is_trained or not any(self.models.values()):
            return {
                "error": "Models not trained",
                "prediction": "X",
                "confidence": 0,
                "probabilities": {"1": 33.3, "X": 33.3, "2": 33.3},
                "score_prediction": "1-1",
                "value_bet": {"value_index": 0, "risk": "Unknown", "recommendation": "Model not trained"}
            }
        
        try:
            # Extract features
            match_data = {
                'home_team': home_team,
                'away_team': away_team,
                'odds': odds,
                'league': league
            }
            
            features = self.feature_engineer.extract_features(match_data)
            
            if features is None:
                raise ValueError("Feature extraction failed")
            
            # NaN/Inf handling
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            features = features.reshape(1, -1)
            
            # Scale
            if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            # Ensemble prediction
            predictions = []
            confidences = []
            
            for name, model in self.models.items():
                if model is None:
                    continue
                
                try:
                    proba = model.predict_proba(features_scaled)[0]
                    predictions.append(proba)
                    confidences.append(max(proba))
                except Exception as e:
                    logger.warning(f"⚠️ {name} prediction failed: {e}")
                    continue
            
            if not predictions:
                raise ValueError("No models available for prediction")
            
            # Average probabilities
            avg_proba = np.mean(predictions, axis=0)
            predicted_class = int(np.argmax(avg_proba))
            confidence = float(np.max(avg_proba) * 100)
            
            ms_map = {0: "1", 1: "X", 2: "2"}
            predicted_result = ms_map[predicted_class]
            
            probabilities = {
                "1": float(avg_proba[0] * 100),
                "X": float(avg_proba[1] * 100),
                "2": float(avg_proba[2] * 100)
            }
            
            # Score prediction
            score_pred = "1-1"
            alt_scores = []
            
            if self.score_predictor and self.score_predictor.model:
                try:
                    score_pred, alt_scores = self.score_predictor.predict_score(
                        features_scaled, predicted_result
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Score prediction failed: {e}")
            
            # Value bet analysis
            value_bet = self._calculate_value_bet(probabilities, odds, confidence)
            
            return {
                "prediction": predicted_result,
                "confidence": round(confidence, 2),
                "probabilities": {k: round(v, 2) for k, v in probabilities.items()},
                "score_prediction": score_pred,
                "alternative_scores": alt_scores[:3],
                "value_bet": value_bet
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}", exc_info=True)
            return {
                "error": str(e),
                "prediction": "X",
                "confidence": 0,
                "probabilities": {"1": 33.3, "X": 33.3, "2": 33.3},
                "score_prediction": "1-1",
                "value_bet": {"value_index": 0, "risk": "Error", "recommendation": str(e)}
            }
    
    def _calculate_value_bet(
        self, 
        probabilities: Dict[str, float], 
        odds: Dict[str, float],
        confidence: float
    ) -> Dict[str, Any]:
        """Calculate value bet index"""
        try:
            value_bets = {}
            
            for outcome in ["1", "X", "2"]:
                prob = probabilities.get(outcome, 0) / 100.0
                odd = float(odds.get(outcome, 1.0))
                
                if prob > 0 and odd > 1.0:
                    expected_value = (prob * odd) - 1.0
                    value_bets[outcome] = expected_value
            
            if not value_bets:
                return {
                    "value_index": 0,
                    "risk": "Unknown",
                    "recommendation": "No value detected"
                }
            
            best_outcome = max(value_bets, key=value_bets.get)
            best_value = value_bets[best_outcome]
            
            # Risk assessment
            if best_value >= 0.15:
                risk = "Low Risk"
                rec = "Excellent Value Bet"
            elif best_value >= 0.10:
                risk = "Medium Risk"
                rec = "Good Value Bet"
            elif best_value >= 0.05:
                risk = "Medium Risk"
                rec = "Moderate Value"
            else:
                risk = "High Risk"
                rec = "Low Value"
            
            return {
                "value_index": round(best_value, 3),
                "best_outcome": best_outcome,
                "risk": risk,
                "recommendation": rec,
                "confidence_factor": round(confidence / 100.0, 2)
            }
            
        except Exception as e:
            logger.error(f"Value bet calculation error: {e}")
            return {
                "value_index": 0,
                "risk": "Error",
                "recommendation": str(e)
            }


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = MLPredictionEngine()
    
    print(f"\n{'='*60}")
    print(f"Sklearn Version: {sklearn.__version__}")
    print(f"Models Trained: {engine.is_trained}")
    print(f"Available Models: {[n for n, m in engine.models.items() if m is not None]}")
    print(f"{'='*60}\n")
    
    if not engine.is_trained:
        print("⚠️  Models not trained!")
        print("💡 Run training: POST /api/training/start")
    else:
        # Test prediction
        test_result = engine.predict_match(
            "Barcelona", "Real Madrid",
            {"1": 2.10, "X": 3.40, "2": 3.20},
            "La Liga"
        )
        
        print("Test Prediction:")
        print(f"  Result: {test_result['prediction']}")
        print(f"  Confidence: {test_result['confidence']}%")
        print(f"  Score: {test_result['score_prediction']}")
