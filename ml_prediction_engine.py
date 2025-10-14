#!/usr/bin/env python3
"""
ML Prediction Engine with Sklearn Compatibility Fix
FULLY FIXED VERSION - Feature Engineer Import & Error Handling
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
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except (ValueError, AttributeError) as e:
            if "node array" in str(e) or "dtype" in str(e):
                logger.warning(f"Sklearn uyumsuzluğu tespit edildi: {e}")
                logger.warning("Model dosyası eski sklearn versiyonuyla kaydedilmiş")
                logger.warning("Çözüm: Modelleri yeniden eğitin (Training endpoint)")
                return None
            raise


class MLPredictionEngine:
    """ML Prediction Engine with version compatibility & feature engineer safety"""
    
    def __init__(self, model_path: str = "data/ai_models_v3"):
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        self.models = {
            'xgboost': None,
            'gradient_boost': None,
            'random_forest': None
        }
        self.scaler = StandardScaler()
        self.score_predictor = None
        self.is_trained = False
        
        # 🔧 Feature engineer - güvenli yükleme
        self.feature_engineer = None
        self._load_feature_engineer()
        
        # Load models
        self.load_models()
    
    # =====================================================
    # 🔹 Feature Engineer Güvenli Yükleme (Fixed)
    # =====================================================
    def _load_feature_engineer(self):
        """🧠 Feature Engineer'ı güvenli şekilde yükler"""
        try:
            from enhanced_feature_engineer_v4 import EnhancedFeatureEngineer
            self.feature_engineer = EnhancedFeatureEngineer(model_path=str(self.model_path))
            logger.info("✅ Feature Engineer v4.0 loaded (100+ features)")
        except ImportError as e:
            logger.warning(f"⚠️ EnhancedFeatureEngineer import edilemedi: {e}")
            self.feature_engineer = None
        except Exception as e:
            logger.error(f"❌ Feature Engineer init hatası: {e}", exc_info=True)
            self.feature_engineer = None

        # Eğer hala None ise dummy mod başlat
        if self.feature_engineer is None:
            logger.warning("⚠️ Feature Engineer None, fallback dummy mode aktif.")
            self.feature_engineer = self._dummy_feature_engineer()
            logger.info("✅ Dummy Feature Engineer yüklendi (minimal feature set).")
    
    # =====================================================
    # 🔹 Dummy Feature Engineer (Fallback)
    # =====================================================
    def _dummy_feature_engineer(self):
        """Feature Engineer yoksa geçici dummy nesne oluşturur"""
        class DummyFeatureEngineer:
            def extract_features(self, match_data: Dict) -> np.ndarray:
                # Basit 6 özellik: oranlar ve normalize edilmiş olasılıklar
                odds = match_data.get("odds", {"1": 2.0, "X": 3.0, "2": 3.5})
                o1, ox, o2 = float(odds.get("1", 2.0)), float(odds.get("X", 3.0)), float(odds.get("2", 3.5))
                total = (1/o1 + 1/ox + 1/o2)
                prob1, probx, prob2 = (1/o1)/total, (1/ox)/total, (1/o2)/total
                return np.array([o1, ox, o2, prob1, probx, prob2], dtype=np.float32)
        return DummyFeatureEngineer()
    
    # =====================================================
    # 🔹 Model Yükleme (örnek)
    # =====================================================
    def load_models(self):
        """Kayıtlı modelleri yükler"""
        try:
            for name in self.models.keys():
                model_file = self.model_path / f"{name}_model.pkl"
                if model_file.exists():
                    self.models[name] = SklearnCompatLoader.safe_load_pickle(model_file)
                    logger.info(f"✅ {name} modeli yüklendi.")
                else:
                    logger.warning(f"⚠️ {name} modeli bulunamadı: {model_file}")
            self.is_trained = any(m is not None for m in self.models.values())
        except Exception as e:
            logger.error(f"❌ Model yükleme hatası: {e}")
            self.is_trained = False
    
    # =====================================================
    # 🔹 Tahmin Fonksiyonu (örnek)
    # =====================================================
    def predict(self, match_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Tahmin üretir"""
        try:
            if not self.feature_engineer:
                logger.error("❌ Feature Engineer not available - cannot predict")
                return None
            
            features = self.feature_engineer.extract_features(match_data)
            if features is None:
                logger.error("❌ Feature extraction başarısız")
                return None

            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            features_scaled = self.scaler.fit_transform(features)
            
            results = {}
            for name, model in self.models.items():
                if model is not None:
                    pred = model.predict_proba(features_scaled)[0]
                    results[name] = float(np.argmax(pred))
                else:
                    results[name] = None
            
            return results
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}", exc_info=True)
            return None

            
            # 2. Fallback: Enhanced Feature Engineer v3.5
            try:
                from enhanced_feature_engineer import EnhancedFeatureEngineer
                self.feature_engineer = EnhancedFeatureEngineer(model_path=str(self.model_path))
                logger.warning("⚠️ Feature Engineer v3.5 loaded (fallback)")
                return
            except ImportError as e:
                logger.debug(f"v3.5 import hatası: {e}")
            
            # 3. Son çare: Advanced Feature Engineer (eski)
            try:
                from advanced_feature_engineer import AdvancedFeatureEngineer
                self.feature_engineer = AdvancedFeatureEngineer(model_path=str(self.model_path))
                logger.warning("⚠️ Advanced Feature Engineer loaded (old version)")
                return
            except ImportError as e:
                logger.debug(f"Advanced import hatası: {e}")
            
            # 4. Hiçbiri yüklenemedi
            logger.error("❌ CRITICAL: Hiçbir Feature Engineer yüklenemedi!")
            logger.error("📁 Kontrol edilecek dosyalar:")
            logger.error("   - enhanced_feature_engineer_v4.py")
            logger.error("   - enhanced_feature_engineer.py")
            logger.error("   - advanced_feature_engineer.py")
            self.feature_engineer = None
            
        except Exception as e:
            logger.error(f"❌ Feature Engineer yükleme hatası: {e}", exc_info=True)
            self.feature_engineer = None
    
    def is_feature_engineer_available(self) -> bool:
        """🆕 Feature Engineer kullanılabilir mi?"""
        return self.feature_engineer is not None
    
    def get_feature_engineer_info(self) -> Dict[str, Any]:
        """🆕 Feature Engineer bilgisi"""
        if not self.feature_engineer:
            return {
                "available": False,
                "type": "None",
                "version": "N/A"
            }
        
        fe_type = type(self.feature_engineer).__name__
        
        # Version tespiti
        version = "unknown"
        if hasattr(self.feature_engineer, '__module__'):
            if 'v4' in self.feature_engineer.__module__:
                version = "v4.0 (100+ features)"
            elif 'enhanced' in self.feature_engineer.__module__:
                version = "v3.5 (70 features)"
            elif 'advanced' in self.feature_engineer.__module__:
                version = "v2.0 (46 features)"
        
        return {
            "available": True,
            "type": fe_type,
            "version": version,
            "has_update_methods": all([
                hasattr(self.feature_engineer, 'update_match_result'),
                hasattr(self.feature_engineer, 'extract_features')
            ])
        }
    
    def load_models(self) -> bool:
        """Load models with compatibility handling"""
        logger.info("📄 Loading ML models...")
        
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
                        # Score predictor import
                        try:
                            from enhanced_score_predictor import EnhancedRealisticScorePredictor
                            self.score_predictor = EnhancedRealisticScorePredictor(
                                models_dir=str(self.model_path)
                            )
                            logger.info("✅ Enhanced Score Predictor loaded")
                        except ImportError:
                            try:
                                from score_predictor import ScorePredictor
                                self.score_predictor = ScorePredictor()
                                self.score_predictor.model = score_data.get('model')
                                self.score_predictor.space = score_data.get('score_space', [])
                                logger.info(f"✅ Score model loaded ({len(self.score_predictor.space)} classes)")
                            except ImportError:
                                logger.warning("⚠️ ScorePredictor not available")
                    except Exception as e:
                        logger.warning(f"⚠️ Score model yükleme hatası: {e}")
            
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
        Predict match outcome with full error handling
        
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
        # 🔧 1. Feature Engineer kontrolü
        if self.feature_engineer is None:
            logger.error("❌ Feature Engineer not available - cannot predict")
            return self._error_response("Feature Engineer not loaded")
        
        # 🔧 2. Model kontrolü
        if not self.is_trained or not any(self.models.values()):
            logger.error("❌ Models not trained")
            return self._error_response("Models not trained")
        
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
            
            if self.score_predictor:
                try:
                    if hasattr(self.score_predictor, 'predict_with_enhanced_details'):
                        # Enhanced predictor
                        result = self.score_predictor.predict_with_enhanced_details(
                            match_data, predicted_result
                        )
                        score_pred = result.get('predicted_score', '1-1')
                        alt_scores = result.get('alternatives', [])
                    elif hasattr(self.score_predictor, 'predict_score'):
                        # Basic predictor
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
            return self._error_response(str(e))
    
    def _error_response(self, error_msg: str) -> Dict[str, Any]:
        """🆕 Standart hata response"""
        return {
            "error": error_msg,
            "prediction": "X",
            "confidence": 0,
            "probabilities": {"1": 33.3, "X": 33.3, "2": 33.3},
            "score_prediction": "1-1",
            "value_bet": {
                "value_index": 0,
                "risk": "Error",
                "recommendation": error_msg
            }
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
    
    def get_system_info(self) -> Dict[str, Any]:
        """🆕 Sistem bilgisi"""
        fe_info = self.get_feature_engineer_info()
        
        return {
            "sklearn_version": sklearn.__version__ if 'sklearn' in dir() else "N/A",
            "models_loaded": len([m for m in self.models.values() if m is not None]),
            "is_trained": self.is_trained,
            "model_path": str(self.model_path),
            "feature_engineer": fe_info,
            "score_predictor_available": self.score_predictor is not None,
            "available_models": [name for name, model in self.models.items() if model is not None]
        }


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = MLPredictionEngine()
    
    print(f"\n{'='*60}")
    print("🔍 SYSTEM CHECK")
    print(f"{'='*60}")
    
    info = engine.get_system_info()
    print(f"\nSklearn Version: {info['sklearn_version']}")
    print(f"Models Trained: {info['is_trained']}")
    print(f"Models Loaded: {info['models_loaded']}")
    print(f"Available Models: {', '.join(info['available_models']) if info['available_models'] else 'None'}")
    
    print(f"\n📊 Feature Engineer:")
    fe = info['feature_engineer']
    print(f"  Available: {fe['available']}")
    print(f"  Type: {fe['type']}")
    print(f"  Version: {fe['version']}")
    
    print(f"\n⚽ Score Predictor: {'✅ Available' if info['score_predictor_available'] else '❌ Not Available'}")
    
    print(f"\n{'='*60}\n")
    
    if not engine.is_trained:
        print("⚠️  Models not trained!")
        print("💡 Run training: POST /api/training/start")
    elif not engine.is_feature_engineer_available():
        print("⚠️  Feature Engineer not available!")
        print("💡 Check if enhanced_feature_engineer.py exists")
    else:
        print("✅ System ready for predictions!")
        
        # Test prediction
        print("\n🧪 Test Prediction:")
        test_result = engine.predict_match(
            "Barcelona", "Real Madrid",
            {"1": 2.10, "X": 3.40, "2": 3.20},
            "La Liga"
        )
        
        if "error" in test_result:
            print(f"❌ Error: {test_result['error']}")
        else:
            print(f"  Result: {test_result['prediction']}")
            print(f"  Confidence: {test_result['confidence']}%")
            print(f"  Score: {test_result['score_prediction']}")
            print(f"  Value Index: {test_result['value_bet']['value_index']}")
