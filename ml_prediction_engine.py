#!/usr/bin/env python3
"""
ML Prediction Engine - FIXED VERSION
Scaler yükleme problemi çözüldü
"""

import os
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from advanced_feature_engineer import AdvancedFeatureEngineer
from score_predictor import ScorePredictor

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("MLEngine")


class MLPredictionEngine:
    """Production ML Engine with fixed scaler loading"""
    
    def __init__(self, model_path: str = "data/ai_models_v2"):
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)
        
        # Components
        self.feature_engineer = AdvancedFeatureEngineer(model_path=model_path)
        self.score_predictor = ScorePredictor(models_dir=model_path)
        
        # Models
        self.models = {
            'xgboost': None,
            'random_forest': None,
            'gradient_boost': None
        }
        self.scaler = None
        self.is_trained = False
        
        # Load models
        self._load_models()
    
    def _load_models(self) -> bool:
        """Load all models with proper error handling"""
        try:
            # 1. MS Models
            ms_path = Path(self.model_path) / "ensemble_models.pkl"
            if not ms_path.exists():
                logger.warning(f"⚠️ MS models not found: {ms_path}")
                return False
            
            file_size = ms_path.stat().st_size
            if file_size < 1000:
                logger.error(f"❌ Corrupted MS models file ({file_size} bytes)")
                return False
            
            with open(ms_path, 'rb') as f:
                ms_data = pickle.load(f)
            
            self.models = ms_data.get('models', {})
            logger.info(f"✅ MS models loaded ({file_size:,} bytes)")
            
            # 2. Scaler (AYRI DOSYA)
            scaler_path = Path(self.model_path) / "scaler.pkl"
            if not scaler_path.exists():
                logger.warning(f"⚠️ Scaler not found: {scaler_path}")
                # Fallback: sklearn StandardScaler oluştur
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                logger.info("ℹ️ Using default StandardScaler")
            else:
                file_size = scaler_path.stat().st_size
                if file_size < 50:
                    logger.error(f"❌ Corrupted scaler file ({file_size} bytes)")
                    from sklearn.preprocessing import StandardScaler
                    self.scaler = StandardScaler()
                else:
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    logger.info(f"✅ Scaler loaded ({file_size:,} bytes)")
            
            # 3. Check if models are valid
            valid_models = sum(1 for m in self.models.values() if m is not None)
            if valid_models == 0:
                logger.error("❌ No valid MS models found")
                return False
            
            self.is_trained = True
            logger.info(f"✅ Engine ready ({valid_models}/3 models)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading error: {e}", exc_info=True)
            return False
    
    def load_models(self) -> bool:
        """Public method to reload models"""
        return self._load_models()
    
    def predict_match(
        self, 
        home_team: str, 
        away_team: str, 
        odds: Dict[str, float], 
        league: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Predict match outcome with comprehensive error handling
        
        Args:
            home_team: Home team name
            away_team: Away team name
            odds: {"1": float, "X": float, "2": float}
            league: League name
        
        Returns:
            Prediction dictionary with result, confidence, score, etc.
        """
        try:
            if not self.is_trained:
                return self._fallback_prediction(home_team, away_team, odds, "Models not trained")
            
            # Prepare match data
            match_data = {
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'odds': odds,
                'date': '2025-10-10'
            }
            
            # Extract features
            features = self.feature_engineer.extract_features(match_data)
            if features is None:
                return self._fallback_prediction(home_team, away_team, odds, "Feature extraction failed")
            
            # Scale features
            try:
                if self.scaler is not None and hasattr(self.scaler, 'mean_'):
                    features_scaled = self.scaler.transform(features.reshape(1, -1))
                else:
                    # Scaler fitted değilse, features'ı olduğu gibi kullan
                    logger.warning("⚠️ Scaler not fitted, using raw features")
                    features_scaled = features.reshape(1, -1)
            except Exception as e:
                logger.warning(f"⚠️ Scaling error: {e}, using raw features")
                features_scaled = features.reshape(1, -1)
            
            # Get predictions from all models
            predictions = {}
            for name, model in self.models.items():
                if model is not None:
                    try:
                        proba = model.predict_proba(features_scaled)[0]
                        predictions[name] = proba
                    except Exception as e:
                        logger.warning(f"⚠️ {name} prediction failed: {e}")
            
            if not predictions:
                return self._fallback_prediction(home_team, away_team, odds, "All models failed")
            
            # Ensemble prediction (weighted average)
            weights = {
                'xgboost': 0.40,
                'random_forest': 0.30,
                'gradient_boost': 0.30
            }
            
            ensemble_proba = np.zeros(3)
            total_weight = 0
            
            for name, proba in predictions.items():
                weight = weights.get(name, 0.33)
                ensemble_proba += proba * weight
                total_weight += weight
            
            if total_weight > 0:
                ensemble_proba /= total_weight
            
            # Get result
            result_idx = int(np.argmax(ensemble_proba))
            result_map = {0: '1', 1: 'X', 2: '2'}
            result = result_map[result_idx]
            confidence = float(ensemble_proba[result_idx]) * 100
            
            # Score prediction
            score, top3_scores = self.score_predictor.predict(match_data, ms_pred=result)
            
            # Value bet analysis
            value_bet = self._calculate_value_bet(result, confidence, odds)
            
            return {
                'prediction': result,
                'result': result,  # Legacy compatibility
                'confidence': confidence,
                'probabilities': {
                    '1': float(ensemble_proba[0]) * 100,
                    'X': float(ensemble_proba[1]) * 100,
                    '2': float(ensemble_proba[2]) * 100
                },
                'score_prediction': score,
                'alternative_scores': top3_scores,
                'value_bet': value_bet,
                'model_predictions': {
                    name: {
                        '1': float(proba[0]) * 100,
                        'X': float(proba[1]) * 100,
                        '2': float(proba[2]) * 100
                    }
                    for name, proba in predictions.items()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}", exc_info=True)
            return self._fallback_prediction(home_team, away_team, odds, str(e))
    
    def _calculate_value_bet(
        self, 
        prediction: str, 
        confidence: float, 
        odds: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate value bet metrics"""
        try:
            predicted_odd = float(odds.get(prediction, 2.0))
            
            # Model probability
            model_prob = confidence / 100.0
            
            # Implied probability from odds
            implied_prob = 1.0 / predicted_odd if predicted_odd > 1.01 else 0.5
            
            # Value index
            value_index = (model_prob * predicted_odd) - 1.0
            
            # Risk assessment
            if value_index >= 0.15:
                risk = "Low Risk"
                recommendation = "Strong Bet"
            elif value_index >= 0.08:
                risk = "Medium Risk"
                recommendation = "Good Bet"
            elif value_index >= 0.03:
                risk = "Higher Risk"
                recommendation = "Marginal Bet"
            else:
                risk = "High Risk"
                recommendation = "Avoid"
            
            return {
                'value_index': round(value_index, 3),
                'predicted_odd': predicted_odd,
                'model_probability': round(model_prob, 3),
                'implied_probability': round(implied_prob, 3),
                'edge': round((model_prob - implied_prob) * 100, 2),
                'risk': risk,
                'recommendation': recommendation
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Value bet calculation error: {e}")
            return {
                'value_index': 0.0,
                'risk': "Unknown",
                'recommendation': "No Data"
            }
    
    def _fallback_prediction(
        self, 
        home_team: str, 
        away_team: str, 
        odds: Dict[str, float],
        reason: str
    ) -> Dict[str, Any]:
        """Fallback prediction based on odds"""
        try:
            # Find favorite based on odds
            odds_1 = float(odds.get("1", 2.0))
            odds_x = float(odds.get("X", 3.0))
            odds_2 = float(odds.get("2", 3.5))
            
            if odds_1 < odds_x and odds_1 < odds_2:
                result = "1"
                confidence = 55.0
            elif odds_2 < odds_1 and odds_2 < odds_x:
                result = "2"
                confidence = 50.0
            else:
                result = "X"
                confidence = 45.0
            
            # Simple score prediction
            if result == "1":
                score = "2-1"
            elif result == "X":
                score = "1-1"
            else:
                score = "1-2"
            
            return {
                'prediction': result,
                'result': result,
                'confidence': confidence,
                'probabilities': {
                    '1': 33.3,
                    'X': 33.3,
                    '2': 33.4
                },
                'score_prediction': score,
                'alternative_scores': [
                    {'score': score, 'prob': 0.33}
                ],
                'value_bet': {
                    'value_index': 0.0,
                    'risk': 'Unknown',
                    'recommendation': 'Fallback Prediction'
                },
                'fallback': True,
                'fallback_reason': reason
            }
            
        except Exception as e:
            logger.error(f"❌ Fallback prediction error: {e}")
            return {
                'prediction': '1',
                'confidence': 33.3,
                'error': 'Complete failure',
                'fallback': True
            }


# Test
if __name__ == "__main__":
    engine = MLPredictionEngine()
    
    test_match = {
        'home_team': 'Barcelona',
        'away_team': 'Real Madrid',
        'odds': {'1': 2.10, 'X': 3.40, '2': 3.20},
        'league': 'La Liga'
    }
    
    print("\n" + "="*60)
    print(f"TEST: {test_match['home_team']} vs {test_match['away_team']}")
    print("="*60)
    
    result = engine.predict_match(
        test_match['home_team'],
        test_match['away_team'],
        test_match['odds'],
        test_match['league']
    )
    
    print(f"\n🎯 Prediction: {result['prediction']}")
    print(f"📊 Confidence: {result['confidence']:.1f}%")
    print(f"⚽ Score: {result['score_prediction']}")
    print(f"💰 Value Index: {result['value_bet']['value_index']:.3f}")
    print(f"⚠️ Risk: {result['value_bet']['risk']}")
    print(f"💡 Recommendation: {result['value_bet']['recommendation']}")
    
    if result.get('fallback'):
        print(f"\n⚠️ FALLBACK: {result.get('fallback_reason')}")
