#!/usr/bin/env python3
"""
ML Tahmin Motoru + Entegre Skor Tahmini (ScorePredictor ile)
"""

import numpy as np
import logging
from typing import Dict, Any, List
from datetime import datetime
import pickle
import os

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML kütüphaneleri yüklü değil")

from advanced_feature_engineer import AdvancedFeatureEngineer, FEATURE_NAMES

# Skor tahmini için
try:
    from score_predictor import ScorePredictor
    SCORE_AVAILABLE = True
except ImportError:
    SCORE_AVAILABLE = False
    logger.warning("ScorePredictor yüklenemedi")


class MLPredictionEngine:
    def __init__(self, model_path: str = "data/ai_models_v2"):
        self.model_path = model_path
        self.feature_engineer = AdvancedFeatureEngineer()
        
        # MS (Maç Sonucu) modelleri
        self.models = {
            'xgboost': None,
            'random_forest': None,
            'gradient_boost': None
        }
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        self.weights = {
            'xgboost': 0.45,
            'random_forest': 0.30,
            'gradient_boost': 0.25
        }
        
        # Skor tahmin modeli
        self.score_predictor = ScorePredictor(model_path) if SCORE_AVAILABLE else None
        
        os.makedirs(model_path, exist_ok=True)
        self._load_models()
    
    def _initialize_models(self):
        if not ML_AVAILABLE:
            return False
        
        try:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                objective='multi:softprob', num_class=3, random_state=42
            )
            
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=150, max_depth=12, min_samples_split=5,
                min_samples_leaf=2, random_state=42
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
        model_file = os.path.join(self.model_path, "ensemble_models.pkl")
        
        if os.path.exists(model_file):
            try:
                with open(model_file, 'rb') as f:
                    data = pickle.load(f)
                    self.models = data['models']
                    self.scaler = data['scaler']
                    self.is_trained = data['is_trained']
                
                logger.info("✅ MS modelleri yüklendi")
                return True
            except Exception as e:
                logger.warning(f"Model yükleme hatası: {e}")
        
        return self._initialize_models()
    
    def _save_models(self):
        try:
            model_file = os.path.join(self.model_path, "ensemble_models.pkl")
            data = {
                'models': self.models,
                'scaler': self.scaler,
                'is_trained': self.is_trained
            }
            
            with open(model_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info("💾 Modeller kaydedildi")
            return True
            
        except Exception as e:
            logger.error(f"Model kaydetme hatası: {e}")
            return False
    
    def train(self, historical_matches: List[Dict]) -> Dict[str, Any]:
        if not ML_AVAILABLE:
            return {"success": False, "error": "ML kütüphaneleri yüklü değil"}
        
        if len(historical_matches) < 100:
            return {"success": False, "error": f"Yetersiz veri: {len(historical_matches)}"}
        
        try:
            logger.info(f"🎓 {len(historical_matches)} maçla eğitim başlıyor...")
            
            X, y = [], []
            
            for match in historical_matches:
                features = self.feature_engineer.extract_features({
                    'home_team': match['home_team'],
                    'away_team': match['away_team'],
                    'league': match.get('league', 'Unknown'),
                    'odds': match.get('odds', {'1': 2.0, 'X': 3.0, '2': 3.5}),
                    'date': match.get('date', datetime.now().isoformat())
                })
                
                result = match.get('result', 'X')
                label = {'1': 0, 'X': 1, '2': 2}.get(result, 1)
                
                X.append(features)
                y.append(label)
                self._update_feature_history(match)
            
            X = np.array(X)
            y = np.array(y)
            
            X_scaled = self.scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            accuracies = {}
            for name, model in self.models.items():
                if model is None:
                    continue
                
                logger.info(f"📚 {name} eğitiliyor...")
                model.fit(X_train, y_train)
                acc = model.score(X_test, y_test)
                accuracies[name] = acc
                logger.info(f"✅ {name}: %{acc*100:.2f}")
            
            self.is_trained = True
            self._save_models()
            
            ensemble_acc = self._calculate_ensemble_accuracy(X_test, y_test)
            
            return {
                "success": True,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "accuracies": accuracies,
                "ensemble_accuracy": ensemble_acc,
                "features_used": len(FEATURE_NAMES)
            }
            
        except Exception as e:
            logger.error(f"Eğitim hatası: {e}")
            return {"success": False, "error": str(e)}
    
    def _update_feature_history(self, match: Dict):
        """Geçmiş maç verilerini feature engineer'a ekle"""
        home_result = 'W' if match['result'] == '1' else ('D' if match['result'] == 'X' else 'L')
        away_result = 'L' if match['result'] == '1' else ('D' if match['result'] == 'X' else 'W')
        
        self.feature_engineer.update_team_history(match['home_team'], {
            'result': home_result,
            'goals_for': match.get('home_goals', 0),
            'goals_against': match.get('away_goals', 0),
            'date': match.get('date', ''),
            'venue': 'home'
        })
        
        self.feature_engineer.update_team_history(match['away_team'], {
            'result': away_result,
            'goals_for': match.get('away_goals', 0),
            'goals_against': match.get('home_goals', 0),
            'date': match.get('date', ''),
            'venue': 'away'
        })
        
        self.feature_engineer.update_h2h_history(
            match['home_team'], match['away_team'],
            {
                'result': match['result'],
                'home_goals': match.get('home_goals', 0),
                'away_goals': match.get('away_goals', 0)
            }
        )
        
        self.feature_engineer.update_league_results(
            match.get('league', 'Unknown'), match['result']
        )
    
    def _calculate_ensemble_accuracy(self, X_test, y_test) -> float:
        predictions = self._ensemble_predict_proba(X_test)
        y_pred = np.argmax(predictions, axis=1)
        return np.mean(y_pred == y_test)
    
    def _ensemble_predict_proba(self, X) -> np.ndarray:
        if not self.is_trained:
            return np.ones((len(X), 3)) / 3
        
        ensemble_probs = np.zeros((len(X), 3))
        
        for name, model in self.models.items():
            if model is None:
                continue
            weight = self.weights.get(name, 0.33)
            probs = model.predict_proba(X)
            ensemble_probs += weight * probs
        
        ensemble_probs /= ensemble_probs.sum(axis=1, keepdims=True)
        return ensemble_probs
    
    def predict_match(self, home_team: str, away_team: str, odds: Dict, 
                     league: str = "Unknown") -> Dict[str, Any]:
        """
        KAPSAMLI MAÇ TAHMİNİ: MS + SKOR + ANALİZ
        """
        try:
            match_data = {
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'odds': odds,
                'date': datetime.now().isoformat()
            }
            
            features = self.feature_engineer.extract_features(match_data)
            
            if not self.is_trained:
                return self._basic_prediction(home_team, away_team, odds)
            
            X = features.reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # MS (Maç Sonucu) tahmini
            probs = self._ensemble_predict_proba(X_scaled)[0]
            
            prediction_idx = np.argmax(probs)
            prediction = ['1', 'X', '2'][prediction_idx]
            confidence = probs[prediction_idx] * 100
            
            # SKOR TAHMİNİ (ScorePredictor kullanarak)
            score_prediction = "1-1"  # Default
            top3_scores = []
            
            if self.score_predictor and self.score_predictor.model is not None:
                try:
                    score_prediction, top3_scores = self.score_predictor.predict(
                        match_data, ms_pred=prediction
                    )
                    logger.info(f"✅ Skor tahmini: {score_prediction}")
                except Exception as e:
                    logger.warning(f"Skor tahmini hatası: {e}")
                    score_prediction = self._fallback_score(prediction, home_team, away_team)
            else:
                score_prediction = self._fallback_score(prediction, home_team, away_team)
            
            # Value bet analizi
            value_index = self._calculate_value_bet(probs, odds)
            
            # Risk seviyesi
            if confidence >= 70:
                risk = "LOW"
            elif confidence >= 55:
                risk = "MEDIUM"
            else:
                risk = "HIGH"
            
            result = {
                "prediction": prediction,
                "result": prediction,  # Frontend uyumluluğu
                "confidence": round(confidence, 1),
                "score_prediction": score_prediction,
                "top3_scores": top3_scores if top3_scores else [
                    {"score": score_prediction, "prob": confidence/100}
                ],
                "probabilities": {
                    "home_win": round(probs[0] * 100, 1),
                    "draw": round(probs[1] * 100, 1),
                    "away_win": round(probs[2] * 100, 1)
                },
                "value_bet": {
                    "value_index": round(value_index, 3),
                    "rating": self._get_value_rating(value_index)
                },
                "risk_level": risk,
                "recommendation": self._get_recommendation(confidence, value_index),
                "model": "ML Ensemble + ScorePredictor",
                "features_used": len(FEATURE_NAMES),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ {home_team} - {away_team}: {prediction} ({confidence:.1f}%) | Skor: {score_prediction}")
            return result
            
        except Exception as e:
            logger.error(f"Tahmin hatası: {e}", exc_info=True)
            return self._basic_prediction(home_team, away_team, odds)
    
    def _fallback_score(self, ms_pred: str, home_team: str, away_team: str) -> str:
        """Skor modeli yoksa feature'lardan basit tahmin"""
        try:
            home_form = self.feature_engineer._calculate_form(home_team, is_home=True)
            away_form = self.feature_engineer._calculate_form(away_team, is_home=False)
            
            home_avg = home_form['goals_per_match']
            away_avg = away_form['goals_per_match']
            
            if ms_pred == '1':  # Ev sahibi galip
                home_goals = max(1, round(home_avg * 1.2))
                away_goals = max(0, round(away_avg * 0.8))
                if home_goals <= away_goals:
                    home_goals = away_goals + 1
            elif ms_pred == 'X':  # Beraberlik
                avg = (home_avg + away_avg) / 2
                home_goals = away_goals = max(1, round(avg))
            else:  # Deplasman galip
                home_goals = max(0, round(home_avg * 0.8))
                away_goals = max(1, round(away_avg * 1.2))
                if away_goals <= home_goals:
                    away_goals = home_goals + 1
            
            # Makul sınırlar
            home_goals = min(5, max(0, home_goals))
            away_goals = min(5, max(0, away_goals))
            
            return f"{home_goals}-{away_goals}"
        except:
            return "2-1" if ms_pred == '1' else ("1-1" if ms_pred == 'X' else "1-2")
    
    def _calculate_value_bet(self, probs: np.ndarray, odds: Dict) -> float:
        try:
            odds_values = [
                float(odds.get('1', 2.0)),
                float(odds.get('X', 3.0)),
                float(odds.get('2', 3.5))
            ]
            expected_values = probs * odds_values
            max_ev = np.max(expected_values)
            return max_ev - 1
        except:
            return 0.0
    
    def _get_value_rating(self, value_index: float) -> str:
        if value_index > 0.15:
            return "EXCELLENT"
        elif value_index > 0.10:
            return "VERY GOOD"
        elif value_index > 0.05:
            return "GOOD"
        elif value_index > 0.02:
            return "FAIR"
        else:
            return "POOR"
    
    def _get_recommendation(self, confidence: float, value_index: float) -> str:
        if confidence >= 70 and value_index > 0.10:
            return "🔥 STRONG BET"
        elif confidence >= 65 and value_index > 0.05:
            return "✅ RECOMMENDED"
        elif confidence >= 55:
            return "⚠️ CONSIDER"
        else:
            return "❌ SKIP"
    
    def _basic_prediction(self, home_team: str, away_team: str, odds: Dict) -> Dict:
        """Model eğitilmemişse basit tahmin"""
        try:
            odds_1 = float(odds.get('1', 2.0))
            odds_x = float(odds.get('X', 3.0))
            odds_2 = float(odds.get('2', 3.5))
            
            imp_1 = 1 / odds_1
            imp_x = 1 / odds_x
            imp_2 = 1 / odds_2
            total = imp_1 + imp_x + imp_2
            
            prob_1 = (imp_1 / total) * 100
            prob_x = (imp_x / total) * 100
            prob_2 = (imp_2 / total) * 100
            
            probs = {'1': prob_1, 'X': prob_x, '2': prob_2}
            prediction = max(probs, key=probs.get)
            
            # Basit skor
            if prediction == '1':
                score = "2-1"
            elif prediction == 'X':
                score = "1-1"
            else:
                score = "1-2"
            
            return {
                "prediction": prediction,
                "result": prediction,
                "confidence": round(probs[prediction], 1),
                "score_prediction": score,
                "top3_scores": [{"score": score, "prob": probs[prediction]/100}],
                "probabilities": {
                    "home_win": round(prob_1, 1),
                    "draw": round(prob_x, 1),
                    "away_win": round(prob_2, 1)
                },
                "value_bet": {"value_index": 0.0, "rating": "UNKNOWN"},
                "risk_level": "HIGH",
                "recommendation": "⏳ MODEL NOT TRAINED",
                "model": "Basic Odds-Based",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Basic prediction hatası: {e}")
            return {
                "prediction": "X",
                "result": "X",
                "confidence": 33.3,
                "score_prediction": "1-1",
                "top3_scores": [],
                "probabilities": {"home_win": 33.3, "draw": 33.3, "away_win": 33.3},
                "value_bet": {"value_index": 0.0, "rating": "UNKNOWN"},
                "risk_level": "HIGH",
                "recommendation": "❌ ERROR",
                "model": "Fallback"
            }