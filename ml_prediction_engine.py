#!/usr/bin/env python3
"""
Gerçek Makine Öğrenmesi Tabanlı Tahmin Motoru
XGBoost + Random Forest Ensemble
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
    logger.warning("ML kütüphaneleri yüklü değil. pip install scikit-learn xgboost")

from advanced_feature_engineer import AdvancedFeatureEngineer, FEATURE_NAMES


class MLPredictionEngine:
    def __init__(self, model_path: str = "data/ai_models_v2"):
        self.model_path = model_path
        self.feature_engineer = AdvancedFeatureEngineer()
        
        # Ensemble modelleri
        self.models = {
            'xgboost': None,
            'random_forest': None,
            'gradient_boost': None
        }
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Model ağırlıkları (ensemble için)
        self.weights = {
            'xgboost': 0.45,
            'random_forest': 0.30,
            'gradient_boost': 0.25
        }
        
        os.makedirs(model_path, exist_ok=True)
        self._load_models()
    
    def _initialize_models(self):
        """Modelleri ilk kez oluştur"""
        if not ML_AVAILABLE:
            logger.error("ML kütüphaneleri yüklü değil!")
            return False
        
        try:
            # XGBoost - En güçlü model
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                num_class=3,
                random_state=42
            )
            
            # Random Forest - Güvenilir yedek
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            # Gradient Boosting - Destek model
            self.models['gradient_boost'] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            
            logger.info("✅ Modeller başarıyla oluşturuldu")
            return True
            
        except Exception as e:
            logger.error(f"Model oluşturma hatası: {e}")
            return False
    
    def _load_models(self):
        """Kaydedilmiş modelleri yükle"""
        model_file = os.path.join(self.model_path, "ensemble_models.pkl")
        
        if os.path.exists(model_file):
            try:
                with open(model_file, 'rb') as f:
                    data = pickle.load(f)
                    self.models = data['models']
                    self.scaler = data['scaler']
                    self.is_trained = data['is_trained']
                
                logger.info("✅ Modeller diskten yüklendi")
                return True
            except Exception as e:
                logger.warning(f"Model yükleme hatası: {e}")
        
        # Eğer yüklenemezse yeni modeller oluştur
        return self._initialize_models()
    
    def _save_models(self):
        """Modelleri diske kaydet"""
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
        """
        Geçmiş maç verileriyle modelleri eğit
        
        Args:
            historical_matches: Geçmiş maç listesi (home_team, away_team, result, odds, vb.)
        
        Returns:
            Eğitim metrikleri
        """
        if not ML_AVAILABLE:
            return {
                "success": False,
                "error": "ML kütüphaneleri yüklü değil"
            }
        
        if len(historical_matches) < 100:
            return {
                "success": False,
                "error": f"Yetersiz veri: {len(historical_matches)} maç (min 100 gerekli)"
            }
        
        try:
            logger.info(f"🎓 {len(historical_matches)} maçla eğitim başlıyor...")
            
            # Feature extraction
            X = []
            y = []
            
            for match in historical_matches:
                # Feature'ları çıkar
                features = self.feature_engineer.extract_features({
                    'home_team': match['home_team'],
                    'away_team': match['away_team'],
                    'league': match.get('league', 'Unknown'),
                    'odds': match.get('odds', {'1': 2.0, 'X': 3.0, '2': 3.5}),
                    'date': match.get('date', datetime.now().isoformat())
                })
                
                # Sonucu label'a çevir
                result = match.get('result', 'X')
                label = {'1': 0, 'X': 1, '2': 2}.get(result, 1)
                
                X.append(features)
                y.append(label)
                
                # Geçmiş verileri feature engineer'a ekle
                self._update_feature_history(match)
            
            X = np.array(X)
            y = np.array(y)
            
            # Normalizasyon
            X_scaled = self.scaler.fit_transform(X)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Her modeli eğit
            accuracies = {}
            
            for name, model in self.models.items():
                if model is None:
                    continue
                
                logger.info(f"📚 {name} eğitiliyor...")
                model.fit(X_train, y_train)
                
                # Test accuracy
                acc = model.score(X_test, y_test)
                accuracies[name] = acc
                logger.info(f"✅ {name}: %{acc*100:.2f} doğruluk")
            
            self.is_trained = True
            self._save_models()
            
            # Ensemble accuracy
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
            return {
                "success": False,
                "error": str(e)
            }
    
    def _update_feature_history(self, match: Dict):
        """Feature engineer'ın geçmiş verilerini güncelle"""
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
            match['home_team'],
            match['away_team'],
            {
                'result': match['result'],
                'home_goals': match.get('home_goals', 0),
                'away_goals': match.get('away_goals', 0)
            }
        )
        
        self.feature_engineer.update_league_results(
            match.get('league', 'Unknown'),
            match['result']
        )
    
    def _calculate_ensemble_accuracy(self, X_test, y_test) -> float:
        """Ensemble modelinin doğruluğunu hesapla"""
        predictions = self._ensemble_predict_proba(X_test)
        y_pred = np.argmax(predictions, axis=1)
        accuracy = np.mean(y_pred == y_test)
        return accuracy
    
    def _ensemble_predict_proba(self, X) -> np.ndarray:
        """Ensemble tahmin - tüm modellerin ağırlıklı ortalaması"""
        if not self.is_trained:
            # Eğitilmemişse uniform dağılım
            return np.ones((len(X), 3)) / 3
        
        ensemble_probs = np.zeros((len(X), 3))
        
        for name, model in self.models.items():
            if model is None:
                continue
            
            weight = self.weights.get(name, 0.33)
            probs = model.predict_proba(X)
            ensemble_probs += weight * probs
        
        # Normalize
        ensemble_probs /= ensemble_probs.sum(axis=1, keepdims=True)
        
        return ensemble_probs
    
    def predict_match(self, home_team: str, away_team: str, odds: Dict, 
                     league: str = "Unknown") -> Dict[str, Any]:
        """
        Maç tahmini yap
        
        Returns:
            Tahmin sonuçları (prediction, confidence, probabilities, vb.)
        """
        try:
            # Feature extraction
            match_data = {
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'odds': odds,
                'date': datetime.now().isoformat()
            }
            
            features = self.feature_engineer.extract_features(match_data)
            
            if not self.is_trained:
                # Eğitilmemişse temel tahmin
                return self._basic_prediction(home_team, away_team, odds)
            
            # ML tahmin
            X = features.reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Ensemble tahmin
            probs = self._ensemble_predict_proba(X_scaled)[0]
            
            # En yüksek olasılıklı sonuç
            prediction_idx = np.argmax(probs)
            prediction = ['1', 'X', '2'][prediction_idx]
            confidence = probs[prediction_idx] * 100
            
            # Value bet analizi
            value_index = self._calculate_value_bet(probs, odds)
            
            # Risk seviyesi
            if confidence >= 70:
                risk = "LOW"
            elif confidence >= 55:
                risk = "MEDIUM"
            else:
                risk = "HIGH"
            
            return {
                "prediction": prediction,
                "confidence": round(confidence, 1),
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
                "model": "ML Ensemble (XGBoost + RF + GB)",
                "features_used": len(FEATURE_NAMES),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Tahmin hatası: {e}")
            return self._basic_prediction(home_team, away_team, odds)
    
    def _calculate_value_bet(self, probs: np.ndarray, odds: Dict) -> float:
        """Value bet indeksini hesapla"""
        try:
            odds_values = [
                float(odds.get('1', 2.0)),
                float(odds.get('X', 3.0)),
                float(odds.get('2', 3.5))
            ]
            
            # Kelly criterion benzeri
            expected_values = probs * odds_values
            max_ev = np.max(expected_values)
            value_index = max_ev - 1
            
            return value_index
            
        except:
            return 0.0
    
    def _get_value_rating(self, value_index: float) -> str:
        """Value index'e göre rating"""
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
        """Tahmin önerisi"""
        if confidence >= 70 and value_index > 0.10:
            return "🔥 STRONG BET"
        elif confidence >= 65 and value_index > 0.05:
            return "✅ RECOMMENDED"
        elif confidence >= 55:
            return "⚠️ CONSIDER"
        else:
            return "❌ SKIP"
    
    def _basic_prediction(self, home_team: str, away_team: str, odds: Dict) -> Dict:
        """Temel tahmin (model eğitilmemişse)"""
        # Oranlardan basit olasılık
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
            
            return {
                "prediction": prediction,
                "confidence": round(probs[prediction], 1),
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
        except:
            return {
                "prediction": "X",
                "confidence": 33.3,
                "probabilities": {"home_win": 33.3, "draw": 33.3, "away_win": 33.3},
                "model": "Fallback"
            }
