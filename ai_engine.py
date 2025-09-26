#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GELİŞTİRİLMİŞ SÜPER ÖĞRENEN YAPAY ZEKA TAHMİN SİSTEMİ
Advanced machine learning with real-time adaptation and deep feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import joblib
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, Union
import os
import json
import sqlite3
from dataclasses import dataclass
from collections import deque, defaultdict
import warnings
from scipy import stats
import requests
import time

warnings.filterwarnings('ignore')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AdvancedMatchFeatures:
    """Gelişmiş maç özellikleri"""
    # Temel özellikler
    home_team_strength: float
    away_team_strength: float
    home_form_score: float
    away_form_score: float
    goal_difference: float
    position_difference: float
    head_to_head_ratio: float
    home_advantage: float
    importance_factor: float
    
    # İstatistiksel özellikler
    home_attack_strength: float
    away_attack_strength: float
    home_defense_strength: float
    away_defense_strength: float
    home_consistency: float
    away_consistency: float
    momentum_difference: float
    
    # Piyasa özellikleri
    odds_1: float
    odds_x: float
    odds_2: float
    odds_variance: float
    market_confidence: float
    
    # Zaman serisi özellikleri
    home_trend: float
    away_trend: float
    recent_performance_gap: float
    
    # Context özellikleri
    pressure_index: float
    motivation_factor: float
    fatigue_index: float

class EnhancedSuperLearningAI:
    """Geliştirilmiş Süper Öğrenen Yapay Zeka Sistemi"""
    
    def __init__(self, db_manager=None, config_file="ai_config.json"):
        self.db_manager = db_manager
        self.config = self.load_config(config_file)
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_selector = None
        self.model_path = "data/ai_models_v2/"
        
        # Performance tracking
        self.performance_history = []
        self.feature_importance_history = []
        self.learning_curves = {}
        
        # Real-time adaptation
        self.prediction_buffer = deque(maxlen=1000)
        self.adaptation_threshold = 0.02  # %2 performance drop
        self.last_accuracy = 0.0
        
        # Feature engineering
        self.feature_correlations = {}
        self.optimal_feature_set = []
        
        # External data integration
        self.external_sources = {
            'weather_api': 'https://api.weatherapi.com/v1',
            'injury_api': 'https://api.football-data.org/v4'
        }
        
        self.setup_environment()
        self.initialize_models()
        self.load_models()
        
    def load_config(self, config_file: str) -> Dict:
        """AI konfigürasyonunu yükle"""
        default_config = {
            "training": {
                "min_samples": 100,
                "cv_folds": 5,
                "test_size": 0.2,
                "auto_retrain_days": 3,
                "validation_threshold": 0.65
            },
            "models": {
                "ensemble_weights": [0.4, 0.3, 0.2, 0.1],
                "confidence_calibration": True,
                "feature_selection": "rfe"
            },
            "features": {
                "max_features": 30,
                "polynomial_degree": 2,
                "interaction_terms": True
            },
            "adaptation": {
                "learning_rate": 0.1,
                "forgetting_factor": 0.95,
                "drift_detection_window": 100
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                # Deep merge
                return self.deep_merge(default_config, user_config)
        except:
            return default_config
    
    def deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Derinlemesine dictionary birleştirme"""
        result = base.copy()
        for key, value in update.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self.deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def setup_environment(self):
        """Çalışma ortamını kur"""
        try:
            os.makedirs(self.model_path, exist_ok=True)
            os.makedirs("logs/", exist_ok=True)
            os.makedirs("data/cache/", exist_ok=True)
            
            logger.info("AI ortamı başarıyla kuruldu")
            
        except Exception as e:
            logger.error(f"Ortam kurulum hatası: {e}")
            raise
    
    def initialize_models(self):
    """Advanced model ensemble initialization"""
    try:
        # Base models for ensemble
        base_models = [
            ('gbm', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=20,
                random_state=42
            )),
            ('rf', RandomForestRegressor(
                n_estimators=150,
                max_depth=8,
                min_samples_split=15,
                random_state=42
            )),
            ('svm', SVC(
                probability=True,
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42
            )),
            ('lr', LogisticRegression(
                max_iter=1000,
                C=0.1,
                solver='liblinear',
                random_state=42
            ))
        ]
        
        # Main ensemble classifier
        self.models['ensemble_classifier'] = VotingClassifier(
            estimators=base_models,
            voting='soft',
            weights=self.config['models']['ensemble_weights']
        )
        
        # Specialized models
        self.models['confidence_calibrator'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        
        self.models['score_predictor'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.15,
            max_depth=4,
            random_state=42
        )
        
        self.models['trend_analyzer'] = RandomForestRegressor(
            n_estimators=80,
            max_depth=5,
            random_state=42
        )
        
        # Advanced scalers and encoders
        self.scalers = {
            'main': StandardScaler(),
            'confidence': StandardScaler(),
            'features': StandardScaler()
        }
        
        self.encoders = {
            'result': LabelEncoder(),
            'score': LabelEncoder(),
            'trend': LabelEncoder()
        }
        
        # Feature selection - DÜZELTİLMİŞ KISIM
        feature_selection_method = self.config.get('models', {}).get('feature_selection', 'rfe')
        max_features = self.config.get('features', {}).get('max_features', 30)
        
        if feature_selection_method == 'rfe':
            self.feature_selector = RFE(
                estimator=RandomForestRegressor(n_estimators=50),
                n_features_to_select=max_features
            )
        else:
            self.feature_selector = SelectKBest(
                score_func=f_classif,
                k=max_features
            )
        
        logger.info("Advanced modeller başarıyla initialize edildi")
        
    except Exception as e:
        logger.error(f"Model initialization hatası: {e}")
        raise
    
    def extract_advanced_features(self, match_data: Dict) -> np.array:
        """Gelişmiş özellik mühendisliği"""
        try:
            home_stats = match_data.get('home_stats', {})
            away_stats = match_data.get('away_stats', {})
            odds = match_data.get('odds', {})
            context = match_data.get('context', {})
            
            features = []
            
            # 1. Temel Güç Özellikleri
            features.extend([
                self.calculate_advanced_team_strength(home_stats),
                self.calculate_advanced_team_strength(away_stats),
                self.calculate_power_difference(home_stats, away_stats)
            ])
            
            # 2. Form ve Momentum Özellikleri
            features.extend([
                self.calculate_weighted_form_score(home_stats.get('recent_form', [])),
                self.calculate_weighted_form_score(away_stats.get('recent_form', [])),
                self.calculate_momentum(home_stats),
                self.calculate_momentum(away_stats),
                self.calculate_consistency(home_stats.get('recent_form', [])),
                self.calculate_consistency(away_stats.get('recent_form', []))
            ])
            
            # 3. İstatistiksel Özellikler
            features.extend([
                self.calculate_attack_strength(home_stats),
                self.calculate_attack_strength(away_stats),
                self.calculate_defense_strength(home_stats),
                self.calculate_defense_strength(away_stats),
                self.calculate_expected_goals(home_stats, away_stats),
                self.calculate_clean_sheet_probability(home_stats, away_stats)
            ])
            
            # 4. Piyasa ve Oran Özellikleri
            features.extend([
                odds.get('1', 2.0),
                odds.get('X', 3.0),
                odds.get('2', 3.5),
                self.calculate_odds_implied_probability(odds),
                self.calculate_market_efficiency(odds),
                self.calculate_value_bet_indicator(odds, home_stats, away_stats)
            ])
            
            # 5. Context ve Durumsal Özellikler
            features.extend([
                context.get('importance', 1.0),
                context.get('pressure', 0.5),
                context.get('motivation', 0.5),
                self.calculate_fatigue_index(home_stats, away_stats, context),
                self.calculate_travel_impact(home_stats, away_stats),
                self.calculate_referee_impact(context.get('referee_stats', {}))
            ])
            
            # 6. Zaman Serisi Özellikleri
            features.extend([
                self.calculate_performance_trend(home_stats),
                self.calculate_performance_trend(away_stats),
                self.calculate_seasonal_effect(match_data.get('match_date', datetime.now()))
            ])
            
            # 7. Interaction Terms
            if self.config['features']['interaction_terms']:
                features.extend(self.calculate_interaction_terms(features))
            
            feature_array = np.array(features).reshape(1, -1)
            
            # Polynomial features
            if self.config['features']['polynomial_degree'] > 1:
                poly = PolynomialFeatures(
                    degree=self.config['features']['polynomial_degree'],
                    include_bias=False
                )
                feature_array = poly.fit_transform(feature_array)
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Advanced feature extraction hatası: {e}")
            return self.extract_basic_features(match_data)
    
    def calculate_advanced_team_strength(self, team_stats: Dict) -> float:
        """Gelişmiş takım gücü hesaplama"""
        try:
            matches_played = max(team_stats.get('matches_played', 1), 1)
            
            # Çok boyutlu güç indeksi
            components = {
                'win_rate': team_stats.get('wins', 0) / matches_played,
                'points_per_game': team_stats.get('points', 0) / matches_played / 3.0,
                'goal_difference': (team_stats.get('goals_for', 0) - team_stats.get('goals_against', 0)) / matches_played,
                'expected_goals': team_stats.get('xG', 0) / matches_played if team_stats.get('xG') else 0,
                'defensive_solidity': 1.0 / (1.0 + team_stats.get('goals_against', 0) / matches_played)
            }
            
            # Ağırlıklar (machine learning ile optimize edilebilir)
            weights = {
                'win_rate': 0.25,
                'points_per_game': 0.25,
                'goal_difference': 0.20,
                'expected_goals': 0.15,
                'defensive_solidity': 0.15
            }
            
            strength = sum(components[comp] * weights[comp] for comp in components)
            return max(0.0, min(1.0, strength))
            
        except Exception as e:
            logger.debug(f"Advanced strength calculation error: {e}")
            return 0.5
    
    def calculate_expected_goals(self, home_stats: Dict, away_stats: Dict) -> float:
        """Beklenen gol (xG) bazlı tahmin"""
        try:
            home_xg = home_stats.get('xG_for', 0) / max(home_stats.get('matches_played', 1), 1)
            away_xg = away_stats.get('xG_for', 0) / max(away_stats.get('matches_played', 1), 1)
            home_xga = home_stats.get('xG_against', 0) / max(home_stats.get('matches_played', 1), 1)
            away_xga = away_stats.get('xG_against', 0) / max(away_stats.get('matches_played', 1), 1)
            
            # Poisson distribution based expected goals
            home_expected = home_xg * away_xga
            away_expected = away_xg * home_xga
            
            return home_expected - away_expected
            
        except:
            return 0.0
    
    def calculate_interaction_terms(self, base_features: List[float]) -> List[float]:
        """Özellik etkileşim terimleri"""
        interactions = []
        n_features = len(base_features)
        
        # Önemli etkileşimler
        important_pairs = [(0, 1), (2, 3), (4, 5), (0, 4), (1, 5)]  # Örnek indeksler
        
        for i, j in important_pairs:
            if i < n_features and j < n_features:
                interactions.append(base_features[i] * base_features[j])
                interactions.append(base_features[i] + base_features[j])
                interactions.append(abs(base_features[i] - base_features[j]))
        
        return interactions
    
    def train_advanced_models(self, force_retrain: bool = False) -> bool:
        """Gelişmiş model eğitimi"""
        try:
            if not self.db_manager:
                logger.warning("Database manager yok, eğitim yapılamıyor")
                return False
            
            # Veri hazırlama
            X, y, metadata = self.prepare_advanced_training_data()
            
            if len(X) < self.config['training']['min_samples']:
                logger.warning(f"Yetersiz eğitim verisi: {len(X)} örnek")
                return False
            
            logger.info(f"Advanced training başlıyor: {len(X)} örnek, {X.shape[1]} özellik")
            
            # Feature selection
            X_selected = self.apply_feature_selection(X, y)
            self.optimal_feature_set = X_selected.shape[1]
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, 
                test_size=self.config['training']['test_size'],
                stratify=y,
                random_state=42
            )
            
            # Feature scaling
            X_train_scaled = self.scalers['main'].fit_transform(X_train)
            X_test_scaled = self.scalers['main'].transform(X_test)
            
            # Model eğitimi
            self.models['ensemble_classifier'].fit(X_train_scaled, y_train)
            
            # Hyperparameter optimization
            self.optimize_hyperparameters(X_train_scaled, y_train)
            
            # Cross-validation
            cv_scores = self.cross_validate_model(X_selected, y)
            
            # Model calibration
            if self.config['models']['confidence_calibration']:
                self.calibrate_confidence(X_train_scaled, y_train, metadata)
            
            # Performance evaluation
            performance = self.evaluate_model_performance(X_test_scaled, y_test, cv_scores)
            
            # Model kaydetme
            self.save_models()
            self.save_performance_metrics(performance)
            
            logger.info(f"Advanced training tamamlandı. Doğruluk: {performance['accuracy']:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Advanced training hatası: {e}")
            return False
    
    def apply_feature_selection(self, X: np.array, y: np.array) -> np.array:
        """Özellik seçimi uygula"""
        try:
            if self.feature_selector:
                X_selected = self.feature_selector.fit_transform(X, y)
                logger.info(f"Feature selection applied: {X.shape[1]} -> {X_selected.shape[1]} features")
                return X_selected
            return X
        except Exception as e:
            logger.warning(f"Feature selection hatası: {e}")
            return X
    
    def optimize_hyperparameters(self, X: np.array, y: np.array):
        """Hyperparameter optimizasyonu"""
        try:
            param_grid = {
                'gbm__learning_rate': [0.05, 0.1, 0.15],
                'gbm__n_estimators': [100, 200, 300],
                'rf__n_estimators': [100, 150, 200],
                'rf__max_depth': [5, 8, 10]
            }
            
            grid_search = GridSearchCV(
                self.models['ensemble_classifier'],
                param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X, y)
            self.models['ensemble_classifier'] = grid_search.best_estimator_
            
            logger.info(f"Hyperparameter optimization completed. Best score: {grid_search.best_score_:.3f}")
            
        except Exception as e:
            logger.warning(f"Hyperparameter optimization hatası: {e}")
    
    def cross_validate_model(self, X: np.array, y: np.array) -> Dict:
        """Cross-validation ile model validasyonu"""
        try:
            cv = StratifiedKFold(
                n_splits=self.config['training']['cv_folds'],
                shuffle=True,
                random_state=42
            )
            
            scores = cross_val_score(
                self.models['ensemble_classifier'],
                X, y,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1
            )
            
            return {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'fold_scores': scores.tolist()
            }
            
        except Exception as e:
            logger.warning(f"Cross-validation hatası: {e}")
            return {'mean_score': 0, 'std_score': 0, 'fold_scores': []}
    
    def predict_with_confidence(self, match_data: Dict) -> Dict[str, Any]:
        """Güvenilir tahmin sistemi"""
        try:
            # Özellik çıkarımı
            features = self.extract_advanced_features(match_data)
            
            if not self.models_trained():
                return self.advanced_fallback_prediction(match_data)
            
            # Feature selection ve scaling
            if self.feature_selector:
                features = self.feature_selector.transform(features)
            features_scaled = self.scalers['main'].transform(features)
            
            # Ensemble prediction
            result_probs = self.models['ensemble_classifier'].predict_proba(features_scaled)[0]
            result_classes = self.encoders['result'].classes_
            probabilities = dict(zip(result_classes, result_probs))
            
            # Confidence calibration
            calibrated_confidence = self.calibrate_prediction_confidence(
                features_scaled, probabilities, match_data
            )
            
            # Advanced predictions
            predicted_result = max(probabilities, key=probabilities.get)
            iy_prediction = self.predict_first_half_advanced(probabilities, calibrated_confidence)
            score_prediction = self.predict_score_advanced(features_scaled, predicted_result)
            trend_analysis = self.analyze_trend(features_scaled, match_data)
            
            # Risk assessment
            risk_factors = self.assess_prediction_risk(probabilities, match_data)
            
            prediction_result = {
                'result_prediction': predicted_result,
                'confidence': calibrated_confidence,
                'iy_prediction': iy_prediction,
                'score_prediction': score_prediction,
                'probabilities': {k: v * 100 for k, v in probabilities.items()},
                'trend_analysis': trend_analysis,
                'risk_factors': risk_factors,
                'ai_powered': True,
                'model_version': '3.0-advanced',
                'features_used': features.shape[1],
                'prediction_time': datetime.now().isoformat(),
                'certainty_index': self.calculate_certainty_index(probabilities, calibrated_confidence)
            }
            
            # Buffer'a kaydet
            self.prediction_buffer.append({
                'prediction': prediction_result,
                'match_data': match_data,
                'timestamp': datetime.now()
            })
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Advanced prediction hatası: {e}")
            return self.advanced_fallback_prediction(match_data)
    
    def calibrate_prediction_confidence(self, features: np.array, 
                                      probabilities: Dict, 
                                      match_data: Dict) -> float:
        """Tahmin güvenilirliğini kalibre et"""
        try:
            # Base confidence from probabilities
            max_prob = max(probabilities.values())
            base_confidence = max_prob * 100
            
            # Context-based adjustments
            context_factor = self.calculate_context_factor(match_data)
            consistency_factor = self.calculate_consistency_factor(probabilities)
            
            # Market efficiency adjustment
            market_factor = self.assess_market_efficiency(match_data.get('odds', {}))
            
            # Final calibrated confidence
            calibrated = base_confidence * context_factor * consistency_factor * market_factor
            
            return max(30.0, min(95.0, calibrated))
            
        except Exception as e:
            logger.debug(f"Confidence calibration hatası: {e}")
            return max(probabilities.values()) * 100
    
    def calculate_certainty_index(self, probabilities: Dict, confidence: float) -> float:
        """Tahmin kesinlik indeksi"""
        try:
            # Probability distribution analysis
            sorted_probs = sorted(probabilities.values(), reverse=True)
            certainty = 0.0
            
            if len(sorted_probs) >= 2:
                # Gap between top two probabilities
                gap_factor = (sorted_probs[0] - sorted_probs[1]) * 2
                
                # Overall distribution shape
                entropy = -sum(p * np.log(p + 1e-10) for p in probabilities.values())
                distribution_factor = 1.0 / (1.0 + entropy)
                
                certainty = (gap_factor + distribution_factor + confidence / 100) / 3
            
            return max(0.0, min(1.0, certainty))
            
        except:
            return confidence / 100
    
    def assess_prediction_risk(self, probabilities: Dict, match_data: Dict) -> Dict:
        """Tahmin risk değerlendirmesi"""
        risk_factors = {}
        
        try:
            # Probability distribution risk
            sorted_probs = sorted(probabilities.values(), reverse=True)
            if len(sorted_probs) >= 2:
                risk_factors['close_probabilities'] = sorted_probs[0] - sorted_probs[1] < 0.15
            
            # Context risk factors
            risk_factors['high_importance'] = match_data.get('context', {}).get('importance', 0) > 0.8
            risk_factors['unstable_form'] = self.assess_form_stability(match_data)
            risk_factors['market_inefficiency'] = self.detect_market_anomalies(match_data.get('odds', {}))
            
            # Overall risk score
            risk_score = sum(1 for factor in risk_factors.values() if factor)
            risk_factors['overall_risk'] = min(1.0, risk_score / len(risk_factors))
            
        except Exception as e:
            logger.debug(f"Risk assessment hatası: {e}")
        
        return risk_factors
    
    def real_time_adaptation(self):
        """Gerçek zamanlı model adaptasyonu"""
        try:
            if len(self.prediction_buffer) < self.config['adaptation']['drift_detection_window']:
                return
            
            # Performance drift detection
            recent_performance = self.calculate_recent_performance()
            
            if (self.last_accuracy - recent_performance) > self.adaptation_threshold:
                logger.warning(f"Performance drift detected: {self.last_accuracy:.3f} -> {recent_performance:.3f}")
                self.adaptive_retraining()
            
            # Incremental learning
            self.incremental_learning()
            
        except Exception as e:
            logger.error(f"Real-time adaptation hatası: {e}")
    
    def adaptive_retraining(self):
        """Adaptif yeniden eğitim"""
        try:
            logger.info("Adaptive retraining başlıyor...")
            
            # Mevcut model performansını kaydet
            current_performance = self.get_model_performance()
            
            # Yeni verilerle eğitim
            success = self.train_advanced_models(force_retrain=True)
            
            if success:
                new_performance = self.get_model_performance()
                improvement = new_performance['latest_accuracy'] - current_performance['latest_accuracy']
                
                logger.info(f"Adaptive retraining tamamlandı. İyileşme: {improvement:.3f}")
                
                # Eğer iyileşme yoksa rollback
                if improvement < 0:
                    self.rollback_model()
            else:
                logger.error("Adaptive retraining başarısız")
                
        except Exception as e:
            logger.error(f"Adaptive retraining hatası: {e}")
    
    # Diğer yardımcı metodlar...
    def models_trained(self) -> bool:
        return all(hasattr(model, 'predict') for model in self.models.values())
    
    def save_models(self):
        """Modelleri kaydet"""
        try:
            for name, model in self.models.items():
                joblib.dump(model, os.path.join(self.model_path, f"{name}.pkl"))
            
            for name, scaler in self.scalers.items():
                joblib.dump(scaler, os.path.join(self.model_path, f"scaler_{name}.pkl"))
            
            for name, encoder in self.encoders.items():
                joblib.dump(encoder, os.path.join(self.model_path, f"encoder_{name}.pkl"))
            
            if self.feature_selector:
                joblib.dump(self.feature_selector, os.path.join(self.model_path, "feature_selector.pkl"))
            
            # Metadata kaydet
            metadata = {
                'performance_history': self.performance_history,
                'feature_importance': self.feature_importance_history,
                'optimal_features': self.optimal_feature_set,
                'last_training': datetime.now().isoformat(),
                'config': self.config
            }
            
            with open(os.path.join(self.model_path, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("Advanced modeller kaydedildi")
            
        except Exception as e:
            logger.error(f"Model kaydetme hatası: {e}")
    
    def load_models(self):
        """Modelleri yükle"""
        try:
            for name in self.models.keys():
                model_file = os.path.join(self.model_path, f"{name}.pkl")
                if os.path.exists(model_file):
                    self.models[name] = joblib.load(model_file)
            
            for name in self.scalers.keys():
                scaler_file = os.path.join(self.model_path, f"scaler_{name}.pkl")
                if os.path.exists(scaler_file):
                    self.scalers[name] = joblib.load(scaler_file)
            
            for name in self.encoders.keys():
                encoder_file = os.path.join(self.model_path, f"encoder_{name}.pkl")
                if os.path.exists(encoder_file):
                    self.encoders[name] = joblib.load(encoder_file)
            
            selector_file = os.path.join(self.model_path, "feature_selector.pkl")
            if os.path.exists(selector_file):
                self.feature_selector = joblib.load(selector_file)
            
            # Metadata yükle
            metadata_file = os.path.join(self.model_path, "metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.performance_history = metadata.get('performance_history', [])
                    self.feature_importance_history = metadata.get('feature_importance', [])
                    self.optimal_feature_set = metadata.get('optimal_features', 0)
            
            logger.info("Advanced modeller yüklendi")
            
        except Exception as e:
            logger.warning(f"Model yükleme hatası: {e}")
    
    def get_detailed_performance(self) -> Dict:
        """Detaylı performans raporu"""
        if not self.performance_history:
            return {"status": "no_training_data"}
        
        latest = self.performance_history[-1] if self.performance_history else {}
        
        return {
            "current_accuracy": latest.get('accuracy', 0),
            "training_samples": latest.get('training_samples', 0),
            "feature_count": self.optimal_feature_set,
            "cross_validation_score": latest.get('cv_mean', 0),
            "model_stability": latest.get('stability_index', 0),
            "last_training": latest.get('timestamp', ''),
            "adaptation_status": "active" if len(self.prediction_buffer) > 50 else "learning",
            "certainty_trend": self.calculate_certainty_trend(),
            "risk_profile": self.assess_risk_profile()
        }

# Kullanım örneği
if __name__ == "__main__":
    ai_system = EnhancedSuperLearningAI()
    
    # Örnek maç verisi
    sample_match = {
        'home_stats': {
            'position': 3, 'points': 45, 'matches_played': 20, 'wins': 13,
            'goals_for': 35, 'goals_against': 15, 'recent_form': ['G','G','B','G','M'],
            'xG_for': 32.5, 'xG_against': 16.2
        },
        'away_stats': {
            'position': 8, 'points': 28, 'matches_played': 20, 'wins': 7,
            'goals_for': 22, 'goals_against': 25, 'recent_form': ['B','M','G','M','B'],
            'xG_for': 24.1, 'xG_against': 26.8
        },
        'odds': {'1': 1.85, 'X': 3.40, '2': 4.20},
        'context': {
            'importance': 0.7,
            'pressure': 0.6,
            'motivation': 0.8,
            'referee_stats': {'home_win_rate': 0.55, 'avg_cards': 3.2}
        }
    }
    
    prediction = ai_system.predict_with_confidence(sample_match)
    print("Advanced AI Prediction:", json.dumps(prediction, indent=2, ensure_ascii=False))
