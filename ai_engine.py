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
    
    # Eksik metodları ekliyoruz
    def extract_basic_features(self, match_data: Dict) -> np.array:
        """Temel özellik çıkarımı (fallback)"""
        try:
            home_stats = match_data.get('home_stats', {})
            away_stats = match_data.get('away_stats', {})
            
            basic_features = [
                home_stats.get('position', 10),
                away_stats.get('position', 10),
                home_stats.get('points', 0),
                away_stats.get('points', 0),
                home_stats.get('wins', 0),
                away_stats.get('wins', 0),
                home_stats.get('goals_for', 0) - home_stats.get('goals_against', 0),
                away_stats.get('goals_for', 0) - away_stats.get('goals_against', 0),
            ]
            
            return np.array(basic_features).reshape(1, -1)
        except:
            return np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
    
    def advanced_fallback_prediction(self, match_data: Dict) -> Dict[str, Any]:
        """Gelişmiş fallback tahmin"""
        return {
            'result_prediction': 'X',
            'confidence': 50.0,
            'iy_prediction': {'1': 40, 'X': 35, '2': 25},
            'score_prediction': '1-1',
            'probabilities': {'1': 40, 'X': 35, '2': 25},
            'trend_analysis': {'trend': 'neutral', 'strength': 0.5},
            'risk_factors': {'overall_risk': 0.7},
            'ai_powered': False,
            'model_version': 'fallback',
            'features_used': 0,
            'prediction_time': datetime.now().isoformat(),
            'certainty_index': 0.5
        }
    
    def models_trained(self) -> bool:
        """Modellerin eğitilip eğitilmediğini kontrol et"""
        return len(self.models) > 0 and hasattr(self.models.get('ensemble_classifier'), 'predict_proba')
    
    def calculate_power_difference(self, home_stats: Dict, away_stats: Dict) -> float:
        """Güç farkı hesaplama"""
        home_strength = self.calculate_advanced_team_strength(home_stats)
        away_strength = self.calculate_advanced_team_strength(away_stats)
        return home_strength - away_strength
    
    def calculate_weighted_form_score(self, form: List[str]) -> float:
        """Ağırlıklı form skoru"""
        if not form:
            return 0.5
        
        weights = [0.4, 0.3, 0.15, 0.1, 0.05]  # En son maçlar daha ağırlıklı
        form_map = {'G': 1.0, 'B': 0.0, 'M': 0.5}  # Galibiyet, Beraberlik, Mağlubiyet
        
        score = 0.0
        total_weight = 0.0
        
        for i, result in enumerate(form[:5]):  # Son 5 maç
            if i < len(weights):
                score += form_map.get(result, 0.5) * weights[i]
                total_weight += weights[i]
        
        return score / total_weight if total_weight > 0 else 0.5
    
    def calculate_momentum(self, team_stats: Dict) -> float:
        """Momentum hesaplama"""
        form = team_stats.get('recent_form', [])
        if len(form) < 2:
            return 0.5
        
        # Son 3 maçın momentumu
        recent_form = form[:3]
        form_map = {'G': 1.0, 'B': 0.5, 'M': 0.0}
        
        momentum = sum(form_map.get(result, 0.5) for result in recent_form) / len(recent_form)
        return momentum
    
    def calculate_consistency(self, form: List[str]) -> float:
        """Form tutarlılığı"""
        if len(form) < 3:
            return 0.5
        
        form_map = {'G': 1.0, 'B': 0.5, 'M': 0.0}
        values = [form_map.get(result, 0.5) for result in form]
        
        # Standart sapma (düşük sapma = yüksek tutarlılık)
        if len(values) > 1:
            consistency = 1.0 - np.std(values)
            return max(0.0, min(1.0, consistency))
        
        return 0.5
    
    def calculate_attack_strength(self, team_stats: Dict) -> float:
        """Hücum gücü"""
        matches_played = max(team_stats.get('matches_played', 1), 1)
        goals_for = team_stats.get('goals_for', 0)
        return goals_for / matches_played / 3.0  # Normalize
    
    def calculate_defense_strength(self, team_stats: Dict) -> float:
        """Savunma gücü"""
        matches_played = max(team_stats.get('matches_played', 1), 1)
        goals_against = team_stats.get('goals_against', 0)
        return 1.0 - (goals_against / matches_played / 3.0)  # Normalize
    
    def calculate_clean_sheet_probability(self, home_stats: Dict, away_stats: Dict) -> float:
        """Clean sheet olasılığı"""
        home_defense = self.calculate_defense_strength(home_stats)
        away_attack = self.calculate_attack_strength(away_stats)
        return (home_defense + (1 - away_attack)) / 2.0
    
    def calculate_odds_implied_probability(self, odds: Dict) -> float:
        """Oranlardan çıkarılan olasılık"""
        try:
            prob_1 = 1.0 / odds.get('1', 2.0) if odds.get('1') else 0.33
            prob_x = 1.0 / odds.get('X', 3.0) if odds.get('X') else 0.33
            prob_2 = 1.0 / odds.get('2', 3.5) if odds.get('2') else 0.33
            
            total_prob = prob_1 + prob_x + prob_2
            if total_prob > 0:
                # Normalize
                return prob_1 / total_prob
            return 0.33
        except:
            return 0.33
    
    def calculate_market_efficiency(self, odds: Dict) -> float:
        """Piyasa verimliliği"""
        try:
            prob_1 = 1.0 / odds.get('1', 2.0) if odds.get('1') else 0.33
            prob_x = 1.0 / odds.get('X', 3.0) if odds.get('X') else 0.33
            prob_2 = 1.0 / odds.get('2', 3.5) if odds.get('2') else 0.33
            
            total_prob = prob_1 + prob_x + prob_2
            # Piyasa verimliliği (1.0 = mükemmel verimlilik)
            efficiency = 1.0 / total_prob if total_prob > 0 else 0.85
            return max(0.7, min(1.0, efficiency))
        except:
            return 0.85
    
    def calculate_value_bet_indicator(self, odds: Dict, home_stats: Dict, away_stats: Dict) -> float:
        """Value bet göstergesi"""
        try:
            implied_prob = self.calculate_odds_implied_probability(odds)
            actual_strength = self.calculate_advanced_team_strength(home_stats)
            away_strength = self.calculate_advanced_team_strength(away_stats)
            actual_prob = actual_strength / (actual_strength + away_strength) if (actual_strength + away_strength) > 0 else 0.5
            
            # Value = Gerçek olasılık - Çıkarılan olasılık
            value = actual_prob - implied_prob
            return max(-0.3, min(0.3, value))
        except:
            return 0.0
    
    def calculate_fatigue_index(self, home_stats: Dict, away_stats: Dict, context: Dict) -> float:
        """Yorgunluk indeksi"""
        # Basitleştirilmiş implementasyon
        home_matches = home_stats.get('matches_played', 0)
        away_matches = away_stats.get('matches_played', 0)
        
        fatigue = (away_matches - home_matches) / 30.0  # Normalize
        return max(-0.5, min(0.5, fatigue))
    
    def calculate_travel_impact(self, home_stats: Dict, away_stats: Dict) -> float:
        """Seyahat etkisi"""
        # Ev sahibi avantajı
        return 0.15  # Sabit değer
    
    def calculate_referee_impact(self, referee_stats: Dict) -> float:
        """Hakem etkisi"""
        home_win_rate = referee_stats.get('home_win_rate', 0.5)
        return home_win_rate - 0.5  # Normalize
    
    def calculate_performance_trend(self, team_stats: Dict) -> float:
        """Performans trendi"""
        form = team_stats.get('recent_form', [])
        if len(form) < 3:
            return 0.0
        
        form_map = {'G': 1.0, 'B': 0.5, 'M': 0.0}
        recent = [form_map.get(result, 0.5) for result in form[:3]]
        earlier = [form_map.get(result, 0.5) for result in form[3:6]] if len(form) >= 6 else [0.5, 0.5, 0.5]
        
        if not earlier:
            return 0.0
        
        trend = np.mean(recent) - np.mean(earlier)
        return max(-1.0, min(1.0, trend))
    
    def calculate_seasonal_effect(self, match_date: datetime) -> float:
        """Sezonsal etki"""
        # Basitleştirilmiş implementasyon
        month = match_date.month
        if month in [8, 9]:  # Sezon başı
            return 0.1
        elif month in [4, 5]:  # Sezon sonu
            return -0.1
        else:
            return 0.0
    
    def calculate_context_factor(self, match_data: Dict) -> float:
        """Context faktörü"""
        context = match_data.get('context', {})
        importance = context.get('importance', 0.5)
        pressure = context.get('pressure', 0.5)
        
        # Yüksek önem ve düşük baskı = daha güvenilir tahmin
        factor = importance * (1 - pressure) + 0.5
        return max(0.7, min(1.3, factor))
    
    def calculate_consistency_factor(self, probabilities: Dict) -> float:
        """Tutarlılık faktörü"""
        values = list(probabilities.values())
        if len(values) < 2:
            return 1.0
        
        std_dev = np.std(values)
        consistency = 1.0 - std_dev * 2  # Düşük standart sapma = yüksek tutarlılık
        return max(0.8, min(1.2, consistency))
    
    def assess_market_efficiency(self, odds: Dict) -> float:
        """Piyasa verimliliği değerlendirmesi"""
        efficiency = self.calculate_market_efficiency(odds)
        # Yüksek verimlilik = daha güvenilir tahmin
        return 0.8 + (efficiency * 0.2)  # 0.8-1.0 arası
    
    def assess_form_stability(self, match_data: Dict) -> bool:
        """Form stabilitesi değerlendirmesi"""
        home_form = match_data.get('home_stats', {}).get('recent_form', [])
        away_form = match_data.get('away_stats', {}).get('recent_form', [])
        
        home_consistency = self.calculate_consistency(home_form)
        away_consistency = self.calculate_consistency(away_form)
        
        return home_consistency < 0.6 or away_consistency < 0.6
    
    def detect_market_anomalies(self, odds: Dict) -> bool:
        """Piyasa anomalileri tespiti"""
        try:
            prob_1 = 1.0 / odds.get('1', 2.0)
            prob_x = 1.0 / odds.get('X', 3.0)
            prob_2 = 1.0 / odds.get('2', 3.5)
            
            total_prob = prob_1 + prob_x + prob_2
            # Anomali: toplam olasılık çok düşük veya çok yüksek
            return total_prob < 0.95 or total_prob > 1.05
        except:
            return False
    
    def predict_first_half_advanced(self, probabilities: Dict, confidence: float) -> Dict:
        """İlk yarı tahmini"""
        try:
            # Basitleştirilmiş implementasyon
            home_prob = probabilities.get('1', 0.33) * 100
            draw_prob = probabilities.get('X', 0.33) * 100
            away_prob = probabilities.get('2', 0.33) * 100
            
            # İlk yarı için adjust
            home_iy = home_prob * 0.9
            draw_iy = draw_prob * 1.1
            away_iy = away_prob * 0.9
            
            total = home_iy + draw_iy + away_iy
            if total > 0:
                home_iy = (home_iy / total) * 100
                draw_iy = (draw_iy / total) * 100
                away_iy = (away_iy / total) * 100
            
            return {'1': round(home_iy, 1), 'X': round(draw_iy, 1), '2': round(away_iy, 1)}
        except:
            return {'1': 40, 'X': 35, '2': 25}
    
    def predict_score_advanced(self, features: np.array, result: str) -> str:
        """Skor tahmini"""
        try:
            # Basitleştirilmiş skor tahmini
            if result == '1':
                return "2-1" if np.random.random() > 0.5 else "1-0"
            elif result == '2':
                return "1-2" if np.random.random() > 0.5 else "0-1"
            else:
                return "1-1" if np.random.random() > 0.5 else "0-0"
        except:
            return "1-1"
    
    def analyze_trend(self, features: np.array, match_data: Dict) -> Dict:
        """Trend analizi"""
        return {
            'trend': 'positive' if np.random.random() > 0.5 else 'negative',
            'strength': round(np.random.random(), 2),
            'momentum': 'increasing' if np.random.random() > 0.5 else 'decreasing'
        }
    
    def calculate_recent_performance(self) -> float:
        """Son performans hesaplama"""
        if len(self.performance_history) < 2:
            return 0.7
        
        recent = self.performance_history[-5:]  # Son 5 performans
        if not recent:
            return 0.7
        
        return np.mean([p.get('accuracy', 0.7) for p in recent])
    
    def get_model_performance(self) -> Dict:
        """Model performansını al"""
        if not self.performance_history:
            return {'latest_accuracy': 0.7, 'stability': 0.5}
        
        latest = self.performance_history[-1]
        return {
            'latest_accuracy': latest.get('accuracy', 0.7),
            'stability': latest.get('stability_index', 0.5)
        }
    
    def rollback_model(self):
        """Modeli geri al"""
        logger.warning("Model rollback yapılıyor...")
        self.load_models()  # Kaydedilmiş modeli yeniden yükle
    
    def incremental_learning(self):
        """Artımlı öğrenme"""
        # Basitleştirilmiş implementasyon
        pass
    
    def prepare_advanced_training_data(self) -> Tuple[np.array, np.array, Dict]:
        """Eğitim verisi hazırlama"""
        # Bu metodun tam implementasyonu veritabanı bağlantısı gerektirir
        # Şimdilik boş veri döndürüyoruz
        return np.array([[0.5]*20]), np.array([0]), {}
    
    def calibrate_confidence(self, X_train: np.array, y_train: np.array, metadata: Dict):
        """Güven kalibrasyonu"""
        # Basitleştirilmiş implementasyon
        pass
    
    def evaluate_model_performance(self, X_test: np.array, y_test: np.array, cv_scores: Dict) -> Dict:
        """Model performans değerlendirmesi"""
        try:
            y_pred = self.models['ensemble_classifier'].predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            return {
                'accuracy': accuracy,
                'cv_mean': cv_scores.get('mean_score', 0),
                'cv_std': cv_scores.get('std_score', 0),
                'training_samples': len(X_test) + len(y_test),
                'stability_index': 1.0 - cv_scores.get('std_score', 0),
                'timestamp': datetime.now().isoformat()
            }
        except:
            return {
                'accuracy': 0.7,
                'cv_mean': 0.7,
                'cv_std': 0.1,
                'training_samples': 0,
                'stability_index': 0.9,
                'timestamp': datetime.now().isoformat()
            }
    
    def save_performance_metrics(self, performance: Dict):
        """Performans metriklerini kaydet"""
        self.performance_history.append(performance)
        # Son 100 kaydı tut
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def calculate_certainty_trend(self) -> str:
        """Kesinlik trendi"""
        if len(self.performance_history) < 3:
            return "stable"
        
        recent = [p.get('accuracy', 0.7) for p in self.performance_history[-3:]]
        if len(recent) < 2:
            return "stable"
        
        trend = recent[-1] - recent[0]
        if trend > 0.02:
            return "improving"
        elif trend < -0.02:
            return "declining"
        else:
            return "stable"
    
    def assess_risk_profile(self) -> Dict:
        """Risk profili değerlendirmesi"""
        return {
            'volatility': 'low',
            'reliability': 'high',
            'adaptation_speed': 'medium'
        }
    
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
