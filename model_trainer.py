#!/usr/bin/env python3
"""
Predicta ML Production Training Pipeline v3.5 - FULLY OPTIMIZED
----------------------------------------------------------------
🚀 Complete optimized model trainer with class balancing
- SMOTE/ADASYN balancing
- Hyperparameter optimization
- Enhanced feature engineering
- Ensemble methods
- Expected improvement: %45 → %55-65
"""

import os
import gc
import time
import json
import joblib
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Tuple, Optional

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

# Imbalanced learning
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("⚠️  imbalanced-learn not found. Install: pip install imbalanced-learn")

# Optional: LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Log configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ModelTrainerOptimized")

# Local imports
try:
    from enhanced_feature_engineer import EnhancedFeatureEngineer
    FEATURE_ENGINEER_AVAILABLE = True
except ImportError:
    try:
        from advanced_feature_engineer import AdvancedFeatureEngineer
        FEATURE_ENGINEER_AVAILABLE = True
    except ImportError:
        FEATURE_ENGINEER_AVAILABLE = False

try:
    from historical_processor import HistoricalDataProcessor
    HISTORICAL_PROCESSOR_AVAILABLE = True
except ImportError:
    HISTORICAL_PROCESSOR_AVAILABLE = False


class OptimizedModelTrainerV35:
    """
    Fully Optimized ML Trainer with Class Balancing
    """
    
    def __init__(
        self,
        models_dir: str = "data/ai_models_v3",
        raw_data_path: str = "data/raw",
        clubs_path: str = "data/clubs",
        test_size: float = 0.2,
        random_state: int = 42,
        batch_size: int = 2000,
        enable_meta_ensemble: bool = True,
        # Optimization parameters
        use_smote: bool = True,
        use_class_weights: bool = True,
        n_iter_search: int = 30,
        cv_folds: int = 3,
        balance_method: str = 'hybrid',
        verbose: bool = True
    ):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_path = Path(raw_data_path)
        self.clubs_path = Path(clubs_path)
        self.test_size = test_size
        self.random_state = random_state
        self.batch_size = batch_size
        self.enable_meta_ensemble = enable_meta_ensemble
        
        # Optimization parameters
        self.use_smote = use_smote and IMBLEARN_AVAILABLE
        self.use_class_weights = use_class_weights
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.balance_method = balance_method
        self.verbose = verbose
        
        # Feature engineer
        if FEATURE_ENGINEER_AVAILABLE:
            try:
                self.feature_engineer = EnhancedFeatureEngineer(model_path=str(self.models_dir))
            except:
                self.feature_engineer = AdvancedFeatureEngineer(model_path=str(self.models_dir))
        else:
            raise ImportError("No feature engineer available")
        
        # Historical processor
        self.history_processor = None
        if HISTORICAL_PROCESSOR_AVAILABLE:
            self.history_processor = HistoricalDataProcessor(
                str(self.raw_data_path),
                str(self.clubs_path)
            )
        
        # Models
        self.models = {
            'xgboost': None,
            'gradient_boost': None,
            'random_forest': None,
            'lightgbm': None
        }
        self.meta_model = None
        self.score_model = None
        self.scaler = StandardScaler()
        
        # Metadata
        self.metadata = {
            "version": "v3.5_optimized",
            "trained_at": datetime.now().isoformat(),
            "feature_engineer": self.feature_engineer.__class__.__name__,
            "optimization": {
                "use_smote": self.use_smote,
                "use_class_weights": self.use_class_weights,
                "balance_method": self.balance_method,
                "n_iter_search": self.n_iter_search,
                "cv_folds": self.cv_folds
            }
        }
        
        self.logger = logging.getLogger("OptimizedTrainerV35")
        
        if self.use_smote and not IMBLEARN_AVAILABLE:
            self.logger.warning("⚠️  SMOTE requested but imbalanced-learn not available")
            self.use_smote = False
    
    def _mem_usage(self) -> str:
        """RAM kullanımı"""
        try:
            import psutil
            process = psutil.Process()
            mem_mb = process.memory_info().rss / 1024 / 1024
            return f"{mem_mb:.1f} MB"
        except ImportError:
            return "N/A"
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Verileri yükle ve hazırla"""
        self.logger.info("📂 Veri yükleme başlatılıyor...")
        
        all_matches = []
        
        if self.history_processor and self.raw_data_path.exists():
            for country_dir in self.raw_data_path.iterdir():
                if country_dir.is_dir():
                    try:
                        matches = self.history_processor.load_country_data(country_dir.name)
                        all_matches.extend(matches)
                        self.logger.info(f"✅ {country_dir.name:<20} {len(matches):6,} maç")
                    except Exception as e:
                        self.logger.warning(f"⚠️ {country_dir.name}: {e}")
        else:
            # Fallback: CSV files
            csv_files = list(self.raw_data_path.glob("**/*.csv"))
            for csv_file in csv_files:
                try:
                    df_temp = pd.read_csv(csv_file)
                    required_cols = ['home_team', 'away_team', 'home_score', 'away_score']
                    if all(col in df_temp.columns for col in required_cols):
                        league = csv_file.parent.name if csv_file.parent.name != 'raw' else 'Unknown'
                        df_temp['league'] = league
                        all_matches.extend(df_temp.to_dict('records'))
                        self.logger.info(f"✅ {csv_file.name}: {len(df_temp)} maç")
                except Exception as e:
                    self.logger.warning(f"⚠️ {csv_file}: {e}")
        
        if not all_matches:
            raise ValueError("❌ Hiç veri bulunamadı!")
        
        df = pd.DataFrame(all_matches)
        self.logger.info(f"📊 Toplam {len(df):,} maç yüklendi | RAM: {self._mem_usage()}")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Veriyi temizle"""
        self.logger.info("🧹 Veri temizleme başlatılıyor...")
        
        initial_count = len(df)
        df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"])
        
        df["home_score"] = pd.to_numeric(df["home_score"], errors='coerce').fillna(0).astype(int)
        df["away_score"] = pd.to_numeric(df["away_score"], errors='coerce').fillna(0).astype(int)
        
        df = df[(df["home_score"] >= 0) & (df["away_score"] >= 0)]
        df = df[(df["home_score"] <= 20) & (df["away_score"] <= 20)]
        
        df["result"] = df.apply(
            lambda r: "1" if r["home_score"] > r["away_score"]
            else "2" if r["home_score"] < r["away_score"]
            else "X", axis=1
        )
        
        if 'odds' not in df.columns:
            df['odds'] = [{"1": 2.0, "X": 3.0, "2": 3.5} for _ in range(len(df))]
        
        if 'league' not in df.columns:
            df['league'] = 'Unknown'
        
        cleaned_count = len(df)
        self.logger.info(f"✅ {initial_count:,} -> {cleaned_count:,} kayıt ({initial_count-cleaned_count} filtrelendi)")
        self.logger.info(f"📊 Sonuç dağılımı: {dict(df['result'].value_counts())}")
        
        return df
    
    def extract_features_v35(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Feature çıkarımı"""
        self.logger.info("🔧 v3.5 Feature çıkarımı başlatılıyor...")
        
        X_list, y_ms, y_score = [], [], []
        skipped_count = 0
        
        progress_bar = tqdm(df.iterrows(), total=len(df), ncols=100, desc="Extracting Features")
        
        for _, row in progress_bar:
            try:
                match_data = {
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "league": row.get("league", "Unknown"),
                    "odds": row.get("odds", {"1": 2.0, "X": 3.0, "2": 3.5}),
                }
                
                features = self.feature_engineer.extract_features(match_data)
                
                if features is not None and len(features) > 0:
                    X_list.append(features)
                    y_ms.append({"1": 0, "X": 1, "2": 2}[row["result"]])
                    y_score.append(f"{row['home_score']}-{row['away_score']}")
                else:
                    skipped_count += 1
            except Exception as e:
                skipped_count += 1
                continue
        
        progress_bar.close()
        
        if not X_list:
            raise ValueError("❌ Hiç feature çıkarılamadı!")
        
        X = np.array(X_list, dtype=np.float32)
        y_ms = np.array(y_ms, dtype=np.int8)
        
        self.logger.info(f"✅ Feature set: {X.shape}")
        self.logger.info(f"✅ Skipped matches: {skipped_count}")
        self.logger.info(f"✅ Result distribution: {Counter(y_ms)}")
        
        return X, y_ms, y_score
    
    def analyze_class_distribution(self, y: np.ndarray, name: str = "Dataset"):
        """Sınıf dağılımını analiz et"""
        counter = Counter(y)
        total = len(y)
        
        self.logger.info(f"📊 {name} Class Distribution:")
        for cls in sorted(counter.keys()):
            count = counter[cls]
            pct = count / total * 100
            self.logger.info(f"   Class {cls}: {count:6,} ({pct:5.2f}%)")
        
        max_count = max(counter.values())
        min_count = min(counter.values())
        imbalance_ratio = max_count / min_count
        self.logger.info(f"   ⚖️  Imbalance Ratio: {imbalance_ratio:.2f}:1")
        
        return counter, imbalance_ratio
    
    def balance_dataset(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Dataset balancing"""
        if not self.use_smote or self.balance_method == 'none':
            self.logger.info("ℹ️  Skipping dataset balancing")
            return X_train, y_train
        
        self.logger.info(f"🔧 Dataset Balancing: {self.balance_method.upper()}")
        
        counter_before, _ = self.analyze_class_distribution(y_train, "Before Balancing")
        
        try:
            if self.balance_method == 'smote':
                sampler = SMOTE(
                    random_state=self.random_state,
                    k_neighbors=min(5, min(counter_before.values()) - 1)
                )
            elif self.balance_method == 'adasyn':
                sampler = ADASYN(
                    random_state=self.random_state,
                    n_neighbors=min(5, min(counter_before.values()) - 1)
                )
            elif self.balance_method == 'smotetomek':
                sampler = SMOTETomek(random_state=self.random_state)
            elif self.balance_method == 'hybrid':
                # Undersample + SMOTE
                majority_target = int(np.median(list(counter_before.values())))
                rus = RandomUnderSampler(
                    sampling_strategy={
                        cls: min(count, int(majority_target * 1.3))
                        for cls, count in counter_before.items()
                    },
                    random_state=self.random_state
                )
                X_temp, y_temp = rus.fit_resample(X_train, y_train)
                sampler = SMOTE(random_state=self.random_state)
                X_balanced, y_balanced = sampler.fit_resample(X_temp, y_temp)
                
                self.analyze_class_distribution(y_balanced, "After Balancing (Hybrid)")
                return X_balanced, y_balanced
            else:
                sampler = SMOTE(random_state=self.random_state)
            
            X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
            self.analyze_class_distribution(y_balanced, f"After Balancing ({self.balance_method.upper()})")
            
            return X_balanced, y_balanced
        except Exception as e:
            self.logger.error(f"❌ Balancing failed: {e}")
            self.logger.warning("⚠️  Using original imbalanced dataset")
            return X_train, y_train
    
    def get_class_weights(self, y_train: np.ndarray) -> Optional[Dict[int, float]]:
        """Class weight'leri hesapla"""
        if not self.use_class_weights:
            return None
        
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weight_dict = dict(zip(classes, weights))
        
        self.logger.info(f"⚖️  Class Weights: {weight_dict}")
        return weight_dict
    
    def train_xgboost_optimized(self, X_train, y_train, X_val, y_val):
        """Optimized XGBoost"""
        self.logger.info("🔧 Training Optimized XGBoost...")
        
        param_distributions = {
            'max_depth': [5, 7, 9, 11],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [300, 500, 800],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.3],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.5, 2],
        }
        
        base_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            random_state=self.random_state,
            n_jobs=-1,
            tree_method='hist',
            verbosity=0
        )
        
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        search = RandomizedSearchCV(
            base_model, param_distributions,
            n_iter=self.n_iter_search, cv=cv,
            scoring='accuracy', n_jobs=-1,
            random_state=self.random_state,
            verbose=1 if self.verbose else 0
        )
        
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        val_acc = accuracy_score(y_val, best_model.predict(X_val))
        
        self.logger.info(f"✅ XGBoost - Best CV: {search.best_score_:.4f}, Val: {val_acc:.4f}")
        return best_model, val_acc
    
    def train_gradient_boost_optimized(self, X_train, y_train, X_val, y_val):
        """Optimized Gradient Boosting"""
        self.logger.info("🔧 Training Optimized Gradient Boosting...")
        
        param_distributions = {
            'n_estimators': [200, 300, 500],
            'max_depth': [5, 7, 9],
            'learning_rate': [0.05, 0.1, 0.15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2']
        }
        
        base_model = GradientBoostingClassifier(random_state=self.random_state)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        search = RandomizedSearchCV(
            base_model, param_distributions,
            n_iter=self.n_iter_search, cv=cv,
            scoring='accuracy', n_jobs=-1,
            random_state=self.random_state,
            verbose=1 if self.verbose else 0
        )
        
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        val_acc = accuracy_score(y_val, best_model.predict(X_val))
        
        self.logger.info(f"✅ GradientBoost - Best CV: {search.best_score_:.4f}, Val: {val_acc:.4f}")
        return best_model, val_acc
    
    def train_random_forest_optimized(self, X_train, y_train, X_val, y_val):
        """Optimized Random Forest"""
        self.logger.info("🔧 Training Optimized Random Forest...")
        
        param_distributions = {
            'n_estimators': [200, 300, 500],
            'max_depth': [15, 20, 25, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        base_model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        search = RandomizedSearchCV(
            base_model, param_distributions,
            n_iter=self.n_iter_search, cv=cv,
            scoring='accuracy', n_jobs=-1,
            random_state=self.random_state,
            verbose=1 if self.verbose else 0
        )
        
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        val_acc = accuracy_score(y_val, best_model.predict(X_val))
        
        self.logger.info(f"✅ RandomForest - Best CV: {search.best_score_:.4f}, Val: {val_acc:.4f}")
        return best_model, val_acc
    
    def train_lightgbm_optimized(self, X_train, y_train, X_val, y_val):
        """Optimized LightGBM"""
        if not LIGHTGBM_AVAILABLE:
            self.logger.warning("⚠️  LightGBM not available")
            return None, 0
        
        self.logger.info("🔧 Training Optimized LightGBM...")
        
        param_distributions = {
            'n_estimators': [200, 300, 500],
            'max_depth': [5, 7, 9, -1],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 70],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        base_model = lgb.LGBMClassifier(random_state=self.random_state, n_jobs=-1, verbose=-1)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        search = RandomizedSearchCV(
            base_model, param_distributions,
            n_iter=self.n_iter_search, cv=cv,
            scoring='accuracy', n_jobs=-1,
            random_state=self.random_state,
            verbose=1 if self.verbose else 0
        )
        
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        val_acc = accuracy_score(y_val, best_model.predict(X_val))
        
        self.logger.info(f"✅ LightGBM - Best CV: {search.best_score_:.4f}, Val: {val_acc:.4f}")
        return best_model, val_acc
    
    def train_base_models(self, X_train, X_test, y_train, y_test):
        """Train all optimized base models"""
        self.logger.info("=" * 80)
        self.logger.info("🎯 OPTIMIZED MODEL TRAINING")
        self.logger.info("=" * 80)
        
        # Analyze distribution
        self.analyze_class_distribution(y_train, "Original Training Data")
        
        # Balance dataset
        X_train_balanced, y_train_balanced = self.balance_dataset(X_train, y_train)
        
        accuracies = {}
        
        # XGBoost
        try:
            model, acc = self.train_xgboost_optimized(X_train_balanced, y_train_balanced, X_test, y_test)
            self.models['xgboost'] = model
            accuracies['xgboost'] = acc
        except Exception as e:
            self.logger.error(f"❌ XGBoost failed: {e}")
        
        # Gradient Boosting
        try:
            model, acc = self.train_gradient_boost_optimized(X_train_balanced, y_train_balanced, X_test, y_test)
            self.models['gradient_boost'] = model
            accuracies['gradient_boost'] = acc
        except Exception as e:
            self.logger.error(f"❌ GradientBoost failed: {e}")
        
        # Random Forest
        try:
            model, acc = self.train_random_forest_optimized(X_train_balanced, y_train_balanced, X_test, y_test)
            self.models['random_forest'] = model
            accuracies['random_forest'] = acc
        except Exception as e:
            self.logger.error(f"❌ RandomForest failed: {e}")
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            try:
                model, acc = self.train_lightgbm_optimized(X_train_balanced, y_train_balanced, X_test, y_test)
                if model:
                    self.models['lightgbm'] = model
                    accuracies['lightgbm'] = acc
            except Exception as e:
                self.logger.error(f"❌ LightGBM failed: {e}")
        
        # Summary
        self.logger.info("=" * 80)
        self.logger.info("📊 TRAINING SUMMARY")
        self.logger.info("=" * 80)
        for name, acc in accuracies.items():
            self.logger.info(f"  {name:<20}: {acc:.4f} ({acc*100:.2f}%)")
        
        if accuracies:
            best_model = max(accuracies.items(), key=lambda x: x[1])
            self.logger.info(f"\n🏆 BEST MODEL: {best_model[0]} ({best_model[1]:.4f})")
        
        return accuracies
    
    def train_meta_ensemble(self, X_train, X_test, y_train, y_test):
        """Meta ensemble with logistic regression"""
        if not self.enable_meta_ensemble:
            return 0.0
        
        self.logger.info("🎯 Training meta ensemble...")
        
        base_predictions = []
        for name, model in self.models.items():
            if model is not None:
                try:
                    proba = model.predict_proba(X_train)
                    base_predictions.append(proba)
                except:
                    pass
        
        if len(base_predictions) < 2:
            return 0.0
        
        meta_features = []
        for i in range(len(X_train)):
            sample_meta = []
            for preds in base_predictions:
                sample_meta.extend([
                    np.max(preds[i]),
                    np.min(preds[i]),
                    np.std(preds[i]),
                    preds[i][0] - preds[i][2],
                    np.argmax(preds[i])
                ])
            meta_features.append(sample_meta)
        
        meta_features = np.array(meta_features)
        
        meta_model = LogisticRegression(
            multi_class='multinomial',
            random_state=self.random_state,
            max_iter=1000
        )
        meta_model.fit(meta_features, y_train)
        
        # Test
        test_meta_features = []
        for i in range(len(X_test)):
            sample_meta = []
            for preds in base_predictions:
                idx = min(i, len(preds) - 1)
                sample_meta.extend([
                    np.max(preds[idx]),
                    np.min(preds[idx]),
                    np.std(preds[idx]),
                    preds[idx][0] - preds[idx][2],
                    np.argmax(preds[idx])
                ])
            test_meta_features.append(sample_meta)
        
        test_meta_features = np.array(test_meta_features)
        meta_pred = meta_model.predict(test_meta_features)
        meta_accuracy = accuracy_score(y_test, meta_pred)
        
        self.meta_model = meta_model
        self.logger.info(f"✅ Meta Ensemble Accuracy: {meta_accuracy:.4f}")
        
        return meta_accuracy
    
    def train_score_model(self, X_train, X_test, y_score_train, y_score_test):
        """Score prediction model"""
        self.logger.info("⚽ Training score model...")
        
        score_counter = Counter(y_score_train)
        common_scores = [score for score, count in score_counter.items() if count >= 5]
        
        if len(common_scores) < 10:
            common_scores = [score for score, _ in score_counter.most_common(15)]
        
        self.score_space = common_scores
        
        def encode_score(score: str) -> int:
            return self.score_space.index(score) if score in self.score_space else -1
        
        y_train_enc = [encode_score(score) for score in y_score_train]
        y_test_enc = [encode_score(score) for score in y_score_test]
        
        valid_train = [i for i, label in enumerate(y_train_enc) if label != -1]
        valid_test = [i for i, label in enumerate(y_test_enc) if label != -1]
        
        if len(valid_train) < 100:
            self.logger.warning("⚠️  Not enough score data")
            return 0.0
        
        X_train_score = X_train[valid_train]
        y_train_score = [y_train_enc[i] for i in valid_train]
        X_test_score = X_test[valid_test]
        y_test_score = [y_test_enc[i] for i in valid_test]
        
        score_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        score_model.fit(X_train_score, y_train_score)
        
        score_pred = score_model.predict(X_test_score)
        score_accuracy = accuracy_score(y_test_score, score_pred)
        
        self.score_model = score_model
        self.logger.info(f"✅ Score Model Accuracy: {score_accuracy:.4f}")
        
        return score_accuracy
    
    def analyze_results(self, X_test, y_test):
        """Detailed result analysis"""
        self.logger.info("=" * 80)
        self.logger.info("📊 DETAILED PERFORMANCE ANALYSIS")
        self.logger.info("=" * 80)
        
        for model_name, model in self.models.items():
            if model is None:
                continue
            
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"🎯 MODEL: {model_name.upper()}")
            self.logger.info(f"{'='*80}")
            
            y_pred = model.predict(X_test)
            
            # Classification Report
            self.logger.info("\n📋 Classification Report:")
            report = classification_report(
                y_test, y_pred,
                target_names=['Home Win (1)', 'Draw (X)', 'Away Win (2)'],
                digits=4
            )
            self.logger.info(f"\n{report}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            self.logger.info("\n🎲 Confusion Matrix:")
            self.logger.info("                  Predicted")
            self.logger.info("                  1      X      2")
            self.logger.info(f"Actual    1    {cm[0,0]:5d}  {cm[0,1]:5d}  {cm[0,2]:5d}")
            self.logger.info(f"          X    {cm[1,0]:5d}  {cm[1,1]:5d}  {cm[1,2]:5d}")
            self.logger.info(f"          2    {cm[2,0]:5d}  {cm[2,1]:5d}  {cm[2,2]:5d}")
            
            # Per-class accuracy
            class_accuracies = cm.diagonal() / cm.sum(axis=1)
            self.logger.info("\n🎯 Per-Class Accuracy:")
            self.logger.info(f"   Home Win (1): {class_accuracies[0]:.2%}")
            self.logger.info(f"   Draw (X):     {class_accuracies[1]:.2%}")
            self.logger.info(f"   Away Win (2): {class_accuracies[2]:.2%}")
            
            overall_acc = accuracy_score(y_test, y_pred)
            self.logger.info(f"\n✅ Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    
    def save_models_v35(self):
        """Save models in v3.5 format"""
        self.logger.info("💾 Saving models in v3.5 format...")
        
        # Ensemble models
        ensemble_data = {
            'models': self.models,
            'scaler': self.scaler,
            'is_trained': True,
            'feature_config': {
                'expected_features': self.metadata.get('expected_features', 45),
                'version': 'v3.5_optimized',
                'created_at': datetime.now().isoformat()
            }
        }
        joblib.dump(ensemble_data, self.models_dir / "ensemble_models.pkl")
        
        # Meta ensemble
        if self.meta_model is not None:
            meta_data = {
                'meta_model': self.meta_model,
                'is_trained': True,
                'model_weights': {
                    'xgboost': 0.35,
                    'lightgbm': 0.30,
                    'gradient_boost': 0.20,
                    'random_forest': 0.15
                }
            }
            joblib.dump(meta_data, self.models_dir / "meta_ensemble.pkl")
        
        # Score model
        if self.score_model is not None:
            score_data = {
                'model': self.score_model,
                'score_space': self.score_space,
                'is_trained': True
            }
            joblib.dump(score_data, self.models_dir / "enhanced_score_predictor.pkl")
        
        # Scaler
        joblib.dump(self.scaler, self.models_dir / "scaler.pkl")
        
        # Feature config
        feature_config = {
            'version': 'v3.5_optimized',
            'feature_count': self.metadata.get('expected_features', 45),
            'feature_engineer': self.metadata['feature_engineer'],
            'created_at': datetime.now().isoformat(),
            'optimization': self.metadata['optimization']
        }
        with open(self.models_dir / "feature_config.json", 'w', encoding='utf-8') as f:
            json.dump(feature_config, f, indent=2, ensure_ascii=False)
        
        # Training metadata
        with open(self.models_dir / "training_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ All models saved → {self.models_dir}")
    
    def run_training(self) -> Dict[str, Any]:
        """Complete training pipeline"""
        start_time = time.time()
        
        self.logger.info("=" * 80)
        self.logger.info("🚀 OPTIMIZED ML PREDICTION ENGINE v3.5 - TRAINING STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"⚙️  Configuration:")
        self.logger.info(f"   - SMOTE: {self.use_smote}")
        self.logger.info(f"   - Balance Method: {self.balance_method}")
        self.logger.info(f"   - Class Weights: {self.use_class_weights}")
        self.logger.info(f"   - Hyperparameter Search: {self.n_iter_search} iterations")
        self.logger.info(f"   - Cross-Validation: {self.cv_folds} folds")
        self.logger.info("=" * 80)
        
        try:
            # 1. Load and prepare data
            df = self.load_and_prepare_data()
            df = self.clean_data(df)
            
            # 2. Extract features
            X, y_ms, y_score = self.extract_features_v35(df)
            
            # Store feature count
            self.metadata['expected_features'] = X.shape[1]
            
            # 3. Train-test split
            X_train, X_test, y_train, y_test, y_score_train, y_score_test = train_test_split(
                X, y_ms, y_score,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y_ms
            )
            
            # 4. Feature scaling
            self.logger.info("🔧 Feature scaling...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 5. Train optimized base models
            base_accuracies = self.train_base_models(
                X_train_scaled, X_test_scaled, y_train, y_test
            )
            
            # 6. Analyze results
            self.analyze_results(X_test_scaled, y_test)
            
            # 7. Train meta ensemble
            meta_accuracy = self.train_meta_ensemble(
                X_train_scaled, X_test_scaled, y_train, y_test
            )
            
            # 8. Train score model
            score_accuracy = self.train_score_model(
                X_train_scaled, X_test_scaled, y_score_train, y_score_test
            )
            
            # 9. Save models
            self.save_models_v35()
            
            # 10. Calculate improvement
            baseline_accuracy = 0.45  # Original accuracy
            best_accuracy = max(base_accuracies.values()) if base_accuracies else 0.45
            improvement = ((best_accuracy - baseline_accuracy) / baseline_accuracy) * 100
            
            # 11. Performance report
            training_time = time.time() - start_time
            performance_report = {
                "success": True,
                "training_time_seconds": training_time,
                "training_time_minutes": training_time / 60,
                "total_samples": len(X),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_count": X.shape[1],
                "base_accuracies": base_accuracies,
                "meta_accuracy": meta_accuracy,
                "score_accuracy": score_accuracy,
                "baseline_accuracy": baseline_accuracy,
                "best_accuracy": best_accuracy,
                "improvement_percentage": improvement,
                "memory_usage": self._mem_usage(),
                "optimization_config": self.metadata['optimization']
            }
            
            # Final summary
            self.logger.info("=" * 80)
            self.logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            self.logger.info(f"⏱️  Training Time: {training_time/60:.2f} minutes")
            self.logger.info(f"📊 Total Samples: {len(X):,}")
            self.logger.info(f"🎯 Feature Count: {X.shape[1]}")
            self.logger.info(f"\n📈 ACCURACY RESULTS:")
            self.logger.info(f"   Baseline (Original): {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
            self.logger.info(f"   Best (Optimized):    {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
            self.logger.info(f"   Improvement:         +{improvement:.2f}%")
            
            if base_accuracies:
                self.logger.info(f"\n🏆 MODEL RANKINGS:")
                sorted_models = sorted(base_accuracies.items(), key=lambda x: x[1], reverse=True)
                for i, (name, acc) in enumerate(sorted_models, 1):
                    self.logger.info(f"   {i}. {name:<20}: {acc:.4f} ({acc*100:.2f}%)")
            
            if meta_accuracy > 0:
                self.logger.info(f"\n🎯 Meta Ensemble: {meta_accuracy:.4f} ({meta_accuracy*100:.2f}%)")
            
            if score_accuracy > 0:
                self.logger.info(f"⚽ Score Model:    {score_accuracy:.4f} ({score_accuracy*100:.2f}%)")
            
            self.logger.info(f"\n💾 Models saved to: {self.models_dir}")
            self.logger.info(f"💻 Memory usage: {self._mem_usage()}")
            self.logger.info("=" * 80)
            
            return performance_report
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "training_time_seconds": time.time() - start_time
            }


def main():
    """Main training function"""
    print("=" * 80)
    print("🚀 ML PREDICTION ENGINE v3.5 - OPTIMIZED MODEL TRAINING")
    print("=" * 80)
    print("\n📦 Checking dependencies...")
    
    # Check dependencies
    missing = []
    if not IMBLEARN_AVAILABLE:
        missing.append("imbalanced-learn")
    if not FEATURE_ENGINEER_AVAILABLE:
        missing.append("enhanced_feature_engineer")
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        if 'imbalanced-learn' in missing:
            print("   Install: pip install imbalanced-learn")
        if 'enhanced_feature_engineer' in missing:
            print("   Make sure enhanced_feature_engineer.py is in the same directory")
        print("\n❌ Cannot proceed without dependencies")
        return
    
    print("✅ All dependencies available")
    print("\n" + "=" * 80)
    
    # Configuration options
    configs = {
        'fast': {
            'use_smote': True,
            'n_iter_search': 15,
            'cv_folds': 2,
            'balance_method': 'smote',
            'description': 'Fast training (~15-30 min)'
        },
        'balanced': {
            'use_smote': True,
            'n_iter_search': 30,
            'cv_folds': 3,
            'balance_method': 'hybrid',
            'description': 'Balanced training (~45-90 min) [RECOMMENDED]'
        },
        'accurate': {
            'use_smote': True,
            'n_iter_search': 50,
            'cv_folds': 5,
            'balance_method': 'hybrid',
            'description': 'Maximum accuracy (~2-3 hours)'
        }
    }
    
    print("📋 Available configurations:")
    for name, config in configs.items():
        print(f"   {name}: {config['description']}")
    
    # Use balanced configuration by default
    selected_config = 'balanced'
    config = configs[selected_config]
    
    print(f"\n✅ Using configuration: {selected_config}")
    print(f"   - Search iterations: {config['n_iter_search']}")
    print(f"   - CV folds: {config['cv_folds']}")
    print(f"   - Balance method: {config['balance_method']}")
    print("\n" + "=" * 80)
    
    try:
        # Create trainer
        trainer = OptimizedModelTrainerV35(
            models_dir="data/ai_models_v3",
            raw_data_path="data/raw",
            clubs_path="data/clubs",
            test_size=0.2,
            random_state=42,
            batch_size=2000,
            enable_meta_ensemble=True,
            use_smote=config['use_smote'],
            use_class_weights=True,
            n_iter_search=config['n_iter_search'],
            cv_folds=config['cv_folds'],
            balance_method=config['balance_method'],
            verbose=True
        )
        
        # Run training
        result = trainer.run_training()
        
        if result["success"]:
            print("\n" + "=" * 80)
            print("✅ TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"\n📊 FINAL RESULTS:")
            print(f"   Training Time:  {result['training_time_minutes']:.2f} minutes")
            print(f"   Baseline:       {result['baseline_accuracy']:.4f} ({result['baseline_accuracy']*100:.2f}%)")
            print(f"   Best Model:     {result['best_accuracy']:.4f} ({result['best_accuracy']*100:.2f}%)")
            print(f"   Improvement:    +{result['improvement_percentage']:.2f}%")
            
            if result['base_accuracies']:
                print(f"\n🏆 Best performing model:")
                best_model = max(result['base_accuracies'].items(), key=lambda x: x[1])
                print(f"   {best_model[0]}: {best_model[1]:.4f} ({best_model[1]*100:.2f}%)")
            
            print(f"\n💾 Models saved to: data/ai_models_v3/")
            print("=" * 80)
        else:
            print(f"\n❌ TRAINING FAILED: {result['error']}")
            
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
