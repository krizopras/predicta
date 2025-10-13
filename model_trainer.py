#!/usr/bin/env python3
"""
Model Trainer v3.5 - SMOTE ve Class Balancing Düzeltilmiş
%45 → %55-65 hedefi için optimize edildi
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
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    balanced_accuracy_score, f1_score, precision_recall_fscore_support
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

# Imbalanced learning
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    from imblearn.combine import SMOTETomek, SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("⚠️  imbalanced-learn yüklü değil. Yükleyin: pip install imbalanced-learn")

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Log ayarları
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ModelTrainerFixed")

# Local imports
try:
    from enhanced_feature_engineer import EnhancedFeatureEngineer
    FEATURE_ENGINEER_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEER_AVAILABLE = False
    logger.error("❌ enhanced_feature_engineer bulunamadı!")

try:
    from historical_processor import HistoricalDataProcessor
    HISTORICAL_PROCESSOR_AVAILABLE = True
except ImportError:
    HISTORICAL_PROCESSOR_AVAILABLE = False


class ImprovedModelTrainer:
    """
    Geliştirilmiş Model Trainer
    - ✅ Düzgün SMOTE uygulaması
    - ✅ Balanced accuracy metrikleri
    - ✅ Daha iyi hyperparameter search
    - ✅ Stacking ensemble
    """
    
    def __init__(
        self,
        models_dir: str = "data/ai_models_v3",
        raw_data_path: str = "data/raw",
        clubs_path: str = "data/clubs",
        test_size: float = 0.2,
        random_state: int = 42,
        # Optimization parameters
        use_smote: bool = True,
        balance_method: str = 'smote_tomek',  # En iyi yöntem
        n_iter_search: int = 50,
        cv_folds: int = 5,
        verbose: bool = True
    ):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_path = Path(raw_data_path)
        self.clubs_path = Path(clubs_path)
        self.test_size = test_size
        self.random_state = random_state
        
        # Optimization
        self.use_smote = use_smote and IMBLEARN_AVAILABLE
        self.balance_method = balance_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.verbose = verbose
        
        # Feature engineer
        if not FEATURE_ENGINEER_AVAILABLE:
            raise ImportError("❌ EnhancedFeatureEngineer bulunamadı!")
        
        self.feature_engineer = EnhancedFeatureEngineer(model_path=str(self.models_dir))
        
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
        self.stacking_model = None
        self.score_model = None
        self.scaler = StandardScaler()
        
        # Metadata
        self.metadata = {
            "version": "v3.5_improved",
            "trained_at": datetime.now().isoformat(),
            "optimization": {
                "use_smote": self.use_smote,
                "balance_method": self.balance_method,
                "n_iter_search": self.n_iter_search,
                "cv_folds": self.cv_folds
            }
        }
        
        self.logger = logger
        
        if self.use_smote and not IMBLEARN_AVAILABLE:
            self.logger.warning("⚠️  SMOTE istendi ama imbalanced-learn yüklü değil")
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
        """Veriyi yükle"""
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
        self.logger.info(f"✅ {initial_count:,} → {cleaned_count:,} kayıt ({initial_count-cleaned_count} filtrelendi)")
        self.logger.info(f"📊 Sonuç dağılımı: {dict(df['result'].value_counts())}")
        
        return df
    
    def extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Feature çıkarımı"""
        self.logger.info("🔧 Feature çıkarımı başlatılıyor...")
        
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
        
        self.logger.info(f"📊 {name} Sınıf Dağılımı:")
        for cls in sorted(counter.keys()):
            count = counter[cls]
            pct = count / total * 100
            class_name = {0: "Ev Sahibi (1)", 1: "Beraberlik (X)", 2: "Deplasman (2)"}[cls]
            self.logger.info(f"   {class_name}: {count:6,} ({pct:5.2f}%)")
        
        max_count = max(counter.values())
        min_count = min(counter.values())
        imbalance_ratio = max_count / min_count
        self.logger.info(f"   ⚖️  Dengesizlik Oranı: {imbalance_ratio:.2f}:1")
        
        return counter, imbalance_ratio
    
    def balance_dataset(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Dataset balancing - DÜZELTİLMİŞ VERSİYON"""
        if not self.use_smote or self.balance_method == 'none':
            self.logger.info("ℹ️  Dataset balancing atlanıyor")
            return X_train, y_train
        
        self.logger.info("=" * 80)
        self.logger.info(f"🔧 DATASET BALANCING BAŞLIYOR: {self.balance_method.upper()}")
        self.logger.info("=" * 80)
        
        counter_before, imbalance_before = self.analyze_class_distribution(y_train, "BALANCING ÖNCESİ")
        
        try:
            # En az örnek sayısını bul
            min_samples = min(counter_before.values())
            
            if self.balance_method == 'smote':
                # Basit SMOTE
                k_neighbors = min(5, min_samples - 1)
                sampler = SMOTE(
                    random_state=self.random_state,
                    k_neighbors=k_neighbors,
                    sampling_strategy='auto'  # Tüm sınıfları majority'ye eşitle
                )
                X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
                
            elif self.balance_method == 'borderline_smote':
                # Borderline-SMOTE (sınır örneklere odaklanır)
                k_neighbors = min(5, min_samples - 1)
                sampler = BorderlineSMOTE(
                    random_state=self.random_state,
                    k_neighbors=k_neighbors,
                    sampling_strategy='auto'
                )
                X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
                
            elif self.balance_method == 'adasyn':
                # ADASYN (adaptive synthetic sampling)
                n_neighbors = min(5, min_samples - 1)
                sampler = ADASYN(
                    random_state=self.random_state,
                    n_neighbors=n_neighbors,
                    sampling_strategy='auto'
                )
                X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
                
            elif self.balance_method == 'smote_tomek':
                # SMOTE + Tomek Links (EN İYİSİ!)
                # Over-sample sonrası boundary noise'u temizler
                k_neighbors = min(5, min_samples - 1)
                sampler = SMOTETomek(
                    smote=SMOTE(
                        random_state=self.random_state,
                        k_neighbors=k_neighbors,
                        sampling_strategy='auto'
                    ),
                    random_state=self.random_state
                )
                X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
                
            elif self.balance_method == 'smote_enn':
                # SMOTE + Edited Nearest Neighbours
                k_neighbors = min(5, min_samples - 1)
                sampler = SMOTEENN(
                    smote=SMOTE(
                        random_state=self.random_state,
                        k_neighbors=k_neighbors,
                        sampling_strategy='auto'
                    ),
                    random_state=self.random_state
                )
                X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
                
            elif self.balance_method == 'hybrid':
                # Undersample + SMOTE
                # Önce majority class'ı azalt, sonra SMOTE uygula
                majority_target = int(np.median(list(counter_before.values())) * 1.5)
                
                rus = RandomUnderSampler(
                    sampling_strategy={
                        cls: min(count, majority_target)
                        for cls, count in counter_before.items()
                    },
                    random_state=self.random_state
                )
                X_temp, y_temp = rus.fit_resample(X_train, y_train)
                
                # Şimdi SMOTE uygula
                temp_counter = Counter(y_temp)
                min_temp = min(temp_counter.values())
                k_neighbors = min(5, min_temp - 1)
                
                smote = SMOTE(
                    random_state=self.random_state,
                    k_neighbors=k_neighbors,
                    sampling_strategy='auto'
                )
                X_balanced, y_balanced = smote.fit_resample(X_temp, y_temp)
                
            else:
                # Fallback
                self.logger.warning(f"⚠️  Bilinmeyen method: {self.balance_method}, SMOTE kullanılıyor")
                sampler = SMOTE(random_state=self.random_state)
                X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
            
            # Sonuçları göster
            self.logger.info("=" * 80)
            counter_after, imbalance_after = self.analyze_class_distribution(y_balanced, "BALANCING SONRASI")
            self.logger.info("=" * 80)
            
            self.logger.info(f"📈 Değişim:")
            self.logger.info(f"   Örnek Sayısı: {len(y_train):,} → {len(y_balanced):,} (+{len(y_balanced)-len(y_train):,})")
            self.logger.info(f"   Dengesizlik: {imbalance_before:.2f}:1 → {imbalance_after:.2f}:1")
            self.logger.info("=" * 80)
            
            return X_balanced, y_balanced
            
        except Exception as e:
            self.logger.error(f"❌ Balancing hatası: {e}")
            self.logger.warning("⚠️  Orijinal dengesiz dataset kullanılıyor")
            return X_train, y_train
    
    def train_xgboost_optimized(self, X_train, y_train, X_val, y_val):
        """Optimize XGBoost - Daha agresif arama"""
        self.logger.info("🔧 XGBoost eğitiliyor (optimized)...")
        
        # Class weights hesapla
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        scale_pos_weight = class_weights[0] / class_weights[2]  # Home vs Away
        
        param_distributions = {
            'max_depth': [3, 5, 7, 9, 11],
            'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
            'n_estimators': [500, 800, 1000, 1500],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.3, 0.5],
            'reg_alpha': [0, 0.1, 0.5, 1],
            'reg_lambda': [1, 1.5, 2, 3],
            'scale_pos_weight': [0.5, 1.0, scale_pos_weight, 2.0]
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
            scoring='balanced_accuracy',  # ÖNEMLİ DEĞİŞİKLİK!
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1 if self.verbose else 0
        )
        
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        
        # Değerlendirme
        val_pred = best_model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        val_balanced_acc = balanced_accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='macro')
        
        self.logger.info(f"✅ LightGBM:")
        self.logger.info(f"   CV Score: {search.best_score_:.4f}")
        self.logger.info(f"   Accuracy: {val_acc:.4f}")
        self.logger.info(f"   Balanced Accuracy: {val_balanced_acc:.4f} ⭐")
        self.logger.info(f"   F1-Macro: {val_f1:.4f}")
        
        return best_model, val_balanced_acc
    
    def train_base_models(self, X_train, X_test, y_train, y_test):
        """Tüm base modelleri eğit"""
        self.logger.info("=" * 80)
        self.logger.info("🎯 BASE MODEL EĞİTİMİ BAŞLIYOR")
        self.logger.info("=" * 80)
        
        # Orijinal dağılımı göster
        self.analyze_class_distribution(y_train, "Orijinal Training Data")
        
        # Dataset'i dengele
        X_train_balanced, y_train_balanced = self.balance_dataset(X_train, y_train)
        
        accuracies = {}
        
        # XGBoost
        try:
            model, acc = self.train_xgboost_optimized(X_train_balanced, y_train_balanced, X_test, y_test)
            self.models['xgboost'] = model
            accuracies['xgboost'] = acc
        except Exception as e:
            self.logger.error(f"❌ XGBoost hatası: {e}")
        
        # Gradient Boosting
        try:
            model, acc = self.train_gradient_boost_optimized(X_train_balanced, y_train_balanced, X_test, y_test)
            self.models['gradient_boost'] = model
            accuracies['gradient_boost'] = acc
        except Exception as e:
            self.logger.error(f"❌ GradientBoost hatası: {e}")
        
        # Random Forest
        try:
            model, acc = self.train_random_forest_optimized(X_train_balanced, y_train_balanced, X_test, y_test)
            self.models['random_forest'] = model
            accuracies['random_forest'] = acc
        except Exception as e:
            self.logger.error(f"❌ RandomForest hatası: {e}")
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            try:
                model, acc = self.train_lightgbm_optimized(X_train_balanced, y_train_balanced, X_test, y_test)
                if model:
                    self.models['lightgbm'] = model
                    accuracies['lightgbm'] = acc
            except Exception as e:
                self.logger.error(f"❌ LightGBM hatası: {e}")
        
        # Özet
        self.logger.info("=" * 80)
        self.logger.info("📊 EĞİTİM ÖZETİ")
        self.logger.info("=" * 80)
        for name, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
            self.logger.info(f"  {name:<20}: {acc:.4f} ({acc*100:.2f}%)")
        
        if accuracies:
            best_model = max(accuracies.items(), key=lambda x: x[1])
            self.logger.info(f"\n🏆 EN İYİ MODEL: {best_model[0]} ({best_model[1]:.4f})")
        
        return accuracies
    
    def train_stacking_ensemble(self, X_train, X_test, y_train, y_test):
        """Stacking Ensemble - Meta modellerden daha iyi"""
        if not self.models or len([m for m in self.models.values() if m is not None]) < 2:
            self.logger.warning("⚠️  Stacking için yeterli model yok")
            return 0.0
        
        self.logger.info("=" * 80)
        self.logger.info("🎯 STACKING ENSEMBLE EĞİTİMİ")
        self.logger.info("=" * 80)
        
        try:
            # Base estimators
            estimators = []
            for name, model in self.models.items():
                if model is not None:
                    estimators.append((name, model))
            
            # Final estimator
            final_estimator = xgb.XGBClassifier(
                max_depth=3,
                learning_rate=0.1,
                n_estimators=100,
                random_state=self.random_state
            )
            
            # Stacking Classifier
            stacking = StackingClassifier(
                estimators=estimators,
                final_estimator=final_estimator,
                cv=5,
                n_jobs=-1,
                verbose=1 if self.verbose else 0
            )
            
            self.logger.info(f"📦 {len(estimators)} model ile stacking oluşturuluyor...")
            stacking.fit(X_train, y_train)
            
            # Değerlendirme
            pred = stacking.predict(X_test)
            acc = accuracy_score(y_test, pred)
            balanced_acc = balanced_accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, average='macro')
            
            self.stacking_model = stacking
            
            self.logger.info(f"✅ Stacking Ensemble:")
            self.logger.info(f"   Accuracy: {acc:.4f}")
            self.logger.info(f"   Balanced Accuracy: {balanced_acc:.4f} ⭐")
            self.logger.info(f"   F1-Macro: {f1:.4f}")
            
            return balanced_acc
            
        except Exception as e:
            self.logger.error(f"❌ Stacking hatası: {e}")
            return 0.0
    
    def analyze_results(self, X_test, y_test):
        """Detaylı sonuç analizi"""
        self.logger.info("=" * 80)
        self.logger.info("📊 DETAYLI PERFORMANS ANALİZİ")
        self.logger.info("=" * 80)
        
        for model_name, model in self.models.items():
            if model is None:
                continue
            
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"🎯 MODEL: {model_name.upper()}")
            self.logger.info(f"{'='*80}")
            
            y_pred = model.predict(X_test)
            
            # Metrikler
            acc = accuracy_score(y_test, y_pred)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            
            self.logger.info(f"\n📈 Metrikler:")
            self.logger.info(f"   Accuracy:          {acc:.4f} ({acc*100:.2f}%)")
            self.logger.info(f"   Balanced Accuracy: {balanced_acc:.4f} ({balanced_acc*100:.2f}%) ⭐")
            self.logger.info(f"   F1-Macro:          {f1_macro:.4f}")
            self.logger.info(f"   F1-Weighted:       {f1_weighted:.4f}")
            
            # Classification Report
            self.logger.info("\n📋 Classification Report:")
            report = classification_report(
                y_test, y_pred,
                target_names=['Ev Sahibi (1)', 'Beraberlik (X)', 'Deplasman (2)'],
                digits=4
            )
            self.logger.info(f"\n{report}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            self.logger.info("\n🎲 Confusion Matrix:")
            self.logger.info("                  Tahmin Edilen")
            self.logger.info("                  1      X      2")
            self.logger.info(f"Gerçek    1    {cm[0,0]:5d}  {cm[0,1]:5d}  {cm[0,2]:5d}")
            self.logger.info(f"          X    {cm[1,0]:5d}  {cm[1,1]:5d}  {cm[1,2]:5d}")
            self.logger.info(f"          2    {cm[2,0]:5d}  {cm[2,1]:5d}  {cm[2,2]:5d}")
            
            # Sınıf bazlı accuracy
            class_accuracies = cm.diagonal() / cm.sum(axis=1)
            self.logger.info("\n🎯 Sınıf Bazlı Accuracy:")
            self.logger.info(f"   Ev Sahibi (1): {class_accuracies[0]:.2%}")
            self.logger.info(f"   Beraberlik (X): {class_accuracies[1]:.2%}")
            self.logger.info(f"   Deplasman (2): {class_accuracies[2]:.2%}")
    
    def train_score_model(self, X_train, X_test, y_score_train, y_score_test):
        """Skor tahmin modeli"""
        self.logger.info("⚽ Skor tahmin modeli eğitiliyor...")
        
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
            self.logger.warning("⚠️  Yeterli skor verisi yok")
            return 0.0
        
        X_train_score = X_train[valid_train]
        y_train_score = [y_train_enc[i] for i in valid_train]
        X_test_score = X_test[valid_test]
        y_test_score = [y_test_enc[i] for i in valid_test]
        
        score_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        score_model.fit(X_train_score, y_train_score)
        
        score_pred = score_model.predict(X_test_score)
        score_accuracy = accuracy_score(y_test_score, score_pred)
        
        self.score_model = score_model
        self.logger.info(f"✅ Skor Model Accuracy: {score_accuracy:.4f}")
        
        return score_accuracy
    
    def save_models(self):
        """Modelleri kaydet"""
        self.logger.info("💾 Modeller kaydediliyor...")
        
        # Ensemble models
        ensemble_data = {
            'models': self.models,
            'scaler': self.scaler,
            'is_trained': True,
            'feature_config': {
                'expected_features': self.metadata.get('expected_features', 45),
                'version': 'v3.5_improved',
                'created_at': datetime.now().isoformat()
            }
        }
        joblib.dump(ensemble_data, self.models_dir / "ensemble_models.pkl")
        
        # Stacking ensemble
        if self.stacking_model is not None:
            stacking_data = {
                'model': self.stacking_model,
                'is_trained': True
            }
            joblib.dump(stacking_data, self.models_dir / "stacking_ensemble.pkl")
        
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
            'version': 'v3.5_improved',
            'feature_count': self.metadata.get('expected_features', 45),
            'created_at': datetime.now().isoformat(),
            'optimization': self.metadata['optimization']
        }
        with open(self.models_dir / "feature_config.json", 'w', encoding='utf-8') as f:
            json.dump(feature_config, f, indent=2, ensure_ascii=False)
        
        # Training metadata
        with open(self.models_dir / "training_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ Tüm modeller kaydedildi → {self.models_dir}")
    
    def run_training(self) -> Dict[str, Any]:
        """Tam eğitim pipeline'ı"""
        start_time = time.time()
        
        self.logger.info("=" * 80)
        self.logger.info("🚀 GELİŞTİRİLMİŞ ML TAHMİN SİSTEMİ v3.5 - EĞİTİM BAŞLIYOR")
        self.logger.info("=" * 80)
        self.logger.info(f"⚙️  Konfigürasyon:")
        self.logger.info(f"   - SMOTE: {self.use_smote}")
        self.logger.info(f"   - Balance Method: {self.balance_method}")
        self.logger.info(f"   - Hyperparameter Search: {self.n_iter_search} iterasyon")
        self.logger.info(f"   - Cross-Validation: {self.cv_folds} fold")
        self.logger.info(f"   - Scoring Metric: balanced_accuracy ⭐")
        self.logger.info("=" * 80)
        
        try:
            # 1. Veriyi yükle ve temizle
            df = self.load_and_prepare_data()
            df = self.clean_data(df)
            
            # 2. Feature extraction
            X, y_ms, y_score = self.extract_features(df)
            
            # Feature sayısını kaydet
            self.metadata['expected_features'] = X.shape[1]
            
            # 3. Train-test split
            X_train, X_test, y_train, y_test, y_score_train, y_score_test = train_test_split(
                X, y_ms, y_score,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y_ms
            )
            
            # 4. Feature scaling
            self.logger.info("🔧 Feature scaling uygulanıyor...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 5. Base modelleri eğit
            base_accuracies = self.train_base_models(
                X_train_scaled, X_test_scaled, y_train, y_test
            )
            
            # 6. Detaylı analiz
            self.analyze_results(X_test_scaled, y_test)
            
            # 7. Stacking ensemble
            stacking_accuracy = self.train_stacking_ensemble(
                X_train_scaled, X_test_scaled, y_train, y_test
            )
            
            # 8. Skor modeli
            score_accuracy = self.train_score_model(
                X_train_scaled, X_test_scaled, y_score_train, y_score_test
            )
            
            # 9. Modelleri kaydet
            self.save_models()
            
            # 10. Performans raporu
            training_time = time.time() - start_time
            baseline_accuracy = 0.45
            best_accuracy = max(base_accuracies.values()) if base_accuracies else 0.45
            improvement = ((best_accuracy - baseline_accuracy) / baseline_accuracy) * 100
            
            performance_report = {
                "success": True,
                "training_time_seconds": training_time,
                "training_time_minutes": training_time / 60,
                "total_samples": len(X),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_count": X.shape[1],
                "base_accuracies": base_accuracies,
                "stacking_accuracy": stacking_accuracy,
                "score_accuracy": score_accuracy,
                "baseline_accuracy": baseline_accuracy,
                "best_accuracy": best_accuracy,
                "improvement_percentage": improvement,
                "memory_usage": self._mem_usage(),
                "optimization_config": self.metadata['optimization']
            }
            
            # Final özet
            self.logger.info("=" * 80)
            self.logger.info("🎉 EĞİTİM BAŞARIYLA TAMAMLANDI")
            self.logger.info("=" * 80)
            self.logger.info(f"⏱️  Eğitim Süresi: {training_time/60:.2f} dakika")
            self.logger.info(f"📊 Toplam Örnek: {len(X):,}")
            self.logger.info(f"🎯 Feature Sayısı: {X.shape[1]}")
            self.logger.info(f"\n📈 BALANCED ACCURACY SONUÇLARI:")
            self.logger.info(f"   Baseline (Önceki): {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
            self.logger.info(f"   En İyi (Yeni):     {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
            self.logger.info(f"   Gelişme:           +{improvement:.2f}%")
            
            if base_accuracies:
                self.logger.info(f"\n🏆 MODEL SIRALAMASI:")
                sorted_models = sorted(base_accuracies.items(), key=lambda x: x[1], reverse=True)
                for i, (name, acc) in enumerate(sorted_models, 1):
                    self.logger.info(f"   {i}. {name:<20}: {acc:.4f} ({acc*100:.2f}%)")
            
            if stacking_accuracy > 0:
                self.logger.info(f"\n🎯 Stacking Ensemble: {stacking_accuracy:.4f} ({stacking_accuracy*100:.2f}%)")
            
            if score_accuracy > 0:
                self.logger.info(f"⚽ Skor Modeli:        {score_accuracy:.4f} ({score_accuracy*100:.2f}%)")
            
            self.logger.info(f"\n💾 Modeller kaydedildi: {self.models_dir}")
            self.logger.info(f"💻 RAM Kullanımı: {self._mem_usage()}")
            self.logger.info("=" * 80)
            
            return performance_report
            
        except Exception as e:
            self.logger.error(f"❌ Eğitim hatası: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "training_time_seconds": time.time() - start_time
            }


def main():
    """Ana eğitim fonksiyonu"""
    print("=" * 80)
    print("🚀 GELİŞTİRİLMİŞ ML TAHMİN SİSTEMİ v3.5 - MODEL EĞİTİMİ")
    print("=" * 80)
    print("\n📦 Bağımlılıklar kontrol ediliyor...")
    
    # Bağımlılık kontrolü
    missing = []
    if not IMBLEARN_AVAILABLE:
        missing.append("imbalanced-learn")
    if not FEATURE_ENGINEER_AVAILABLE:
        missing.append("enhanced_feature_engineer")
    
    if missing:
        print(f"\n⚠️  Eksik bağımlılıklar: {', '.join(missing)}")
        if 'imbalanced-learn' in missing:
            print("   Yüklemek için: pip install imbalanced-learn")
        if 'enhanced_feature_engineer' in missing:
            print("   enhanced_feature_engineer.py dosyasının aynı dizinde olduğundan emin olun")
        print("\n❌ Eksik bağımlılıklar yüzünden devam edilemiyor")
        return
    
    print("✅ Tüm bağımlılıklar mevcut")
    print("\n" + "=" * 80)
    
    # Konfigurasyon seçenekleri
    configs = {
        'test': {
            'use_smote': True,
            'balance_method': 'smote',
            'n_iter_search': 10,
            'cv_folds': 3,
            'description': 'Hızlı test (~10-20 dakika)'
        },
        'balanced': {
            'use_smote': True,
            'balance_method': 'smote_tomek',
            'n_iter_search': 50,
            'cv_folds': 5,
            'description': 'Dengeli eğitim (~1-2 saat) [ÖNERİLEN]'
        },
        'aggressive': {
            'use_smote': True,
            'balance_method': 'smote_tomek',
            'n_iter_search': 100,
            'cv_folds': 5,
            'description': 'Maksimum doğruluk (~3-4 saat)'
        }
    }
    
    print("📋 Mevcut konfigürasyonlar:")
    for name, config in configs.items():
        print(f"   {name}: {config['description']}")
    
    # Balanced kullan (önerilen)
    selected_config = 'balanced'
    config = configs[selected_config]
    
    print(f"\n✅ Kullanılan konfigürasyon: {selected_config}")
    print(f"   - Balancing Method: {config['balance_method']}")
    print(f"   - Search Iterations: {config['n_iter_search']}")
    print(f"   - CV Folds: {config['cv_folds']}")
    print("\n" + "=" * 80)
    
    try:
        # Trainer oluştur
        trainer = ImprovedModelTrainer(
            models_dir="data/ai_models_v3",
            raw_data_path="data/raw",
            clubs_path="data/clubs",
            test_size=0.2,
            random_state=42,
            use_smote=config['use_smote'],
            balance_method=config['balance_method'],
            n_iter_search=config['n_iter_search'],
            cv_folds=config['cv_folds'],
            verbose=True
        )
        
        # Eğitimi başlat
        result = trainer.run_training()
        
        if result["success"]:
            print("\n" + "=" * 80)
            print("✅ EĞİTİM BAŞARIYLA TAMAMLANDI!")
            print("=" * 80)
            print(f"\n📊 SONUÇLAR:")
            print(f"   Eğitim Süresi:  {result['training_time_minutes']:.2f} dakika")
            print(f"   Baseline:       {result['baseline_accuracy']:.4f} ({result['baseline_accuracy']*100:.2f}%)")
            print(f"   En İyi Model:   {result['best_accuracy']:.4f} ({result['best_accuracy']*100:.2f}%)")
            print(f"   Gelişme:        +{result['improvement_percentage']:.2f}%")
            
            if result['base_accuracies']:
                print(f"\n🏆 En iyi performans gösteren model:")
                best_model = max(result['base_accuracies'].items(), key=lambda x: x[1])
                print(f"   {best_model[0]}: {best_model[1]:.4f} ({best_model[1]*100:.2f}%)")
            
            print(f"\n💾 Modeller kaydedildi: data/ai_models_v3/")
            print("=" * 80)
        else:
            print(f"\n❌ EĞİTİM BAŞARISIZ: {result['error']}")
            
    except Exception as e:
        print(f"\n❌ Kritik hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
        
        self.logger.info(f"✅ XGBoost:")
        self.logger.info(f"   CV Score: {search.best_score_:.4f}")
        self.logger.info(f"   Accuracy: {val_acc:.4f}")
        self.logger.info(f"   Balanced Accuracy: {val_balanced_acc:.4f} ⭐")
        self.logger.info(f"   F1-Macro: {val_f1:.4f}")
        
        return best_model, val_balanced_acc
    
    def train_gradient_boost_optimized(self, X_train, y_train, X_val, y_val):
        """Optimize Gradient Boosting"""
        self.logger.info("🔧 Gradient Boosting eğitiliyor...")
        
        param_distributions = {
            'n_estimators': [200, 300, 500, 800],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2', 0.8]
        }
        
        base_model = GradientBoostingClassifier(random_state=self.random_state)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        search = RandomizedSearchCV(
            base_model, param_distributions,
            n_iter=self.n_iter_search, cv=cv,
            scoring='balanced_accuracy',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1 if self.verbose else 0
        )
        
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        
        val_pred = best_model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        val_balanced_acc = balanced_accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='macro')
        
        self.logger.info(f"✅ GradientBoost:")
        self.logger.info(f"   CV Score: {search.best_score_:.4f}")
        self.logger.info(f"   Accuracy: {val_acc:.4f}")
        self.logger.info(f"   Balanced Accuracy: {val_balanced_acc:.4f} ⭐")
        self.logger.info(f"   F1-Macro: {val_f1:.4f}")
        
        return best_model, val_balanced_acc
    
    def train_random_forest_optimized(self, X_train, y_train, X_val, y_val):
        """Optimize Random Forest"""
        self.logger.info("🔧 Random Forest eğitiliyor...")
        
        param_distributions = {
            'n_estimators': [200, 300, 500, 800],
            'max_depth': [10, 15, 20, 25, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', 0.8],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }
        
        base_model = RandomForestClassifier(
            random_state=self.random_state, 
            n_jobs=-1,
            bootstrap=True
        )
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        search = RandomizedSearchCV(
            base_model, param_distributions,
            n_iter=self.n_iter_search, cv=cv,
            scoring='balanced_accuracy',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1 if self.verbose else 0
        )
        
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        
        val_pred = best_model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        val_balanced_acc = balanced_accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='macro')
        
        self.logger.info(f"✅ RandomForest:")
        self.logger.info(f"   CV Score: {search.best_score_:.4f}")
        self.logger.info(f"   Accuracy: {val_acc:.4f}")
        self.logger.info(f"   Balanced Accuracy: {val_balanced_acc:.4f} ⭐")
        self.logger.info(f"   F1-Macro: {val_f1:.4f}")
        
        return best_model, val_balanced_acc
    
    def train_lightgbm_optimized(self, X_train, y_train, X_val, y_val):
        """Optimize LightGBM"""
        if not LIGHTGBM_AVAILABLE:
            self.logger.warning("⚠️  LightGBM yüklü değil")
            return None, 0
        
        self.logger.info("🔧 LightGBM eğitiliyor...")
        
        param_distributions = {
            'n_estimators': [200, 300, 500, 800],
            'max_depth': [3, 5, 7, 9, -1],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [31, 50, 70, 100],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_samples': [5, 10, 20],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
        }
        
        base_model = lgb.LGBMClassifier(
            random_state=self.random_state, 
            n_jobs=-1, 
            verbose=-1,
            class_weight='balanced'
        )
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        search = RandomizedSearchCV(
            base_model, param_distributions,
            n_iter=self.n_iter_search, cv=cv,
            scoring='balanced_accuracy',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1 if self.verbose else 0
        )
        
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        
        val_pred = best_model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        val_balanced_acc = balanced_accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='macro')
