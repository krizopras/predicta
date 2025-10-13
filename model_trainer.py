#!/usr/bin/env python3
"""
Model Trainer v3.5 - Temiz ve Çalışır Versiyon
Tüm girinti hataları düzeltildi
"""

import os
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
    balanced_accuracy_score, f1_score
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek, SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("⚠️  pip install imbalanced-learn")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ImprovedTrainer")

try:
    from enhanced_feature_engineer import EnhancedFeatureEngineer
    FEATURE_ENGINEER_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEER_AVAILABLE = False
    logger.error("❌ enhanced_feature_engineer.py bulunamadı!")

try:
    from historical_processor import HistoricalDataProcessor
    HISTORICAL_PROCESSOR_AVAILABLE = True
except ImportError:
    HISTORICAL_PROCESSOR_AVAILABLE = False


class ImprovedModelTrainer:
    
    def __init__(
        self,
        models_dir: str = "data/ai_models_v3",
        raw_data_path: str = "data/raw",
        clubs_path: str = "data/clubs",
        test_size: float = 0.2,
        random_state: int = 42,
        use_smote: bool = True,
        balance_method: str = 'smote_tomek',
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
        
        self.use_smote = use_smote and IMBLEARN_AVAILABLE
        self.balance_method = balance_method
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.verbose = verbose
        
        if not FEATURE_ENGINEER_AVAILABLE:
            raise ImportError("❌ EnhancedFeatureEngineer bulunamadı!")
        
        self.feature_engineer = EnhancedFeatureEngineer(model_path=str(self.models_dir))
        
        self.history_processor = None
        if HISTORICAL_PROCESSOR_AVAILABLE:
            self.history_processor = HistoricalDataProcessor(
                str(self.raw_data_path),
                str(self.clubs_path)
            )
        
        self.models = {
            'xgboost': None,
            'gradient_boost': None,
            'random_forest': None,
            'lightgbm': None
        }
        self.stacking_model = None
        self.score_model = None
        self.scaler = StandardScaler()
        
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
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        self.logger.info("📂 Veri yükleniyor...")
        
        all_matches = []
        
        if self.history_processor and self.raw_data_path.exists():
            for country_dir in self.raw_data_path.iterdir():
                if country_dir.is_dir():
                    try:
                        matches = self.history_processor.load_country_data(country_dir.name)
                        all_matches.extend(matches)
                        self.logger.info(f"✅ {country_dir.name}: {len(matches)} maç")
                    except Exception as e:
                        self.logger.warning(f"⚠️ {country_dir.name}: {e}")
        else:
            csv_files = list(self.raw_data_path.glob("**/*.csv"))
            for csv_file in csv_files:
                try:
                    df_temp = pd.read_csv(csv_file)
                    required_cols = ['home_team', 'away_team', 'home_score', 'away_score']
                    if all(col in df_temp.columns for col in required_cols):
                        league = csv_file.parent.name if csv_file.parent.name != 'raw' else 'Unknown'
                        df_temp['league'] = league
                        all_matches.extend(df_temp.to_dict('records'))
                except Exception as e:
                    self.logger.warning(f"⚠️ {csv_file}: {e}")
        
        if not all_matches:
            raise ValueError("❌ Veri bulunamadı!")
        
        df = pd.DataFrame(all_matches)
        self.logger.info(f"📊 Toplam {len(df):,} maç")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("🧹 Veri temizleniyor...")
        
        initial = len(df)
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
        
        self.logger.info(f"✅ {initial:,} → {len(df):,} kayıt")
        self.logger.info(f"📊 Dağılım: {dict(df['result'].value_counts())}")
        
        return df
    
    def extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        self.logger.info("🔧 Feature extraction...")
        
        X_list, y_ms, y_score = [], [], []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Features"):
            try:
                match_data = {
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "league": row.get("league", "Unknown"),
                    "odds": row.get("odds", {"1": 2.0, "X": 3.0, "2": 3.5}),
                }
                
                features = self.feature_engineer.extract_features(match_data)
                
                if features is not None:
                    X_list.append(features)
                    y_ms.append({"1": 0, "X": 1, "2": 2}[row["result"]])
                    y_score.append(f"{row['home_score']}-{row['away_score']}")
            except:
                continue
        
        X = np.array(X_list, dtype=np.float32)
        y_ms = np.array(y_ms, dtype=np.int8)
        
        self.logger.info(f"✅ Shape: {X.shape}")
        return X, y_ms, y_score
    
    def analyze_class_distribution(self, y: np.ndarray, name: str = "Dataset"):
        counter = Counter(y)
        total = len(y)
        
        self.logger.info(f"📊 {name}:")
        for cls in sorted(counter.keys()):
            count = counter[cls]
            pct = count / total * 100
            class_name = {0: "Ev (1)", 1: "Berab (X)", 2: "Depl (2)"}[cls]
            self.logger.info(f"   {class_name}: {count:6,} ({pct:5.2f}%)")
        
        imbalance = max(counter.values()) / min(counter.values())
        self.logger.info(f"   Dengesizlik: {imbalance:.2f}:1")
        return counter, imbalance
    
    def balance_dataset(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.use_smote:
            return X_train, y_train
        
        self.logger.info("=" * 80)
        self.logger.info(f"🔧 BALANCING: {self.balance_method.upper()}")
        self.logger.info("=" * 80)
        
        counter_before, _ = self.analyze_class_distribution(y_train, "ÖNCESİ")
        
        try:
            min_samples = min(counter_before.values())
            k_neighbors = min(5, min_samples - 1)
            
            if self.balance_method == 'smote':
                sampler = SMOTE(random_state=self.random_state, k_neighbors=k_neighbors)
            elif self.balance_method == 'smote_tomek':
                sampler = SMOTETomek(
                    smote=SMOTE(random_state=self.random_state, k_neighbors=k_neighbors),
                    random_state=self.random_state
                )
            elif self.balance_method == 'hybrid':
                rus = RandomUnderSampler(random_state=self.random_state)
                X_temp, y_temp = rus.fit_resample(X_train, y_train)
                sampler = SMOTE(random_state=self.random_state)
                X_balanced, y_balanced = sampler.fit_resample(X_temp, y_temp)
                self.analyze_class_distribution(y_balanced, "SONRASI")
                return X_balanced, y_balanced
            else:
                sampler = SMOTE(random_state=self.random_state)
            
            X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
            self.analyze_class_distribution(y_balanced, "SONRASI")
            self.logger.info("=" * 80)
            
            return X_balanced, y_balanced
            
        except Exception as e:
            self.logger.error(f"❌ Balancing hatası: {e}")
            return X_train, y_train
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        self.logger.info("🔧 XGBoost...")
        
        param_dist = {
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [500, 800, 1000],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        base = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        search = RandomizedSearchCV(
            base, param_dist,
            n_iter=self.n_iter_search,
            cv=cv,
            scoring='balanced_accuracy',
            n_jobs=-1,
            random_state=self.random_state
        )
        
        search.fit(X_train, y_train)
        model = search.best_estimator_
        
        pred = model.predict(X_val)
        acc = accuracy_score(y_val, pred)
        bal_acc = balanced_accuracy_score(y_val, pred)
        f1 = f1_score(y_val, pred, average='macro')
        
        self.logger.info(f"✅ XGBoost: Acc={acc:.4f}, BalAcc={bal_acc:.4f}, F1={f1:.4f}")
        return model, bal_acc
    
    def train_gradient_boost(self, X_train, y_train, X_val, y_val):
        self.logger.info("🔧 GradientBoost...")
        
        param_dist = {
            'n_estimators': [200, 300, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        base = GradientBoostingClassifier(random_state=self.random_state)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        search = RandomizedSearchCV(
            base, param_dist,
            n_iter=self.n_iter_search,
            cv=cv,
            scoring='balanced_accuracy',
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        model = search.best_estimator_
        
        pred = model.predict(X_val)
        bal_acc = balanced_accuracy_score(y_val, pred)
        
        self.logger.info(f"✅ GradientBoost: BalAcc={bal_acc:.4f}")
        return model, bal_acc
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        self.logger.info("🔧 RandomForest...")
        
        param_dist = {
            'n_estimators': [200, 300, 500],
            'max_depth': [15, 20, 25],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', None]
        }
        
        base = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        search = RandomizedSearchCV(
            base, param_dist,
            n_iter=self.n_iter_search,
            cv=cv,
            scoring='balanced_accuracy',
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        model = search.best_estimator_
        
        pred = model.predict(X_val)
        bal_acc = balanced_accuracy_score(y_val, pred)
        
        self.logger.info(f"✅ RandomForest: BalAcc={bal_acc:.4f}")
        return model, bal_acc
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        if not LIGHTGBM_AVAILABLE:
            return None, 0
        
        self.logger.info("🔧 LightGBM...")
        
        param_dist = {
            'n_estimators': [200, 300, 500],
            'max_depth': [5, 7, 9],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [31, 50, 70]
        }
        
        base = lgb.LGBMClassifier(random_state=self.random_state, n_jobs=-1, verbose=-1)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        search = RandomizedSearchCV(
            base, param_dist,
            n_iter=self.n_iter_search,
            cv=cv,
            scoring='balanced_accuracy',
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        model = search.best_estimator_
        
        pred = model.predict(X_val)
        bal_acc = balanced_accuracy_score(y_val, pred)
        
        self.logger.info(f"✅ LightGBM: BalAcc={bal_acc:.4f}")
        return model, bal_acc
    
    def train_base_models(self, X_train, X_test, y_train, y_test):
        self.logger.info("=" * 80)
        self.logger.info("🎯 BASE MODEL EĞİTİMİ")
        self.logger.info("=" * 80)
        
        self.analyze_class_distribution(y_train, "Original")
        X_train_bal, y_train_bal = self.balance_dataset(X_train, y_train)
        
        accuracies = {}
        
        try:
            model, acc = self.train_xgboost(X_train_bal, y_train_bal, X_test, y_test)
            self.models['xgboost'] = model
            accuracies['xgboost'] = acc
        except Exception as e:
            self.logger.error(f"❌ XGBoost: {e}")
        
        try:
            model, acc = self.train_gradient_boost(X_train_bal, y_train_bal, X_test, y_test)
            self.models['gradient_boost'] = model
            accuracies['gradient_boost'] = acc
        except Exception as e:
            self.logger.error(f"❌ GradientBoost: {e}")
        
        try:
            model, acc = self.train_random_forest(X_train_bal, y_train_bal, X_test, y_test)
            self.models['random_forest'] = model
            accuracies['random_forest'] = acc
        except Exception as e:
            self.logger.error(f"❌ RandomForest: {e}")
        
        if LIGHTGBM_AVAILABLE:
            try:
                model, acc = self.train_lightgbm(X_train_bal, y_train_bal, X_test, y_test)
                if model:
                    self.models['lightgbm'] = model
                    accuracies['lightgbm'] = acc
            except Exception as e:
                self.logger.error(f"❌ LightGBM: {e}")
        
        self.logger.info("=" * 80)
        self.logger.info("📊 SONUÇLAR:")
        for name, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
            self.logger.info(f"  {name}: {acc:.4f}")
        
        return accuracies
    
    def train_stacking(self, X_train, X_test, y_train, y_test):
        if len([m for m in self.models.values() if m]) < 2:
            return 0.0
        
        self.logger.info("🎯 Stacking Ensemble...")
        
        estimators = [(name, model) for name, model in self.models.items() if model]
        final = xgb.XGBClassifier(max_depth=3, random_state=self.random_state)
        
        stacking = StackingClassifier(estimators=estimators, final_estimator=final, cv=5, n_jobs=-1)
        stacking.fit(X_train, y_train)
        
        pred = stacking.predict(X_test)
        bal_acc = balanced_accuracy_score(y_test, pred)
        
        self.stacking_model = stacking
        self.logger.info(f"✅ Stacking: {bal_acc:.4f}")
        return bal_acc
    
    def analyze_results(self, X_test, y_test):
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📊 DETAYLI ANALİZ")
        self.logger.info("=" * 80)
        
        for name, model in self.models.items():
            if not model:
                continue
            
            pred = model.predict(X_test)
            
            self.logger.info(f"\n{name.upper()}:")
            self.logger.info(f"  Accuracy: {accuracy_score(y_test, pred):.4f}")
            self.logger.info(f"  Balanced Acc: {balanced_accuracy_score(y_test, pred):.4f}")
            self.logger.info(f"  F1-Macro: {f1_score(y_test, pred, average='macro'):.4f}")
            
            cm = confusion_matrix(y_test, pred)
            self.logger.info(f"\n  Confusion Matrix:")
            self.logger.info(f"  {cm[0]}")
            self.logger.info(f"  {cm[1]}")
            self.logger.info(f"  {cm[2]}")
    
    def save_models(self):
        self.logger.info("💾 Kaydediliyor...")
        
        joblib.dump({
            'models': self.models,
            'scaler': self.scaler,
            'is_trained': True
        }, self.models_dir / "ensemble_models.pkl")
        
        if self.stacking_model:
            joblib.dump({'model': self.stacking_model}, self.models_dir / "stacking_ensemble.pkl")
        
        with open(self.models_dir / "feature_config.json", 'w') as f:
            json.dump({"version": "v3.5_improved"}, f)
        
        self.logger.info(f"✅ Kaydedildi: {self.models_dir}")
    
    def run_training(self):
        start = time.time()
        
        self.logger.info("=" * 80)
        self.logger.info("🚀 IMPROVED ML TRAINER v3.5")
        self.logger.info("=" * 80)
        
        try:
            df = self.load_and_prepare_data()
            df = self.clean_data(df)
            
            X, y_ms, y_score = self.extract_features(df)
            self.metadata['expected_features'] = X.shape[1]
            
            X_train, X_test, y_train, y_test, _, _ = train_test_split(
                X, y_ms, y_score,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y_ms
            )
            
            self.logger.info("🔧 Scaling...")
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            
            base_acc = self.train_base_models(X_train, X_test, y_train, y_test)
            self.analyze_results(X_test, y_test)
            
            stacking_acc = self.train_stacking(X_train, X_test, y_train, y_test)
            
            self.save_models()
            
            elapsed = time.time() - start
            best_acc = max(base_acc.values()) if base_acc else 0.45
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("🎉 EĞİTİM TAMAMLANDI")
            self.logger.info("=" * 80)
            self.logger.info(f"⏱️  Süre: {elapsed/60:.1f} dakika")
            self.logger.info(f"🎯 En İyi: {best_acc:.4f} ({best_acc*100:.2f}%)")
            
            if stacking_acc > 0:
                self.logger.info(f"🎯 Stacking: {stacking_acc:.4f}")
            
            return {
                "success": True,
                "best_accuracy": best_acc,
                "stacking_accuracy": stacking_acc,
                "time_minutes": elapsed/60
            }
            
        except Exception as e:
            self.logger.error(f"❌ Hata: {e}", exc_info=True)
            return {"success": False, "error": str(e)}


def main():
    print("=" * 80)
    print("🚀 IMPROVED ML TRAINER v3.5")
    print("=" * 80)
    
    if not IMBLEARN_AVAILABLE:
        print("\n❌ imbalanced-learn yüklü değil!")
        print("Yüklemek için: pip install imbalanced-learn")
        return
    
    if not FEATURE_ENGINEER_AVAILABLE:
        print("\n❌ enhanced_feature_engineer.py bulunamadı!")
        return
    
    print("✅ Bağımlılıklar OK\n")
    
    trainer = ImprovedModelTrainer(
        use_smote=True,
        balance_method='smote_tomek',
        n_iter_search=50,
        cv_folds=5
    )
    
    result = trainer.run_training()
    
    if result["success"]:
        print(f"\n✅ BAŞARILI!")
        print(f"🎯 En İyi: {result['best_accuracy']:.4f}")
    else:
        print(f"\n❌ HATA: {result['error']}")


if __name__ == "__main__":
    main()
