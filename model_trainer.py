#!/usr/bin/env python3
"""
Predicta ML Production Training Pipeline v2.2
----------------------------------------------
✨ Profesyonel İyileştirmeler:
- ✅ Veri sızıntısı fix (Scaler train'de fit, test'te transform)
- ✅ XGBoost early stopping + optimizasyonlar
- ✅ Confusion matrix (verbose mode)
- ✅ --no-limit flag (SINIRSIZ veri)
- ✅ İyileştirilmiş hata yönetimi
- ✅ Batch processing + NaN/Inf auto-fix

Usage:
    python model_trainer.py --verbose
    python model_trainer.py --no-limit --verbose  # TÜM VERİ
    python model_trainer.py --countries turkey england spain
    python model_trainer.py --max-matches 30000 --batch-size 500
"""

import os
import sys
import argparse
import logging
import pickle
import json
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import xgboost as xgb
    import sklearn
    ML_AVAILABLE = True
except ImportError as e:
    print(f"❌ ML kütüphaneleri eksik: {e}")
    print("📦 Yüklemek için: pip install scikit-learn xgboost pandas numpy")
    sys.exit(1)

# Local imports
try:
    from historical_processor import HistoricalDataProcessor
    from advanced_feature_engineer import AdvancedFeatureEngineer, FEATURE_NAMES
except ImportError as e:
    print(f"❌ Local modül yüklenemedi: {e}")
    print("💡 Script'i proje ana dizininden çalıştırın")
    sys.exit(1)


class ProductionModelTrainer:
    """Production-grade model training pipeline v2.2"""
    
    def __init__(
        self,
        models_dir: str = "data/ai_models_v2",
        raw_data_path: str = "data/raw",
        clubs_path: str = "data/clubs",
        min_matches: int = 10,
        max_matches: int = None,
        test_size: float = 0.2,
        random_state: int = 42,
        version_archive: bool = False,
        verbose: bool = True,
        batch_size: int = 1000,
        early_stopping: bool = True
    ):
        self.models_dir = Path(models_dir)
        self.raw_data_path = Path(raw_data_path)
        self.clubs_path = Path(clubs_path)
        self.min_matches = min_matches
        self.max_matches = max_matches
        self.test_size = test_size
        self.random_state = random_state
        self.version_archive = version_archive
        self.verbose = verbose
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        
        # Setup logging
        self._setup_logging()
        
        # Versiyonlama
        if version_archive:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = self.models_dir / f"v_{timestamp}"
            self.logger.info(f"📁 Versioned output: {self.output_dir}")
        else:
            self.output_dir = self.models_dir
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.feature_engineer = AdvancedFeatureEngineer(model_path=str(self.models_dir))
        self.history_processor = HistoricalDataProcessor(
            str(self.raw_data_path),
            str(self.clubs_path)
        )
        
        # Models
        self.ms_models = {}
        self.score_model = None
        self.score_space = []
        self.scaler = StandardScaler()
        
        # Enhanced metadata
        self.metadata = {
            'training_date': datetime.now().isoformat(),
            'version': 'v2.2',
            'random_state': random_state,
            'test_size': test_size,
            'min_matches': min_matches,
            'max_matches': max_matches if max_matches else 'unlimited',
            'batch_size': batch_size,
            'early_stopping': early_stopping,
            'feature_names': FEATURE_NAMES.copy(),
            'feature_count': len(FEATURE_NAMES),
            'sklearn_version': sklearn.__version__,
            'xgboost_version': xgb.__version__,
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
        }
    
    def _setup_logging(self):
        """Configure logging"""
        log_format = "%(asctime)s [%(levelname)8s] %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        level = logging.INFO if self.verbose else logging.WARNING
        
        logging.basicConfig(
            level=level,
            format=log_format,
            datefmt=date_format,
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        
        self.logger = logging.getLogger("ProductionTrainer")
    
    def load_historical_data(self, countries: List[str] = None) -> pd.DataFrame:
        """Load and validate historical data"""
        self.logger.info("=" * 70)
        self.logger.info("📂 LOADING HISTORICAL DATA (UNLIMITED MODE)")
        self.logger.info("=" * 70)
        
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data path not found: {self.raw_data_path}")
        
        all_countries = [
            d.name for d in self.raw_data_path.iterdir()
            if d.is_dir() and d.name.lower() != "clubs"
        ]
        
        if not all_countries:
            raise ValueError(f"No country folders found in {self.raw_data_path}")
        
        if countries:
            target_countries = [c for c in all_countries if c.lower() in [x.lower() for x in countries]]
            if not target_countries:
                raise ValueError(f"None of {countries} found in {all_countries}")
        else:
            target_countries = all_countries
        
        self.logger.info(f"🌍 Countries: {len(target_countries)} → {', '.join(target_countries[:5])}")
        if len(target_countries) > 5:
            self.logger.info(f"             ... and {len(target_countries) - 5} more")
        self.logger.info(f"🔓 Data limit: {'UNLIMITED' if not self.max_matches else f'{self.max_matches:,} matches'}")
        
        all_matches = []
        stats = {'loaded': 0, 'failed': 0, 'countries': {}}
        
        for country in target_countries:
            try:
                matches = self.history_processor.load_country_data(country)
                
                if matches:
                    all_matches.extend(matches)
                    stats['loaded'] += len(matches)
                    stats['countries'][country] = len(matches)
                    self.logger.info(f"   ✅ {country:20s}: {len(matches):6,} matches")
                else:
                    stats['failed'] += 1
                    stats['countries'][country] = 0
                    
            except Exception as e:
                stats['failed'] += 1
                stats['countries'][country] = 0
                self.logger.error(f"   ❌ {country:20s}: {e}")
        
        if not all_matches:
            raise ValueError("No matches loaded from any country")
        
        df = pd.DataFrame(all_matches)
        
        # Max limit kontrolü
        if self.max_matches and len(df) > self.max_matches:
            self.logger.warning(f"⚠️  Applying max limit: {len(df):,} → {self.max_matches:,}")
            df = df.sample(n=self.max_matches, random_state=self.random_state)
        
        # Metadata
        self.metadata['data_stats'] = stats
        self.metadata['total_raw_matches'] = len(df)
        self.metadata['countries_count'] = len([c for c in stats['countries'].values() if c > 0])
        self.metadata['leagues'] = df['league'].nunique() if 'league' in df.columns else 0
        
        self.logger.info(f"\n📊 RAW DATA SUMMARY")
        self.logger.info(f"   Total matches: {len(df):,}")
        self.logger.info(f"   Countries: {self.metadata['countries_count']}")
        self.logger.info(f"   Unique leagues: {self.metadata['leagues']}")
        
        if 'date' in df.columns:
            self.logger.info(f"   Date range: {df['date'].min()} to {df['date'].max()}")
            self.metadata['date_range'] = {'min': str(df['date'].min()), 'max': str(df['date'].max())}
        
        return df
    
    def clean_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate match data"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("🧹 CLEANING AND VALIDATION")
        self.logger.info("=" * 70)
        
        initial_count = len(df)
        
        # Duplicates
        df = df.drop_duplicates(subset=['home_team', 'away_team', 'date'], keep='first')
        dup_removed = initial_count - len(df)
        if dup_removed > 0:
            self.logger.info(f"   Duplicates removed: {dup_removed:,}")
        
        # Required columns
        required = ['home_team', 'away_team', 'home_score', 'away_score']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Missing data
        before = len(df)
        df = df.dropna(subset=required)
        missing_removed = before - len(df)
        if missing_removed > 0:
            self.logger.info(f"   Missing data removed: {missing_removed:,}")
        
        # Score conversion
        df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce').fillna(0).astype(int)
        df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce').fillna(0).astype(int)
        
        # Unrealistic scores
        before = len(df)
        df = df[(df['home_score'] <= 10) & (df['away_score'] <= 10)]
        unrealistic = before - len(df)
        if unrealistic > 0:
            self.logger.info(f"   Unrealistic scores removed: {unrealistic:,}")
        
        # Result calculation
        def calc_result(row):
            if row['home_score'] > row['away_score']:
                return '1'
            elif row['home_score'] < row['away_score']:
                return '2'
            else:
                return 'X'
        
        if 'result' not in df.columns or df['result'].isna().any():
            df['result'] = df.apply(calc_result, axis=1)
        
        valid_results = ['1', 'X', '2']
        df = df[df['result'].isin(valid_results)]
        
        # Defaults
        if 'odds' not in df.columns:
            df['odds'] = df.apply(lambda x: {'1': 2.0, 'X': 3.0, '2': 3.5}, axis=1)
        
        if 'league' not in df.columns:
            df['league'] = 'Unknown'
        df['league'] = df['league'].fillna('Unknown')
        
        # Statistics
        home_wins = (df['result'] == '1').sum()
        draws = (df['result'] == 'X').sum()
        away_wins = (df['result'] == '2').sum()
        
        avg_home_score = df['home_score'].mean()
        avg_away_score = df['away_score'].mean()
        avg_total_goals = (df['home_score'] + df['away_score']).mean()
        
        self.logger.info(f"\n✅ CLEANED DATA")
        self.logger.info(f"   Final count: {len(df):,} matches")
        self.logger.info(f"   Home wins: {home_wins:,} ({home_wins/len(df)*100:.1f}%)")
        self.logger.info(f"   Draws: {draws:,} ({draws/len(df)*100:.1f}%)")
        self.logger.info(f"   Away wins: {away_wins:,} ({away_wins/len(df)*100:.1f}%)")
        self.logger.info(f"   Avg goals: Home={avg_home_score:.2f}, Away={avg_away_score:.2f}, Total={avg_total_goals:.2f}")
        
        # Metadata
        self.metadata['total_clean_matches'] = len(df)
        self.metadata['result_distribution'] = {
            'home_wins': int(home_wins),
            'draws': int(draws),
            'away_wins': int(away_wins),
            'home_win_pct': round(home_wins/len(df)*100, 2),
            'draw_pct': round(draws/len(df)*100, 2),
            'away_win_pct': round(away_wins/len(df)*100, 2)
        }
        self.metadata['average_scores'] = {
            'home': round(avg_home_score, 2),
            'away': round(avg_away_score, 2),
            'total': round(avg_total_goals, 2)
        }
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract features with batch processing and error tracking"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("🔧 FEATURE ENGINEERING (BATCH MODE)")
        self.logger.info("=" * 70)
        
        total_rows = len(df)
        num_batches = (total_rows + self.batch_size - 1) // self.batch_size
        
        self.logger.info(f"   📦 Batches: {num_batches} (size={self.batch_size:,})")
        
        X_list = []
        y_ms_list = []
        y_score_list = []
        
        error_stats = {
            'none_returns': 0,
            'nan_inf_fixed': 0,
            'exceptions': 0,
            'total_skipped': 0
        }
        
        for batch_num in range(num_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx]
            
            if (batch_num + 1) % 10 == 0 or batch_num == num_batches - 1:
                self.logger.info(f"   Processing batch {batch_num + 1}/{num_batches} ({start_idx:,}-{end_idx:,})")
            
            for _, row in batch_df.iterrows():
                try:
                    match_data = {
                        'home_team': row['home_team'],
                        'away_team': row['away_team'],
                        'league': row['league'],
                        'odds': row['odds'],
                        'date': row.get('date', datetime.now().isoformat())
                    }
                    
                    features = self.feature_engineer.extract_features(match_data)
                    
                    if features is None:
                        error_stats['none_returns'] += 1
                        error_stats['total_skipped'] += 1
                        continue
                    
                    # NaN/Inf auto-fix
                    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                        error_stats['nan_inf_fixed'] += 1
                    
                    ms_label = {'1': 0, 'X': 1, '2': 2}[row['result']]
                    score_label = f"{row['home_score']}-{row['away_score']}"
                    
                    X_list.append(features)
                    y_ms_list.append(ms_label)
                    y_score_list.append(score_label)
                    
                    self._update_feature_history(row)
                    
                except KeyError:
                    error_stats['exceptions'] += 1
                    error_stats['total_skipped'] += 1
                except Exception:
                    error_stats['exceptions'] += 1
                    error_stats['total_skipped'] += 1
        
        if not X_list:
            raise ValueError("No valid features extracted!")
        
        X = np.array(X_list, dtype=np.float32)
        y_ms = np.array(y_ms_list, dtype=np.int32)
        
        skip_rate = error_stats['total_skipped'] / total_rows * 100
        
        self.logger.info(f"\n✅ FEATURES READY")
        self.logger.info(f"   Valid samples: {len(X):,}")
        self.logger.info(f"   Features per sample: {X.shape[1]}")
        self.logger.info(f"   Skipped: {error_stats['total_skipped']:,} ({skip_rate:.2f}%)")
        self.logger.info(f"      - None returns: {error_stats['none_returns']:,}")
        self.logger.info(f"      - NaN/Inf fixed: {error_stats['nan_inf_fixed']:,}")
        self.logger.info(f"      - Exceptions: {error_stats['exceptions']:,}")
        
        # Save feature data
        self.feature_engineer._save_data()
        self.logger.info(f"   💾 Feature history saved")
        
        # Metadata
        self.metadata['training_samples'] = len(X)
        self.metadata['features_count'] = X.shape[1]
        self.metadata['error_stats'] = error_stats
        self.metadata['skip_rate_pct'] = round(skip_rate, 2)
        
        return X, y_ms, y_score_list
    
    def _update_feature_history(self, row):
        """Update feature engineer history"""
        result = row['result']
        
        home_result = 'W' if result == '1' else ('D' if result == 'X' else 'L')
        away_result = 'L' if result == '1' else ('D' if result == 'X' else 'W')
        
        self.feature_engineer.update_team_history(row['home_team'], {
            'result': home_result,
            'goals_for': int(row['home_score']),
            'goals_against': int(row['away_score']),
            'date': row.get('date', ''),
            'venue': 'home'
        })
        
        self.feature_engineer.update_team_history(row['away_team'], {
            'result': away_result,
            'goals_for': int(row['away_score']),
            'goals_against': int(row['home_score']),
            'date': row.get('date', ''),
            'venue': 'away'
        })
        
        self.feature_engineer.update_h2h_history(
            row['home_team'], row['away_team'],
            {
                'result': result,
                'home_goals': int(row['home_score']),
                'away_goals': int(row['away_score'])
            }
        )
        
        self.feature_engineer.update_league_results(row['league'], result)
    
    def train_match_result_models(self, X_train, X_test, y_train, y_test) -> Dict[str, float]:
        """Train MS prediction models with early stopping"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("🎯 TRAINING MATCH RESULT MODELS")
        self.logger.info("=" * 70)
        
        accuracies = {}
        
        # XGBoost with early stopping
        self.logger.info("\n📚 XGBoost...")
        self.ms_models['xgboost'] = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,
            tree_method='hist',
            eval_metric='mlogloss',
            early_stopping_rounds=15 if self.early_stopping else None,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )
        
        if self.early_stopping:
            eval_set = [(X_test, y_test)]
            self.ms_models['xgboost'].fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            best_iter = self.ms_models['xgboost'].best_iteration
            self.logger.info(f"   Best iteration: {best_iter}")
        else:
            self.ms_models['xgboost'].fit(X_train, y_train)
        
        acc = self.ms_models['xgboost'].score(X_test, y_test)
        accuracies['xgboost'] = acc
        self.logger.info(f"   ✅ Accuracy: {acc*100:.2f}%")
        
        # Random Forest
        self.logger.info("\n🌲 Random Forest...")
        self.ms_models['random_forest'] = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.ms_models['random_forest'].fit(X_train, y_train)
        acc = self.ms_models['random_forest'].score(X_test, y_test)
        accuracies['random_forest'] = acc
        self.logger.info(f"   ✅ Accuracy: {acc*100:.2f}%")
        
        # Gradient Boost
        self.logger.info("\n⚡ Gradient Boost...")
        self.ms_models['gradient_boost'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=self.random_state
        )
        self.ms_models['gradient_boost'].fit(X_train, y_train)
        acc = self.ms_models['gradient_boost'].score(X_test, y_test)
        accuracies['gradient_boost'] = acc
        self.logger.info(f"   ✅ Accuracy: {acc*100:.2f}%")
        
        # Detailed report (verbose mode)
        if self.verbose:
            y_pred = self.ms_models['xgboost'].predict(X_test)
            self.logger.info(f"\n📊 XGBoost Classification Report:")
            report = classification_report(y_test, y_pred, target_names=['Home Win', 'Draw', 'Away Win'])
            for line in report.split('\n'):
                if line.strip():
                    self.logger.info(f"   {line}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            self.logger.info(f"\n📊 Confusion Matrix:")
            self.logger.info(f"   {'':10s} {'Pred:1':>10s} {'Pred:X':>10s} {'Pred:2':>10s}")
            self.logger.info(f"   {'True:1':10s} {cm[0][0]:10d} {cm[0][1]:10d} {cm[0][2]:10d}")
            self.logger.info(f"   {'True:X':10s} {cm[1][0]:10d} {cm[1][1]:10d} {cm[1][2]:10d}")
            self.logger.info(f"   {'True:2':10s} {cm[2][0]:10d} {cm[2][1]:10d} {cm[2][2]:10d}")
        
        return accuracies
    
    def train_score_model(self, X_train, X_test, y_score_train, y_score_test) -> float:
        """Train score prediction model"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("⚽ TRAINING SCORE MODEL")
        self.logger.info("=" * 70)
        
        score_counts = Counter(y_score_train)
        common_scores = [s for s, c in score_counts.items() if c >= 1]
        
        if len(common_scores) < 10:
            self.logger.warning("   ⚠️  Low score diversity, using all scores")
            common_scores = list(score_counts.keys())
        
        self.score_space = sorted(common_scores) + ["OTHER"]
        
        self.logger.info(f"   Score classes: {len(self.score_space)}")
        self.logger.info(f"   Top 20 scores: {sorted(score_counts.items(), key=lambda x: x[1], reverse=True)[:20]}")
        
        # Metadata
        self.metadata['score_distribution'] = dict(sorted(score_counts.items(), key=lambda x: x[1], reverse=True)[:20])
        self.metadata['score_classes_count'] = len(self.score_space)
        
        def encode_score(score):
            return self.score_space.index(score) if score in self.score_space else len(self.score_space) - 1
        
        y_train_enc = [encode_score(s) for s in y_score_train]
        y_test_enc = [encode_score(s) for s in y_score_test]
        
        self.logger.info("\n🎯 Training Random Forest Score Model...")
        self.score_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.score_model.fit(X_train, y_train_enc)
        acc = self.score_model.score(X_test, y_test_enc)
        
        self.logger.info(f"   ✅ Accuracy: {acc*100:.2f}%")
        
        return acc
    
    def save_models_atomic(self):
        """Atomically save all models + scaler"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("💾 SAVING MODELS (ATOMIC)")
        self.logger.info("=" * 70)
        
        # 1. MS models (with scaler embedded)
        ms_path = self.output_dir / "ensemble_models.pkl"
        ms_temp = self.output_dir / "ensemble_models.pkl.tmp"
        
        try:
            ms_data = {
                'models': self.ms_models,
                'scaler': self.scaler,  # ✅ Embedded
                'is_trained': True,
                'metadata': self.metadata.copy()
            }
            
            with open(ms_temp, 'wb') as f:
                pickle.dump(ms_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            if ms_temp.stat().st_size < 1000:
                raise ValueError("MS model file too small")
            
            ms_temp.replace(ms_path)
            size = ms_path.stat().st_size
            self.logger.info(f"   ✅ MS models: {ms_path.name} ({size:,} bytes)")
            
        except Exception as e:
            self.logger.error(f"   ❌ MS models save failed: {e}")
            if ms_temp.exists():
                ms_temp.unlink()
            raise
        
        # 2. Scaler (separate - backward compatibility)
        scaler_path = self.output_dir / "scaler.pkl"
        scaler_temp = self.output_dir / "scaler.pkl.tmp"
        
        try:
            with open(scaler_temp, 'wb') as f:
                pickle.dump(self.scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            if scaler_temp.stat().st_size < 100:
                raise ValueError("Scaler file too small")
            
            scaler_temp.replace(scaler_path)
            size = scaler_path.stat().st_size
            self.logger.info(f"   ✅ Scaler: {scaler_path.name} ({size:,} bytes)")
            
        except Exception as e:
            self.logger.error(f"   ❌ Scaler save failed: {e}")
            if scaler_temp.exists():
                scaler_temp.unlink()
            raise
        
        # 3. Score model
        score_path = self.output_dir / "score_model.pkl"
        score_temp = self.output_dir / "score_model.pkl.tmp"
        
        try:
            score_data = {
                'model': self.score_model,
                'score_space': self.score_space,
                'metadata': self.metadata.copy()
            }
            
            with open(score_temp, 'wb') as f:
                pickle.dump(score_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            if score_temp.stat().st_size < 1000:
                raise ValueError("Score model file too small")
            
            score_temp.replace(score_path)
            size = score_path.stat().st_size
            self.logger.info(f"   ✅ Score model: {score_path.name} ({size:,} bytes)")
            
        except Exception as e:
            self.logger.error(f"   ❌ Score model save failed: {e}")
            if score_temp.exists():
                score_temp.unlink()
            raise
        
        # 4. Metadata JSON
        meta_path = self.output_dir / "training_metadata.json"
        try:
            with open(meta_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            self.logger.info(f"   ✅ Metadata: {meta_path.name}")
        except Exception as e:
            self.logger.warning(f"   ⚠️  Metadata save failed: {e}")
        
        # 5. Versiyonlama
        if self.version_archive and self.output_dir != self.models_dir:
            self.logger.info(f"\n📋 Copying to main directory: {self.models_dir}")
            try:
                for file in [ms_path, scaler_path, score_path, meta_path]:
                    if file.exists():
                        target = self.models_dir / file.name
                        shutil.copy2(file, target)
                        self.logger.info(f"   ✅ Copied: {file.name}")
            except Exception as e:
                self.logger.warning(f"   ⚠️  Copy failed: {e}")
    
    def run_full_pipeline(self, countries: List[str] = None) -> Dict[str, Any]:
        """Execute full training pipeline"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🚀 PREDICTA ML - PRODUCTION TRAINING v2.2")
        self.logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # 1. Load
            df = self.load_historical_data(countries)
            
            # 2. Clean
            df = self.clean_and_validate(df)
            
            if len(df) < self.min_matches:
                raise ValueError(f"Insufficient data: {len(df):,} < {self.min_matches:,}")
            
            # 3. Features
            X, y_ms, y_score = self.prepare_features(df)
            
            # 4. ✅ FIX: Split BEFORE scaling (prevent data leakage)
            X_train, X_test, y_ms_train, y_ms_test, y_score_train, y_score_test = train_test_split(
                X, y_ms, y_score,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y_ms
            )
            
            # 5. ✅ FIX: Fit scaler on train only, transform both
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.logger.info(f"\n📊 TRAIN/TEST SPLIT")
            self.logger.info(f"   Train: {len(X_train):,} samples")
            self.logger.info(f"   Test: {len(X_test):,} samples")
            self.logger.info(f"   Scaler fitted on train only ✅")
            
            self.metadata['train_samples'] = len(X_train)
            self.metadata['test_samples'] = len(X_test)
            
            # 6. Train MS
            ms_acc = self.train_match_result_models(X_train_scaled, X_test_scaled, y_ms_train, y_ms_test)
            self.metadata['ms_accuracies'] = ms_acc
            
            # 7. Train score
            score_acc = self.train_score_model(X_train_scaled, X_test_scaled, y_score_train, y_score_test)
            self.metadata['score_accuracy'] = score_acc
            
            # 8. Save
            self.save_models_atomic()
            
            # Summary
            duration = (datetime.now() - start_time).total_seconds()
            self.metadata['training_duration_seconds'] = round(duration, 2)
            self.metadata['success'] = True
            self.metadata['completion_time'] = datetime.now().isoformat()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("✅ TRAINING COMPLETE!")
            self.logger.info("=" * 80)
            self.logger.info(f"⏱️  Duration: {duration:.1f}s ({duration/60:.1f} min)")
            self.logger.info(f"📊 Total Matches: {len(df):,}")
            self.logger.info(f"🎯 Train Set: {len(X_train):,} samples")
            self.logger.info(f"🧪 Test Set: {len(X_test):,} samples")
            self.logger.info(f"\n📈 Match Result Model Accuracies:")
            for model, acc in ms_acc.items():
                self.logger.info(f"   {model:20s}: {acc*100:5.2f}%")
            self.logger.info(f"\n⚽ Score Model Accuracy: {score_acc*100:.2f}%")
            self.logger.info(f"\n📁 Output Directory: {self.output_dir}")
            
            if self.version_archive:
                self.logger.info(f"📦 Archive: {self.output_dir}")
                self.logger.info(f"🔗 Active: {self.models_dir}")
            
            self.logger.info("=" * 80)
            
            # Final metadata kaydet
            final_meta_path = self.output_dir / "training_metadata.json"
            try:
                with open(final_meta_path, 'w') as f:
                    json.dump(self.metadata, f, indent=2)
            except:
                pass
            
            return {
                'success': True,
                'metadata': self.metadata,
                'output_dir': str(self.output_dir)
            }
            
        except Exception as e:
            self.logger.error(f"\n❌ TRAINING FAILED: {e}", exc_info=True)
            self.metadata['success'] = False
            self.metadata['error'] = str(e)
            return {
                'success': False,
                'error': str(e),
                'metadata': self.metadata
            }


def parse_args():
    """Parse CLI arguments"""
    parser = argparse.ArgumentParser(
        description="Predicta ML Production Training Pipeline v2.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sınırsız veri ile tüm ülkeler
  python model_trainer.py --verbose
  
  # --no-limit bayrağı ile (açıkça belirtmek için)
  python model_trainer.py --no-limit --verbose
  
  # Belirli ülkeler
  python model_trainer.py --countries turkey england spain --verbose
  
  # Maximum veri limiti (bellek sınırlaması için)
  python model_trainer.py --max-matches 30000 --verbose
  
  # Versiyonlama ile backup
  python model_trainer.py --version-archive --verbose
  
  # Early stopping kapalı
  python model_trainer.py --no-early-stopping --verbose
        """
    )
    
    parser.add_argument(
        '--models-dir',
        default='data/ai_models_v2',
        help='Model output directory (default: data/ai_models_v2)'
    )
    
    parser.add_argument(
        '--raw-data',
        default='data/raw',
        help='Raw historical data path (default: data/raw)'
    )
    
    parser.add_argument(
        '--clubs',
        default='data/clubs',
        help='Clubs data path (default: data/clubs)'
    )
    
    parser.add_argument(
        '--countries',
        nargs='+',
        default=None,
        help='Specific countries to train on (default: all)'
    )
    
    parser.add_argument(
        '--min-matches',
        type=int,
        default=10,
        help='Minimum matches required (default: 10)'
    )
    
    parser.add_argument(
        '--max-matches',
        type=int,
        default=None,
        help='Maximum matches to use (default: None = unlimited)'
    )
    
    parser.add_argument(
        '--no-limit',
        action='store_true',
        help='Explicitly use unlimited data (same as --max-matches=None)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size (default: 0.2)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for feature processing (default: 1000)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random state seed (default: 42)'
    )
    
    parser.add_argument(
        '--version-archive',
        action='store_true',
        help='Create versioned backup (v_YYYYMMDD_HHMMSS)'
    )
    
    parser.add_argument(
        '--no-early-stopping',
        action='store_true',
        help='Disable XGBoost early stopping'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output with detailed reports'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # --no-limit flag ile max_matches'i None yap
    if args.no_limit:
        args.max_matches = None
    
    print("\n" + "=" * 80)
    print("🚀 PREDICTA ML - PRODUCTION TRAINING v2.2")
    print("=" * 80)
    print(f"📊 Min matches: {args.min_matches:,}")
    print(f"📊 Max matches: {'UNLIMITED ✅' if args.max_matches is None else f'{args.max_matches:,}'}")
    print(f"🎯 Batch size: {args.batch_size:,}")
    print(f"🔄 Early stopping: {'✅ Enabled' if not args.no_early_stopping else '❌ Disabled'}")
    print(f"📁 Data path: {args.raw_data}")
    print("=" * 80 + "\n")
    
    trainer = ProductionModelTrainer(
        models_dir=args.models_dir,
        raw_data_path=args.raw_data,
        clubs_path=args.clubs,
        min_matches=args.min_matches,
        max_matches=args.max_matches,
        test_size=args.test_size,
        random_state=args.seed,
        version_archive=args.version_archive,
        verbose=args.verbose,
        batch_size=args.batch_size,
        early_stopping=not args.no_early_stopping
    )
    
    result = trainer.run_full_pipeline(countries=args.countries)
    
    if result['success']:
        print("\n✅ Training successful!")
        print(f"💡 Models saved to: {result['output_dir']}")
        print(f"🚀 Start server: python main.py")
        sys.exit(0)
    else:
        print(f"\n❌ Training failed: {result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
