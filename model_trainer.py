#!/usr/bin/env python3
"""
Predicta ML Production Training Pipeline v2
--------------------------------------------
✨ Yeni özellikler:
- Scaler ayrı dosya
- Versiyonlama desteği
- Gelişmiş metadata
- NaN/None güvenli feature extraction
- Detaylı istatistikler

Usage:
    python train_models.py
    python train_models.py --countries turkey england spain
    python train_models.py --version-archive
    python train_models.py --min-matches 500 --test-size 0.25 --seed 42
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
    """Production-grade model training pipeline with versioning"""
    
    def __init__(
        self,
        models_dir: str = "data/ai_models_v2",
        raw_data_path: str = "data/raw",
        clubs_path: str = "data/clubs",
        min_matches: int = 100,
        test_size: float = 0.2,
        random_state: int = 42,
        version_archive: bool = False,
        verbose: bool = True
    ):
        self.models_dir = Path(models_dir)
        self.raw_data_path = Path(raw_data_path)
        self.clubs_path = Path(clubs_path)
        self.min_matches = min_matches
        self.test_size = test_size
        self.random_state = random_state
        self.version_archive = version_archive
        self.verbose = verbose
        
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
            'training_timestamp': datetime.now().timestamp(),
            'version': 'v2.0',
            'random_state': random_state,
            'test_size': test_size,
            'min_matches': min_matches,
            'feature_names': FEATURE_NAMES.copy(),
            'feature_count': len(FEATURE_NAMES),
            'sklearn_version': sklearn.__version__,
            'xgboost_version': xgb.__version__,
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
            'output_directory': str(self.output_dir)
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
        self.logger.info("📂 LOADING HISTORICAL DATA")
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
        
        self.logger.info(f"🌍 Loading {len(target_countries)} countries: {', '.join(target_countries)}")
        
        all_matches = []
        stats = {'loaded': 0, 'failed': 0, 'countries': {}}
        
        for country in target_countries:
            try:
                self.logger.info(f"📄 Processing {country}...")
                matches = self.history_processor.load_country_data(country)
                
                if matches:
                    all_matches.extend(matches)
                    stats['loaded'] += len(matches)
                    stats['countries'][country] = len(matches)
                    self.logger.info(f"   ✅ {country}: {len(matches)} matches")
                else:
                    stats['failed'] += 1
                    stats['countries'][country] = 0
                    self.logger.warning(f"   ⚠️  {country}: No data")
                    
            except Exception as e:
                stats['failed'] += 1
                stats['countries'][country] = 0
                self.logger.error(f"   ❌ {country}: {e}")
        
        if not all_matches:
            raise ValueError("No matches loaded from any country")
        
        df = pd.DataFrame(all_matches)
        
        # Metadata güncellemeleri
        self.metadata['data_stats'] = stats
        self.metadata['total_raw_matches'] = len(df)
        self.metadata['countries_count'] = len([c for c in stats['countries'].values() if c > 0])
        self.metadata['leagues'] = df['league'].nunique() if 'league' in df.columns else 0
        
        self.logger.info(f"\n📊 RAW DATA SUMMARY")
        self.logger.info(f"   Total matches: {len(df)}")
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
        
        df = df.drop_duplicates(subset=['home_team', 'away_team', 'date'], keep='first')
        self.logger.info(f"   Duplicates removed: {initial_count - len(df)}")
        
        required = ['home_team', 'away_team', 'home_score', 'away_score']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        before = len(df)
        df = df.dropna(subset=required)
        self.logger.info(f"   Rows with missing data removed: {before - len(df)}")
        
        df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce').fillna(0).astype(int)
        df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce').fillna(0).astype(int)
        
        before = len(df)
        df = df[(df['home_score'] <= 10) & (df['away_score'] <= 10)]
        self.logger.info(f"   Unrealistic scores removed: {before - len(df)}")
        
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
        
        if 'odds' not in df.columns:
            df['odds'] = df.apply(lambda x: {'1': 2.0, 'X': 3.0, '2': 3.5}, axis=1)
        
        if 'league' not in df.columns:
            df['league'] = 'Unknown'
        df['league'] = df['league'].fillna('Unknown')
        
        # İstatistikler
        home_wins = (df['result'] == '1').sum()
        draws = (df['result'] == 'X').sum()
        away_wins = (df['result'] == '2').sum()
        
        avg_home_score = df['home_score'].mean()
        avg_away_score = df['away_score'].mean()
        avg_total_goals = (df['home_score'] + df['away_score']).mean()
        
        self.logger.info(f"\n✅ CLEANED DATA")
        self.logger.info(f"   Final count: {len(df)} matches")
        self.logger.info(f"   Home wins: {home_wins} ({home_wins/len(df)*100:.1f}%)")
        self.logger.info(f"   Draws: {draws} ({draws/len(df)*100:.1f}%)")
        self.logger.info(f"   Away wins: {away_wins} ({away_wins/len(df)*100:.1f}%)")
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
        """Extract features with NaN/None safety"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("🔧 FEATURE ENGINEERING")
        self.logger.info("=" * 70)
        
        X_list = []
        y_ms_list = []
        y_score_list = []
        
        skipped = 0
        none_returns = 0
        nan_features = 0
        
        for idx, row in df.iterrows():
            try:
                match_data = {
                    'home_team': row['home_team'],
                    'away_team': row['away_team'],
                    'league': row['league'],
                    'odds': row['odds'],
                    'date': row.get('date', datetime.now().isoformat())
                }
                
                # Feature extraction - NaN/None güvenli
                features = self.feature_engineer.extract_features(match_data)
                
                # None kontrolü
                if features is None:
                    none_returns += 1
                    skipped += 1
                    continue
                
                # NaN/Inf kontrolü
                if np.any(np.isnan(feature_array)) or np.any(np.isinf(feature_array)):
                    logger.warning(f"NaN/Inf düzeltildi: {features}")
                    feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)

                
                ms_label = {'1': 0, 'X': 1, '2': 2}[row['result']]
                score_label = f"{row['home_score']}-{row['away_score']}"
                
                X_list.append(features)
                y_ms_list.append(ms_label)
                y_score_list.append(score_label)
                
                self._update_feature_history(row)
                
                if (idx + 1) % 1000 == 0:
                    self.logger.info(f"   Processed: {idx + 1}/{len(df)}")
                
            except Exception as e:
                self.logger.warning(f"   ⚠️  Row {idx} error: {e}")
                skipped += 1
                continue
        
        if not X_list:
            raise ValueError("No valid features extracted!")
        
        X = np.array(X_list, dtype=np.float32)
        y_ms = np.array(y_ms_list, dtype=np.int32)
        
        self.logger.info(f"\n✅ FEATURES READY")
        self.logger.info(f"   Valid samples: {len(X)}")
        self.logger.info(f"   Features per sample: {X.shape[1]}")
        self.logger.info(f"   Skipped total: {skipped}")
        self.logger.info(f"      - None returns: {none_returns}")
        self.logger.info(f"      - NaN/Inf: {nan_features}")
        self.logger.info(f"      - Other errors: {skipped - none_returns - nan_features}")
        
        # Feature data kaydet
        self.feature_engineer._save_data()
        self.logger.info(f"   💾 Feature history saved")
        
        # Metadata
        self.metadata['training_samples'] = len(X)
        self.metadata['features_count'] = X.shape[1]
        self.metadata['samples_skipped'] = skipped
        self.metadata['skip_reasons'] = {
            'none_returns': none_returns,
            'nan_inf': nan_features,
            'other': skipped - none_returns - nan_features
        }
        
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
        """Train MS prediction models"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("🎯 TRAINING MATCH RESULT MODELS")
        self.logger.info("=" * 70)
        
        accuracies = {}
        
        # XGBoost
        self.logger.info("\n📚 XGBoost...")
        self.ms_models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,
            random_state=self.random_state,
            verbosity=0
        )
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
        
        if self.verbose:
            y_pred = self.ms_models['xgboost'].predict(X_test)
            self.logger.info(f"\n📊 XGBoost Classification Report:")
            report = classification_report(y_test, y_pred, target_names=['Home Win', 'Draw', 'Away Win'])
            for line in report.split('\n'):
                if line.strip():
                    self.logger.info(f"   {line}")
        
        return accuracies
    
    def train_score_model(self, X_train, X_test, y_score_train, y_score_test) -> float:
        """Train score prediction model"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("⚽ TRAINING SCORE MODEL")
        self.logger.info("=" * 70)
        
        score_counts = Counter(y_score_train)
        common_scores = [s for s, c in score_counts.items() if c >= 15]
        
        if len(common_scores) < 10:
            self.logger.warning("   ⚠️  Low score diversity, using all scores")
            common_scores = list(score_counts.keys())
        
        self.score_space = sorted(common_scores) + ["OTHER"]
        
        self.logger.info(f"   Score classes: {len(self.score_space)}")
        self.logger.info(f"   Top 10 scores: {sorted(score_counts.items(), key=lambda x: x[1], reverse=True)[:10]}")
        
        # Metadata - skor dağılımı
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
        """Atomically save all models + scaler separately"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("💾 SAVING MODELS (ATOMIC)")
        self.logger.info("=" * 70)
        
        # 1. MS models
        ms_path = self.output_dir / "ensemble_models.pkl"
        ms_temp = self.output_dir / "ensemble_models.pkl.tmp"
        
        try:
            ms_data = {
                'models': self.ms_models,
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
        
        # 2. Scaler (AYRI DOSYA)
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
        
        # 5. Versiyonlama yapıldıysa, main dizine de kopyala
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
        self.logger.info("🚀 PREDICTA ML - PRODUCTION TRAINING PIPELINE v2")
        self.logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # 1. Load
            df = self.load_historical_data(countries)
            
            # 2. Clean
            df = self.clean_and_validate(df)
            
            if len(df) < self.min_matches:
                raise ValueError(f"Insufficient data: {len(df)} < {self.min_matches}")
            
            # 3. Features
            X, y_ms, y_score = self.prepare_features(df)
            
            # 4. Scale and split
            X_scaled = self.scaler.fit_transform(X)
            
            X_train, X_test, y_ms_train, y_ms_test, y_score_train, y_score_test = train_test_split(
                X_scaled, y_ms, y_score,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y_ms
            )
            
            self.logger.info(f"\n📊 TRAIN/TEST SPLIT")
            self.logger.info(f"   Train: {len(X_train)} samples")
            self.logger.info(f"   Test: {len(X_test)} samples")
            
            self.metadata['train_samples'] = len(X_train)
            self.metadata['test_samples'] = len(X_test)
            
            # 5. Train MS
            ms_acc = self.train_match_result_models(X_train, X_test, y_ms_train, y_ms_test)
            self.metadata['ms_accuracies'] = ms_acc
            
            # 6. Train score
            score_acc = self.train_score_model(X_train, X_test, y_score_train, y_score_test)
            self.metadata['score_accuracy'] = score_acc
            
            # 7. Save
            self.save_models_atomic()
            
            # Summary
            duration = (datetime.now() - start_time).total_seconds()
            self.metadata['training_duration_seconds'] = round(duration, 2)
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("✅ TRAINING COMPLETE!")
            self.logger.info("=" * 80)
            self.logger.info(f"⏱️  Duration: {duration:.1f}s")
            self.logger.info(f"📊 Total Matches: {len(df)}")
            self.logger.info(f"🎯 Train Set: {len(X_train)} samples")
            self.logger.info(f"🧪 Test Set: {len(X_test)} samples")
            self.logger.info(f"\n📈 Match Result Model Accuracies:")
            for model, acc in ms_acc.items():
                self.logger.info(f"   {model:20s}: {acc*100:5.2f}%")
            self.logger.info(f"\n⚽ Score Model Accuracy: {score_acc*100:.2f}%")
            self.logger.info(f"\n📁 Output Directory: {self.output_dir}")
            
            if self.version_archive:
                self.logger.info(f"📦 Archive: {self.output_dir}")
                self.logger.info(f"🔗 Active: {self.models_dir}")
            
            self.logger.info("=" * 80)
            
            # Son metadata güncellemesi
            self.metadata['success'] = True
            self.metadata['completion_time'] = datetime.now().isoformat()
            
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
            return {
                'success': False,
                'error': str(e)
            }


def parse_args():
    """Parse CLI arguments"""
    parser = argparse.ArgumentParser(
        description="Predicta ML Production Training Pipeline v2",
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        default=100,
        help='Minimum matches required (default: 100)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size (default: 0.2)'
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
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    trainer = ProductionModelTrainer(
        models_dir=args.models_dir,
        raw_data_path=args.raw_data,
        clubs_path=args.clubs,
        min_matches=args.min_matches,
        test_size=args.test_size,
        random_state=args.seed,
        version_archive=args.version_archive,
        verbose=args.verbose
    )
    
    result = trainer.run_full_pipeline(countries=args.countries)
    
    if result['success']:
        print("\n✅ Training successful!")
        print(f"💡 Start server: python main.py")
        sys.exit(0)
    else:
        print(f"\n❌ Training failed: {result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
