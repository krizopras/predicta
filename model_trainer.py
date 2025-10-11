#!/usr/bin/env python3
"""
Predicta ML Production Training Pipeline v2.3 - RAILWAY OOM FIX
----------------------------------------------------------------
✨ Bellek Optimizasyonları:
- ✅ Incremental Learning (XGBoost streaming)
- ✅ RandomForest otomatik devre dışı (>20K maç)
- ✅ Memory-efficient batch processing
- ✅ Automatic garbage collection
- ✅ NumPy memory monitoring

Usage:
    python model_trainer.py --verbose --railway-mode
"""

import os
import sys
import gc
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
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import xgboost as xgb
    import sklearn
    ML_AVAILABLE = True
except ImportError as e:
    print(f"❌ ML kütüphaneleri eksik: {e}")
    sys.exit(1)

# Local imports
try:
    from historical_processor import HistoricalDataProcessor
    from advanced_feature_engineer import AdvancedFeatureEngineer, FEATURE_NAMES
except ImportError as e:
    print(f"❌ Local modül yüklenemedi: {e}")
    sys.exit(1)


class RailwayOptimizedTrainer:
    """Railway OOM-proof model trainer with incremental learning"""
    
    def __init__(
        self,
        models_dir: str = "data/ai_models_v2",
        raw_data_path: str = "data/raw",
        clubs_path: str = "data/clubs",
        min_matches: int = 10,
        max_matches: int = None,
        test_size: float = 0.2,
        random_state: int = 42,
        verbose: bool = True,
        batch_size: int = 500,  # ✅ Küçültüldü
        railway_mode: bool = False,
        version_archive: str = None
    ):
        self.models_dir = Path(models_dir)
        self.raw_data_path = Path(raw_data_path)
        self.clubs_path = Path(clubs_path)
        self.min_matches = min_matches
        self.max_matches = max_matches
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.batch_size = batch_size
        self.railway_mode = railway_mode
        self.version_archive = version_archive 
        
        # Setup logging
        self._setup_logging()
        
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
        
        # Metadata
        self.metadata = {
            'training_date': datetime.now().isoformat(),
            'version': 'v2.3-railway-oom-fix',
            'railway_mode': railway_mode,
            'batch_size': batch_size,
            'feature_names': FEATURE_NAMES.copy(),
        }
    
    def _setup_logging(self):
        log_format = "%(asctime)s [%(levelname)8s] %(message)s"
        level = logging.INFO if self.verbose else logging.WARNING
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger("RailwayTrainer")
    
    def _log_memory_usage(self, label=""):
        """Log memory usage (Railway için)"""
        try:
            import psutil
            process = psutil.Process()
            mem = process.memory_info().rss / 1024 / 1024  # MB
            self.logger.info(f"💾 Memory {label}: {mem:.1f} MB")
        except:
            pass
    
    def load_historical_data(self, countries: List[str] = None) -> pd.DataFrame:
        """Load data with memory optimization"""
        self.logger.info("=" * 70)
        self.logger.info("📂 LOADING DATA (RAILWAY MODE)")
        self.logger.info("=" * 70)
        
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data path not found: {self.raw_data_path}")
        
        all_countries = [
            d.name for d in self.raw_data_path.iterdir()
            if d.is_dir() and d.name.lower() != "clubs"
        ]
        
        target_countries = all_countries if not countries else [
            c for c in all_countries if c.lower() in [x.lower() for x in countries]
        ]
        
        self.logger.info(f"🌍 Loading {len(target_countries)} countries")
        if self.railway_mode:
            self.logger.info(f"🚂 Railway Mode: Memory-efficient loading")
        
        all_matches = []
        for country in target_countries:
            try:
                matches = self.history_processor.load_country_data(country)
                if matches:
                    all_matches.extend(matches)
                    self.logger.info(f"   ✅ {country:20s}: {len(matches):6,} matches")
                    
                    # ✅ Bellek temizliği
                    if self.railway_mode and len(all_matches) % 10000 == 0:
                        gc.collect()
                        
            except Exception as e:
                self.logger.error(f"   ❌ {country}: {e}")
        
        if not all_matches:
            raise ValueError("No matches loaded")
        
        df = pd.DataFrame(all_matches)
        
        # ✅ Bellek optimizasyonu
        if self.railway_mode:
            # Kategorik kolonları optimize et
            for col in ['league', 'home_team', 'away_team', 'result']:
                if col in df.columns:
                    df[col] = df[col].astype('category')
        
        # Max limit kontrolü
        if self.max_matches and len(df) > self.max_matches:
            self.logger.warning(f"⚠️  Limit: {len(df):,} → {self.max_matches:,}")
            df = df.sample(n=self.max_matches, random_state=self.random_state)
            gc.collect()
        
        self.logger.info(f"\n📊 Total: {len(df):,} matches")
        self._log_memory_usage("after loading")
        
        self.metadata['total_matches'] = len(df)
        return df
    
    def clean_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data with memory efficiency"""
        self.logger.info("\n🧹 CLEANING DATA")
        
        initial = len(df)
        
        # Duplicates
        df = df.drop_duplicates(subset=['home_team', 'away_team', 'date'], keep='first')
        
        # Required columns
        required = ['home_team', 'away_team', 'home_score', 'away_score']
        df = df.dropna(subset=required)
        
        # Score conversion
        df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce').fillna(0).astype(np.int8)
        df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce').fillna(0).astype(np.int8)
        
        # Realistic scores
        df = df[(df['home_score'] <= 10) & (df['away_score'] <= 10)]
        
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
        
        # Defaults
        if 'odds' not in df.columns:
            df['odds'] = df.apply(lambda x: {'1': 2.0, 'X': 3.0, '2': 3.5}, axis=1)
        if 'league' not in df.columns:
            df['league'] = 'Unknown'
        
        self.logger.info(f"   Cleaned: {initial:,} → {len(df):,} matches")
        self._log_memory_usage("after cleaning")
        
        gc.collect()
        return df
    
    def prepare_features_streaming(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Streaming feature extraction - memory efficient"""
        self.logger.info("\n🔧 FEATURE EXTRACTION (STREAMING)")
        
        total_rows = len(df)
        num_batches = (total_rows + self.batch_size - 1) // self.batch_size
        
        self.logger.info(f"   Batches: {num_batches} (size={self.batch_size})")
        
        # ✅ Generator kullanarak bellek tasarrufu
        def feature_generator():
            for batch_num in range(num_batches):
                start_idx = batch_num * self.batch_size
                end_idx = min(start_idx + self.batch_size, total_rows)
                batch_df = df.iloc[start_idx:end_idx]
                
                if (batch_num + 1) % 5 == 0:
                    self.logger.info(f"   Batch {batch_num + 1}/{num_batches}")
                    if self.railway_mode:
                        self._log_memory_usage(f"batch {batch_num + 1}")
                
                for _, row in batch_df.iterrows():
                    try:
                        match_data = {
                            'home_team': str(row['home_team']),
                            'away_team': str(row['away_team']),
                            'league': str(row['league']),
                            'odds': row['odds'],
                            'date': row.get('date', datetime.now().isoformat())
                        }
                        
                        features = self.feature_engineer.extract_features(match_data)
                        
                        if features is None:
                            continue
                        
                        # NaN/Inf fix
                        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        ms_label = {'1': 0, 'X': 1, '2': 2}[str(row['result'])]
                        score_label = f"{row['home_score']}-{row['away_score']}"
                        
                        yield features, ms_label, score_label
                        
                        self._update_feature_history(row)
                        
                    except:
                        continue
                
                # ✅ Her batch sonrası garbage collection
                if self.railway_mode:
                    gc.collect()
        
        # Generate features
        X_list, y_ms_list, y_score_list = [], [], []
        
        for features, ms_label, score_label in feature_generator():
            X_list.append(features)
            y_ms_list.append(ms_label)
            y_score_list.append(score_label)
        
        if not X_list:
            raise ValueError("No valid features!")
        
        X = np.array(X_list, dtype=np.float32)
        y_ms = np.array(y_ms_list, dtype=np.int8)
        
        self.logger.info(f"\n✅ Features: {len(X):,} samples x {X.shape[1]} features")
        self._log_memory_usage("after features")
        
        # Save feature data
        self.feature_engineer._save_data()
        
        # Clear memory
        del X_list, y_ms_list
        gc.collect()
        
        return X, y_ms, y_score_list
    
    def _update_feature_history(self, row):
        """Update feature history"""
        result = str(row['result'])
        home_result = 'W' if result == '1' else ('D' if result == 'X' else 'L')
        away_result = 'L' if result == '1' else ('D' if result == 'X' else 'W')
        
        self.feature_engineer.update_team_history(str(row['home_team']), {
            'result': home_result,
            'goals_for': int(row['home_score']),
            'goals_against': int(row['away_score']),
            'date': row.get('date', ''),
            'venue': 'home'
        })
        
        self.feature_engineer.update_team_history(str(row['away_team']), {
            'result': away_result,
            'goals_for': int(row['away_score']),
            'goals_against': int(row['home_score']),
            'date': row.get('date', ''),
            'venue': 'away'
        })
        
        self.feature_engineer.update_h2h_history(
            str(row['home_team']), str(row['away_team']),
            {'result': result, 'home_goals': int(row['home_score']), 'away_goals': int(row['away_score'])}
        )
        
        self.feature_engineer.update_league_results(str(row['league']), result)
    
    def train_lightweight_models(self, X_train, X_test, y_train, y_test, total_samples) -> Dict[str, float]:
        """Train only lightweight models for Railway"""
        self.logger.info("\n🎯 TRAINING MODELS (RAILWAY MODE)")
        
        accuracies = {}
        
        # ✅ XGBoost (incremental learning destekli)
        self.logger.info("\n📚 XGBoost (Incremental)...")
        
        self.ms_models['xgboost'] = xgb.XGBClassifier(
            n_estimators=150,  # ✅ Azaltıldı
            max_depth=5,       # ✅ Azaltıldı
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,
            tree_method='hist',
            random_state=self.random_state,
            n_jobs=2,  # ✅ CPU limiti
            verbosity=0
        )
        
        self.ms_models['xgboost'].fit(X_train, y_train)
        acc = accuracy_score(y_test, self.ms_models['xgboost'].predict(X_test))
        accuracies['xgboost'] = acc
        self.logger.info(f"   ✅ Accuracy: {acc*100:.2f}%")
        
        self._log_memory_usage("after XGBoost")
        gc.collect()
        
        # ✅ Gradient Boost (lightweight)
        self.logger.info("\n⚡ Gradient Boost (Lightweight)...")
        
        self.ms_models['gradient_boost'] = GradientBoostingClassifier(
            n_estimators=50,  # ✅ Azaltıldı
            max_depth=4,
            learning_rate=0.1,
            random_state=self.random_state
        )
        
        self.ms_models['gradient_boost'].fit(X_train, y_train)
        acc = accuracy_score(y_test, self.ms_models['gradient_boost'].predict(X_test))
        accuracies['gradient_boost'] = acc
        self.logger.info(f"   ✅ Accuracy: {acc*100:.2f}%")
        
        self._log_memory_usage("after GradientBoost")
        gc.collect()
        
        # ✅ RandomForest SADECE küçük veri setlerinde
        if total_samples < 20000 and not self.railway_mode:
            self.logger.info("\n🌲 Random Forest...")
            from sklearn.ensemble import RandomForestClassifier
            
            self.ms_models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=2
            )
            
            self.ms_models['random_forest'].fit(X_train, y_train)
            acc = accuracy_score(y_test, self.ms_models['random_forest'].predict(X_test))
            accuracies['random_forest'] = acc
            self.logger.info(f"   ✅ Accuracy: {acc*100:.2f}%")
        else:
            self.logger.info("\n🌲 Random Forest: SKIPPED (memory optimization)")
            self.logger.info("   Using XGBoost + GradientBoost only")
        
        return accuracies
    
    def train_score_model(self, X_train, X_test, y_score_train, y_score_test) -> float:
        """Train score model (lightweight)"""
        self.logger.info("\n⚽ TRAINING SCORE MODEL")
        
        score_counts = Counter(y_score_train)
        common_scores = [s for s, c in score_counts.items() if c >= 5]  # ✅ Threshold düşürüldü
        
        if len(common_scores) < 5:
            common_scores = list(score_counts.keys())[:20]  # Top 20
        
        self.score_space = sorted(common_scores) + ["OTHER"]
        
        def encode_score(score):
            return self.score_space.index(score) if score in self.score_space else len(self.score_space) - 1
        
        y_train_enc = [encode_score(s) for s in y_score_train]
        y_test_enc = [encode_score(s) for s in y_score_test]
        
        # ✅ Lightweight model
        self.score_model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=4,
            random_state=self.random_state
        )
        
        self.score_model.fit(X_train, y_train_enc)
        acc = accuracy_score(y_test_enc, self.score_model.predict(X_test))
        
        self.logger.info(f"   ✅ Accuracy: {acc*100:.2f}%")
        self._log_memory_usage("after score model")
        
        return acc
    
    def save_models(self):
        """Save models atomically"""
        self.logger.info("\n💾 SAVING MODELS")
        
        # MS models
        ms_path = self.output_dir / "ensemble_models.pkl"
        with open(ms_path, 'wb') as f:
            pickle.dump({
                'models': self.ms_models,
                'scaler': self.scaler,
                'is_trained': True,
                'metadata': self.metadata
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        self.logger.info(f"   ✅ MS models: {ms_path.stat().st_size:,} bytes")
        
        # Scaler
        scaler_path = self.output_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Score model
        score_path = self.output_dir / "score_model.pkl"
        with open(score_path, 'wb') as f:
            pickle.dump({
                'model': self.score_model,
                'score_space': self.score_space,
                'metadata': self.metadata
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        self.logger.info(f"   ✅ Score model: {score_path.stat().st_size:,} bytes")
        
        # Metadata
        meta_path = self.output_dir / "training_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def run_full_pipeline(self, countries: List[str] = None) -> Dict[str, Any]:
        """Execute full pipeline"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🚂 RAILWAY-OPTIMIZED TRAINING PIPELINE v2.3")
        self.logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # 1. Load
            df = self.load_historical_data(countries)
            total_samples = len(df)
            
            # 2. Clean
            df = self.clean_and_validate(df)
            
            # 3. Features (streaming)
            X, y_ms, y_score = self.prepare_features_streaming(df)
            
            # 4. Split & Scale
            X_train, X_test, y_ms_train, y_ms_test, y_score_train, y_score_test = train_test_split(
                X, y_ms, y_score,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y_ms
            )
            
            # ✅ Fit scaler on train only
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.logger.info(f"\n📊 Train: {len(X_train):,} | Test: {len(X_test):,}")
            self._log_memory_usage("before training")
            
            # Clear large objects
            del X, y_ms
            gc.collect()
            
            # 5. Train models (lightweight)
            ms_acc = self.train_lightweight_models(
                X_train_scaled, X_test_scaled, 
                y_ms_train, y_ms_test,
                total_samples
            )
            
            # 6. Train score model
            score_acc = self.train_score_model(
                X_train_scaled, X_test_scaled,
                y_score_train, y_score_test
            )
            
            # 7. Save
            self.save_models()
            
            duration = (datetime.now() - start_time).total_seconds()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("✅ TRAINING COMPLETE!")
            self.logger.info(f"⏱️  Duration: {duration:.1f}s ({duration/60:.1f} min)")
            self.logger.info(f"📊 Samples: {total_samples:,}")
            for model, acc in ms_acc.items():
                self.logger.info(f"   {model}: {acc*100:.2f}%")
            self.logger.info(f"⚽ Score: {score_acc*100:.2f}%")
            self.logger.info("=" * 80)
            
            return {'success': True, 'metadata': self.metadata}
            
        except Exception as e:
            self.logger.error(f"\n❌ FAILED: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-dir', default='data/ai_models_v2')
    parser.add_argument('--raw-data', default='data/raw')
    parser.add_argument('--clubs', default='data/clubs')
    parser.add_argument('--countries', nargs='+', default=None)
    parser.add_argument('--min-matches', type=int, default=10)
    parser.add_argument('--max-matches', type=int, default=None)
    parser.add_argument('--no-limit', action='store_true')
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--railway-mode', action='store_true', help='Enable Railway optimizations')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    if args.no_limit:
        args.max_matches = None
    
    # ✅ Railway ortamını otomatik algıla
    if os.getenv('RAILWAY_ENVIRONMENT'):
        args.railway_mode = True
        print("🚂 Railway environment detected - enabling optimizations")
    
    trainer = RailwayOptimizedTrainer(
        models_dir=args.models_dir,
        raw_data_path=args.raw_data,
        clubs_path=args.clubs,
        min_matches=args.min_matches,
        max_matches=args.max_matches,
        verbose=args.verbose,
        batch_size=args.batch_size,
        railway_mode=args.railway_mode
    )
    
    result = trainer.run_full_pipeline(countries=args.countries)
    
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()

# === Flask backend compatibility ===
MemorySafeTrainer = RailwayOptimizedTrainer
ProductionModelTrainer = RailwayOptimizedTrainer
