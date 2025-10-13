#!/usr/bin/env python3
"""
Predicta ML Production Training Pipeline v3.5 - FULL COMPATIBLE
----------------------------------------------------------------
🚀 ML Prediction Engine v3.5 ile tam uyumlu model eğitimi:
- EnhancedFeatureEngineer ile feature uyumu
- Meta ensemble model eğitimi
- Feature konfigürasyonu kaydetme
- Version 3.5 formatında model kaydetme
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Local imports - Enhanced components for v3.5
try:
    from enhanced_feature_engineer import EnhancedFeatureEngineer
    FEATURE_ENGINEER_AVAILABLE = True
    logger.info("✅ EnhancedFeatureEngineer loaded")
except ImportError:
    try:
        from advanced_feature_engineer import AdvancedFeatureEngineer
        FEATURE_ENGINEER_AVAILABLE = True
        logger.info("✅ AdvancedFeatureEngineer loaded (fallback)")
    except ImportError:
        FEATURE_ENGINEER_AVAILABLE = False
        logger.error("❌ No feature engineer available")

try:
    from historical_processor import HistoricalDataProcessor
    HISTORICAL_PROCESSOR_AVAILABLE = True
except ImportError:
    HISTORICAL_PROCESSOR_AVAILABLE = False
    logger.warning("⚠️ HistoricalDataProcessor not available")


class ModelTrainerV35:
    """ML Prediction Engine v3.5 ile tam uyumlu model eğitici"""

    def __init__(
        self,
        models_dir: str = "data/ai_models_v3",  # v3.5 uyumlu dizin
        raw_data_path: str = "data/raw",
        clubs_path: str = "data/clubs",
        test_size: float = 0.2,
        random_state: int = 42,
        batch_size: int = 2000,
        enable_meta_ensemble: bool = True,
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
        self.verbose = verbose

        # Enhanced feature engineer for v3.5
        if FEATURE_ENGINEER_AVAILABLE:
            try:
                self.feature_engineer = EnhancedFeatureEngineer(model_path=str(self.models_dir))
            except:
                self.feature_engineer = AdvancedFeatureEngineer(model_path=str(self.models_dir))
        else:
            raise ImportError("No feature engineer available")

        # Historical processor (if available)
        self.history_processor = None
        if HISTORICAL_PROCESSOR_AVAILABLE:
            self.history_processor = HistoricalDataProcessor(
                str(self.raw_data_path),
                str(self.clubs_path)
            )

        # Models for v3.5
        self.models = {
            'xgboost': None,
            'gradient_boost': None, 
            'random_forest': None,
            'lightgbm': None
        }
        self.meta_model = None
        self.score_model = None
        self.scaler = StandardScaler()
        
        # Training metadata
        self.metadata = {
            "version": "v3.5",
            "trained_at": datetime.now().isoformat(),
            "feature_engineer": self.feature_engineer.__class__.__name__,
            "expected_features": getattr(self.feature_engineer, 'expected_features', 45),
            "batch_size": batch_size,
            "enable_meta_ensemble": enable_meta_ensemble,
            "model_types": list(self.models.keys())
        }

        # Setup logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("ModelTrainerV35")

    def _mem_usage(self) -> str:
        """RAM kullanımı (MB)"""
        try:
            import psutil
            process = psutil.Process()
            mem_mb = process.memory_info().rss / 1024 / 1024
            return f"{mem_mb:.1f} MB"
        except ImportError:
            return "N/A"

    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Verileri yükle ve hazırla
        """
        self.logger.info("📂 Veri yükleme başlatılıyor...")
        
        all_matches = []
        
        # Historical processor kullanılıyorsa
        if self.history_processor and self.raw_data_path.exists():
            for country_dir in self.raw_data_path.iterdir():
                if country_dir.is_dir():
                    try:
                        matches = self.history_processor.load_country_data(country_dir.name)
                        all_matches.extend(matches)
                        self.logger.info(f"✅ {country_dir.name:<20} {len(matches):6,} maç")
                    except Exception as e:
                        self.logger.warning(f"❌ {country_dir.name}: {e}")
        else:
            # Fallback: CSV dosyalarını direkt oku
            csv_files = list(self.raw_data_path.glob("**/*.csv"))
            for csv_file in csv_files:
                try:
                    df_temp = pd.read_csv(csv_file)
                    # Temel sütun kontrolü
                    required_cols = ['home_team', 'away_team', 'home_score', 'away_score']
                    if all(col in df_temp.columns for col in required_cols):
                        # League bilgisini dosya adından çıkar
                        league = csv_file.parent.name if csv_file.parent.name != 'raw' else 'Unknown'
                        df_temp['league'] = league
                        all_matches.extend(df_temp.to_dict('records'))
                        self.logger.info(f"✅ {csv_file.name}: {len(df_temp)} maç")
                except Exception as e:
                    self.logger.warning(f"❌ {csv_file}: {e}")

        if not all_matches:
            raise ValueError("❌ Hiç veri bulunamadı!")
            
        df = pd.DataFrame(all_matches)
        self.logger.info(f"📊 Toplam {len(df):,} maç yüklendi | RAM: {self._mem_usage()}")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Veriyi temizle ve sonuçları hesapla
        """
        self.logger.info("🧹 Veri temizleme başlatılıyor...")
        
        # Temel temizlik
        initial_count = len(df)
        df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"])
        
        # Skorları integer'a çevir
        df["home_score"] = pd.to_numeric(df["home_score"], errors='coerce').fillna(0).astype(int)
        df["away_score"] = pd.to_numeric(df["away_score"], errors='coerce').fillna(0).astype(int)
        
        # Geçersiz skorları filtrele
        df = df[(df["home_score"] >= 0) & (df["away_score"] >= 0)]
        df = df[(df["home_score"] <= 20) & (df["away_score"] <= 20)]  # Makul skor aralığı
        
        # Sonuç hesapla
        df["result"] = df.apply(
            lambda r: "1" if r["home_score"] > r["away_score"]
            else "2" if r["home_score"] < r["away_score"]
            else "X", axis=1
        )
        
        # Odds bilgisi yoksa varsayılan değerleri ekle
        if 'odds' not in df.columns:
            df['odds'] = [{"1": 2.0, "X": 3.0, "2": 3.5} for _ in range(len(df))]
        
        # League bilgisi yoksa ekle
        if 'league' not in df.columns:
            df['league'] = 'Unknown'
            
        cleaned_count = len(df)
        self.logger.info(f"✅ {initial_count:,} -> {cleaned_count:,} kayıt ({initial_count-cleaned_count} filtrelendi)")
        self.logger.info(f"📊 Sonuç dağılımı: {dict(df['result'].value_counts())}")
        
        return df

    def extract_features_v35(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        v3.5 feature çıkarımı - EnhancedFeatureEngineer ile
        """
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
                
                # Enhanced feature extraction
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

    def train_base_models(self, X_train: np.ndarray, X_test: np.ndarray, 
                         y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Temel modelleri eğit
        """
        self.logger.info("🎯 Temel modeller eğitiliyor...")
        
        accuracies = {}
        
        # 1. XGBoost
        self.logger.info("🔧 XGBoost eğitiliyor...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            random_state=self.random_state,
            n_jobs=4,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        xgb_pred = xgb_model.predict(X_test)
        acc_xgb = accuracy_score(y_test, xgb_pred)
        self.models["xgboost"] = xgb_model
        accuracies["xgboost"] = acc_xgb
        self.logger.info(f"✅ XGBoost Accuracy: {acc_xgb*100:.2f}%")

        # 2. Gradient Boosting
        self.logger.info("🔧 Gradient Boosting eğitiliyor...")
        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state
        )
        gb_model.fit(X_train, y_train)
        gb_pred = gb_model.predict(X_test)
        acc_gb = accuracy_score(y_test, gb_pred)
        self.models["gradient_boost"] = gb_model
        accuracies["gradient_boost"] = acc_gb
        self.logger.info(f"✅ GradientBoost Accuracy: {acc_gb*100:.2f}%")

        # 3. Random Forest
        self.logger.info("🔧 Random Forest eğitiliyor...")
        rf_model = RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=4
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        acc_rf = accuracy_score(y_test, rf_pred)
        self.models["random_forest"] = rf_model
        accuracies["random_forest"] = acc_rf
        self.logger.info(f"✅ RandomForest Accuracy: {acc_rf*100:.2f}%")

        # 4. LightGBM (if available)
        try:
            import lightgbm as lgb
            self.logger.info("🔧 LightGBM eğitiliyor...")
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                num_leaves=31,
                random_state=self.random_state,
                n_jobs=4,
                verbose=-1
            )
            lgb_model.fit(X_train, y_train)
            lgb_pred = lgb_model.predict(X_test)
            acc_lgb = accuracy_score(y_test, lgb_pred)
            self.models["lightgbm"] = lgb_model
            accuracies["lightgbm"] = acc_lgb
            self.logger.info(f"✅ LightGBM Accuracy: {acc_lgb*100:.2f}%")
        except ImportError:
            self.logger.warning("⚠️ LightGBM not available, skipping...")
            self.models["lightgbm"] = None

        # Model performans özeti
        self.logger.info("📊 Model Performans Özeti:")
        for model_name, accuracy in accuracies.items():
            self.logger.info(f"   {model_name:<15}: {accuracy*100:.2f}%")

        return accuracies

    def train_meta_ensemble(self, X_train: np.ndarray, X_test: np.ndarray,
                           y_train: np.ndarray, y_test: np.ndarray) -> float:
        """
        Meta ensemble model eğit
        """
        if not self.enable_meta_ensemble:
            self.logger.info("⏩ Meta ensemble disabled, skipping...")
            return 0.0

        self.logger.info("🎯 Meta ensemble model eğitiliyor...")
        
        # Base model tahminlerini topla
        base_predictions = []
        valid_models = []
        
        for model_name, model in self.models.items():
            if model is not None:
                try:
                    proba = model.predict_proba(X_train)
                    base_predictions.append(proba)
                    valid_models.append(model_name)
                except Exception as e:
                    self.logger.warning(f"⚠️ {model_name} meta features failed: {e}")
        
        if len(base_predictions) < 2:
            self.logger.warning("⚠️ Yeterli model yok, meta ensemble atlanıyor...")
            return 0.0
        
        # Meta features oluştur
        meta_features = []
        for i in range(len(X_train)):
            sample_meta = []
            for preds in base_predictions:
                sample_meta.extend([
                    np.max(preds[i]),           # Max probability
                    np.min(preds[i]),           # Min probability
                    np.std(preds[i]),           # Std of probabilities
                    preds[i][0] - preds[i][2],  # Home vs Away strength
                    np.argmax(preds[i])         # Predicted class
                ])
            meta_features.append(sample_meta)
        
        meta_features = np.array(meta_features)
        
        # Meta model eğit (Logistic Regression)
        meta_model = LogisticRegression(
            multi_class='multinomial',
            random_state=self.random_state,
            max_iter=1000
        )
        meta_model.fit(meta_features, y_train)
        
        # Test accuracy
        test_meta_features = []
        for i in range(len(X_test)):
            sample_meta = []
            for preds in base_predictions:
                test_proba = preds[i] if i < len(preds) else np.array([0.33, 0.33, 0.33])
                sample_meta.extend([
                    np.max(test_proba),
                    np.min(test_proba),
                    np.std(test_proba),
                    test_proba[0] - test_proba[2],
                    np.argmax(test_proba)
                ])
            test_meta_features.append(sample_meta)
        
        test_meta_features = np.array(test_meta_features)
        meta_pred = meta_model.predict(test_meta_features)
        meta_accuracy = accuracy_score(y_test, meta_pred)
        
        self.meta_model = meta_model
        self.logger.info(f"✅ Meta Ensemble Accuracy: {meta_accuracy*100:.2f}%")
        
        return meta_accuracy

    def train_score_model(self, X_train: np.ndarray, X_test: np.ndarray,
                         y_score_train: List[str], y_score_test: List[str]) -> float:
        """
        Skor tahmin modeli eğit
        """
        self.logger.info("⚽ Skor tahmin modeli eğitiliyor...")
        
        # En yaygın skorları belirle
        score_counter = Counter(y_score_train)
        common_scores = [score for score, count in score_counter.items() if count >= 5]
        
        # Yeterli skor yoksa en yaygın 15'ini al
        if len(common_scores) < 10:
            common_scores = [score for score, _ in score_counter.most_common(15)]
        
        self.score_space = common_scores
        self.logger.info(f"📊 Skor uzayı: {len(self.score_space)} skor")
        self.logger.info(f"📊 En yaygın skorlar: {score_counter.most_common(10)}")
        
        # Skorları encode et
        def encode_score(score: str) -> int:
            return self.score_space.index(score) if score in self.score_space else -1
        
        y_train_enc = [encode_score(score) for score in y_score_train]
        y_test_enc = [encode_score(score) for score in y_score_test]
        
        # Geçerli skorları filtrele
        valid_train = [i for i, label in enumerate(y_train_enc) if label != -1]
        valid_test = [i for i, label in enumerate(y_test_enc) if label != -1]
        
        if len(valid_train) < 100 or len(valid_test) < 50:
            self.logger.warning("⚠️ Yeterli skor verisi yok, skor modeli atlanıyor...")
            return 0.0
        
        X_train_score = X_train[valid_train]
        y_train_score = [y_train_enc[i] for i in valid_train]
        X_test_score = X_test[valid_test] 
        y_test_score = [y_test_enc[i] for i in valid_test]
        
        # Skor modeli eğit
        score_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=4
        )
        score_model.fit(X_train_score, y_train_score)
        
        # Accuracy hesapla
        score_pred = score_model.predict(X_test_score)
        score_accuracy = accuracy_score(y_test_score, score_pred)
        
        self.score_model = score_model
        self.logger.info(f"✅ Skor Modeli Accuracy: {score_accuracy*100:.2f}%")
        self.logger.info(f"✅ Skor dağılımı: {len(set(y_train_score))} sınıf")
        
        return score_accuracy

    def save_models_v35(self):
        """
        v3.5 formatında modelleri kaydet
        """
        self.logger.info("💾 v3.5 formatında modeller kaydediliyor...")
        
        # 1. Ensemble models (ana modeller)
        ensemble_data = {
            'models': self.models,
            'scaler': self.scaler,
            'is_trained': True,
            'feature_config': {
                'expected_features': self.metadata['expected_features'],
                'version': 'v3.5',
                'created_at': datetime.now().isoformat()
            }
        }
        joblib.dump(ensemble_data, self.models_dir / "ensemble_models.pkl")
        
        # 2. Meta ensemble model
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
        
        # 3. Score model
        if self.score_model is not None:
            score_data = {
                'model': self.score_model,
                'score_space': self.score_space,
                'is_trained': True
            }
            joblib.dump(score_data, self.models_dir / "enhanced_score_predictor.pkl")
        
        # 4. Scaler (ayrıca)
        joblib.dump(self.scaler, self.models_dir / "scaler.pkl")
        
        # 5. Feature config (JSON formatında)
        feature_config = {
            'version': 'v3.5',
            'feature_count': self.metadata['expected_features'],
            'feature_engineer': self.metadata['feature_engineer'],
            'created_at': datetime.now().isoformat(),
            'feature_groups': ['odds', 'team_strength', 'historical', 'context', 'statistical']
        }
        with open(self.models_dir / "feature_config.json", 'w', encoding='utf-8') as f:
            json.dump(feature_config, f, indent=2, ensure_ascii=False)
        
        # 6. Training metadata
        self.metadata['feature_config'] = feature_config
        with open(self.models_dir / "training_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ Tüm modeller kaydedildi → {self.models_dir}")

    def run_training(self) -> Dict[str, Any]:
        """
        Tam eğitim pipeline'ını çalıştır
        """
        start_time = time.time()
        self.logger.info("🚀 ML Prediction Engine v3.5 Eğitim Başlatılıyor...")
        
        try:
            # 1. Veri yükleme ve hazırlık
            df = self.load_and_prepare_data()
            df = self.clean_data(df)
            
            # 2. Feature çıkarımı
            X, y_ms, y_score = self.extract_features_v35(df)
            
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
            
            # 5. Temel modelleri eğit
            base_accuracies = self.train_base_models(X_train_scaled, X_test_scaled, y_train, y_test)
            
            # 6. Meta ensemble eğit
            meta_accuracy = self.train_meta_ensemble(X_train_scaled, X_test_scaled, y_train, y_test)
            
            # 7. Skor modeli eğit
            score_accuracy = self.train_score_model(X_train_scaled, X_test_scaled, y_score_train, y_score_test)
            
            # 8. Modelleri kaydet
            self.save_models_v35()
            
            # 9. Performans raporu
            training_time = time.time() - start_time
            performance_report = {
                "success": True,
                "training_time_seconds": training_time,
                "training_time_minutes": training_time / 60,
                "total_samples": len(X),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "base_accuracies": base_accuracies,
                "meta_accuracy": meta_accuracy,
                "score_accuracy": score_accuracy,
                "feature_count": X.shape[1],
                "memory_usage": self._mem_usage()
            }
            
            self.logger.info("📈 Eğitim Performans Raporu:")
            self.logger.info(f"   ⏱️  Süre: {training_time/60:.2f} dakika")
            self.logger.info(f"   📊 Toplam Örnek: {len(X):,}")
            self.logger.info(f"   🎯 Base Model Doğrulukları:")
            for model, acc in base_accuracies.items():
                self.logger.info(f"      {model}: {acc*100:.2f}%")
            if meta_accuracy > 0:
                self.logger.info(f"   🎯 Meta Ensemble: {meta_accuracy*100:.2f}%")
            if score_accuracy > 0:
                self.logger.info(f"   ⚽ Skor Modeli: {score_accuracy*100:.2f}%")
            
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
    print("🚀 ML Prediction Engine v3.5 - Model Eğitimi")
    print("=" * 60)
    
    try:
        # Trainer oluştur
        trainer = ModelTrainerV35(
            models_dir="data/ai_models_v3",
            raw_data_path="data/raw",
            clubs_path="data/clubs", 
            test_size=0.2,
            batch_size=2000,
            enable_meta_ensemble=True,
            verbose=True
        )
        
        # Eğitimi başlat
        result = trainer.run_training()
        
        if result["success"]:
            print("✅ Eğitim başarıyla tamamlandı!")
            print(f"📊 En iyi model: {max(result['base_accuracies'].items(), key=lambda x: x[1])}")
        else:
            print(f"❌ Eğitim başarısız: {result['error']}")
            
    except Exception as e:
        print(f"❌ Kritik hata: {e}")


if __name__ == "__main__":
    main()
