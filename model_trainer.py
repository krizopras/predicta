# model_trainer.py - FIXED VERSION
# Predicta Europe ML v2 - Otomatik eğitim

import os
import json
import pickle
import logging
from datetime import datetime
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from advanced_feature_engineer import AdvancedFeatureEngineer, FEATURE_NAMES
from ml_prediction_engine import MLPredictionEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = "data/ai_models_v2"
os.makedirs(MODELS_DIR, exist_ok=True)

def _to_score_label(hg: int, ag: int) -> str:
    return f"{hg}-{ag}"

def _build_score_space(labels, top_k=20):
    """En sık görülen skor sınıfları"""
    counts = Counter(labels).most_common(top_k)
    top_scores = [s for s, _ in counts]
    if "OTHER" not in top_scores:
        top_scores.append("OTHER")
    return top_scores

def _score_map(space):
    return {lab: i for i, lab in enumerate(space)}

def _clamp_score(label: str, allowed: set) -> str:
    return label if label in allowed else "OTHER"

def _load_all_matches_sync(raw_path: str = "data/raw"):
    """
    TXT dosyalarından maçları SYNC olarak yükle
    (asyncio yerine - Railway uyumluluğu için)
    """
    matches = []
    raw_dir = Path(raw_path)
    
    if not raw_dir.exists():
        logger.warning(f"⚠️ Raw data dizini bulunamadı: {raw_path}")
        return matches
    
    logger.info(f"📂 Raw data okunuyor: {raw_path}")
    
    for country_dir in raw_dir.iterdir():
        if not country_dir.is_dir():
            continue
        
        for txt_file in country_dir.glob("*.txt"):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        
                        # Format: DATE|HOME|AWAY|HOME_SCORE|AWAY_SCORE
                        parts = line.split("|")
                        if len(parts) >= 5:
                            date_str, home, away, h_score, a_score = parts[:5]
                            
                            try:
                                hg = int(h_score)
                                ag = int(a_score)
                            except ValueError:
                                continue
                            
                            res = '1' if hg > ag else ('X' if hg == ag else '2')
                            
                            matches.append({
                                "date": date_str,
                                "home_team": home.strip(),
                                "away_team": away.strip(),
                                "home_goals": hg,
                                "away_goals": ag,
                                "result": res,
                                "league": txt_file.stem,  # Dosya adından lig
                                "odds": {"1": 2.0, "X": 3.0, "2": 3.5}
                            })
            
            except Exception as e:
                logger.error(f"❌ Dosya okuma hatası ({txt_file}): {e}")
    
    logger.info(f"✅ {len(matches)} maç yüklendi")
    return matches

def _featurize(matches, engineer: AdvancedFeatureEngineer):
    """Maçlardan feature matrisi oluştur"""
    X, y_ms, y_score_raw = [], [], []
    
    for i, m in enumerate(matches):
        try:
            # Geçmişi güncelle
            home_res = 'W' if m['result'] == '1' else ('D' if m['result'] == 'X' else 'L')
            away_res = 'L' if m['result'] == '1' else ('D' if m['result'] == 'X' else 'W')
            
            engineer.update_team_history(m['home_team'], {
                "result": home_res,
                "goals_for": m['home_goals'],
                "goals_against": m['away_goals'],
                "date": m.get("date", ""),
                "venue": "home"
            })
            
            engineer.update_team_history(m['away_team'], {
                "result": away_res,
                "goals_for": m['away_goals'],
                "goals_against": m['home_goals'],
                "date": m.get("date", ""),
                "venue": "away"
            })
            
            engineer.update_h2h_history(m['home_team'], m['away_team'], {
                "result": m['result'],
                "home_goals": m['home_goals'],
                "away_goals": m['away_goals']
            })
            
            engineer.update_league_results(m['league'], m['result'])
            
            # Feature üret
            feats = engineer.extract_features({
                "home_team": m["home_team"],
                "away_team": m["away_team"],
                "league": m["league"],
                "odds": m["odds"],
                "date": m.get("date", "")
            })
            
            X.append(feats)
            y_ms.append({'1': 0, 'X': 1, '2': 2}[m['result']])
            y_score_raw.append(_to_score_label(m["home_goals"], m["away_goals"]))
        
        except Exception as e:
            logger.error(f"❌ Maç featurize hatası ({i}): {e}")
            continue
    
    return np.array(X, dtype=np.float32), np.array(y_ms, dtype=np.int64), y_score_raw

def train_all(raw_path: str = "data/raw", top_scores_k: int = 20):
    """ANA EĞİTİM FONKSİYONU"""
    
    logger.info("🎯 MODEL EĞİTİMİ BAŞLIYOR")
    logger.info(f"📂 Raw data: {raw_path}")
    
    # 1) Veri yükleme (SYNC)
    try:
        matches = _load_all_matches_sync(raw_path)
    except Exception as e:
        logger.error(f"❌ Veri yükleme hatası: {e}")
        return {"success": False, "error": str(e)}
    
    if len(matches) < 100:
        err_msg = f"Yetersiz veri: {len(matches)} maç (min 100 gerekli)"
        logger.error(f"❌ {err_msg}")
        return {"success": False, "error": err_msg}
    
    logger.info(f"✅ {len(matches)} maç yüklendi")
    
    # 2) Feature Engineer
    engineer = AdvancedFeatureEngineer()
    
    # 3) Feature extraction
    logger.info("🔧 Feature'lar üretiliyor...")
    try:
        X, y_ms, y_sc_raw = _featurize(matches, engineer)
    except Exception as e:
        logger.error(f"❌ Feature üretim hatası: {e}")
        return {"success": False, "error": str(e)}
    
    if len(X) == 0:
        return {"success": False, "error": "Feature üretilemedi"}
    
    logger.info(f"✅ {len(X)} sample feature üretildi")
    
    # 4) Skor sınıf uzayı
    score_space = _build_score_space(y_sc_raw, top_k=top_scores_k)
    s2i = _score_map(score_space)
    allowed = set(score_space)
    y_sc = np.array([s2i[_clamp_score(lbl, allowed)] for lbl in y_sc_raw], dtype=np.int64)
    
    # 5) Train/test split
    try:
        X_tr, X_te, y_ms_tr, y_ms_te, y_sc_tr, y_sc_te = train_test_split(
            X, y_ms, y_sc, test_size=0.2, random_state=42, stratify=y_ms
        )
    except Exception as e:
        logger.error(f"❌ Split hatası: {e}")
        return {"success": False, "error": str(e)}
    
    # 6) MS modeli - MLPredictionEngine kullan
    logger.info("🤖 MS (Maç Sonucu) modeli eğitiliyor...")
    
    hist_for_engine = []
    for m in matches:
        hist_for_engine.append({
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "result": m["result"],
            "odds": m["odds"],
            "league": m["league"],
            "date": m.get("date", ""),
            "home_goals": m["home_goals"],
            "away_goals": m["away_goals"],
        })
    
    try:
        engine = MLPredictionEngine(model_path=MODELS_DIR)
        ms_report = engine.train(hist_for_engine)
        
        if not ms_report.get("success", False):
            logger.error(f"❌ MS model eğitim hatası: {ms_report.get('error')}")
            return {"success": False, "error": ms_report.get("error", "MS training failed")}
    
    except Exception as e:
        logger.error(f"❌ MS model eğitim exception: {e}")
        return {"success": False, "error": str(e)}
    
    logger.info(f"✅ MS model accuracy: {ms_report.get('ensemble_accuracy', 0):.3f}")
    
    # 7) Skor modeli
    logger.info("⚽ Skor modeli eğitiliyor...")
    try:
        score_clf = RandomForestClassifier(
            n_estimators=250,
            max_depth=16,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        score_clf.fit(X_tr, y_sc_tr)
        y_sc_pred = score_clf.predict(X_te)
        score_acc = accuracy_score(y_sc_te, y_sc_pred)
        
        logger.info(f"✅ Skor model accuracy: {score_acc:.3f}")
    
    except Exception as e:
        logger.error(f"❌ Skor model eğitim hatası: {e}")
        return {"success": False, "error": str(e)}
    
    # 8) Skor modelini kaydet
    try:
        with open(os.path.join(MODELS_DIR, "score_model.pkl"), "wb") as f:
            pickle.dump({
                "model": score_clf,
                "score_space": score_space,
                "feature_names": FEATURE_NAMES
            }, f)
        logger.info("💾 Skor modeli kaydedildi")
    except Exception as e:
        logger.error(f"❌ Skor model kaydetme hatası: {e}")
    
    # 9) Rapor
    summary = {
        "success": True,
        "samples_total": len(matches),
        "samples_used": len(X),
        "ms_engine_report": ms_report,
        "ensemble_accuracy": ms_report.get("ensemble_accuracy", 0),
        "score_accuracy": float(score_acc),
        "score_classes": score_space,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        with open(os.path.join(MODELS_DIR, "last_training_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"❌ Rapor kaydetme hatası: {e}")
    
    logger.info("=" * 60)
    logger.info("✅ EĞİTİM TAMAMLANDI")
    logger.info(f"• MS Ensemble accuracy: {ms_report.get('ensemble_accuracy', 0):.3f}")
    logger.info(f"• Score RF accuracy:    {score_acc:.3f}")
    logger.info(f"• Skor sınıf sayısı:    {len(score_space)}")
    logger.info("=" * 60)
    
    return summary

if __name__ == "__main__":
    result = train_all(raw_path="data/raw", top_scores_k=20)
    if result.get("success"):
        print("✅ Eğitim başarılı")
    else:
        print(f"❌ Hata: {result.get('error')}")
