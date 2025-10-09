#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Trainer - FULL SYNC VERSION
- Ülke bazlı geçmiş veriyi (CSV/TXT/football.db) okur
- Kulüp alias (data/clubs) desteğiyle takım isimlerini normalize eder
- MS (1-X-2) için ensemble modelleri (MLPredictionEngine) eğitir ve kaydeder
- Skor tahmini için ayrı bir RandomForest modelini eğitir ve score_model.pkl olarak kaydeder
"""

import os
import json
import pickle
from datetime import datetime
from collections import Counter
from argparse import ArgumentParser

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

from advanced_feature_engineer import AdvancedFeatureEngineer, FEATURE_NAMES
from ml_prediction_engine import MLPredictionEngine
from historical_processor import HistoricalDataProcessor

# ============================== CONFIG ==============================

MODELS_DIR = os.environ.get("PREDICTA_MODELS_DIR", "data/ai_models_v2")
os.makedirs(MODELS_DIR, exist_ok=True)

DEFAULT_RAW_PATH = "data/raw"
DEFAULT_CLUBS_PATH = "data/clubs"

MIN_SAMPLES_MS = 100           # MS modeli için minimum örnek
MIN_SAMPLES_SCORE = 100        # Skor modeli için minimum örnek
TOP_SCORES_K = 24              # Skor sınıf uzayı (en sık görülen K skor + OTHER)

RANDOM_STATE = 42

# ============================== HELPERS ==============================

def _to_score_label(hg: int, ag: int) -> str:
    return f"{int(hg)}-{int(ag)}"

def _build_score_space(labels, top_k=20):
    """En sık görülen TOP-K skor sınıfları + OTHER"""
    counts = Counter(labels).most_common(top_k)
    top_scores = [s for s, _ in counts]
    if "OTHER" not in top_scores:
        top_scores.append("OTHER")
    return top_scores

def _score_space_map(space):
    return {lab: i for i, lab in enumerate(space)}

def _clamp_score(label: str, allowed: set) -> str:
    return label if label in allowed else "OTHER"

def _result_from_goals(hg: int, ag: int) -> str:
    return '1' if hg > ag else ('X' if hg == ag else '2')

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

# ============================== DATA LOADING ==============================

def load_all_matches(raw_path: str, clubs_path: str):
    """
    HistoricalDataProcessor ile TÜM ülkeleri yükle ve normalize et.
    Dönüş: matches list[dict] (home_team, away_team, home_goals, away_goals, result, league, date, odds)
    """
    proc = HistoricalDataProcessor(raw_data_path=raw_path, clubs_path=clubs_path)

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data path not found: {raw_path}")

    # 'clubs' dizinini ülke olarak ele almayalım
    countries = [
        d for d in os.listdir(raw_path)
        if os.path.isdir(os.path.join(raw_path, d)) and d.lower() != "clubs"
    ]
    countries.sort()

    all_matches = []
    for country in countries:
        country_matches = proc.load_country_data(country)
        for m in country_matches:
            hg = int(m.get("home_score", 0))
            ag = int(m.get("away_score", 0))
            res = m.get("result")
            if res not in ('1', 'X', '2'):
                res = _result_from_goals(hg, ag)

            all_matches.append({
                "date": m.get("date", ""),
                "home_team": m.get("home_team", ""),
                "away_team": m.get("away_team", ""),
                "home_goals": hg,
                "away_goals": ag,
                "result": res,
                "league": m.get("league", "Unknown"),
                # Odds yoksa makul default ver
                "odds": {
                    "1": 2.0,
                    "X": 3.0,
                    "2": 3.5
                }
            })

    return all_matches

# ============================== FEATURE PIPELINE ==============================

def featurize_for_both(matches, engineer: AdvancedFeatureEngineer):
    """
    MS (1-X-2) ve Skor sınıflandırma için tek geçişte feature çıkar.
    Dönüş:
      X -> np.ndarray[n_samples, n_features]
      y_ms -> np.ndarray[int] (0:'1', 1:'X', 2:'2')
      y_sc_raw -> list[str]  (örn '2-1', '1-1', ...)
    Ayrıca engineer'ın hafızaları (form, h2h, lig) güncellenir.
    """
    X, y_ms, y_sc_raw = [], [], []

    for m in matches:
        # Engineer hafızasını güncelle
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
        engineer.update_league_results(m.get("league", "Unknown"), m['result'])

        # Feature çıkar
        feats = engineer.extract_features({
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "league": m.get("league", "Unknown"),
            "odds": m.get("odds", {"1": 2.0, "X": 3.0, "2": 3.5}),
            "date": m.get("date", "")
        })

        X.append(feats)
        y_ms.append({'1': 0, 'X': 1, '2': 2}[m['result']])
        y_sc_raw.append(_to_score_label(m["home_goals"], m["away_goals"]))

    return np.array(X, dtype=np.float32), np.array(y_ms, dtype=np.int64), y_sc_raw

# ============================== TRAINERS ==============================

def train_score_model(matches, top_scores_k=TOP_SCORES_K):
    """
    Skor sınıflandırma için RandomForest modeli eğit ve kaydet.
    """
    if len(matches) < MIN_SAMPLES_SCORE:
        return {
            "success": False,
            "error": f"Yetersiz veri (score): {len(matches)} örnek (min {MIN_SAMPLES_SCORE})"
        }

    engineer = AdvancedFeatureEngineer()
    X, _, y_sc_raw = featurize_for_both(matches, engineer)

    # Sınıf uzayı
    score_space = _build_score_space(y_sc_raw, top_k=top_scores_k)
    s2i = _score_space_map(score_space)
    allowed = set(score_space)
    y_sc = np.array([s2i[_clamp_score(lbl, allowed)] for lbl in y_sc_raw], dtype=np.int64)

    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_sc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_sc
    )

    # Model
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample"
    )
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, zero_division=0, output_dict=True)

    # Kaydet
    out_path = os.path.join(MODELS_DIR, "score_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "model": clf,
            "score_space": score_space,
            "trained_at": datetime.now().isoformat(),
            "feature_names": FEATURE_NAMES
        }, f)

    return {
        "success": True,
        "model_path": out_path,
        "accuracy": acc,
        "score_classes": score_space,
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "report": report
    }

def train_ms_models(matches):
    """
    MS (1-X-2) için ensemble modellerini MLPredictionEngine ile eğit.
    """
    if len(matches) < MIN_SAMPLES_MS:
        return {
            "success": False,
            "error": f"Yetersiz veri (MS): {len(matches)} örnek (min {MIN_SAMPLES_MS})"
        }

    engine = MLPredictionEngine(model_path=MODELS_DIR)
    result = engine.train(matches)  # engine, modelleri kendi içinde kaydeder
    return result

# ============================== ORCHESTRATOR ==============================

def train_all(raw_path=DEFAULT_RAW_PATH, clubs_path=DEFAULT_CLUBS_PATH, top_scores_k=TOP_SCORES_K):
    """
    Tüm eğitimi uçtan uca çalıştır.
    """
    print("🎓 Eğitim süreci başlıyor...")
    print(f"📁 Raw Path   : {raw_path}")
    print(f"📁 Clubs Path : {clubs_path}")
    print(f"💾 Models Dir : {MODELS_DIR}")

    # 1) Veri Yükle
    print("📊 Geçmiş veriler yükleniyor...")
    matches = load_all_matches(raw_path, clubs_path)
    print(f"✅ Toplam {len(matches)} maç yüklendi")

    if len(matches) < min(MIN_SAMPLES_MS, MIN_SAMPLES_SCORE):
        return {
            "success": False,
            "error": f"Genel veri yetersiz: {len(matches)} maç"
        }

    # 2) MS (1-X-2) modelleri
    print("\n🔧 MS (1-X-2) ensemble modelleri eğitiliyor...")
    ms_info = train_ms_models(matches)
    if not ms_info.get("success", False):
        print(f"⚠️ MS eğitim uyarısı: {ms_info.get('error')}")
    else:
        print(f"✅ Ensemble doğruluk: %{ms_info.get('ensemble_accuracy', 0)*100:.2f}")

    # 3) Skor modeli
    print("\n🔢 Skor sınıflandırıcı eğitiliyor...")
    sc_info = train_score_model(matches, top_scores_k=top_scores_k)
    if not sc_info.get("success", False):
        print(f"⚠️ Skor eğitim uyarısı: {sc_info.get('error')}")
    else:
        print(f"✅ Skor doğruluk: %{sc_info.get('accuracy', 0)*100:.2f}")
        print(f"🎯 Sınıf uzayı: {', '.join(sc_info.get('score_classes', [])[:10])} ...")

    # 4) Raporu kaydet
    summary = {
        "trained_at": datetime.now().isoformat(),
        "models_dir": MODELS_DIR,
        "samples_total": len(matches),
        "ms": ms_info,
        "score": {k: v for k, v in sc_info.items() if k != "report"},
    }
    with open(os.path.join(MODELS_DIR, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n💾 Rapor: data/ai_models_v2/training_summary.json")
    print("🏁 Eğitim tamamlandı.")
    return {"success": True, **summary}

# ============================== CLI ==============================

def _parse_args():
    ap = ArgumentParser(description="Predicta Europe ML - Full Model Trainer")
    ap.add_argument("--raw", type=str, default=DEFAULT_RAW_PATH, help="Geçmiş verilerin kök klasörü (data/raw)")
    ap.add_argument("--clubs", type=str, default=DEFAULT_CLUBS_PATH, help="Kulüp alias klasörü (data/clubs)")
    ap.add_argument("--topk", type=int, default=TOP_SCORES_K, help="Skor sınıfları için TOP-K")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    train_all(raw_path=args.raw, clubs_path=args.clubs, top_scores_k=args.topk)
