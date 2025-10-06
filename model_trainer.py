# model_trainer.py
# Predicta Europe ML v2 - Otomatik eğitim (MS 1/X/2 + Tam Skor modeli)
import os
import json
import asyncio
import pickle
from datetime import datetime
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

from txt_data_processor import TXTDataProcessor
from advanced_feature_engineer import AdvancedFeatureEngineer, FEATURE_NAMES
from ml_prediction_engine import MLPredictionEngine

MODELS_DIR = "data/ai_models_v2"
os.makedirs(MODELS_DIR, exist_ok=True)

def _to_score_label(hg: int, ag: int) -> str:
    return f"{hg}-{ag}"

def _build_score_space(labels, top_k=20):
    """Eğitim setindeki en sık görülen TOP-K skor sınıfları + OTHER"""
    counts = Counter(labels).most_common(top_k)
    top_scores = [s for s, _ in counts]
    if "OTHER" not in top_scores:
        top_scores.append("OTHER")
    return top_scores

def _score_map(space):
    return {lab: i for i, lab in enumerate(space)}

def _clamp_score(label: str, allowed: set) -> str:
    return label if label in allowed else "OTHER"

async def _load_all_matches(raw_path: str = "data/raw"):
    """TXTDataProcessor ile tüm ülkeleri okuyup normalize eder."""
    proc = TXTDataProcessor(raw_data_path=raw_path)
    out = await proc.process_all_countries()  # {"matches":[...], "team_stats": {...}, ...}
    matches = out.get("matches", [])
    norm = []
    for m in matches:
        hg = int(m.get("home_score", 0))
        ag = int(m.get("away_score", 0))
        res = '1' if hg > ag else ('X' if hg == ag else '2')
        norm.append({
            "date": m.get("date", ""),
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "home_goals": hg,
            "away_goals": ag,
            "result": m.get("result", res),
            "league": m.get("league", "Unknown"),
            "odds": {"1": 2.0, "X": 3.0, "2": 3.5}  # arşivde oran yoksa default
        })
    return norm

def _featurize(matches, engineer: AdvancedFeatureEngineer):
    X, y_ms, y_score_raw = [], [], []
    for m in matches:
        # Engineer hafızasını güncelle (form/momentum vs.)
        home_res = 'W' if m['result'] == '1' else ('D' if m['result'] == 'X' else 'L')
        away_res = 'L' if m['result'] == '1' else ('D' if m['result'] == 'X' else 'W')
        engineer.update_team_history(m['home_team'], {
            "result": home_res, "goals_for": m['home_goals'], "goals_against": m['away_goals'],
            "date": m.get("date",""), "venue": "home"
        })
        engineer.update_team_history(m['away_team'], {
            "result": away_res, "goals_for": m['away_goals'], "goals_against": m['home_goals'],
            "date": m.get("date",""), "venue": "away"
        })
        engineer.update_h2h_history(m['home_team'], m['away_team'], {
            "result": m['result'], "home_goals": m['home_goals'], "away_goals": m['away_goals']
        })
        engineer.update_league_results(m['league'], m['result'])

        feats = engineer.extract_features({
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "league": m["league"],
            "odds": m["odds"],
            "date": m.get("date","")
        })
        X.append(feats)
        y_ms.append({'1':0,'X':1,'2':2}[m['result']])
        y_score_raw.append(_to_score_label(m["home_goals"], m["away_goals"]))

    return np.array(X, dtype=np.float32), np.array(y_ms, dtype=np.int64), y_score_raw

def train_all(raw_path: str = "data/raw", top_scores_k: int = 20):
    # 1) Veri
    matches = asyncio.run(_load_all_matches(raw_path))
    if len(matches) < 100:
        raise RuntimeError(f"Yetersiz veri: {len(matches)}")

    # 2) Feature Engineer
    engineer = AdvancedFeatureEngineer()

    # 3) Feature + etiketler
    X, y_ms, y_sc_raw = _featurize(matches, engineer)

    # 4) Skor sınıf uzayı
    score_space = _build_score_space(y_sc_raw, top_k=top_scores_k)
    s2i = _score_map(score_space)
    allowed = set(score_space)
    y_sc = np.array([s2i[_clamp_score(lbl, allowed)] for lbl in y_sc_raw], dtype=np.int64)

    # 5) Split (aynı split ikisi için)
    X_tr, X_te, y_ms_tr, y_ms_te, y_sc_tr, y_sc_te = train_test_split(
        X, y_ms, y_sc, test_size=0.2, random_state=42, stratify=y_ms
    )

    # 6) MS modeli → mevcut ensemble sınıfın kendi train akışıyla
    #    Onun beklediği format = historical_matches listesi
    hist_for_engine = []
    for m in matches:
        hist_for_engine.append({
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "result": m["result"],
            "odds": m["odds"],
            "league": m["league"],
            "date": m.get("date",""),
            "home_goals": m["home_goals"],
            "away_goals": m["away_goals"],
        })
    engine = MLPredictionEngine(model_path=MODELS_DIR)
    ms_report = engine.train(hist_for_engine)  # kendi içinde scale+fit+save yapar

    # 7) Skor modeli (çok sınıflı RF)
    score_clf = RandomForestClassifier(
        n_estimators=250, max_depth=16, min_samples_split=4, min_samples_leaf=2, random_state=42
    )
    score_clf.fit(X_tr, y_sc_tr)
    y_sc_pred = score_clf.predict(X_te)
    score_acc = accuracy_score(y_sc_te, y_sc_pred)

    # 8) Skor modelini kaydet
    with open(os.path.join(MODELS_DIR, "score_model.pkl"), "wb") as f:
        pickle.dump({
            "model": score_clf,
            "score_space": score_space,
            "feature_names": FEATURE_NAMES
        }, f)

    # 9) Rapor kaydet
    summary = {
        "samples_total": len(matches),
        "ms_engine_report": ms_report,
        "score_accuracy": float(score_acc),
        "score_classes": score_space,
        "timestamp": datetime.now().isoformat()
    }
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "last_training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("✅ Eğitim tamamlandı")
    print(f"• MS Ensemble (engine): {ms_report.get('ensemble_accuracy')}")
    print(f"• Score RF accuracy:    {score_acc:.3f}")
    print(f"• Sınıf sayısı (skor):  {len(score_space)}")
    return summary

if __name__ == "__main__":
    train_all(raw_path="data/raw", top_scores_k=20)
