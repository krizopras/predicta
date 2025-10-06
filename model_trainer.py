# model_trainer.py
# Predicta Europe ML v2 - Otomatik eğitim (MS + Skor modeli)
import os
import json
import asyncio
import pickle
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np

# ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# Yerel modüller
from txt_data_processor import TXTDataProcessor  # Senin dosyan
from advanced_feature_engineer import AdvancedFeatureEngineer, FEATURE_NAMES
from ml_prediction_engine import MLPredictionEngine


MODELS_DIR = "data/ai_models_v2"
os.makedirs(MODELS_DIR, exist_ok=True)

# ---- Yardımcılar -----------------------------------------------------------

def to_score_label(hg: int, ag: int) -> str:
    return f"{hg}-{ag}"

def result_from_goals(hg: int, ag: int) -> str:
    return '1' if hg > ag else ('X' if hg == ag else '2')

def build_score_space(labels: list, top_k: int = 20):
    """Eğitim setindeki en sık görülen TOP-K skor sınıfları + 'OTHER'"""
    counts = Counter(labels).most_common(top_k)
    top_scores = [s for s, _ in counts]
    if "OTHER" not in top_scores:
        top_scores.append("OTHER")
    return top_scores

def score_label_map(score_space: list):
    return {lab: idx for idx, lab in enumerate(score_space)}

def clamp_score_label(label: str, allowed: set) -> str:
    return label if label in allowed else "OTHER"

# ---- Veri Yükleme & Feature Hazırlama -------------------------------------

async def load_all_matches_from_raw(raw_path: str = "data/raw"):
    """
    Tüm ülkeleri paralel işler; satırları parse edip unified bir liste döndürür.
    Beklenen alanlar: date, home_team, away_team, home_score, away_score, result, country, league
    """
    proc = TXTDataProcessor(raw_data_path=raw_path)
    out = await proc.process_all_countries()  # {"matches": [...], "team_stats": {...}, ...}
    matches = out.get("matches", [])
    # tarih normalize etme (varsa), odds yoksa default
    normalized = []
    for m in matches:
        date = m.get("date", "")
        # ISO'ya zorlamayalım; feature_engineer zaten fallback'li çalışır
        normalized.append({
            "date": date,
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "home_goals": int(m.get("home_score", 0)),
            "away_goals": int(m.get("away_score", 0)),
            "result": m.get("result") or result_from_goals(int(m.get("home_score", 0)),
                                                           int(m.get("away_score", 0))),
            "league": m.get("league", "Unknown"),
            "odds": {"1": 2.0, "X": 3.0, "2": 3.5}  # arşivde yoksa defaults
        })
    return normalized

def featurize_matches(matches: list, engineer: AdvancedFeatureEngineer):
    """
    Feature çıkarırken takım/h2h/lig hafızasını da güncelliyoruz (online şekilde).
    """
    X, y_res, y_score = [], [], []
    for m in matches:
        # önce geçmişi güncelle (engineer momentum/form için)
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
            "result": m["result"],
            "home_goals": m["home_goals"],
            "away_goals": m["away_goals"]
        })
        engineer.update_league_results(m["league"], m["result"])

        # feature çıkar
        feats = engineer.extract_features({
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "league": m["league"],
            "odds": m["odds"],
            "date": m.get("date", "")
        })
        X.append(feats)
        y_res.append({"1": 0, "X": 1, "2": 2}[m["result"]])
        y_score.append(to_score_label(m["home_goals"], m["away_goals"]))

    X = np.array(X, dtype=np.float32)
    y_res = np.array(y_res, dtype=np.int64)
    return X, y_res, y_score

# ---- Eğitim Akışı ----------------------------------------------------------

def train_all(raw_path: str = "data/raw", top_scores_k: int = 20):
    """
    - Arşiv verisini yükler
    - Features + etiketleri hazırlar
    - MLPredictionEngine ile MS (1/X/2) ensemble eğitim
    - Ayrı bir RandomForest ile 'tam skor' çok-sınıflı eğitim
    - Modelleri kaydeder
    """
    # 1) Veri
    matches = asyncio.run(load_all_matches_from_raw(raw_path))
    if len(matches) < 100:
        raise RuntimeError(f"Yetersiz veri: {len(matches)}")

    # 2) Feature Engineer
    engineer = AdvancedFeatureEngineer()

    # 3) Feature çıkar + etiketler
    X, y_res, y_score_raw = featurize_matches(matches, engineer)

    # 4) Skor sınıf uzayı (TOP-K + OTHER)
    score_space = build_score_space(y_score_raw, top_k=top_scores_k)
    s2i = score_label_map(score_space)
    score_allowed = set(score_space)

    y_score = [clamp_score_label(lbl, score_allowed) for lbl in y_score_raw]
    y_score = np.array([s2i[lbl] for lbl in y_score], dtype=np.int64)

    # 5) Train-test split (aynı split’i hem MS hem skor modeli kullanacak)
    X_train, X_test, y_res_train, y_res_test, y_sc_train, y_sc_test = train_test_split(
        X, y_res, y_score, test_size=0.2, random_state=42, stratify=y_res
    )

    # 6) MS modeli (ensemble) -> mevcut sınıfınız üzerinden
    engine = MLPredictionEngine(model_path=MODELS_DIR)
    # Burada engine kendi scaler'ını içinde fit ediyor — önce çağıracağız
    # engine.train() method’u scaled + ensemble fit + kaydet yapıyor.
    # Ama bizim X_train elimizde; engine.train beklediği format 'historical_matches' listesi.
    # O yüzden hızlı bir "uygulama datası" hazırlayalım:
    historical_for_engine = []
    for m in matches:
        historical_for_engine.append({
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "result": m["result"],
            "odds": m["odds"],
            "league": m["league"],
            "date": m.get("date", ""),
            "home_goals": m["home_goals"],
            "away_goals": m["away_goals"],
        })
    train_report = engine.train(historical_for_engine)  # içinde split+fit+save var

    # 7) Skor modeli (çok sınıflı RF) — features doğrudan kullanır
    score_clf = RandomForestClassifier(
        n_estimators=250, max_depth=16, min_samples_split=4, min_samples_leaf=2, random_state=42
    )
    score_clf.fit(X_train, y_sc_train)
    y_sc_pred = score_clf.predict(X_test)
    score_acc = accuracy_score(y_sc_test, y_sc_pred)

    # 8) Kaydet (skor modeli + label space)
    with open(os.path.join(MODELS_DIR, "score_model.pkl"), "wb") as f:
        pickle.dump({
            "model": score_clf,
            "score_space": score_space,
            "feature_names": FEATURE_NAMES
        }, f)

    # 9) Rapor
    score_report = classification_report(
        y_sc_test, y_sc_pred,
        target_names=[str(x) for x in range(len(score_space))],
        zero_division=0
    )

    summary = {
        "samples_total": len(matches),
        "ms_engine_report": train_report,
        "score_accuracy": float(score_acc),
        "score_classes": score_space,
        "timestamp": datetime.now().isoformat()
    }

    with open(os.path.join(MODELS_DIR, "last_training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("✅ Eğitim tamamlandı")
    print(f"• MS Ensemble (engine): {train_report.get('ensemble_accuracy')}")
    print(f"• Score RF accuracy:    {score_acc:.3f}")
    print(f"• Sınıf sayısı (skor):  {len(score_space)}")
    return summary


if __name__ == "__main__":
    # Komut satırından:  python model_trainer.py
    train_all(raw_path="data/raw", top_scores_k=20)
