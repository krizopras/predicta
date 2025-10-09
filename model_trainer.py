#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_trainer.py
Predicta Europe ML - Ana Model Eğitici
======================================
- Tüm geçmiş maç verilerini yükler
- Özellikleri çıkarır
- MS (1X2) ve Skor tahmin modellerini eğitir
- Ensemble modelleri kaydeder
- training_summary.json oluşturur
"""

import os
import json
import traceback
from datetime import datetime

# ML motorlarını import et
from ml_prediction_engine import MLPredictionEngine
from score_predictor import ScorePredictor


# ===============================================================
# 1️⃣ VERİ YÜKLEYİCİ
# ===============================================================
def load_all_matches(raw_path="data/raw", clubs_path="data/clubs"):
    """
    Tüm geçmiş maçları yükler (ülke bazlı klasörlerden)
    Beklenen sütunlar:
      home_team, away_team, result, home_goals, away_goals, league, odds
    """
    matches = []
    try:
        for country in os.listdir(raw_path):
            cpath = os.path.join(raw_path, country)
            if not os.path.isdir(cpath):
                continue

            for file in os.listdir(cpath):
                if not file.endswith(('.csv', '.txt')):
                    continue

                fpath = os.path.join(cpath, file)
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()

                for line in lines[1:]:  # ilk satır başlık kabul edilir
                    parts = line.strip().split(',')
                    if len(parts) < 6:
                        continue

                    try:
                        home_team, away_team, result, hg, ag, league = parts[:6]
                        odds = {'1': 2.0, 'X': 3.0, '2': 3.5}
                        if len(parts) >= 9:
                            odds = {'1': float(parts[6]), 'X': float(parts[7]), '2': float(parts[8])}
                    except Exception:
                        continue

                    matches.append({
                        'home_team': home_team.strip(),
                        'away_team': away_team.strip(),
                        'result': result.strip(),
                        'home_goals': int(hg or 0),
                        'away_goals': int(ag or 0),
                        'league': league.strip(),
                        'odds': odds,
                        'date': datetime.now().isoformat()
                    })

        return matches
    except Exception as e:
        print(f"❌ load_all_matches() hata: {e}")
        traceback.print_exc()
        return []


# ===============================================================
# 2️⃣ ANA EĞİTİM FONKSİYONU
# ===============================================================
def train_all(raw_path="data/raw", clubs_path="data/clubs", top_scores_k=24):
    """
    Tüm modelleri (MS + Skor) eğitir ve kaydeder
    """
    summary = {
        "success": False,
        "samples_total": 0,
        "timestamp": datetime.now().isoformat()
    }

    try:
        # 1️⃣ Verileri yükle
        matches = load_all_matches(raw_path, clubs_path)
        summary["samples_total"] = len(matches)

        if len(matches) < 100:
            raise ValueError(f"Yetersiz veri ({len(matches)} maç). En az 100 maç gerekli.")

        # 2️⃣ ML Prediction Engine (MS tahmini)
        print("\n⚙️  MLPredictionEngine eğitiliyor...")
        ml_engine = MLPredictionEngine()
        ms_result = ml_engine.train(matches)
        summary["ms"] = ms_result

        # 3️⃣ Skor Tahmin Modeli
        print("\n⚙️  ScorePredictor eğitiliyor...")
        score_predictor = ScorePredictor("data/ai_models_v2")
        score_result = score_predictor.train(matches, top_k=top_scores_k)
        summary["score"] = score_result

        # 4️⃣ Eğitim özeti
        summary["success"] = True
        summary["timestamp"] = datetime.now().isoformat()

        # 5️⃣ Raporu kaydet
        os.makedirs("data/ai_models_v2", exist_ok=True)
        with open("data/ai_models_v2/training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\n💾 Eğitim özeti kaydedildi: data/ai_models_v2/training_summary.json")
        return summary

    except Exception as e:
        print(f"\n❌ Eğitim sırasında hata: {e}")
        traceback.print_exc()
        summary["error"] = str(e)
        summary["success"] = False
        return summary


# ===============================================================
# 3️⃣ DOĞRUDAN ÇALIŞTIRMA (manuel test)
# ===============================================================
if __name__ == "__main__":
    print("🧠 PREDICTA Europe Model Trainer başlatıldı...\n")

    result = train_all()

    if result.get("success"):
        print("\n✅ Eğitim başarıyla tamamlandı!")
        print(f"Toplam Maç: {result.get('samples_total')}")
        if 'ms' in result and result['ms'].get('success'):
            print(f"MS Ensemble Accuracy: %{result['ms'].get('ensemble_accuracy', 0)*100:.2f}")
        if 'score' in result and result['score'].get('success'):
            print(f"Skor Model Accuracy: %{result['score'].get('accuracy', 0)*100:.2f}")
    else:
        print("\n❌ Eğitim başarısız:", result.get("error", "Bilinmeyen hata"))
