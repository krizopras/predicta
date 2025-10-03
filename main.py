#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predicta AI v5.5 — FutureMatch + Prediction Hybrid
Oynanmamış futbol maçlarını (Nesine CDN'den) toplar ve ImprovedPredictionEngine ile tahmin üretir.
"""
import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------------------------------
# Import Tahmin Motoru (örnek placeholder)
# ----------------------------------------------------
try:
    from improved_prediction_engine import ImprovedPredictionEngine
except ImportError:
    # Basit mock tahmin motoru (örnek)
    class ImprovedPredictionEngine:
        def predict_match(self, home_team, away_team, odds, league="default"):
            import random
            probs = {"1": 0.0, "X": 0.0, "2": 0.0}
            base = [1 / (odds.get("1", 2.0)), 1 / (odds.get("X", 3.0)), 1 / (odds.get("2", 3.5))]
            s = sum(base)
            probs["1"], probs["X"], probs["2"] = [round(b / s, 3) for b in base]
            pred = max(probs, key=probs.get)
            return {
                "predicted_result": pred,
                "confidence": probs[pred],
                "probabilities": probs,
                "suggestion": {
                    "1": "Ev sahibi kazanır",
                    "X": "Beraberlik",
                    "2": "Deplasman kazanır"
                }[pred]
            }

# ----------------------------------------------------
# Global Ayarlar
# ----------------------------------------------------
DATA_DIR = "./data"
DATA_FILE = os.path.join(DATA_DIR, "future_matches.json")
DAYS_AHEAD = 14
REFRESH_INTERVAL_SECONDS = 6 * 3600  # 6 saat
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

API_ENDPOINT = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"

# ----------------------------------------------------
# Logger
# ----------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("predicta-v5.5")

# ----------------------------------------------------
# FastAPI App
# ----------------------------------------------------
app = FastAPI(title="Predicta AI v5.5 - Hybrid", version="5.5")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_cached_matches: List[Dict[str, Any]] = []
_cache_lock = threading.Lock()
predictor = ImprovedPredictionEngine()

# ----------------------------------------------------
# Yardımcı Fonksiyonlar
# ----------------------------------------------------
def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def save_matches_to_disk(matches: List[Dict[str, Any]]):
    ensure_data_dir()
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(matches)} matches to {DATA_FILE}")

def load_matches_from_disk() -> List[Dict[str, Any]]:
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def parse_nesine_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    matches = []
    if not isinstance(data, dict):
        return matches
    sg = data.get("sg", {})
    ea = sg.get("EA", [])
    for m in ea:
        if m.get("GT") != 1:
            continue
        home = m.get("HN", "").strip()
        away = m.get("AN", "").strip()
        if not home or not away:
            continue
        odds = {"1": 2.0, "X": 3.0, "2": 3.5}
        for bahis in m.get("MA", []):
            if bahis.get("MTID") == 1:
                oranlar = bahis.get("OCA", [])
                if len(oranlar) >= 3:
                    try:
                        odds["1"] = float(oranlar[0].get("O", 2.0))
                        odds["X"] = float(oranlar[1].get("O", 3.0))
                        odds["2"] = float(oranlar[2].get("O", 3.5))
                    except:
                        pass
        matches.append({
            "home_team": home,
            "away_team": away,
            "league": m.get("LC", "Bilinmeyen"),
            "date": m.get("D", datetime.now().strftime("%Y-%m-%d")),
            "time": m.get("T", "20:00"),
            "odds": odds,
            "is_live": False
        })
    return matches

def fetch_future_matches(days_ahead: int = DAYS_AHEAD) -> List[Dict[str, Any]]:
    all_matches = []
    s = requests.Session()
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    for i in range(days_ahead):
        date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
        url = f"{API_ENDPOINT}?date={date}"
        try:
            r = s.get(url, headers=headers, timeout=20)
            if r.status_code == 200:
                data = r.json()
                parsed = parse_nesine_json(data)
                all_matches.extend(parsed)
            else:
                logger.warning(f"{url} -> {r.status_code}")
        except Exception as e:
            logger.warning(f"{url} bağlantı hatası: {e}")
            continue
    # duplicate filtrele
    seen = set()
    uniq = []
    for m in all_matches:
        key = f"{m['home_team']}|{m['away_team']}|{m['date']}"
        if key not in seen:
            seen.add(key)
            uniq.append(m)
    uniq.sort(key=lambda x: (x["date"], x["time"]))
    logger.info(f"Toplam {len(uniq)} oynanmamış maç bulundu")
    return uniq

def background_refresh():
    global _cached_matches
    while True:
        try:
            matches = fetch_future_matches(DAYS_AHEAD)
            with _cache_lock:
                _cached_matches = matches
            save_matches_to_disk(matches)
        except Exception as e:
            logger.error(f"Background refresh error: {e}")
        time.sleep(REFRESH_INTERVAL_SECONDS)

# ----------------------------------------------------
# FastAPI Routes
# ----------------------------------------------------
@app.on_event("startup")
def on_startup():
    global _cached_matches
    _cached_matches = load_matches_from_disk()
    t = threading.Thread(target=background_refresh, daemon=True)
    t.start()
    logger.info("Background refresh thread started.")

@app.get("/")
def root():
    return {"service": "Predicta AI v5.5 Hybrid", "timestamp": datetime.now().isoformat()}

@app.get("/api/matches/upcoming")
def get_upcoming(limit: Optional[int] = 0):
    with _cache_lock:
        data = list(_cached_matches)
    if limit:
        data = data[:limit]
    return {"success": True, "count": len(data), "matches": data}

@app.post("/api/matches/refresh")
def manual_refresh():
    try:
        matches = fetch_future_matches(DAYS_AHEAD)
        with _cache_lock:
            global _cached_matches
            _cached_matches = matches
        save_matches_to_disk(matches)
        return {"success": True, "count": len(matches)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions/upcoming")
def get_predictions(limit: Optional[int] = 50):
    with _cache_lock:
        matches = list(_cached_matches)
    if not matches:
        raise HTTPException(status_code=503, detail="Henüz veri yok.")
    if limit:
        matches = matches[:limit]

    predictions = []
    for m in matches:
        try:
            p = predictor.predict_match(m["home_team"], m["away_team"], m["odds"], m["league"])
            m2 = m.copy()
            m2["ai_prediction"] = p
            predictions.append(m2)
        except Exception as e:
            logger.error(f"Tahmin hatası: {e}")
    return {
        "success": True,
        "count": len(predictions),
        "matches": predictions,
        "engine": "ImprovedPredictionEngine",
        "timestamp": datetime.now().isoformat()
    }

# ----------------------------------------------------
# Run
# ----------------------------------------------------
if __name__ == "__main__":
    ensure_data_dir()
    try:
        _cached_matches = fetch_future_matches(DAYS_AHEAD)
        save_matches_to_disk(_cached_matches)
    except Exception as e:
        logger.warning(f"Initial fetch failed: {e}")
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
