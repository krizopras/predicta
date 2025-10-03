#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predicta AI v5.4 - FutureMatch Collector
Çekirdek: Nesine'den bugünkü + ileri tarihli oynanmamış futbol maçlarını toplar,
diskte JSON olarak saklar ve FastAPI üzerinden servis eder.
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

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = os.environ.get("PREDICTA_DATA_DIR", "./data")
DATA_FILE = os.path.join(DATA_DIR, "future_matches.json")
DAYS_AHEAD = int(os.environ.get("PREDICTA_DAYS_AHEAD", "3"))  # kaç gün ileriye topla
REFRESH_INTERVAL_SECONDS = int(os.environ.get("PREDICTA_REFRESH_SECONDS", str(60 * 60 * 6)))  # default 6 saat
USER_AGENT = os.environ.get("PREDICTA_USER_AGENT",
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

API_ENDPOINTS = [
    "https://cdnbulten.nesine.com/api/bulten/getprebultenfull",
    "https://www.nesine.com/api/prematch/bulletin",
    "https://api.nesine.com/api/bulletin/prematch",
    "https://www.nesine.com/futbol/mac-programi/json"
]

MOBILE_IDDAA = "https://m.nesine.com/iddaa"
DESKTOP_IDDAA = "https://www.nesine.com/iddaa"

# -------------------------
# LOGGER
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("predicta-v5.4")

# -------------------------
# APP + GLOBAL CACHE
# -------------------------
app = FastAPI(title="Predicta AI v5.4 - FutureMatch Collector", version="5.4")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_cached_matches: List[Dict[str, Any]] = []
_cache_lock = threading.Lock()


# -------------------------
# UTIL: Save / Load disk
# -------------------------
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
            data = json.load(f)
            if isinstance(data, list):
                logger.info(f"Loaded {len(data)} matches from disk")
                return data
    except Exception as e:
        logger.warning(f"Failed to load disk data: {e}")
    return []


# -------------------------
# PARSERS
# -------------------------
def parse_nesine_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Nesine JSON parser — sg.EA / sg.CA yapılarını kullanır.
    Sadece GT == 1 (futbol) maçlarını döndürür.
    """
    matches = []
    try:
        if not isinstance(data, dict):
            return matches

        sg = data.get("sg") or data.get("data") or {}
        ea = sg.get("EA", []) if isinstance(sg, dict) else []
        ca = sg.get("CA", []) if isinstance(sg, dict) else []
        all_entries = []
        if ea:
            all_entries.extend(ea)
        if ca:
            all_entries.extend(ca)

        for m in all_entries:
            try:
                # GT==1 => futbol
                if m.get("GT") is not None and int(m.get("GT")) != 1:
                    continue

                home = (m.get("HN") or m.get("home") or m.get("home_team") or "").strip()
                away = (m.get("AN") or m.get("away") or m.get("away_team") or "").strip()
                date = m.get("D") or m.get("date") or ""
                time_str = m.get("T") or m.get("time") or ""

                # odds
                odds = {'1': None, 'X': None, '2': None}
                for bahis in m.get("MA", []):
                    if bahis.get("MTID") == 1:
                        oranlar = bahis.get("OCA", [])
                        if len(oranlar) >= 3:
                            try:
                                odds['1'] = float(oranlar[0].get('O') or oranlar[0].get('o') or 0)
                                odds['X'] = float(oranlar[1].get('O') or oranlar[1].get('o') or 0)
                                odds['2'] = float(oranlar[2].get('O') or oranlar[2].get('o') or 0)
                            except:
                                pass
                        break

                if not home or not away:
                    continue

                matches.append({
                    "home_team": home,
                    "away_team": away,
                    "league": m.get("LC") or m.get("league") or "Bilinmeyen",
                    "date": _normalize_date(date),
                    "time": _normalize_time(time_str),
                    "odds": odds,
                    "is_live": bool(m.get("S") == 1)
                })
            except Exception:
                continue

    except Exception as e:
        logger.debug(f"parse_nesine_json error: {e}")
    return matches


def _normalize_date(d: Optional[str]) -> str:
    """Basit normalize: eğer 'YYYY-MM-DD' benzeri yoksa today string döndürür"""
    if not d:
        return datetime.now().strftime("%Y-%m-%d")
    # bazı yanıtlar farklı formatta olabilir; basit regex ile al
    m = re.search(r"(\d{4}-\d{2}-\d{2})", str(d))
    if m:
        return m.group(1)
    # alternatif: dd.mm.yyyy
    m2 = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", str(d))
    if m2:
        return f"{m2.group(3)}-{m2.group(2)}-{m2.group(1)}"
    return str(d)


def _normalize_time(t: Optional[str]) -> str:
    if not t:
        return "00:00"
    m = re.search(r"(\d{1,2}:\d{2})", str(t))
    if m:
        tm = m.group(1)
        if len(tm.split(":")[0]) == 1:
            # 9:00 -> 09:00
            hh, mm = tm.split(":")
            return f"{int(hh):02d}:{mm}"
        return tm
    return "00:00"


# -------------------------
# HTML FALLBACK (desktop + mobile)
# -------------------------
def extract_matches_from_html(html_text: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html_text, "html.parser")
    matches = []
    seen = set()

    # 1) Scriptlerde JSON arama (basit)
    for script in soup.find_all("script"):
        txt = script.string or ""
        if not txt:
            continue
        # common patterns
        patterns = [
            r'window\.__INITIAL_STATE__\s*=\s*({.*?});',
            r'window\.APP_DATA\s*=\s*({.*?});',
            r'var\s+matches\s*=\s*(\[.*?\]);'
        ]
        for p in patterns:
            m = re.search(p, txt, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(1))
                    extracted = _extract_from_obj_recursive(obj)
                    for e in extracted:
                        key = f"{e['home_team']}|{e['away_team']}|{e['date']}|{e['time']}"
                        if key not in seen:
                            seen.add(key)
                            matches.append(e)
                except Exception:
                    continue

    # 2) Table veya text'ten "Team - Team" pattern
    text = soup.get_text(" ", strip=True)
    team_pattern = re.compile(r'([A-ZÇĞİÖŞÜ][a-zçğıöşü\.]{2,30})\s+(?:-|vs\.?|–)\s+([A-ZÇĞİÖŞÜ][a-zçğıöşü\.]{2,30})')
    for m in team_pattern.finditer(text):
        home = m.group(1).strip()
        away = m.group(2).strip()
        key = f"{home}|{away}"
        if key in seen:
            continue
        seen.add(key)
        matches.append({
            "home_team": home,
            "away_team": away,
            "league": "Bilinmeyen",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": "00:00",
            "odds": {"1": None, "X": None, "2": None},
            "is_live": False
        })

    return matches


def _extract_from_obj_recursive(obj: Any, depth: int = 0) -> List[Dict[str, Any]]:
    if depth > 6:
        return []
    found = []
    if isinstance(obj, dict):
        # quick detection
        if all(k in obj for k in ("home", "away")) or ("HN" in obj and "AN" in obj):
            home = str(obj.get("home") or obj.get("HN") or "")
            away = str(obj.get("away") or obj.get("AN") or "")
            date = _normalize_date(obj.get("date") or obj.get("D"))
            time_str = _normalize_time(obj.get("time") or obj.get("T"))
            if home and away:
                found.append({
                    "home_team": home.strip(),
                    "away_team": away.strip(),
                    "league": obj.get("league") or obj.get("LC") or "Bilinmeyen",
                    "date": date,
                    "time": time_str,
                    "odds": obj.get("odds", {"1": None, "X": None, "2": None}),
                    "is_live": bool(obj.get("is_live") or obj.get("S") == 1)
                })
        for v in obj.values():
            found.extend(_extract_from_obj_recursive(v, depth + 1))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_extract_from_obj_recursive(item, depth + 1))
    return found


# -------------------------
# NETWORK: fetch per-date
# -------------------------
def fetch_date_from_endpoints(date_str: str) -> List[Dict[str, Any]]:
    """
    Belirli bir tarih için API endpoint'lerini sırayla dener.
    Dönen tüm maçları (parsed) union halinde döndürür.
    """
    all_matches = []
    session = requests.Session()
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json, text/html"}
    for base in API_ENDPOINTS:
        try:
            # bazı endpoint'ler date param kabul eder - ekle
            if "?" in base:
                url = f"{base}&date={date_str}"
            else:
                url = f"{base}?date={date_str}"
            logger.debug(f"Trying API: {url}")
            r = session.get(url, headers=headers, timeout=10)
            logger.info(f"API {base} -> status {r.status_code}, len {len(r.text)}")
            if r.status_code == 200:
                # try json first
                try:
                    data = r.json()
                    parsed = parse_nesine_json(data)
                    if parsed:
                        all_matches.extend(parsed)
                except Exception:
                    # fallback to html parse if returned HTML
                    html_parsed = extract_matches_from_html(r.text)
                    if html_parsed:
                        all_matches.extend(html_parsed)
            else:
                # non-200 -> skip
                pass
        except Exception as e:
            logger.debug(f"Endpoint {base} failed: {e}")
            continue
    return all_matches


def fetch_html_fallback(date_str: str) -> List[Dict[str, Any]]:
    """Mobil + desktop sayfaları kontrol et (ek maç bulmak için)"""
    found = []
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(MOBILE_IDDAA, headers=headers, timeout=10)
        if r.status_code == 200:
            found.extend(extract_matches_from_html(r.text))
    except Exception:
        pass

    try:
        r = requests.get(DESKTOP_IDDAA, headers=headers, timeout=10)
        if r.status_code == 200:
            found.extend(extract_matches_from_html(r.text))
    except Exception:
        pass

    return found


# -------------------------
# MAIN: Collect future matches
# -------------------------
def is_future_match(m: Dict[str, Any]) -> bool:
    """Match'in datetime'ı şu anın ilerisi mi kontrol et"""
    date_str = m.get("date") or ""
    time_str = m.get("time") or "00:00"
    try:
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        return dt > datetime.now()
    except Exception:
        # eğer parse edilemez ama date >= today string ise kabul et
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
            return d.date() >= datetime.now().date()
        except Exception:
            return True


def deduplicate_matches(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for m in matches:
        key = f"{m.get('home_team','').strip().lower()}|{m.get('away_team','').strip().lower()}|{m.get('date','')}|{m.get('time','')}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(m)
    return unique


def collect_future_matches(days_ahead: int = DAYS_AHEAD) -> List[Dict[str, Any]]:
    logger.info(f"Collecting future matches for next {days_ahead} days...")
    all_matches = []
    for i in range(days_ahead):
        date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            # 1) Try API endpoints with date param
            parsed = fetch_date_from_endpoints(date)
            if parsed:
                all_matches.extend(parsed)

            # 2) HTML fallback (mobile/desktop)
            fallback = fetch_html_fallback(date)
            if fallback:
                all_matches.extend(fallback)

        except Exception as e:
            logger.debug(f"Date {date} fetch error: {e}")
            continue

    # filter only football-like matches (we already tried to ensure GT==1 in parser, but ensure again)
    filtered = []
    for m in all_matches:
        # simple heuristic: home/away must contain alphabetic chars & not include UI words
        home = m.get("home_team","")
        away = m.get("away_team","")
        if not home or not away:
            continue
        if any(bad in home.lower() for bad in ["hesabim","giris","menu","kupon","casino"]):
            continue
        filtered.append(m)

    # only future matches
    future = [m for m in filtered if is_future_match(m)]

    # dedupe
    unique = deduplicate_matches(future)

    # sort by date/time
    def sort_key(x):
        try:
            return datetime.strptime(f"{x.get('date','')} {x.get('time','00:00')}", "%Y-%m-%d %H:%M")
        except Exception:
            return datetime.max
    unique.sort(key=sort_key)

    logger.info(f"Collected {len(unique)} unique future matches")
    return unique


# -------------------------
# BACKGROUND REFRESH LOOP (thread)
# -------------------------
def _refresh_loop():
    global _cached_matches
    while True:
        try:
            matches = collect_future_matches(DAYS_AHEAD)
            with _cache_lock:
                _cached_matches = matches
            save_matches_to_disk(matches)
        except Exception as e:
            logger.error(f"Background refresh failed: {e}")
        time.sleep(REFRESH_INTERVAL_SECONDS)


# -------------------------
# FASTAPI EVENTS & ENDPOINTS
# -------------------------
@app.on_event("startup")
def startup_event():
    # load disk cache first
    global _cached_matches
    _cached_matches = load_matches_from_disk()
    # start background thread
    t = threading.Thread(target=_refresh_loop, daemon=True)
    t.start()
    logger.info("Background refresh thread started")


@app.get("/")
def root():
    return {"service": "Predicta AI v5.4 - FutureMatch Collector", "timestamp": datetime.now().isoformat()}


@app.get("/api/matches/upcoming")
def get_upcoming(limit: Optional[int] = 0):
    """Return upcoming matches from cache. limit=0 => all"""
    with _cache_lock:
        data = list(_cached_matches)
    if limit and limit > 0:
        data = data[:limit]
    return {"success": True, "count": len(data), "matches": data, "timestamp": datetime.now().isoformat()}


@app.post("/api/matches/refresh")
def manual_refresh():
    """Manual refresh (runs a synchronous collection, updates cache & disk)"""
    try:
        matches = collect_future_matches(DAYS_AHEAD)
        with _cache_lock:
            global _cached_matches
            _cached_matches = matches
        save_matches_to_disk(matches)
        return {"success": True, "count": len(matches)}
    except Exception as e:
        logger.error(f"manual refresh failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/matches/raw")
def raw_cached():
    with _cache_lock:
        return {"count": len(_cached_matches), "matches": _cached_matches}


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    ensure_data_dir()
    # on start do an initial blocking collect to populate cache (so first responses are meaningful)
    try:
        initial = collect_future_matches(DAYS_AHEAD)
        with _cache_lock:
            _cached_matches = initial
        save_matches_to_disk(initial)
    except Exception as e:
        logger.warning(f"Initial collect failed: {e}")

    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
