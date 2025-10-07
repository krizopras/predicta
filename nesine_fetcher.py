# ==============================================================
#  nesine_fetcher.py
#  Günlük maçları Nesine CDN PreBülten API'sinden çeker.
#  Çıktı formatı: [{home_team, away_team, league, start_time, odds:{"1":..,"X":..,"2":..}}]
# ==============================================================

from __future__ import annotations
import datetime as dt
import time
import json
from typing import List, Dict, Any, Optional
import requests

PREBULTEN_URL = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull?date={date}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PredictaBot/1.0; +https://krizopras.github.io/predicta/)",
    "Accept": "application/json,text/plain,*/*",
}

def _get(url: str, retries: int = 3, timeout: int = 12) -> Optional[Dict[str, Any]]:
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.ok:
                return r.json()
            last_err = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:
            last_err = e
        time.sleep(0.8 * (i+1))
    print(f"[nesine_fetcher] GET failed: {last_err}")
    return None


def _extract_1x2(market_list: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """
    Nesine JSON'unda bir event için 1-X-2 oranlarını bulur.
    Piyasalar farklı anahtarlarla gelebilir; en yaygınları:
      - "Futbol Maç Sonucu" / "Maç Sonucu"
      - marketCode: "MS" / "MatchResult" vb.
    """
    if not market_list:
        return None

    candidates = []
    for m in market_list:
        name = (m.get("name") or m.get("n") or "").lower()
        code = (m.get("code") or m.get("c") or "").lower()
        if any(k in name for k in ["maç sonucu", "futbol maç sonucu", "mac sonucu", "match result"]) or code in {"ms", "matchresult"}:
            candidates.append(m)

    # aday yoksa tüm piyasalara bak
    for m in (candidates or market_list):
        outcomes = m.get("outcomes") or m.get("o") or []
        odds_map = {}
        for o in outcomes:
            oname = (o.get("name") or o.get("n") or "").strip().lower()
            odd = o.get("odd") or o.get("v") or o.get("oddValue")
            try:
                odd = float(odd)
            except Exception:
                continue
            if oname in {"1", "ev", "home", "home team"}:
                odds_map["1"] = odd
            elif oname in {"x", "beraberlik", "draw"}:
                odds_map["X"] = odd
            elif oname in {"2", "dep", "away", "away team"}:
                odds_map["2"] = odd
        if {"1","X","2"}.issubset(odds_map.keys()):
            return odds_map
    return None


def fetch_matches_for_date(date: dt.date) -> List[Dict[str, Any]]:
    """
    Verilen tarih için tüm maçları döner.
    - Tarih formatı: YYYY-MM-DD
    - Çıktıda lig adı, takımlar, başlama saati (HH:MM) ve 1X2 oranları bulunur.
    """
    date_str = date.strftime("%Y-%m-%d")
    url = PREBULTEN_URL.format(date=date_str)
    data = _get(url)
    if not data:
        return []

    events = []
    # Nesine dökümünde genellikle "Leagues" -> "Events" bulunur, varyasyonlara karşı korumalıyız.
    leagues = data.get("Leagues") or data.get("leagues") or data.get("Data") or []
    for lg in leagues:
        league_name = lg.get("LeagueName") or lg.get("name") or lg.get("N") or "—"
        league_events = lg.get("Events") or lg.get("events") or lg.get("E") or []
        for ev in league_events:
            home = ev.get("HomeTeamName") or ev.get("home") or ev.get("H") or ""
            away = ev.get("AwayTeamName") or ev.get("away") or ev.get("A") or ""
            # saat
            stime = ev.get("StartDate") or ev.get("startTime") or ev.get("S") or ""
            if isinstance(stime, str):
                # "2025-10-06T19:45:00" -> "19:45"
                hhmm = stime[11:16] if len(stime) >= 16 else stime
            else:
                hhmm = ""

            markets = ev.get("Markets") or ev.get("markets") or ev.get("M") or []
            odds = _extract_1x2(markets)
            if not odds:
                # 1x2 bulunamadıysa pas geç
                continue

            events.append({
                "home_team": str(home).strip(),
                "away_team": str(away).strip(),
                "league": str(league_name).strip(),
                "start_time": hhmm,
                "odds": odds
            })
    return events


def fetch_today() -> List[Dict[str, Any]]:
    """Bugünün maçlarını döner (yerel tarihe göre)."""
    today = dt.date.today()
    return fetch_matches_for_date(today)


if __name__ == "__main__":
    out = fetch_today()
    print(json.dumps(out[:5], ensure_ascii=False, indent=2))
    print(f"Toplam: {len(out)} maç")
