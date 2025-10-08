#!/usr/bin/env python3
"""
Nesine API'den canlı maç çekme - 2025 API FORMATI
"""

import datetime as dt
import time
import json
from typing import List, Dict, Any, Optional
import requests

PREBULTEN_URL = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull?date={date}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "tr-TR,tr;q=0.9",
    "Referer": "https://www.nesine.com/"
}

def _get(url: str, retries: int = 3, timeout: int = 15) -> Optional[Dict[str, Any]]:
    """HTTP GET isteği"""
    last_err = None
    for i in range(retries):
        try:
            print(f"[Nesine] İstek: {url}")
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            print(f"[Nesine] Status: {r.status_code}")
            
            if r.ok:
                data = r.json()
                print(f"[Nesine] Veri boyutu: {len(str(data))} karakter")
                return data
            else:
                last_err = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:
            last_err = e
            print(f"[Nesine] Hata (deneme {i+1}/{retries}): {e}")
        
        if i < retries - 1:
            time.sleep(1.5 * (i+1))
    
    print(f"[Nesine] GET başarısız: {last_err}")
    return None


def _extract_1x2_new(outcome_list: List) -> Optional[Dict[str, float]]:
    """
    YENİ API FORMATI: Outcomes listesinden 1-X-2 oranlarını çıkar
    Format: [{"N": "1", "OD": "2.05"}, {"N": "X", "OD": "3.30"}, ...]
    """
    if not outcome_list or not isinstance(outcome_list, list):
        return None
    
    odds_map = {}
    
    for outcome in outcome_list:
        if not isinstance(outcome, dict):
            continue
        
        # N: Outcome adı (1, X, 2)
        # OD veya O: Oran değeri
        name = str(outcome.get("N", "")).strip().upper()
        odd_value = outcome.get("OD") or outcome.get("O")
        
        if not odd_value:
            continue
        
        try:
            odd = float(odd_value)
        except:
            continue
        
        # 1-X-2 eşleştirme
        if name in {"1", "EV", "HOME"}:
            odds_map["1"] = odd
        elif name in {"X", "0", "DRAW", "BER"}:
            odds_map["X"] = odd
        elif name in {"2", "DEP", "AWAY"}:
            odds_map["2"] = odd
    
    # 3 oran da bulundu mu?
    if len(odds_map) == 3 and "1" in odds_map and "X" in odds_map and "2" in odds_map:
        return odds_map
    
    return None


def fetch_matches_for_date(date: dt.date, filter_leagues: bool = True) -> List[Dict[str, Any]]:
    """
    YENİ API FORMATI İÇİN PARSER
    
    API Yapısı:
    {
      "C": "code",
      "MN": "league_name",
      "E": [  # Events (maçlar)
        {
          "N": "home_team - away_team",
          "D": "2025-10-08T19:00:00",
          "O": [  # Outcomes (oranlar)
            {"N": "MS", "OC": [{"N": "1", "OD": "2.05"}, ...]}
          ]
        }
      ]
    }
    """
    date_str = date.strftime("%Y-%m-%d")
    url = PREBULTEN_URL.format(date=date_str)
    
    data = _get(url)
    
    if not data:
        print(f"[Nesine] {date_str} için veri alınamadı!")
        return []

    events = []
    
    # Ana anahtarları debug et
    print(f"[Nesine] Ana anahtarlar: {list(data.keys())[:10]}")
    
    # Yeni format: Top-level array veya dict içinde array
    leagues_data = None
    
    if isinstance(data, list):
        leagues_data = data
    elif isinstance(data, dict):
        # Olası anahtarlar: Data, data, leagues, Leagues, items vb
        for key in ["Data", "data", "D", "items", "leagues", "Leagues"]:
            if key in data and isinstance(data[key], list):
                leagues_data = data[key]
                break
        
        # Eğer hala bulamadıysak, tüm list değerleri dene
        if not leagues_data:
            for v in data.values():
                if isinstance(v, list) and len(v) > 0:
                    leagues_data = v
                    break
    
    if not leagues_data:
        print("[Nesine] Lig verisi bulunamadı! API yapısı:")
        print(json.dumps(data, indent=2, ensure_ascii=False)[:1000])
        return []
    
    print(f"[Nesine] {len(leagues_data)} lig bulundu")
    
    # Lig filtresi (sadece futbol)
    football_keywords = [
        "süper lig", "1. lig", "premier", "championship", "laliga", "serie a", 
        "bundesliga", "ligue 1", "eredivisie", "liga", "league", "futbol"
    ]
    
    for league in leagues_data:
        if not isinstance(league, dict):
            continue
        
        # Lig adı
        league_name = (
            league.get("MN") or 
            league.get("N") or 
            league.get("LeagueName") or 
            league.get("name") or 
            "Bilinmeyen"
        )
        
        # Lig filtresi
        if filter_leagues:
            league_lower = str(league_name).lower()
            if not any(kw in league_lower for kw in football_keywords):
                continue
        
        # Maçlar
        matches = (
            league.get("E") or 
            league.get("Events") or 
            league.get("events") or 
            league.get("matches") or
            []
        )
        
        print(f"[Nesine] {league_name}: {len(matches)} maç")
        
        for match in matches:
            if not isinstance(match, dict):
                continue
            
            # Takım isimleri
            match_name = str(match.get("N") or match.get("Name") or "").strip()
            
            # Format: "Home Team - Away Team"
            if " - " in match_name:
                parts = match_name.split(" - ", 1)
                home_team = parts[0].strip()
                away_team = parts[1].strip()
            else:
                # Alternatif format
                home_team = str(match.get("H") or match.get("HT") or "").strip()
                away_team = str(match.get("A") or match.get("AT") or "").strip()
            
            if not home_team or not away_team:
                continue
            
            # Saat
            start_time = match.get("D") or match.get("Date") or match.get("StartDate") or ""
            if isinstance(start_time, str) and len(start_time) >= 16:
                hhmm = start_time[11:16]
            else:
                hhmm = "??:??"
            
            # Oranlar (O veya OC)
            outcomes = match.get("O") or match.get("OC") or match.get("Outcomes") or []
            
            # MS (Maç Sonucu) piyasasını bul
            odds = None
            for market in outcomes:
                if not isinstance(market, dict):
                    continue
                
                market_name = str(market.get("N") or market.get("Name") or "").upper()
                
                # MS piyasası
                if "MS" in market_name or "1X2" in market_name or market_name in {"MATCHRESULT", "RESULT"}:
                    outcomes_list = market.get("OC") or market.get("Outcomes") or []
                    odds = _extract_1x2_new(outcomes_list)
                    if odds:
                        break
            
            if not odds:
                print(f"[Nesine] {home_team} - {away_team}: Oran bulunamadı, varsayılan kullanılıyor")
                odds = {"1": 2.00, "X": 3.20, "2": 3.50}
            
            events.append({
                "home_team": home_team,
                "away_team": away_team,
                "league": league_name,
                "start_time": hhmm,
                "odds": odds
            })
    
    print(f"[Nesine] Toplam {len(events)} geçerli maç çekildi")
    return events


def fetch_today(filter_leagues: bool = True) -> List[Dict[str, Any]]:
    """Bugünün maçlarını çek"""
    today = dt.date.today()
    print(f"[Nesine] Bugünün maçları çekiliyor: {today}")
    return fetch_matches_for_date(today, filter_leagues=filter_leagues)


if __name__ == "__main__":
    print("=" * 50)
    print("Nesine Fetcher Test (YENİ FORMAT)")
    print("=" * 50)
    
    # Önce bugün
    matches = fetch_today(filter_leagues=True)
    
    if matches:
        print(f"\n✅ {len(matches)} maç bulundu\n")
        print(json.dumps(matches[:3], ensure_ascii=False, indent=2))
    else:
        print("\n⚠️ Bugün maç yok, yarını deneyelim...")
        tomorrow = dt.date.today() + dt.timedelta(days=1)
        matches = fetch_matches_for_date(tomorrow, filter_leagues=True)
        
        if matches:
            print(f"\n✅ Yarın {len(matches)} maç var\n")
            print(json.dumps(matches[:3], ensure_ascii=False, indent=2))
        else:
            print("\n❌ Maç bulunamadı!")
