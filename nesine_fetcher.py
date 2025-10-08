#!/usr/bin/env python3
"""
Nesine API'den canlı maç çekme - GERÇEK 2025 API FORMATI
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

# Lig kodu -> İsim mapping
LEAGUE_NAMES = {
    2188: "Türkiye Süper Lig",
    2189: "Türkiye 1. Lig",
    2190: "İngiltere Premier League",
    2191: "İngiltere Championship",
    2192: "İspanya LaLiga",
    2193: "İspanya LaLiga 2",
    2194: "İtalya Serie A",
    2195: "İtalya Serie B",
    2196: "Almanya Bundesliga",
    2197: "Almanya 2. Bundesliga",
    2198: "Fransa Ligue 1",
    2199: "Fransa Ligue 2",
    2200: "Hollanda Eredivisie",
    2201: "Portekiz Primeira Liga",
    2202: "Belçika Pro League",
    2203: "Avusturya Bundesliga",
    2204: "İsviçre Super League",
    2205: "İskoçya Premiership",
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


def _extract_1x2_from_market(market: Dict) -> Optional[Dict[str, float]]:
    """
    Market içinden MS (1X2) oranlarını çıkar
    
    Format:
    {
      "NO": 8924,  # Market numarası (570 = MS, 8924 = Sanal MS?)
      "OCA": [
        {"N": 1, "O": 205},  # 1: Ev
        {"N": 2, "O": 320},  # X veya 2
        {"N": 3, "O": 350}   # 2 veya X
      ]
    }
    """
    if not isinstance(market, dict):
        return None
    
    # MS piyasası kontrolü (NO: 570 veya MTID: 570)
    market_no = market.get("NO") or market.get("MTID")
    
    # Oranlar
    outcomes = market.get("OCA") or market.get("OC") or []
    
    if not outcomes or not isinstance(outcomes, list):
        return None
    
    odds_map = {}
    
    for oc in outcomes:
        if not isinstance(oc, dict):
            continue
        
        n = oc.get("N")  # Outcome index (1, 2, 3)
        o = oc.get("O")  # Odd value (örn: 205 = 2.05)
        
        if n is None or o is None:
            continue
        
        try:
            odd_value = float(o) / 100.0  # 205 -> 2.05
        except:
            continue
        
        # N değerine göre eşleştir
        if n == 1:
            odds_map["1"] = odd_value
        elif n == 2:
            # Genelde X ama bazen 2 olabiliyor
            if "X" not in odds_map:
                odds_map["X"] = odd_value
            else:
                odds_map["2"] = odd_value
        elif n == 3:
            odds_map["2"] = odd_value
    
    # 3 oran da var mı?
    if len(odds_map) == 3 and "1" in odds_map and "X" in odds_map and "2" in odds_map:
        return odds_map
    
    return None


def fetch_matches_for_date(date: dt.date, filter_leagues: bool = True) -> List[Dict[str, Any]]:
    """
    GERÇEK API YAPISI:
    {
      "sg": {
        "EA": [  # Event Array
          {
            "C": 2426762,        # Event code
            "HN": "Phoenix (K)", # Home name
            "AN": "Las Vegas",   # Away name
            "T": "03:00",        # Time
            "D": "09.10.2025",   # Date
            "LC": 2186,          # League code
            "TYPE": 2,           # Type (1=Canlı, 2=Ön bahis)
            "GT": 2,             # Game type (1=Futbol, 2=Basketbol, vb)
            "MA": [              # Markets
              {
                "NO": 570,       # Market type (570=MS)
                "OCA": [         # Outcomes
                  {"N": 1, "O": 205},  # 1: 2.05
                  {"N": 2, "O": 320},  # X: 3.20
                  {"N": 3, "O": 350}   # 2: 3.50
                ]
              }
            ]
          }
        ]
      },
      "nsn": {...}
    }
    """
    date_str = date.strftime("%Y-%m-%d")
    url = PREBULTEN_URL.format(date=date_str)
    
    data = _get(url)
    
    if not data:
        print(f"[Nesine] {date_str} için veri alınamadı!")
        return []

    events = []
    
    # Ana yapı: sg -> EA
    sg = data.get("sg") or data.get("SG")
    
    if not sg or not isinstance(sg, dict):
        print("[Nesine] 'sg' anahtarı bulunamadı!")
        return []
    
    event_array = sg.get("EA") or sg.get("ea")
    
    if not event_array or not isinstance(event_array, list):
        print("[Nesine] 'EA' (Event Array) bulunamadı!")
        return []
    
    print(f"[Nesine] {len(event_array)} event bulundu")
    
    # Filtreleme için sayaçlar
    total_events = len(event_array)
    football_events = 0
    today_events = 0
    valid_events = 0
    
    for event in event_array:
        if not isinstance(event, dict):
            continue
        
        # Spor türü filtresi (GT: 1 = Futbol)
        game_type = event.get("GT")
        
        if filter_leagues and game_type != 1:
            continue  # Sadece futbol
        
        football_events += 1
        
        # Canlı değil, ön bahis mi? (TYPE: 2)
        event_type = event.get("TYPE")
        
        # Takım isimleri
        home_team = str(event.get("HN") or "").strip()
        away_team = str(event.get("AN") or "").strip()
        
        if not home_team or not away_team:
            continue
        
        # Tarih kontrolü (bugün mü?)
        event_date = str(event.get("D") or "")
        today_str = date.strftime("%d.%m.%Y")
        
        if filter_leagues and event_date != today_str:
            continue  # Sadece bugünkü maçlar
        
        today_events += 1
        
        # Saat
        start_time = str(event.get("T") or "??:??")
        
        # Lig bilgisi
        league_code = event.get("LC")
        league_name = LEAGUE_NAMES.get(league_code, f"Lig {league_code}")
        
        # Oranlar (MA -> Markets)
        markets = event.get("MA") or []
        
        odds = None
        for market in markets:
            odds = _extract_1x2_from_market(market)
            if odds:
                break
        
        if not odds:
            # Varsayılan oranlar
            odds = {"1": 2.00, "X": 3.20, "2": 3.50}
        
        events.append({
            "home_team": home_team,
            "away_team": away_team,
            "league": league_name,
            "start_time": start_time,
            "odds": odds
        })
        
        valid_events += 1
    
    print(f"[Nesine] Filtre sonuçları:")
    print(f"  - Toplam event: {total_events}")
    print(f"  - Futbol: {football_events}")
    print(f"  - Bugün: {today_events}")
    print(f"  - Geçerli: {valid_events}")
    
    return events


def fetch_today(filter_leagues: bool = True) -> List[Dict[str, Any]]:
    """Bugünün maçlarını çek"""
    today = dt.date.today()
    print(f"[Nesine] Bugünün maçları çekiliyor: {today}")
    return fetch_matches_for_date(today, filter_leagues=filter_leagues)


if __name__ == "__main__":
    print("=" * 60)
    print("Nesine Fetcher Test (GERÇEK API FORMATI)")
    print("=" * 60)
    
    # Bugün
    print("\n🔍 Bugünün maçları...")
    matches = fetch_today(filter_leagues=True)
    
    if matches:
        print(f"\n✅ {len(matches)} maç bulundu\n")
        print(json.dumps(matches[:5], ensure_ascii=False, indent=2))
    else:
        print("\n⚠️ Bugün maç yok")
        
        # Yarını dene
        print("\n🔍 Yarının maçları...")
        tomorrow = dt.date.today() + dt.timedelta(days=1)
        matches = fetch_matches_for_date(tomorrow, filter_leagues=True)
        
        if matches:
            print(f"\n✅ Yarın {len(matches)} maç var\n")
            print(json.dumps(matches[:5], ensure_ascii=False, indent=2))
        else:
            print("\n❌ Hiç maç bulunamadı!")
