#!/usr/bin/env python3
"""
Nesine API'den canlı maç çekme - GÜNCELLENMİŞ VERSİYON
"""

import datetime as dt
import time
import json
from typing import List, Dict, Any, Optional
import requests

# Alternatif API endpointleri
PREBULTEN_URL = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull?date={date}"
ALTERNATIVE_URL = "https://www.nesine.com/iddaa/pre-bulten?date={date}"

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
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last_err = e
            print(f"[Nesine] Hata (deneme {i+1}/{retries}): {e}")
        
        if i < retries - 1:
            time.sleep(1.5 * (i+1))
    
    print(f"[Nesine] GET başarısız: {last_err}")
    return None


def _extract_1x2(market_list: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """
    1-X-2 oranlarını çıkar - GELİŞTİRİLMİŞ VERSİYON
    """
    if not market_list:
        return None

    # Tüm olası piyasa isimlerini dene
    market_keywords = [
        "maç sonucu", "mac sonucu", "match result", "ms", "1x2",
        "futbol maç sonucu", "3 yol"
    ]
    
    candidates = []
    for m in market_list:
        name = str(m.get("name") or m.get("n") or "").lower()
        code = str(m.get("code") or m.get("c") or "").lower()
        
        if any(kw in name for kw in market_keywords) or any(kw in code for kw in market_keywords):
            candidates.append(m)
    
    # Önce adayları, sonra tüm piyasaları dene
    for m in (candidates if candidates else market_list):
        outcomes = m.get("outcomes") or m.get("o") or m.get("Outcomes") or []
        odds_map = {}
        
        for o in outcomes:
            # Farklı anahtar isimleri dene
            oname = str(o.get("name") or o.get("n") or o.get("Name") or "").strip().lower()
            odd = o.get("odd") or o.get("v") or o.get("oddValue") or o.get("Odd")
            
            try:
                odd = float(odd)
            except:
                continue
            
            # Sonuç eşleştirme
            if oname in {"1", "ev", "home", "home team", "ev sahibi"}:
                odds_map["1"] = odd
            elif oname in {"x", "0", "beraberlik", "draw", "berabere"}:
                odds_map["X"] = odd
            elif oname in {"2", "dep", "away", "away team", "deplasman"}:
                odds_map["2"] = odd
        
        if len(odds_map) == 3 and "1" in odds_map and "X" in odds_map and "2" in odds_map:
            return odds_map
    
    return None


def fetch_matches_for_date(date: dt.date) -> List[Dict[str, Any]]:
    """
    Verilen tarih için maçları çek - GELİŞTİRİLMİŞ VERSİYON
    """
    date_str = date.strftime("%Y-%m-%d")
    url = PREBULTEN_URL.format(date=date_str)
    
    data = _get(url)
    
    if not data:
        print(f"[Nesine] {date_str} için veri alınamadı!")
        return []

    events = []
    
    # API yapısını debug et
    print(f"[Nesine] Ana anahtarlar: {list(data.keys())}")
    
    # Farklı API yapılarını dene
    leagues = (
        data.get("Leagues") or 
        data.get("leagues") or 
        data.get("Data") or 
        data.get("data") or
        []
    )
    
    if not leagues:
        print("[Nesine] Lig verisi bulunamadı! API yapısı değişmiş olabilir.")
        print(f"[Nesine] Ham veri: {json.dumps(data, indent=2, ensure_ascii=False)[:500]}")
        return []
    
    print(f"[Nesine] {len(leagues)} lig bulundu")
    
    for lg in leagues:
        league_name = (
            lg.get("LeagueName") or 
            lg.get("name") or 
            lg.get("N") or 
            lg.get("Name") or 
            "Bilinmeyen Lig"
        )
        
        league_events = (
            lg.get("Events") or 
            lg.get("events") or 
            lg.get("E") or
            lg.get("Matches") or
            []
        )
        
        print(f"[Nesine] {league_name}: {len(league_events)} maç")
        
        for ev in league_events:
            home = str(ev.get("HomeTeamName") or ev.get("home") or ev.get("H") or ev.get("HomeTeam") or "").strip()
            away = str(ev.get("AwayTeamName") or ev.get("away") or ev.get("A") or ev.get("AwayTeam") or "").strip()
            
            if not home or not away:
                continue
            
            # Saat bilgisi
            stime = ev.get("StartDate") or ev.get("startTime") or ev.get("S") or ev.get("Date") or ""
            if isinstance(stime, str) and len(stime) >= 16:
                hhmm = stime[11:16]
            else:
                hhmm = "??:??"
            
            # Oranları çek
            markets = ev.get("Markets") or ev.get("markets") or ev.get("M") or ev.get("Odds") or []
            odds = _extract_1x2(markets)
            
            if not odds:
                # Oran bulunamazsa varsayılan değerler
                print(f"[Nesine] {home} - {away}: Oran bulunamadı, varsayılan kullanılıyor")
                odds = {"1": 2.00, "X": 3.20, "2": 3.50}
            
            events.append({
                "home_team": home,
                "away_team": away,
                "league": league_name,
                "start_time": hhmm,
                "odds": odds
            })
    
    print(f"[Nesine] Toplam {len(events)} geçerli maç çekildi")
    return events


def fetch_today() -> List[Dict[str, Any]]:
    """Bugünün maçlarını çek"""
    # Bugünün gerçek tarihi
    today = dt.date.today()
    print(f"[Nesine] Bugünün maçları çekiliyor: {today}")
    
    # Not: Eğer test için farklı bir tarih gerekiyorsa:
    # today = dt.date(2025, 10, 8)
    
    return fetch_matches_for_date(today)


# Test için
if __name__ == "__main__":
    print("=" * 50)
    print("Nesine Fetcher Test")
    print("=" * 50)
    
    matches = fetch_today()
    
    if matches:
        print(f"\n✅ {len(matches)} maç bulundu\n")
        print(json.dumps(matches[:3], ensure_ascii=False, indent=2))
    else:
        print("\n❌ Maç bulunamadı! API'yi manuel kontrol edin:")
        print("https://cdnbulten.nesine.com/api/bulten/getprebultenfull?date=" + dt.date.today().strftime("%Y-%m-%d"))