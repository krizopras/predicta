#!/usr/bin/env python3
"""
Nesine API Fetcher - YENİ FORMAT (2025)
API yapısı: {"g":"...", "C":"...", "N":"...", ...}
"""

from __future__ import annotations
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
    """HTTP GET request"""
    last_err = None
    for i in range(retries):
        try:
            print(f"[Nesine] İstek: {url}")
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            print(f"[Nesine] Status: {r.status_code}")
            
            if r.ok:
                data = r.json()
                print(f"[Nesine] Veri boyutu: {len(str(data))} karakter")
                print(f"[Nesine] Ana anahtarlar: {list(data.keys())[:10]}")
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


def parse_nesine_new_format(data: Dict[str, Any], filter_leagues: bool = True) -> List[Dict[str, Any]]:
    """
    Yeni Nesine API formatını parse et
    Format: Her maç bir anahtar-değer çifti, anahtarlar rakamlardan oluşuyor
    """
    events = []
    
    # Önemli ligler filtresi
    important_keywords = [
        'süper lig', 'super lig', '1. lig', 'premier league', 'la liga', 'laliga',
        'serie a', 'bundesliga', 'ligue 1', 'eredivisie', 'primeira liga',
        'champions league', 'şampiyonlar', 'europa league', 'avrupa ligi',
        'türkiye kupası', 'ziraat', 'championship', 'pro league'
    ]
    
    match_count = 0
    filtered_count = 0
    
    # Ana veri yapısını tara
    for key, value in data.items():
        # Maç verisi dict olmalı
        if not isinstance(value, dict):
            continue
        
        # Maç bilgilerini çıkar (farklı anahtar isimlerini dene)
        try:
            # Ev sahibi takım
            home = (
                value.get("HT") or 
                value.get("HomeTeam") or 
                value.get("home") or 
                value.get("H") or
                ""
            )
            
            # Deplasman takım
            away = (
                value.get("AT") or 
                value.get("AwayTeam") or 
                value.get("away") or 
                value.get("A") or
                ""
            )
            
            # Lig ismi
            league = (
                value.get("LN") or 
                value.get("LeagueName") or 
                value.get("league") or
                value.get("L") or
                ""
            )
            
            # Maç saati
            match_time = (
                value.get("D") or 
                value.get("Date") or 
                value.get("StartDate") or
                value.get("time") or
                ""
            )
            
            # Takım isimleri boş ise atla
            if not home or not away:
                continue
            
            # String'e çevir
            home = str(home).strip()
            away = str(away).strip()
            league = str(league).strip()
            
            if not home or not away or home == "None" or away == "None":
                continue
            
            match_count += 1
            
            # Lig filtresi
            if filter_leagues and league:
                league_lower = league.lower()
                is_important = any(kw in league_lower for kw in important_keywords)
                
                if not is_important:
                    filtered_count += 1
                    continue
            
            # Saat parse et
            start_time = "??:??"
            if isinstance(match_time, str) and len(match_time) >= 16:
                start_time = match_time[11:16]
            elif isinstance(match_time, str) and ":" in match_time:
                parts = match_time.split(":")
                if len(parts) >= 2:
                    start_time = f"{parts[0]}:{parts[1]}"
            
            # Oranları çıkar
            odds = {"1": 2.00, "X": 3.20, "2": 3.50}  # Varsayılan
            
            # Farklı oran anahtarlarını dene
            odds_data = value.get("O") or value.get("Odds") or value.get("odds") or {}
            
            if isinstance(odds_data, dict):
                # MS (Maç Sonucu) oranları
                if "1" in odds_data:
                    try:
                        odds["1"] = float(odds_data["1"])
                    except:
                        pass
                
                if "X" in odds_data or "0" in odds_data:
                    try:
                        odds["X"] = float(odds_data.get("X") or odds_data.get("0"))
                    except:
                        pass
                
                if "2" in odds_data:
                    try:
                        odds["2"] = float(odds_data["2"])
                    except:
                        pass
            
            events.append({
                "home_team": home,
                "away_team": away,
                "league": league if league else "Bilinmeyen Lig",
                "start_time": start_time,
                "odds": odds
            })
            
        except Exception as e:
            continue
    
    print(f"[Nesine] Toplam maç bulundu: {match_count}")
    print(f"[Nesine] Filtrelenen maç: {filtered_count}")
    print(f"[Nesine] Geçerli maç: {len(events)}")
    
    return events


def fetch_matches_for_date(date: dt.date, filter_leagues: bool = False) -> List[Dict[str, Any]]:
    """
    Verilen tarih için maçları çek
    filter_leagues: False = TÜM MAÇLAR (varsayılan değişti)
    """
    date_str = date.strftime("%Y-%m-%d")
    url = PREBULTEN_URL.format(date=date_str)
    
    data = _get(url)
    
    if not data:
        print(f"[Nesine] {date_str} için veri alınamadı!")
        return []
    
    # Yeni formatı parse et
    events = parse_nesine_new_format(data, filter_leagues=filter_leagues)
    
    if not events:
        print("[Nesine] ⚠️ Hiç maç parse edilemedi!")
        print("[Nesine] API yanıtının ilk 500 karakteri:")
        print(str(data)[:500])
    
    return events


def fetch_today(filter_leagues: bool = False) -> List[Dict[str, Any]]:
    """
    Bugünün maçlarını çek
    filter_leagues: False = TÜM MAÇLAR (varsayılan)
    """
    today = dt.date.today()
    print(f"[Nesine] Bugünün maçları çekiliyor: {today} (Filtre: {filter_leagues})")
    return fetch_matches_for_date(today, filter_leagues=filter_leagues)


# Test
if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Nesine Fetcher Test - YENİ FORMAT")
    print("=" * 60)
    
    # Bugün
    matches = fetch_today(filter_leagues=True)
    
    if matches:
        print(f"\n✅ {len(matches)} maç bulundu\n")
        print(json.dumps(matches[:3], ensure_ascii=False, indent=2))
    else:
        print("\n❌ Maç bulunamadı!")
        print("\nYarın için deneyin:")
        tomorrow = dt.date.today() + dt.timedelta(days=1)
        matches_tomorrow = fetch_matches_for_date(tomorrow, filter_leagues=True)
        if matches_tomorrow:
            print(f"✅ Yarın {len(matches_tomorrow)} maç var")
            print(json.dumps(matches_tomorrow[:2], ensure_ascii=False, indent=2))
