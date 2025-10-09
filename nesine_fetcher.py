import requests
import json
from datetime import date, datetime

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.nesine.com/",
    "Origin": "https://www.nesine.com",
    "Accept": "application/json, text/plain, */*",
}

# YENİ API ENDPOINT (2025 güncel)
BASE_URL = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"

def fetch_bulletin(target_date=None, filter_leagues=False):
    """
    Nesine bültenini JSON API'den çeker (prematch + canlı).
    
    Args:
        target_date: Tarih (YYYY-MM-DD formatında string veya date objesi)
        filter_leagues: Sadece ana ligleri getir (False = tüm maçlar)
    
    Returns:
        list: Maç listesi
    """
    if not target_date:
        target_date = date.today()
    
    # Tarih formatını düzelt
    if isinstance(target_date, date):
        date_str = target_date.strftime("%Y-%m-%d")
    else:
        date_str = str(target_date)
    
    url = f"{BASE_URL}?date={date_str}"
    
    print(f"\n{'='*60}")
    print(f"📡 Nesine API Çağrısı")
    print(f"{'='*60}")
    print(f"URL: {url}")
    print(f"Tarih: {date_str}")
    print(f"Filtre: {'AÇIK' if filter_leagues else 'KAPALI'}")
    
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        
        # API yanıtını debug et
        print(f"\n🔍 API Yanıt Yapısı:")
        if isinstance(data, dict):
            print(f"   Keys: {list(data.keys())}")
            if "Leagues" in data:
                print(f"   Leagues Count: {len(data.get('Leagues', []))}")
            if "sg" in data:
                sg_keys = list(data.get("sg", {}).keys())
                print(f"   SG Keys: {sg_keys}")
                for key in sg_keys:
                    count = len(data["sg"].get(key, []))
                    print(f"      {key}: {count} maç")
        else:
            print(f"   Type: {type(data)}")
        
    except requests.exceptions.Timeout:
        print(f"❌ API Timeout (15 saniye)")
        return []
    except requests.exceptions.RequestException as e:
        print(f"❌ API İstek Hatası: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ JSON Parse Hatası: {e}")
        return []
    except Exception as e:
        print(f"❌ Bilinmeyen Hata: {e}")
        return []

    matches = []
    
    # FORMAT 1: Yeni API Yapısı (Leagues -> Events)
    if "Leagues" in data and isinstance(data["Leagues"], list):
        print(f"\n🔄 Format 1 (Leagues) işleniyor...")
        
        for league in data.get("Leagues", []):
            league_name = league.get("N", "Bilinmeyen Lig")
            league_id = league.get("I", "")
            
            for match in league.get("Events", []):
                # Sadece futbol (GT=1)
                if match.get("GT") != 1:
                    continue
                
                # Oran bilgisini çek
                odds_dict = {"1": None, "X": None, "2": None}
                try:
                    ocg = match.get("OCG", {})
                    if "1" in ocg and "OC" in ocg["1"]:
                        oc_list = ocg["1"]["OC"]
                        if len(oc_list) >= 3:
                            odds_dict["1"] = oc_list[0].get("O")
                            odds_dict["X"] = oc_list[1].get("O")
                            odds_dict["2"] = oc_list[2].get("O")
                except:
                    pass
                
                # Maç saati
                match_time = match.get("T", "")
                match_date = match.get("D", "")
                
                matches.append({
                    "home_team": match.get("HN", "").strip(),
                    "away_team": match.get("AN", "").strip(),
                    "league": league_name.strip(),
                    "league_id": league_id,
                    "date": match_date,
                    "time": match_time,
                    "start_time": match_time,
                    "match_id": match.get("C", ""),
                    "event_id": match.get("EV", ""),
                    "odds": odds_dict
                })
        
        print(f"   ✅ Format 1'den {len(matches)} maç çekildi")
    
    # FORMAT 2: Eski API Yapısı (sg -> EA/CA)
    elif "sg" in data and isinstance(data["sg"], dict):
        print(f"\n🔄 Format 2 (SG) işleniyor...")
        
        for section in ["EA", "CA"]:  # EA=Prematch, CA=Canlı
            section_matches = data.get("sg", {}).get(section, [])
            print(f"   {section}: {len(section_matches)} maç")
            
            for m in section_matches:
                # Sadece futbol (GT=1)
                if m.get("GT") != 1:
                    continue
                
                # Oran bilgisini çek
                odds_dict = {"1": None, "X": None, "2": None}
                try:
                    ma_list = m.get("MA", [])
                    if ma_list:
                        oca_list = ma_list[0].get("OCA", [])
                        if len(oca_list) >= 3:
                            odds_dict["1"] = oca_list[0].get("O")
                            odds_dict["X"] = oca_list[1].get("O")
                            odds_dict["2"] = oca_list[2].get("O")
                except:
                    pass
                
                match_time = m.get("T", "")
                match_date = m.get("D", "")
                
                matches.append({
                    "home_team": m.get("HN", "").strip(),
                    "away_team": m.get("AN", "").strip(),
                    "league": m.get("LN", "Bilinmeyen Lig").strip(),
                    "league_code": m.get("LC", ""),
                    "league_id": m.get("LID", ""),
                    "date": match_date,
                    "time": match_time,
                    "start_time": match_time,
                    "match_id": m.get("C", ""),
                    "event_id": m.get("EV", ""),
                    "odds": odds_dict,
                    "live": section == "CA"
                })
        
        print(f"   ✅ Format 2'den {len(matches)} maç çekildi")
    
    else:
        print(f"\n⚠️ Bilinmeyen API formatı!")
        print(f"   Data keys: {list(data.keys()) if isinstance(data, dict) else 'NOT_DICT'}")
        return []
    
    # Boş takım isimlerini filtrele
    matches = [m for m in matches if m["home_team"] and m["away_team"]]
    
    # Lig filtreleme (opsiyonel)
    if filter_leagues:
        print(f"\n🔍 Lig filtreleme aktif...")
        
        # Tüm lig isimlerini topla (debug için)
        all_leagues = set(m["league"].lower() for m in matches if m["league"])
        print(f"   Toplam {len(all_leagues)} farklı lig var")
        print(f"   Örnek ligler: {sorted(list(all_leagues))[:10]}")
        
        # Ana ligler (Genişletilmiş liste)
        keywords = [
            "premier", "championship", "league one", "league two",
            "la liga", "laliga", "segunda",
            "bundesliga", "2. bundesliga",
            "serie a", "serie b",
            "ligue 1", "ligue 2",
            "süper lig", "süper", "1. lig",
            "eredivisie",
            "primeira liga",
            "pro league",
            "champions league", "uefa", "europa",
            "world cup", "euro", "nations league",
            "cup", "kupa"
        ]
        
        filtered = []
        for m in matches:
            league_lower = m["league"].lower()
            if any(kw in league_lower for kw in keywords):
                filtered.append(m)
        
        print(f"   ✅ Filtrelenmiş: {len(filtered)} maç")
        matches = filtered
    
    print(f"\n{'='*60}")
    print(f"✅ SONUÇ: {len(matches)} maç bulundu")
    print(f"{'='*60}\n")
    
    return matches


def fetch_today(filter_leagues=False):
    """Bugünkü maçları çek"""
    return fetch_bulletin(date.today(), filter_leagues)


def fetch_matches_for_date(target_date, filter_leagues=False):
    """Belirli bir tarihteki maçları çek"""
    return fetch_bulletin(target_date, filter_leagues)


# ====================================================
# 🧪 Test Modu
# ====================================================
if __name__ == "__main__":
    print("\n🎯 NESINE FETCHER TEST")
    print("="*60)
    
    # Test 1: Bugünkü tüm maçlar
    print("\n📅 TEST 1: Bugünkü TÜM maçlar (filtre KAPALI)")
    all_matches = fetch_today(filter_leagues=False)
    print(f"\n📊 Sonuç: {len(all_matches)} maç bulundu")
    
    if all_matches:
        print("\n📋 İlk 3 maç:")
        for i, m in enumerate(all_matches[:3], 1):
            print(f"\n{i}. {m['home_team']} - {m['away_team']}")
            print(f"   Lig: {m['league']}")
            print(f"   Saat: {m['time']}")
            print(f"   Oranlar: 1={m['odds']['1']} X={m['odds']['X']} 2={m['odds']['2']}")
    
    # Test 2: Filtrelenmiş maçlar
    print("\n" + "="*60)
    print("\n📅 TEST 2: Bugünkü ANA LİG maçları (filtre AÇIK)")
    filtered_matches = fetch_today(filter_leagues=True)
    print(f"\n📊 Sonuç: {len(filtered_matches)} maç bulundu")
    
    # JSON'a kaydet
    output = {
        "date": date.today().strftime("%Y-%m-%d"),
        "total_matches": len(all_matches),
        "filtered_matches": len(filtered_matches),
        "matches": all_matches
    }
    
    filename = f"nesine_matches_{date.today().strftime('%Y%m%d')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 {filename} dosyası oluşturuldu!")
    print("\n✅ Test tamamlandı!")
