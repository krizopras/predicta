import requests
import json
from datetime import date, datetime

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.nesine.com/",
    "Origin": "https://www.nesine.com",
    "Accept": "application/json, text/plain, */*",
}

BASE_URL = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"

def fetch_bulletin(target_date=None, filter_leagues=False):
    """
    Nesine API'den maç bilgilerini çeker.
    
    Args:
        target_date: date veya YYYY-MM-DD formatında string
        filter_leagues: True ise sadece büyük ligleri döndürür
    
    Returns:
        list: Maç listesi
    """
    if not target_date:
        target_date = date.today()
    
    if isinstance(target_date, date):
        date_str = target_date.strftime("%Y-%m-%d")
    else:
        date_str = str(target_date)
    
    url = f"{BASE_URL}?date={date_str}"
    
    print(f"\n{'='*60}")
    print(f"📡 NESINE API ÇAĞRISI")
    print(f"{'='*60}")
    print(f"URL: {url}")
    print(f"Tarih: {date_str}")
    print(f"Filtre: {'AÇIK' if filter_leagues else 'KAPALI'}")
    
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        
        print(f"\n🔍 API Yanıt Yapısı:")
        if isinstance(data, dict):
            print(f"   Ana Anahtarlar: {list(data.keys())}")
        
    except requests.exceptions.Timeout:
        print(f"❌ API Zaman Aşımı (15 saniye)")
        return []
    except requests.exceptions.RequestException as e:
        print(f"❌ API İstek Hatası: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ JSON Ayrıştırma Hatası: {e}")
        return []
    except Exception as e:
        print(f"❌ Bilinmeyen Hata: {e}")
        return []

    matches = []
    
    # YENİ FORMAT: sg -> EA/CA yapısı
    if "sg" in data and isinstance(data["sg"], dict):
        print(f"\n📄 SG Formatı İşleniyor...")
        
        sg_data = data["sg"]
        
        # EA = Maç Öncesi, CA = Canlı
        for section in ["EA", "CA"]:
            if section not in sg_data:
                continue
                
            section_matches = sg_data[section]
            if not isinstance(section_matches, list):
                continue
            
            print(f"   {section}: {len(section_matches)} maç")
            
            for m in section_matches:
                # Sadece futbol maçları (GT=1)
                if m.get("GT") != 1:
                    continue
                
                home_team = m.get("HN", "").strip()
                away_team = m.get("AN", "").strip()
                
                # Boş takım isimlerini atla
                if not home_team or not away_team:
                    continue
                
                # Lig bilgisi
                league = m.get("LN", "Bilinmeyen Lig").strip()
                
                # Oranları çek
                odds_dict = {"1": None, "X": None, "2": None}
                try:
                    ma_list = m.get("MA", [])
                    if ma_list and len(ma_list) > 0:
                        # İlk market (genellikle 1X2)
                        market = ma_list[0]
                        oca_list = market.get("OCA", [])
                        
                        # 1X2 oranları
                        if len(oca_list) >= 3:
                            odds_dict["1"] = oca_list[0].get("O")
                            odds_dict["X"] = oca_list[1].get("O")
                            odds_dict["2"] = oca_list[2].get("O")
                except Exception as e:
                    print(f"   ⚠️ Oran çekme hatası: {home_team} - {away_team} ({e})")
                
                # Tarih ve saat
                match_time = m.get("T", "")
                match_date = m.get("D", "")
                
                match_info = {
                    "home_team": home_team,
                    "away_team": away_team,
                    "league": league,
                    "league_code": m.get("LC", ""),
                    "league_id": m.get("LID", ""),
                    "date": match_date,
                    "time": match_time,
                    "start_time": f"{match_date} {match_time}",
                    "match_id": m.get("C", ""),
                    "event_id": m.get("EV", ""),
                    "odds": odds_dict,
                    "live": section == "CA"
                }
                
                matches.append(match_info)
        
        print(f"   ✅ Toplam {len(matches)} maç bulundu")
    
    # ESKİ FORMAT: Leagues yapısı (yedek)
    elif "Leagues" in data and isinstance(data["Leagues"], list):
        print(f"\n📄 Leagues Formatı İşleniyor...")
        
        for league in data.get("Leagues", []):
            league_name = league.get("N", "Bilinmeyen Lig")
            league_id = league.get("I", "")
            
            for match in league.get("Events", []):
                if match.get("GT") != 1:
                    continue
                
                home_team = match.get("HN", "").strip()
                away_team = match.get("AN", "").strip()
                
                if not home_team or not away_team:
                    continue
                
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
                
                match_time = match.get("T", "")
                match_date = match.get("D", "")
                
                matches.append({
                    "home_team": home_team,
                    "away_team": away_team,
                    "league": league_name.strip(),
                    "league_id": league_id,
                    "date": match_date,
                    "time": match_time,
                    "start_time": f"{match_date} {match_time}",
                    "match_id": match.get("C", ""),
                    "event_id": match.get("EV", ""),
                    "odds": odds_dict
                })
        
        print(f"   ✅ Toplam {len(matches)} maç bulundu")
    
    else:
        print(f"\n⚠️ Bilinmeyen API formatı!")
        print(f"   Data anahtarları: {list(data.keys()) if isinstance(data, dict) else 'DİZİN DEĞİL'}")
        return []
    
    # Lig filtresi
    if filter_leagues and matches:
        print(f"\n🔍 Lig filtresi uygulanıyor...")
        
        keywords = [
            "premier", "championship", "league one", "league two",
            "la liga", "laliga", "segunda",
            "bundesliga", "2. bundesliga",
            "serie a", "serie b",
            "ligue 1", "ligue 2",
            "süper lig", "super lig", "1. lig",
            "eredivisie",
            "primeira liga",
            "pro league",
            "champions league", "uefa", "europa",
            "world cup", "euro", "nations league",
            "cup", "kupa", "kupası"
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
    """Bugünün maçlarını çek."""
    return fetch_bulletin(date.today(), filter_leagues)


def fetch_matches_for_date(target_date, filter_leagues=False):
    """Belirli bir tarihteki maçları çek."""
    return fetch_bulletin(target_date, filter_leagues)


# ====================================================
# 🧪 Test Modu
# ====================================================
if __name__ == "__main__":
    print("\n🎯 NESINE FETCHER TEST")
    print("="*60)
    
    # Test 1: Bugünün tüm maçları
    print("\n📅 TEST 1: TÜM MAÇLAR (filtre KAPALI)")
    all_matches = fetch_today(filter_leagues=False)
    print(f"\n📊 Sonuç: {len(all_matches)} maç bulundu")
    
    if all_matches:
        print("\n📋 İlk 5 maç:")
        for i, m in enumerate(all_matches[:5], 1):
            print(f"\n{i}. {m['home_team']} - {m['away_team']}")
            print(f"   Lig: {m['league']}")
            print(f"   Saat: {m['time']}")
            print(f"   Oranlar: 1={m['odds']['1']} X={m['odds']['X']} 2={m['odds']['2']}")
            print(f"   Canlı: {'Evet' if m.get('live') else 'Hayır'}")
    
    # Test 2: Büyük ligler
    print("\n" + "="*60)
    print("\n📅 TEST 2: BÜYÜK LİGLER (filtre AÇIK)")
    filtered_matches = fetch_today(filter_leagues=True)
    print(f"\n📊 Sonuç: {len(filtered_matches)} maç bulundu")
    
    # Ligleri grupla
    if filtered_matches:
        leagues = {}
        for m in filtered_matches:
            lg = m.get("league", "Bilinmeyen")
            leagues[lg] = leagues.get(lg, 0) + 1
        
        print(f"\n📊 Lig Dağılımı:")
        for lg, count in sorted(leagues.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {lg}: {count} maç")
    
    # JSON'a kaydet
    output = {
        "tarih": date.today().strftime("%Y-%m-%d"),
        "toplam_mac": len(all_matches),
        "filtrelenmis_mac": len(filtered_matches),
        "maclar": all_matches[:50]  # İlk 50 maç
    }
    
    filename = f"nesine_maclar_{date.today().strftime('%Y%m%d')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 {filename} dosyası oluşturuldu!")
    print("\n✅ Test tamamlandı!")
