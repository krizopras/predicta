import requests
import json
from datetime import date, datetime

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.nesine.com/",
    "Origin": "https://www.nesine.com",
    "Accept": "application/json, text/plain, */*",
}

# UPDATED API ENDPOINT (2025 version)
BASE_URL = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"

def fetch_bulletin(target_date=None, filter_leagues=False):
    """
    Fetches the Nesine bulletin from the official JSON API (prematch + live).

    Args:
        target_date: date or string in YYYY-MM-DD format
        filter_leagues: If True, returns only major leagues (False = all matches)

    Returns:
        list: List of matches
    """
    if not target_date:
        target_date = date.today()
    
    # Normalize date
    if isinstance(target_date, date):
        date_str = target_date.strftime("%Y-%m-%d")
    else:
        date_str = str(target_date)
    
    url = f"{BASE_URL}?date={date_str}"
    
    print(f"\n{'='*60}")
    print(f"📡 NESINE API CALL")
    print(f"{'='*60}")
    print(f"URL: {url}")
    print(f"Date: {date_str}")
    print(f"Filter: {'ON' if filter_leagues else 'OFF'}")
    
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        
        # Debug API response
        print(f"\n🔍 API Response Structure:")
        if isinstance(data, dict):
            print(f"   Keys: {list(data.keys())}")
            if "Leagues" in data:
                print(f"   Leagues Count: {len(data.get('Leagues', []))}")
            if "sg" in data:
                sg_keys = list(data.get("sg", {}).keys())
                print(f"   SG Keys: {sg_keys}")
                for key in sg_keys:
                    count = len(data["sg"].get(key, []))
                    print(f"      {key}: {count} matches")
        else:
            print(f"   Type: {type(data)}")
        
    except requests.exceptions.Timeout:
        print(f"❌ API Timeout (15 seconds)")
        return []
    except requests.exceptions.RequestException as e:
        print(f"❌ API Request Error: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ JSON Parsing Error: {e}")
        return []
    except Exception as e:
        print(f"❌ Unknown Error: {e}")
        return []

    matches = []
    
    # FORMAT 1: New API structure (Leagues -> Events)
    if "Leagues" in data and isinstance(data["Leagues"], list):
        print(f"\n🔄 Processing Format 1 (Leagues)...")
        
        for league in data.get("Leagues", []):
            league_name = league.get("N", "Unknown League")
            league_id = league.get("I", "")
            
            for match in league.get("Events", []):
                # Only football (GT=1)
                if match.get("GT") != 1:
                    continue
                
                # Extract odds
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
        
        print(f"   ✅ {len(matches)} matches retrieved from Format 1")
    
    # FORMAT 2: Old API structure (sg -> EA/CA)
    elif "sg" in data and isinstance(data["sg"], dict):
        print(f"\n🔄 Processing Format 2 (SG)...")
        
        for section in ["EA", "CA"]:  # EA = Prematch, CA = Live
            section_matches = data.get("sg", {}).get(section, [])
            print(f"   {section}: {len(section_matches)} matches")
            
            for m in section_matches:
                if m.get("GT") != 1:
                    continue
                
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
                    "league": m.get("LN", "Unknown League").strip(),
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
        
        print(f"   ✅ {len(matches)} matches retrieved from Format 2")
    
    else:
        print(f"\n⚠️ Unknown API format!")
        print(f"   Data keys: {list(data.keys()) if isinstance(data, dict) else 'NOT_DICT'}")
        return []
    
    # Filter out incomplete data
    matches = [m for m in matches if m["home_team"] and m["away_team"]]
    
    # Optional league filtering
    if filter_leagues:
        print(f"\n🔍 League filtering enabled...")
        
        all_leagues = set(m["league"].lower() for m in matches if m["league"])
        print(f"   Total {len(all_leagues)} leagues found")
        print(f"   Example leagues: {sorted(list(all_leagues))[:10]}")
        
        # Major leagues (extended list)
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
            "cup", "kupa"
        ]
        
        filtered = []
        for m in matches:
            league_lower = m["league"].lower()
            if any(kw in league_lower for kw in keywords):
                filtered.append(m)
        
        print(f"   ✅ Filtered: {len(filtered)} matches")
        matches = filtered
    
    print(f"\n{'='*60}")
    print(f"✅ RESULT: {len(matches)} matches found")
    print(f"{'='*60}\n")
    
    return matches


def fetch_today(filter_leagues=False):
    """Fetch today's matches."""
    return fetch_bulletin(date.today(), filter_leagues)


def fetch_matches_for_date(target_date, filter_leagues=False):
    """Fetch matches for a specific date."""
    return fetch_bulletin(target_date, filter_leagues)


# ====================================================
# 🧪 Test Mode
# ====================================================
if __name__ == "__main__":
    print("\n🎯 NESINE FETCHER TEST")
    print("="*60)
    
    # Test 1: All matches for today
    print("\n📅 TEST 1: ALL matches (filter OFF)")
    all_matches = fetch_today(filter_leagues=False)
    print(f"\n📊 Result: {len(all_matches)} matches found")
    
    if all_matches:
        print("\n📋 First 3 matches:")
        for i, m in enumerate(all_matches[:3], 1):
            print(f"\n{i}. {m['home_team']} - {m['away_team']}")
            print(f"   League: {m['league']}")
            print(f"   Time: {m['time']}")
            print(f"   Odds: 1={m['odds']['1']} X={m['odds']['X']} 2={m['odds']['2']}")
    
    # Test 2: Filtered major leagues
    print("\n" + "="*60)
    print("\n📅 TEST 2: MAJOR LEAGUES ONLY (filter ON)")
    filtered_matches = fetch_today(filter_leagues=True)
    print(f"\n📊 Result: {len(filtered_matches)} matches found")
    
    # Save to JSON
    output = {
        "date": date.today().strftime("%Y-%m-%d"),
        "total_matches": len(all_matches),
        "filtered_matches": len(filtered_matches),
        "matches": all_matches
    }
    
    filename = f"nesine_matches_{date.today().strftime('%Y%m%d')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 {filename} file created!")
    print("\n✅ Test completed!")
