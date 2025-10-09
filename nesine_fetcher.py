import requests
import json
from datetime import date

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.nesine.com/",
    "Origin": "https://www.nesine.com",
    "Accept": "application/json, text/plain, */*",
}

BASE_URL = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"

def fetch_bulletin(target_date=None, filter_leagues=False):
    if not target_date:
        target_date = date.today()

    if isinstance(target_date, date):
        date_str = target_date.strftime("%Y-%m-%d")
    else:
        date_str = str(target_date)

    url = f"{BASE_URL}?date={date_str}"
    print(f"\n{'='*60}\n📡 NESINE API CALL\n{'='*60}")
    print(f"URL: {url}\nDate: {date_str}\nFilter: {'ON' if filter_leagues else 'OFF'}")

    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        print(f"🔍 API Response Structure:")
        if isinstance(data, dict):
            print(f"   Keys: {list(data.keys())}")
            if "sg" in data:
                sg_keys = list(data.get("sg", {}).keys())
                print(f"   SG Keys: {sg_keys}")
        else:
            print(f"   Type: {type(data)}")
    except Exception as e:
        print(f"❌ API Error: {e}")
        return []

    matches = []

    # === Format 2: Old API (sg → EA/CA)
    if "sg" in data and isinstance(data["sg"], dict):
        print("\n🔄 Processing Format 2 (SG)...")
        for section in ["EA", "CA"]:
            section_matches = data.get("sg", {}).get(section, [])
            if not isinstance(section_matches, list):
                print(f"⚠️ {section} section is not a list, skipping.")
                continue
            print(f"   {section}: {len(section_matches)} matches")

            for m in section_matches:
                if not isinstance(m, dict):
                    continue
                if m.get("GT") != 1:
                    continue

                odds_dict = {"1": None, "X": None, "2": None}
                try:
                    ma_list = m.get("MA", [])
                    if isinstance(ma_list, list) and ma_list:
                        oca_list = ma_list[0].get("OCA", [])
                        if isinstance(oca_list, list) and len(oca_list) >= 3:
                            odds_dict["1"] = oca_list[0].get("O")
                            odds_dict["X"] = oca_list[1].get("O")
                            odds_dict["2"] = oca_list[2].get("O")
                except Exception as e:
                    print(f"   ⚠️ Odds parse error: {e}")
                    continue

                matches.append({
                    "home_team": m.get("HN", "").strip(),
                    "away_team": m.get("AN", "").strip(),
                    "league": m.get("LN", "Unknown League").strip(),
                    "date": m.get("D", ""),
                    "time": m.get("T", ""),
                    "match_id": m.get("C", ""),
                    "odds": odds_dict,
                    "live": section == "CA",
                })
        print(f"   ✅ Retrieved {len(matches)} matches from Format 2")
    else:
        print("\n⚠️ Unknown API format!")
        return []

    # Filter empty matches
    matches = [m for m in matches if m["home_team"] and m["away_team"]]
    print(f"✅ Total matches found: {len(matches)}")
    return matches
