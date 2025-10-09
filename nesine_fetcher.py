import requests
import json
from datetime import date

# HTTP Headers (avoid 403)
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.nesine.com/",
    "Origin": "https://www.nesine.com",
    "Accept": "application/json, text/plain, */*",
}

# ✅ Latest Nesine API endpoint (2025)
BASE_URL = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"


def fetch_bulletin(target_date=None, filter_leagues=False):
    """
    Fetch the daily Nesine bulletin (prematch + live)
    Returns a unified match list for ML pipeline.
    """
    if not target_date:
        target_date = date.today()

    date_str = target_date.strftime("%Y-%m-%d") if isinstance(target_date, date) else str(target_date)
    url = f"{BASE_URL}?date={date_str}"

    print(f"\n{'='*60}")
    print(f"📡 NESINE API CALL")
    print(f"{'='*60}")
    print(f"URL: {url}")
    print(f"Date: {date_str}")
    print(f"Filter: {'ON' if filter_leagues else 'OFF'}")

    # --- Request and parse JSON ---
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        print(f"\n🔍 API Response Structure:")
        if isinstance(data, dict):
            print(f"   Keys: {list(data.keys())}")
            if "sg" in data:
                print(f"   SG Keys: {list(data['sg'].keys())}")
        else:
            print(f"   Unexpected type: {type(data)}")
    except requests.exceptions.Timeout:
        print(f"❌ API Timeout (15 seconds)")
        return []
    except Exception as e:
        print(f"❌ API Request Error: {e}")
        return []

    matches = []

    # === FORMAT 2: Old API (sg → EA/CA)
    if "sg" in data and isinstance(data["sg"], dict):
        print(f"\n🔄 Processing Format 2 (SG)...")

        for section in ["EA", "CA"]:  # EA = Prematch, CA = Live
            section_data = data.get("sg", {}).get(section, [])
            if not isinstance(section_data, list):
                continue
            print(f"   {section}: {len(section_data)} matches")

            for m in section_data:
                if not isinstance(m, dict):
                    continue
                if m.get("GT") != 1:  # only football
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
                    print(f"⚠️ Odds parse error: {e}")

                matches.append({
                    "home_team": m.get("HN", "").strip(),
                    "away_team": m.get("AN", "").strip(),
                    "league": m.get("LN", "Unknown League").strip(),
                    "league_code": m.get("LC", ""),
                    "league_id": m.get("LID", ""),
                    "date": m.get("D", ""),
                    "time": m.get("T", ""),
                    "match_id": m.get("C", ""),
                    "event_id": m.get("EV", ""),
                    "odds": odds_dict,
                    "live": section == "CA",
                })

        print(f"   ✅ Retrieved {len(matches)} matches from Format 2")

    else:
        print(f"\n⚠️ Unknown API format: {list(data.keys()) if isinstance(data, dict) else 'Not dict'}")
        return []

    # Remove entries without teams
    matches = [m for m in matches if m["home_team"] and m["away_team"]]

    # Optional league filter (Premier, Serie A, etc.)
    if filter_leagues:
        keywords = [
            "premier", "la liga", "bundesliga", "serie a", "ligue 1", "super lig", "championship",
            "europa", "champions", "world cup", "nations league", "cup", "kupa"
        ]
        filtered = []
        for m in matches:
            league_lower = m["league"].lower()
            if any(kw in league_lower for kw in keywords):
                filtered.append(m)
        print(f"   🔍 League Filter Active: {len(filtered)} matches kept")
        matches = filtered

    print(f"\n✅ FINAL RESULT: {len(matches)} matches found\n{'='*60}\n")
    return matches


def fetch_today(filter_leagues=False):
    """Shortcut: fetch matches for today"""
    return fetch_bulletin(date.today(), filter_leagues)


def fetch_matches_for_date(target_date, filter_leagues=False):
    """Fetch matches for a given date"""
    return fetch_bulletin(target_date, filter_leagues)


# === Manual Test ===
if __name__ == "__main__":
    print("\n🎯 NESINE FETCHER TEST MODE")
    all_matches = fetch_today(filter_leagues=False)
    print(f"\n📊 Total matches: {len(all_matches)}")

    if all_matches:
        print("\n📋 Sample Matches:")
        for i, m in enumerate(all_matches[:3], 1):
            print(f"{i}. {m['home_team']} - {m['away_team']} ({m['league']})")
            print(f"   Odds: 1={m['odds']['1']} X={m['odds']['X']} 2={m['odds']['2']}")

    output = {
        "date": str(date.today()),
        "total_matches": len(all_matches),
        "matches": all_matches
    }
    filename = f"nesine_today_{date.today().strftime('%Y%m%d')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n💾 {filename} created successfully.")
