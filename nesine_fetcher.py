import requests
import json
from datetime import date, datetime

# ====================================================
#  NESINE FETCHER 2025 - FULL BULLETIN COMPATIBLE
#  Author: Olcay Arslan | Version: 2.3
# ====================================================

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.nesine.com/",
    "Origin": "https://www.nesine.com",
    "Accept": "application/json, text/plain, */*",
}

BASE_URL = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"


def fetch_bulletin(target_date=None, filter_leagues=False):
    """
    ✅ Fetches Nesine pre-match and live bulletin (JSON API)
    Works with both Leagues and SG-based formats.
    """
    if not target_date:
        target_date = date.today()

    if isinstance(target_date, date):
        date_str = target_date.strftime("%Y-%m-%d")
    else:
        date_str = str(target_date)

    url = f"{BASE_URL}?date={date_str}"

    print("\n" + "=" * 60)
    print("📡 NESINE API CALL")
    print("=" * 60)
    print(f"URL: {url}")
    print(f"Date: {date_str}")
    print(f"Filter: {'ON' if filter_leagues else 'OFF'}")

    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.Timeout:
        print("❌ API Timeout (15s)")
        return []
    except Exception as e:
        print(f"❌ Request error: {e}")
        return []

    print("\n🔍 API Response Structure:")
    if isinstance(data, dict):
        print(f"   Keys: {list(data.keys())}")
        if "sg" in data:
            sg_keys = list(data.get("sg", {}).keys())
            print(f"   SG Keys: {sg_keys}")
    else:
        print(f"   Type: {type(data)}")
        return []

    matches = []

    # ============================================================
    # FORMAT 1: Leagues -> Events (Classic Pre-Bulletin)
    # ============================================================
    if "Leagues" in data and isinstance(data["Leagues"], list):
        print(f"\n🔄 Processing Format 1 (Leagues)...")

        for league in data.get("Leagues", []):
            league_name = league.get("N", "Unknown League")
            for match in league.get("Events", []):
                if match.get("GT") != 1:  # Only football
                    continue

                odds_dict = {"1": None, "X": None, "2": None}
                try:
                    ocg = match.get("OCG", {})
                    if "1" in ocg and "OC" in ocg["1"]:
                        oc_list = ocg["1"]["OC"]
                        if isinstance(oc_list, list) and len(oc_list) >= 3:
                            odds_dict["1"] = oc_list[0].get("O")
                            odds_dict["X"] = oc_list[1].get("O")
                            odds_dict["2"] = oc_list[2].get("O")
                except Exception as e:
                    print(f"⚠️ Odds parse error: {e}")

                matches.append({
                    "home_team": match.get("HN", "").strip(),
                    "away_team": match.get("AN", "").strip(),
                    "league": league_name.strip(),
                    "date": match.get("D", ""),
                    "time": match.get("T", ""),
                    "odds": odds_dict,
                    "event_id": match.get("EV", ""),
                })

        print(f"✅ {len(matches)} matches processed from Leagues format.")

    # ============================================================
    # FORMAT 2: SG -> EA / CA (New API structure)
    # ============================================================
    elif "sg" in data and isinstance(data["sg"], dict):
        print(f"\n🔄 Processing Format 2 (SG)...")

        for section in ["EA", "CA"]:  # EA = Prematch, CA = Live
            section_data = data["sg"].get(section, [])
            if not isinstance(section_data, list):
                print(f"⚠️ {section} is not a list (type={type(section_data).__name__}) → skipping")
                continue

            print(f"   {section}: {len(section_data)} matches")

            for m in section_data:
                if not isinstance(m, dict):
                    continue

                # only football (GT == 1)
                if m.get("GT") != 1:
                    continue

                odds_dict = {"1": None, "X": None, "2": None}
                try:
                    # Normal MA structure
                    ma_list = m.get("MA", [])
                    if isinstance(ma_list, list) and ma_list:
                        oca_list = ma_list[0].get("OCA", [])
                        if isinstance(oca_list, list) and len(oca_list) >= 3:
                            odds_dict["1"] = oca_list[0].get("O")
                            odds_dict["X"] = oca_list[1].get("O")
                            odds_dict["2"] = oca_list[2].get("O")

                    # Championship / Special markets (MSA)
                    elif "MSA" in m and isinstance(m["MSA"], list) and m["MSA"]:
                        oca_list = m["MSA"][0].get("OCA", [])
                        if isinstance(oca_list, list) and len(oca_list) >= 3:
                            odds_dict["1"] = oca_list[0].get("O")
                            odds_dict["X"] = oca_list[1].get("O")
                            odds_dict["2"] = oca_list[2].get("O")
                except Exception as e:
                    print(f"      ⚠️ Odds parse error: {e}")

                matches.append({
                    "home_team": m.get("HN", "").strip() or m.get("ON", "").strip(),
                    "away_team": m.get("AN", "").strip(),
                    "league": m.get("LN", m.get("ENN", "Unknown League")).strip(),
                    "date": m.get("D", ""),
                    "time": m.get("T", ""),
                    "odds": odds_dict,
                    "event_id": m.get("EV", ""),
                    "live": section == "CA"
                })

        print(f"✅ {len(matches)} matches processed from SG format.")

    else:
        print("⚠️ Unknown API format! Returning empty list.")
        print(f"   Data keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
        return []

    # Filter empty or invalid
    matches = [m for m in matches if m.get("home_team") and m.get("away_team")]

    print("\n" + "=" * 60)
    print(f"✅ RESULT: {len(matches)} total matches fetched.")
    print("=" * 60 + "\n")

    return matches


# ====================================================
# Helper functions
# ====================================================
def fetch_today(filter_leagues=False):
    """Fetch today’s matches"""
    return fetch_bulletin(date.today(), filter_leagues)


def fetch_matches_for_date(target_date, filter_leagues=False):
    """Fetch matches for a given date"""
    return fetch_bulletin(target_date, filter_leagues)


# ====================================================
# Test Mode
# ====================================================
if __name__ == "__main__":
    print("\n🎯 NESINE FETCHER TEST MODE")
    print("=" * 60)

    today_matches = fetch_today(filter_leagues=False)
    print(f"📊 Found {len(today_matches)} matches")

    if today_matches:
        print("\n📋 First 3 matches:")
        for i, m in enumerate(today_matches[:3], 1):
            print(f"{i}. {m['home_team']} vs {m['away_team']} ({m['league']}) | {m['time']}")
            print(f"   Odds: 1={m['odds']['1']} X={m['odds']['X']} 2={m['odds']['2']}")
