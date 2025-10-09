#!/usr/bin/env python3
"""
Nesine Bulletin Fetcher v3
Full bulletin retriever for Predicta Europe ML
Compatible with both legacy (sg -> EA/CA) and new (FB) structures.
"""

import requests
import json
from datetime import date
import traceback

# =============================
# Main Fetch Function
# =============================

def fetch_bulletin(fetch_date=None, filter_leagues=False):
    """
    Fetches the full Nesine bulletin for a given date.
    Works with all known API structures (FB, EA/CA, etc.)
    """
    if fetch_date is None:
        fetch_date = date.today()

    url = f"https://cdnbulten.nesine.com/api/bulten/getprebultenfull?date={fetch_date}"
    print("=" * 60)
    print("📡 NESINE API CALL")
    print("=" * 60)
    print(f"URL: {url}")
    print(f"Date: {fetch_date}")
    print(f"Filter: {'ON' if filter_leagues else 'OFF'}")

    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            print(f"❌ HTTP Error: {resp.status_code}")
            return []

        data = resp.json()
        matches = []

        # Inspect structure
        print("🔍 API Response Structure:")
        if isinstance(data, dict):
            print(f"   Keys: {list(data.keys())}")
            if "sg" in data:
                print(f"   SG Keys: {list(data['sg'].keys())}")
        else:
            print("   Response is not a dictionary!")
            return []

        # ======================================================
        # FORMAT 1: New API (FB) - usually contains prematch data
        # ======================================================
        if "FB" in data and isinstance(data["FB"], list):
            print("\n🔄 Processing Format 1 (FB)...")
            for m in data["FB"]:
                if m.get("GT") != 1:  # Only football
                    continue

                odds_dict = {"1": None, "X": None, "2": None}
                try:
                    ma_list = m.get("MA", [])
                    if isinstance(ma_list, list) and len(ma_list) > 0:
                        oca_list = ma_list[0].get("OCA", [])
                        if isinstance(oca_list, list) and len(oca_list) >= 3:
                            odds_dict["1"] = oca_list[0].get("O")
                            odds_dict["X"] = oca_list[1].get("O")
                            odds_dict["2"] = oca_list[2].get("O")
                except Exception as e:
                    print(f"⚠️ Odds parse error (FB): {e}")

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
                    "live": False
                })

            print(f"   ✅ {len(matches)} matches retrieved from Format 1 (FB)")

        # ======================================================
        # FORMAT 2: Old API (sg -> EA / CA)
        # ======================================================
        elif "sg" in data and isinstance(data["sg"], dict):
            print(f"\n🔄 Processing Format 2 (SG)...")

            sg_data = data.get("sg", {})
            valid_sections = [k for k, v in sg_data.items() if isinstance(v, list)]

            print(f"   Detected sections: {list(sg_data.keys())}")
            print(f"   ✅ Valid sections: {valid_sections}")

            for section in valid_sections:
                section_matches = sg_data.get(section, [])
                print(f"   {section}: {len(section_matches)} matches")

                for m in section_matches:
                    if m.get("GT") != 1:
                        continue

                    odds_dict = {"1": None, "X": None, "2": None}
                    try:
                        ma_list = m.get("MA", [])
                        if isinstance(ma_list, list) and len(ma_list) > 0:
                            oca_list = ma_list[0].get("OCA", [])
                            if isinstance(oca_list, list) and len(oca_list) >= 3:
                                odds_dict["1"] = oca_list[0].get("O")
                                odds_dict["X"] = oca_list[1].get("O")
                                odds_dict["2"] = oca_list[2].get("O")
                    except Exception as e:
                        print(f"      ⚠️ Odds parse error in {section}: {e}")

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
                        "live": section == "CA"
                    })

            print(f"   ✅ {len(matches)} total matches successfully parsed (SG format)")

        else:
            print(f"\n⚠️ Unknown API format!")
            print(f"   Data keys: {list(data.keys()) if isinstance(data, dict) else 'NOT_DICT'}")
            return []

        print("=" * 60)
        print(f"✅ RESULT: {len(matches)} matches found")
        print("=" * 60)

        return matches

    except Exception as e:
        print("❌ Exception during fetch:")
        print(traceback.format_exc())
        return []

# =============================
# Convenience Wrappers
# =============================

def fetch_today(filter_leagues=False):
    """Shortcut for today's bulletin"""
    return fetch_bulletin(date.today(), filter_leagues=filter_leagues)


def fetch_matches_for_date(date_obj):
    """Fetch bulletin for a specific date object"""
    return fetch_bulletin(date_obj, filter_leagues=False)


# =============================
# Manual Test
# =============================

if __name__ == "__main__":
    print("🧪 Running manual test for today's matches...")
    data = fetch_today(filter_leagues=False)
    print(f"\nTotal Matches: {len(data)}")
    if data:
        print(f"Sample:\n{json.dumps(data[:3], indent=2, ensure_ascii=False)}")
