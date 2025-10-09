#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nesine Bulletin Fetcher
Fully compatible with new CDN API structure (as of 2025)
Author: Olcay Arslan
"""

import requests
from datetime import date

def fetch_bulletin(target_date=None, filter_leagues=False):
    """
    Fetch full pre-bulletin data from Nesine CDN.
    :param target_date: date object or None (defaults to today)
    :param filter_leagues: bool - whether to apply filtering by leagues
    :return: list of parsed matches
    """
    if target_date is None:
        target_date = date.today()
    date_str = target_date.strftime("%Y-%m-%d")

    url = f"https://cdnbulten.nesine.com/api/bulten/getprebultenfull?date={date_str}"
    print("============================================================")
    print("📡 NESINE API CALL")
    print("============================================================")
    print(f"URL: {url}")
    print(f"Date: {date_str}")
    print(f"Filter: {'ON' if filter_leagues else 'OFF'}")

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"❌ API connection or parse error: {e}")
        return []

    matches = []

    # --- FORMAT 1: New API structure (sg -> EA -> MSA)
    if "sg" in data and isinstance(data["sg"], dict):
        sg_data = data["sg"]
        sections = sg_data.get("EA", [])
        print(f"🔍 API Response Structure:")
        print(f"   Keys: {list(data.keys())}")
        print(f"   SG Keys: {list(sg_data.keys())}")

        for section in sections:
            try:
                # Some events (champion bets, futures) have TYPE != 1 — skip them
                if section.get("TYPE", 1) not in [0, 1]:
                    continue

                league = section.get("ENN", "").strip() or "Unknown League"
                home = section.get("HN", "").strip()
                away = section.get("AN", "").strip()
                match_date = section.get("D", date_str)
                match_time = section.get("T", "")

                # odds
                odds_dict = {"1": None, "X": None, "2": None}
                market_list = section.get("MA", []) or section.get("MSA", [])
                if market_list:
                    first_market = market_list[0]
                    oca_list = first_market.get("OCA", [])
                    if len(oca_list) >= 3:
                        odds_dict["1"] = oca_list[0].get("O")
                        odds_dict["X"] = oca_list[1].get("O")
                        odds_dict["2"] = oca_list[2].get("O")

                matches.append({
                    "league": league,
                    "home_team": home,
                    "away_team": away,
                    "date": match_date,
                    "time": match_time,
                    "odds": odds_dict
                })
            except Exception as e:
                print(f"⚠️ Parse error in EA section: {e}")
                continue

        print(f"✅ Parsed {len(matches)} matches (Format 1: EA).")

    else:
        print("⚠️ Unknown or unsupported API format detected.")
        print(f"   Data keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
        return []

    if not matches:
        print("⚠️ No matches found in bulletin.")
    return matches


def fetch_today(filter_leagues=False):
    """Shortcut for today's matches"""
    return fetch_bulletin(date.today(), filter_leagues)


def fetch_matches_for_date(target_date):
    """Fetch matches for a specific date"""
    return fetch_bulletin(target_date, filter_leagues=False)


# --- Test Run ---
if __name__ == "__main__":
    print("🧪 Testing Nesine Fetcher...")
    matches = fetch_today()
    print(f"✅ Total matches fetched: {len(matches)}")
    for m in matches[:5]:
        print(f"🏆 {m['league']}: {m['home_team']} vs {m['away_team']} | Odds: {m['odds']}")
