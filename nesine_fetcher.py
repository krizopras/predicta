import requests
import json
from datetime import date

headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.nesine.com/",
    "Origin": "https://www.nesine.com",
}

BASE_URL = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"

def fetch_bulletin(target_date=None, filter_leagues=True):
    """Nesine bültenini JSON API'den çeker (prematch + canlı)."""
    if not target_date:
        target_date = date.today().strftime("%Y-%m-%d")
    url = f"{BASE_URL}?date={target_date}"

    print(f"📡 Nesine API çağrısı: {url}")
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"❌ API hatası: {e}")
        return []

    matches = []

    # Format 1: Yeni (Leagues -> Events)
    if "Leagues" in data:
        for league in data.get("Leagues", []):
            league_name = league.get("N", "Bilinmeyen Lig")
            for match in league.get("Events", []):
                if match.get("GT") != 1:  # sadece futbol
                    continue
                matches.append({
                    "home": match.get("HN", ""),
                    "away": match.get("AN", ""),
                    "league": league_name,
                    "date": match.get("D", ""),
                    "time": match.get("T", ""),
                    "match_id": match.get("C", ""),
                    "event_id": match.get("EV", ""),
                    "odds": [
                        o.get("O") for o in match.get("OCG", {}).get("1", {}).get("OC", [])[:3]
                        if o.get("O")
                    ]
                })

    # Format 2: Eski (sg -> EA/CA)
    elif "sg" in data:
        for section in ["EA", "CA"]:
            for m in data.get("sg", {}).get(section, []):
                if m.get("GT") != 1:  # sadece futbol
                    continue
                match_info = {
                    "home": m.get("HN", ""),
                    "away": m.get("AN", ""),
                    "league_code": m.get("LC", ""),
                    "league_id": m.get("LID", ""),
                    "date": m.get("D", ""),
                    "time": m.get("T", ""),
                    "match_id": m.get("C", ""),
                    "event_id": m.get("EV", ""),
                    "odds": []
                }
                for bahis in m.get("MA", []):
                    odds = [oca.get("O") for oca in bahis.get("OCA", []) if oca.get("O")]
                    if odds:
                        match_info["odds"].extend(odds[:3])
                matches.append(match_info)

    # Lig filtresi (isteğe bağlı)
    if filter_leagues:
        matches = [
            m for m in matches
            if any(k in m.get("league", "").lower() for k in [
                "premier", "liga", "bundesliga", "serie", "ligue",
                "süper", "eredivisie", "primeira"
            ])
        ]

    print(f"✅ {len(matches)} maç bulundu ({target_date})")
    return matches

# ====================================================
# 🧪 Test modu
# ====================================================
if __name__ == "__main__":
    matches = fetch_bulletin()
    print(f"Toplam {len(matches)} futbol maçı bulundu.")

    with open("prematch_matches.json", "w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)

    print("💾 prematch_matches.json dosyası oluşturuldu!")
