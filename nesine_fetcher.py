import requests, json
from datetime import date

BASE_URL = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.nesine.com/",
    "Origin": "https://www.nesine.com",
}

def fetch_bulletin(target_date=None, filter_leagues=True):
    """Nesine bültenini JSON API'den çeker."""
    if not target_date:
        target_date = date.today().strftime("%Y-%m-%d")
    url = f"{BASE_URL}?date={target_date}"

    print(f"📡 Nesine API çağrısı: {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"❌ API hatası: {e}")
        return []

    matches = []
    if "Leagues" in data:
        for league in data.get("Leagues", []):
            lname = league.get("N", "Bilinmeyen Lig")
            for m in league.get("Events", []):
                if m.get("GT") != 1:  # sadece futbol
                    continue
                odds = []
                ocg = m.get("OCG", {}).get("1", {}).get("OC", [])
                for o in ocg[:3]:
                    if o.get("O"):
                        odds.append(o.get("O"))
                matches.append({
                    "home_team": m.get("HN", ""),
                    "away_team": m.get("AN", ""),
                    "league": lname,
                    "date": m.get("D", ""),
                    "time": m.get("T", ""),
                    "match_id": m.get("C", ""),
                    "event_id": m.get("EV", ""),
                    "odds": {"1": float(odds[0]) if len(odds)>0 else 0,
                             "X": float(odds[1]) if len(odds)>1 else 0,
                             "2": float(odds[2]) if len(odds)>2 else 0}
                })

    # Lig filtresi (isteğe bağlı)
    if filter_leagues:
        matches = [m for m in matches if any(k in m["league"].lower() for k in [
            "premier","bundes","liga","serie","ligue","super","eredivisie","primeira"
        ])]

    print(f"✅ {len(matches)} maç bulundu ({target_date})")
    return matches

# Eski uyumluluk için alias fonksiyonlar:
def fetch_today(filter_leagues=True):
    """Bugünkü maçları döndürür (fetch_bulletin alias'ı)."""
    return fetch_bulletin(date.today().strftime("%Y-%m-%d"), filter_leagues=filter_leagues)

def fetch_matches_for_date(target_date):
    """Belirli tarih için Nesine maçları döndürür."""
    return fetch_bulletin(target_date.strftime("%Y-%m-%d"))
