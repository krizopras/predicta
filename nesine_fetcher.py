import requests
import json
from datetime import date, datetime

BASE_API_URL = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"

def fetch_today(filter_leagues: bool = True):
    """Bugünkü maçları Nesine JSON API'sinden çeker."""
    today = date.today().strftime("%Y-%m-%d")
    return _fetch_from_api(today, filter_leagues)

def fetch_matches_for_date(target_date, filter_leagues: bool = True):
    """Belirli bir tarihteki maçları Nesine JSON API'sinden çeker."""
    if isinstance(target_date, (date, datetime)):
        target_date = target_date.strftime("%Y-%m-%d")
    return _fetch_from_api(target_date, filter_leagues)

def _fetch_from_api(target_date: str, filter_leagues: bool):
    """Nesine’nin JSON API'sinden maç listesini getirir."""
    url = f"{BASE_API_URL}?date={target_date}"
    print(f"📡 Nesine API çağrısı: {url}")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"❌ Nesine API erişim hatası: {e}")
        return []

    matches = []

    # Anahtar: "Leagues" → Lig listesi
    for league in data.get("Leagues", []):
        league_name = league.get("N", "Bilinmeyen Lig")
        for match in league.get("Events", []):
            try:
                home = match.get("HN", "").strip()
                away = match.get("AN", "").strip()

                odds = match.get("OCG", {}).get("1", {}).get("OC", [])
                odds_1x2 = []
                for o in odds:
                    val = o.get("O")
                    if val:
                        odds_1x2.append(val)

                if len(odds_1x2) < 3:
                    continue

                matches.append({
                    "home_team": home,
                    "away_team": away,
                    "odds": odds_1x2[:3],
                    "league": league_name
                })
            except Exception as e:
                print(f"⚠️ Maç parse hatası: {e}")
                continue

    # Lig filtresi
    if filter_leagues:
        matches = [
            m for m in matches
            if any(k in m["league"].lower() for k in [
                "premier", "liga", "bundesliga", "serie", "ligue",
                "süper", "eredivisie", "primeira", "championship"
            ])
        ]

    print(f"✅ {len(matches)} maç bulundu ({target_date})")
    return matches

# ============================================================
# 🧪 Test modu
# ============================================================
if __name__ == "__main__":
    print("📡 Nesine verisi çekiliyor...")
    today_matches = fetch_today()
    print(f"✅ {len(today_matches)} maç bulundu\n")

    if today_matches:
        for m in today_matches[:5]:
            print(f"{m['home_team']} vs {m['away_team']} | {m['odds']} | {m['league']}")

        with open("nesine_today.json", "w", encoding="utf-8") as f:
            json.dump(today_matches, f, ensure_ascii=False, indent=2)
        print("\n💾 nesine_today.json dosyası oluşturuldu.")
    else:
        print("⚠️ Henüz aktif bülten bulunamadı.")
