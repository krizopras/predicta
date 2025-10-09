import requests
import json
from datetime import date, datetime

# ============================================================
# 🔧 Global Ayarlar
# ============================================================
BASE_URL = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.nesine.com/",
    "Origin": "https://www.nesine.com",
}

# ============================================================
# 🎯 Ana Fonksiyonlar
# ============================================================

def fetch_bulletin(target_date=None, filter_leagues=True):
    """
    Nesine'nin resmi JSON API'sinden bülteni çeker (prematch + canlı).
    Geri dönüş formatı PredictaIQ backend yapısına uyumludur.
    """
    if not target_date:
        target_date = date.today().strftime("%Y-%m-%d")
    elif isinstance(target_date, (date, datetime)):
        target_date = target_date.strftime("%Y-%m-%d")

    url = f"{BASE_URL}?date={target_date}"
    print(f"📡 Nesine API çağrısı: {url}")

    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"❌ Nesine API erişim hatası: {e}")
        return []

    matches = []

    # =====================================================
    # 🧩 Format 1: Yeni JSON (Leagues -> Events)
    # =====================================================
    if "Leagues" in data:
        for league in data.get("Leagues", []):
            league_name = league.get("N", "Bilinmeyen Lig")

            for match in league.get("Events", []):
                if match.get("GT") != 1:  # sadece futbol
                    continue

                ocg = match.get("OCG", {}).get("1", {}).get("OC", [])
                odds = []
                for o in ocg[:3]:
                    if o.get("O"):
                        try:
                            odds.append(round(float(o.get("O")), 2))
                        except:
                            continue

                if len(odds) < 3:
                    continue

                matches.append({
                    "home_team": match.get("HN", "").strip(),
                    "away_team": match.get("AN", "").strip(),
                    "league": league_name,
                    "date": match.get("D", ""),
                    "time": match.get("T", ""),
                    "match_id": match.get("C", ""),
                    "event_id": match.get("EV", ""),
                    "odds": {
                        "MS1": odds[0],
                        "MS0": odds[1],
                        "MS2": odds[2]
                    }
                })

    # =====================================================
    # 🧩 Format 2: Eski JSON (sg -> EA/CA)
    # =====================================================
    elif "sg" in data:
        for section in ["EA", "CA"]:
            for m in data.get("sg", {}).get(section, []):
                if m.get("GT") != 1:
                    continue

                odds = []
                for bahis in m.get("MA", []):
                    for oca in bahis.get("OCA", []):
                        val = oca.get("O")
                        if val:
                            try:
                                odds.append(round(float(val), 2))
                            except:
                                continue

                if len(odds) < 3:
                    continue

                matches.append({
                    "home_team": m.get("HN", ""),
                    "away_team": m.get("AN", ""),
                    "league": m.get("LC", "Bilinmeyen Lig"),
                    "date": m.get("D", ""),
                    "time": m.get("T", ""),
                    "match_id": m.get("C", ""),
                    "event_id": m.get("EV", ""),
                    "odds": {
                        "MS1": odds[0],
                        "MS0": odds[1],
                        "MS2": odds[2]
                    }
                })

    # =====================================================
    # 🔎 Lig Filtresi (isteğe bağlı)
    # =====================================================
    if filter_leagues:
        matches = [
            m for m in matches
            if any(k in m.get("league", "").lower() for k in [
                "premier", "bundes", "liga", "serie",
                "ligue", "super", "süper", "eredivisie", "primeira"
            ])
        ]

    print(f"✅ {len(matches)} maç bulundu ({target_date})")
    return matches

# ============================================================
# ⚙️ PredictaIQ Uyumluluk Katmanı
# ============================================================

def fetch_today(filter_leagues: bool = True):
    """
    Bugünkü maçları getirir.
    PredictaIQ backend'i bu fonksiyonu doğrudan çağırır.
    """
    return fetch_bulletin(date.today(), filter_leagues)


def fetch_matches_for_date(target_date, filter_leagues: bool = True):
    """
    Belirli bir tarih için Nesine maçlarını getirir.
    PredictaIQ backend'inde tarihli sorgular bu fonksiyonu kullanır.
    """
    return fetch_bulletin(target_date, filter_leagues)


# ============================================================
# 🧪 Test (lokal çalıştırma)
# ============================================================
if __name__ == "__main__":
    print("📡 Nesine bülteni çekiliyor...\n")
    today_matches = fetch_today()

    print(f"✅ {len(today_matches)} maç bulundu!\n")
    for m in today_matches[:10]:
        print(f"{m['home_team']} vs {m['away_team']} | "
              f"{m['odds']} | {m['league']}")

    # JSON kaydı
    with open("nesine_today.json", "w", encoding="utf-8") as f:
        json.dump(today_matches, f, ensure_ascii=False, indent=2)
    print("\n💾 nesine_today.json dosyası oluşturuldu.")
