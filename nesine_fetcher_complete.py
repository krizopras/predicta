import requests
import json

headers = {
    "User-Agent": "Mozilla/5.0",
    "Authorization": "Basic ...",  # Burada kendi keyin olacak!
    "Referer": "https://www.nesine.com/",
    "Origin": "https://www.nesine.com",
}

url = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"

r = requests.get(url, headers=headers)
print("Status kodu:", r.status_code)
if r.status_code != 200:
    print("Hata! Kod:", r.status_code)
    exit()

data = r.json()
matches = []

# EA: Erken Açılan / Prematch Maçlar
for m in data.get("sg", {}).get("EA", []):
    if m.get("GT") != 1:  # Sadece futbol
        continue
    match_info = {
        "home": m.get("HN", ""),
        "away": m.get("AN", ""),
        "league_code": m.get("LC", ""),
        "league_id": m.get("LID", ""),
        "tarih": m.get("D", ""),
        "saat": m.get("T", ""),
        "mac_id": m.get("C", ""),
        "mac_event_id": m.get("EV", ""),
        "oranlar": []
    }
    for bahis in m.get("MA", []):
        bahis_tipi = bahis.get("MTID")
        oranlar = bahis.get("OCA", [])
        match_info["oranlar"].append({
            "bahis_tipi": bahis_tipi,
            "oranlar": oranlar
        })
    matches.append(match_info)

# CA: Canlıda Açık Maçlar (CANLI)
for m in data.get("sg", {}).get("CA", []):
    if m.get("GT") != 1:  # Sadece futbol
        continue
    match_info = {
        "home": m.get("HN", ""),
        "away": m.get("AN", ""),
        "league_code": m.get("LC", ""),
        "league_id": m.get("LID", ""),
        "tarih": m.get("D", ""),
        "saat": m.get("T", ""),
        "mac_id": m.get("C", ""),
        "mac_event_id": m.get("EV", ""),
        "oranlar": []
    }
    for bahis in m.get("MA", []):
        bahis_tipi = bahis.get("MTID")
        oranlar = bahis.get("OCA", [])
        match_info["oranlar"].append({
            "bahis_tipi": bahis_tipi,
            "oranlar": oranlar
        })
    matches.append(match_info)

# Dosyaya yaz
with open("prematch_matches.json", "w", encoding="utf-8") as f:
    json.dump(matches, f, ensure_ascii=False, indent=2)

print(f"Toplam {len(matches)} futbol maçı (canlı + prematch) kaydedildi! -> prematch_matches.json")
