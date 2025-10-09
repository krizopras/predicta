import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import date

class NesineFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/91.0.4472.124 Safari/537.36'
            )
        }
        self.base_url = "https://www.nesine.com"

    def get_page_content(self, url_path="/iddaa"):
        """Nesine sayfasının HTML içeriğini çeker."""
        try:
            url = f"{self.base_url}{url_path}"
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"❌ Sayfa çekme hatası: {e}")
            return None

    def extract_leagues_and_matches(self, html_content):
        """HTML içeriğinden lig, maç ve tahmin bilgilerini ayrıştırır."""
        soup = BeautifulSoup(html_content, 'html.parser')
        data = {"leagues": [], "matches": [], "predictions": []}

        self._extract_leagues(soup, data)
        self._extract_matches(soup, data)
        self._extract_predictions(soup, data)
        return data

    def _extract_leagues(self, soup, data):
        """Sayfadan lig isimlerini çıkarır."""
        all_text = soup.get_text()
        league_patterns = {
            "Premier League": ["Premier League"],
            "La Liga": ["La Liga"],
            "Bundesliga": ["Bundesliga"],
            "Serie A": ["Serie A"],
            "Ligue 1": ["Ligue 1"],
            "Süper Lig": ["Süper Lig"],
            "Eredivisie": ["Eredivisie"],
            "Primeira Liga": ["Primeira Liga"],
            "Championship": ["Championship"]
        }

        for league_name, keywords in league_patterns.items():
            for keyword in keywords:
                if keyword in all_text:
                    data["leagues"].append({
                        "name": league_name,
                        "original_text": keyword,
                        "category": "AVRUPA LİGLERİ"
                    })
                    break

    def _extract_matches(self, soup, data):
        """Sayfadan maç bilgilerini ayrıştırır."""
        match_elements = soup.find_all(['div', 'tr'], class_=re.compile(r'match|event|game'))
        for element in match_elements:
            match_data = self._parse_match_element(element)
            if match_data:
                data["matches"].append(match_data)

    def _parse_match_element(self, element):
        """HTML elementinden tek bir maçın verilerini çözümler."""
        try:
            teams = element.find_all(['span', 'div'], class_=re.compile(r'team|name'))
            if len(teams) >= 2:
                home_team = teams[0].get_text(strip=True)
                away_team = teams[1].get_text(strip=True)

                odds_elements = element.find_all(['span', 'button'], class_=re.compile(r'odd|rate|value'))
                odds = [odd.get_text(strip=True) for odd in odds_elements[:3]]

                return {
                    "home_team": home_team,
                    "away_team": away_team,
                    "odds": odds if len(odds) == 3 else [],
                    "league": self._detect_league_from_teams(home_team, away_team)
                }
        except Exception as e:
            print(f"⚠️ Maç parse hatası: {e}")
        return None

    def _detect_league_from_teams(self, home_team, away_team):
        """Takım isimlerinden lig tahmini yapar."""
        team_leagues = {
            "Arsenal": "Premier League", "Chelsea": "Premier League", "Liverpool": "Premier League",
            "Manchester": "Premier League", "Tottenham": "Premier League",
            "Real Madrid": "La Liga", "Barcelona": "La Liga", "Atletico": "La Liga",
            "Bayern": "Bundesliga", "Dortmund": "Bundesliga",
            "Milan": "Serie A", "Inter": "Serie A", "Juventus": "Serie A",
            "PSG": "Ligue 1", "Lyon": "Ligue 1",
            "Galatasaray": "Süper Lig", "Fenerbahçe": "Süper Lig", "Beşiktaş": "Süper Lig",
            "Ajax": "Eredivisie", "Porto": "Primeira Liga", "Benfica": "Primeira Liga"
        }

        for team, league in team_leagues.items():
            if team.lower() in home_team.lower() or team.lower() in away_team.lower():
                return league
        return "Bilinmeyen Lig"

    def _extract_predictions(self, soup, data):
        """Sayfadan tahmin bölümü (örneğin Günün En İyi Tahminleri) çıkarır."""
        predictions_text = soup.find(string=re.compile(r"Günün En İyi Tahminleri"))
        if predictions_text:
            prediction_section = predictions_text.find_parent()
            if prediction_section:
                data["predictions"].append({
                    "title": "Günün En İyi Tahminleri",
                    "analyst": "Tolgar Dine",
                    "content": prediction_section.get_text(strip=True)
                })


# =====================================================================
# 🔥 Backend ile uyumlu fonksiyonlar (main.py çağırır)
# =====================================================================

def fetch_today(filter_leagues: bool = True):
    """Bugünkü maçları Nesine'den getirir (opsiyonel lig filtresiyle)."""
    fetcher = NesineFetcher()
    html = fetcher.get_page_content()
    if not html:
        return []

    data = fetcher.extract_leagues_and_matches(html)
    matches = data.get("matches", [])

    # Lig filtreleme (varsayılan: sadece büyük ligler)
    if filter_leagues:
        matches = [
            m for m in matches
            if any(
                keyword in (m.get("league") or "").lower()
                for keyword in [
                    "premier", "liga", "bundesliga", "serie",
                    "ligue", "süper", "eredivisie", "primeira"
                ]
            )
        ]
    return matches


def fetch_matches_for_date(target_date, filter_leagues: bool = True):
    """Belirli bir tarih için Nesine maçlarını getirir."""
    fetcher = NesineFetcher()
    html = fetcher.get_page_content(f"/iddaa?dt={target_date}")
    if not html:
        return []

    data = fetcher.extract_leagues_and_matches(html)
    matches = data.get("matches", [])

    if filter_leagues:
        matches = [
            m for m in matches
            if any(
                keyword in (m.get("league") or "").lower()
                for keyword in [
                    "premier", "liga", "bundesliga", "serie",
                    "ligue", "süper", "eredivisie", "primeira"
                ]
            )
        ]
    return matches


# =====================================================================
# 🧪 Test amaçlı doğrudan çalıştırma
# =====================================================================
if __name__ == "__main__":
    print("📡 Nesine verisi çekiliyor...")
    today_matches = fetch_today()
    print(f"✅ {len(today_matches)} maç bulundu\n")

    for match in today_matches[:5]:
        print(f"{match['home_team']} vs {match['away_team']} | {match['odds']} | {match['league']}")

    with open("nesine_today.json", "w", encoding="utf-8") as f:
        json.dump(today_matches, f, ensure_ascii=False, indent=2)
    print("\n💾 nesine_today.json dosyası oluşturuldu.")
