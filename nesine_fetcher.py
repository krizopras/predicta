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
        """Nesine sayfasının içeriğini çek."""
        try:
            url = f"{self.base_url}{url_path}"
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"❌ Sayfa çekme hatası: {e}")
            return None

    def extract_leagues_and_matches(self, html_content):
        """HTML'den lig ve maç bilgilerini çıkar."""
        soup = BeautifulSoup(html_content, 'html.parser')
        data = {"leagues": [], "matches": [], "predictions": []}

        self._extract_leagues(soup, data)
        self._extract_matches(soup, data)
        self._extract_predictions(soup, data)

        return data

    def _extract_leagues(self, soup, data):
        """Ligleri çıkar."""
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
        """Maç bilgilerini çıkar."""
        match_elements = soup.find_all(['div', 'tr'], class_=re.compile(r'match|event|game'))
        for element in match_elements:
            match_data = self._parse_match_element(element)
            if match_data:
                data["matches"].append(match_data)

    def _parse_match_element(self, element):
        """Maç elementini parse et."""
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
                    "odds": odds,
                    "league": self._detect_league_from_teams(home_team, away_team)
                }
        except Exception as e:
            print(f"Maç parse hatası: {e}")
        return None

    def _detect_league_from_teams(self, home_team, away_team):
        """Takım isimlerinden lig tahmini."""
        team_leagues = {
            "Arsenal": "Premier League", "Chelsea": "Premier League",
            "Manchester": "Premier League", "Liverpool": "Premier League",
            "Real Madrid": "La Liga", "Barcelona": "La Liga",
            "Atletico": "La Liga", "Sevilla": "La Liga",
            "Bayern": "Bundesliga", "Dortmund": "Bundesliga",
            "Milan": "Serie A", "Inter": "Serie A", "Juventus": "Serie A",
            "PSG": "Ligue 1", "Lyon": "Ligue 1",
            "Galatasaray": "Süper Lig", "Fenerbahçe": "Süper Lig", "Beşiktaş": "Süper Lig"
        }

        for team, league in team_leagues.items():
            if team in home_team or team in away_team:
                return league
        return "Bilinmeyen Lig"

    def _extract_predictions(self, soup, data):
        """Tahmin bölümünü çıkar."""
        predictions_text = soup.find(string=re.compile(r"Günün En İyi Tahminleri"))
        if predictions_text:
            prediction_section = predictions_text.find_parent()
            if prediction_section:
                prediction_data = {
                    "title": "Günün En İyi Tahminleri",
                    "analyst": "Tolgar Dine",
                    "content": prediction_section.get_text(strip=True)
                }
                data["predictions"].append(prediction_data)


# 🟢 Yeni eklenen fonksiyonlar (main.py ile tam uyumlu)
def fetch_today():
    """Bugünün Nesine maçlarını getirir."""
    fetcher = NesineFetcher()
    html = fetcher.get_page_content()
    if html:
        return fetcher.extract_leagues_and_matches(html)
    return {}


def fetch_matches_for_date(target_date):
    """Belirli bir tarih için Nesine maçlarını getirir."""
    fetcher = NesineFetcher()
    html = fetcher.get_page_content(f"/iddaa?dt={target_date}")
    if html:
        return fetcher.extract_leagues_and_matches(html)
    return {}


# 🧪 Test çalıştırma (manuel kullanım)
if __name__ == "__main__":
    data = fetch_today()
    print("🟢 LİGLER:")
    for league in data.get("leagues", []):
        print(f"   {league['category']} - {league['name']}")

    print(f"\n⚽ MAÇLAR ({len(data.get('matches', []))} adet):")
    for match in data.get("matches", [])[:5]:
        print(f"   {match['home_team']} vs {match['away_team']}")
        print(f"   Oranlar: {match.get('odds', [])}")
        print(f"   Lig: {match.get('league', 'Bilinmiyor')}")
        print()

    print("🎯 TAHMİNLER:")
    for prediction in data.get("predictions", []):
        print(f"   {prediction['title']} - {prediction['analyst']}")

    # JSON olarak kaydetmek istersen:
    with open("nesine_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("\n✅ nesine_data.json dosyası oluşturuldu.")
