import requests
from bs4 import BeautifulSoup
import json
import re

class NesineScraper:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.base_url = "https://www.nesine.com"
    
    def get_page_content(self, url_path="/iddaa"):
        """Nesine sayfasının içeriğini çek"""
        try:
            url = f"{self.base_url}{url_path}"
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"❌ Sayfa çekme hatası: {e}")
            return None
    
    def extract_leagues_and_matches(self, html_content):
        """HTML'den lig ve maç bilgilerini çıkar"""
        soup = BeautifulSoup(html_content, 'html.parser')
        data = {
            "leagues": [],
            "matches": [],
            "predictions": []
        }
        
        # Ligleri çıkar
        self._extract_leagues(soup, data)
        
        # Maçları çıkar
        self._extract_matches(soup, data)
        
        # Tahminleri çıkar
        self._extract_predictions(soup, data)
        
        return data
    
    def _extract_leagues(self, soup, data):
        """Avrupa liglerini çıkar"""
        # Tüm metni al ve ligleri parse et
        all_text = soup.get_text()
        
        # Lig pattern'leri
        league_patterns = {
            "Premier League": ["Premier League", "hgiflere"],
            "La Liga": ["La Liga", "Epanya"],
            "Bundesliga": ["Bundesliga", "Almanya"],
            "Serie A": ["Serie A", "italya"],
            "Ligue 1": ["Ligue 1", "Fransa"],
            "Süper Lig": ["Süper Lig", "Stiger Lig", "Türkiye"],
            "Eredivisie": ["Erotihiste", "Hollanska"],
            "Primeira Liga": ["Primeira Liga", "Portekiz"],
            "Championship": ["Championalip"]
        }
        
        for league_name, keywords in league_patterns.items():
            for keyword in keywords:
                if keyword in all_text:
                    data["leagues"].append({
                        "name": league_name,
                        "original_text": keyword,
                        "category": "AVRUPA LİGLERİ" if league_name in [
                            "Premier League", "La Liga", "Bundesliga", 
                            "Serie A", "Ligue 1", "Süper Lig"
                        ] else "DİĞER LİGLER"
                    })
                    break
    
    def _extract_matches(self, soup, data):
        """Maç bilgilerini çıkar"""
        # Maç elementlerini bul (nesine.com'un yapısına göre)
        match_elements = soup.find_all(['div', 'tr'], class_=re.compile(r'match|event|game'))
        
        for element in match_elements:
            match_data = self._parse_match_element(element)
            if match_data:
                data["matches"].append(match_data)
    
    def _parse_match_element(self, element):
        """Maç elementini parse et"""
        try:
            # Takım isimlerini bul
            teams = element.find_all(['span', 'div'], class_=re.compile(r'team|name'))
            if len(teams) >= 2:
                home_team = teams[0].get_text(strip=True)
                away_team = teams[1].get_text(strip=True)
                
                # Oranları bul
                odds_elements = element.find_all(['span', 'button'], class_=re.compile(r'odd|rate|value'))
                odds = [odd.get_text(strip=True) for odd in odds_elements[:3]]  # İlk 3 oran
                
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
        """Takım isimlerinden lig tahmini"""
        # Bu kısmı takım veritabanınızla geliştirebilirsiniz
        team_leagues = {
            "Arsenal": "Premier League", "Chelsea": "Premier League",
            "Real Madrid": "La Liga", "Barcelona": "La Liga",
            "Bayern": "Bundesliga", "Dortmund": "Bundesliga",
            "Milan": "Serie A", "Inter": "Serie A",
            "Galatasaray": "Süper Lig", "Fenerbahçe": "Süper Lig"
        }
        
        for team, league in team_leagues.items():
            if team in home_team or team in away_team:
                return league
        
        return "Bilinmeyen Lig"
    
    def _extract_predictions(self, soup, data):
        """Tahmin bölümünü çıkar"""
        # "Günün En İyi Tahminleri" bölümünü bul
        predictions_text = soup.find(string=re.compile(r"Günün En İyi Tahminleri"))
        if predictions_text:
            prediction_section = predictions_text.find_parent()
            if prediction_section:
                # Tahmin detaylarını çıkar
                prediction_data = {
                    "title": "Günün En İyi Tahminleri",
                    "analyst": "Tolgar Dine",  # Sabit görünüyor
                    "content": prediction_section.get_text(strip=True)
                }
                data["predictions"].append(prediction_data)

# Kullanım örneği
def main():
    scraper = NesineScraper()
    
    # Sayfayı çek
    html_content = scraper.get_page_content()
    
    if html_content:
        # Verileri çıkar
        data = scraper.extract_leagues_and_matches(html_content)
        
        # Sonuçları göster
        print("🟢 LİGLER:")
        for league in data["leagues"]:
            print(f"   {league['category']} - {league['name']}")
        
        print(f"\n⚽ MAÇLAR ({len(data['matches'])} adet):")
        for match in data["matches"][:5]:  # İlk 5 maçı göster
            print(f"   {match['home_team']} vs {match['away_team']}")
            print(f"   Oranlar: {match.get('odds', [])}")
            print(f"   Lig: {match.get('league', 'Bilinmiyor')}")
            print()
        
        print("🎯 TAHMİNLER:")
        for prediction in data["predictions"]:
            print(f"   {prediction['title']} - {prediction['analyst']}")

if __name__ == "__main__":
    main()
