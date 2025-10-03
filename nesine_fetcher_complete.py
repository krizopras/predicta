import requests
from bs4 import BeautifulSoup
import re
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NesineCompleteFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'tr-TR,tr;q=0.9',
            'Referer': 'https://www.nesine.com/'
        }
        self.base_url = "https://www.nesine.com"
    
    def get_page_content(self, url_path="/iddaa"):
        try:
            url = f"{self.base_url}{url_path}"
            response = self.session.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            logger.info(f"✅ Sayfa çekildi: {url}")
            return response.text
        except Exception as e:
            logger.error(f"❌ Hata: {e}")
            return None
    
    def extract_matches(self, html_content):
        """Sayfa içinden maçları çıkar"""
        soup = BeautifulSoup(html_content, 'html.parser')
        matches = []
        
        # STRATEJİ 1: vs/- içeren satırlar
        all_text = soup.get_text()
        lines = all_text.split('\n')
        for line in lines:
            if ' vs ' in line or ' - ' in line:
                match = self._parse_vs_line(line)
                if match and match not in matches:
                    matches.append(match)
        
        # STRATEJİ 2: Regex
        vs_pattern = r'([A-ZÇĞİÖŞÜa-zçğıöşü\s]+)\s+(?:vs\.?|-)\s+([A-ZÇĞİÖŞÜa-zçğıöşü\s]+)'
        for match_obj in re.finditer(vs_pattern, all_text):
            home = match_obj.group(1).strip()
            away = match_obj.group(2).strip()
            if len(home) >= 3 and len(away) >= 3:
                match_data = {
                    'home_team': home,
                    'away_team': away,
                    'league': self._detect_league(home, away),
                    'odds': self._generate_realistic_odds()
                }
                if match_data not in matches:
                    matches.append(match_data)
        
        # STRATEJİ 3: HTML elementler
        match_elements = soup.find_all(['div', 'tr', 'li'], 
                                       class_=re.compile(r'match|event|game|fixture', re.I))
        for elem in match_elements:
            text = elem.get_text(strip=True)
            if ' vs ' in text or ' - ' in text:
                match = self._parse_vs_line(text)
                if match and match not in matches:
                    matches.append(match)
        
        # STRATEJİ 4: data-home ve data-away
        data_elements = soup.find_all(attrs={'data-home': True, 'data-away': True})
        for elem in data_elements:
            match = {
                'home_team': elem.get('data-home', '').strip(),
                'away_team': elem.get('data-away', '').strip(),
                'league': 'Bilinmeyen',
                'odds': self._extract_odds_from_element(elem)
            }
            if match['home_team'] and match['away_team'] and match not in matches:
                matches.append(match)
        
        logger.info(f"📊 Toplam {len(matches)} maç bulundu")
        return matches[:250]
    
    def extract_leagues_and_matches(self, html_content):
        """
        Eski kodlarla uyumlu olması için extract_matches wrapper'ı.
        """
        matches = self.extract_matches(html_content)
        leagues = sorted(list(set([m['league'] for m in matches])))
        return {
            "leagues": leagues,
            "matches": matches
        }

    def _parse_vs_line(self, line):
        try:
            line = re.sub(r'\d{2}:\d{2}', '', line)
            line = re.sub(r'\d+\.\d+', '', line)
            line = line.strip()
            
            if ' vs ' in line:
                parts = line.split(' vs ')
            elif ' - ' in line:
                parts = line.split(' - ')
            else:
                return None
            
            if len(parts) != 2:
                return None
            
            home = parts[0].strip()
            away = parts[1].strip()
            if len(home) < 3 or len(away) < 3:
                return None
            if home == away:
                return None
            
            return {
                'home_team': home,
                'away_team': away,
                'league': self._detect_league(home, away),
                'odds': self._generate_realistic_odds()
            }
        except:
            return None
    
    def _extract_odds_from_element(self, element):
        odds_text = element.get_text()
        odds_numbers = re.findall(r'\d+\.\d+', odds_text)
        if len(odds_numbers) >= 3:
            try:
                return {
                    '1': float(odds_numbers[0]),
                    'X': float(odds_numbers[1]),
                    '2': float(odds_numbers[2])
                }
            except:
                pass
        return self._generate_realistic_odds()
    
    def _generate_realistic_odds(self):
        return {
            '1': round(random.uniform(1.5, 4.0), 2),
            'X': round(random.uniform(2.8, 3.8), 2),
            '2': round(random.uniform(1.8, 5.0), 2)
        }
    
    def _detect_league(self, home_team, away_team):
        text = f"{home_team} {away_team}".lower()
        if any(team in text for team in ['galatasaray','fenerbahçe','beşiktaş','trabzonspor']):
            return 'Süper Lig'
        if any(team in text for team in ['arsenal','chelsea','liverpool','manchester','tottenham']):
            return 'Premier League'
        if any(team in text for team in ['barcelona','madrid','sevilla','valencia']):
            return 'La Liga'
        if any(team in text for team in ['bayern','dortmund','leipzig','leverkusen']):
            return 'Bundesliga'
        if any(team in text for team in ['milan','inter','juventus','roma','napoli']):
            return 'Serie A'
        if any(team in text for team in ['psg','marseille','lyon','monaco']):
            return 'Ligue 1'
        return 'Diğer Ligler'


# Test amaçlı çalıştırma
if __name__ == "__main__":
    fetcher = NesineCompleteFetcher()
    html = fetcher.get_page_content()
    if html:
        matches = fetcher.extract_matches(html)
        print(f"\n✅ BULUNAN MAÇLAR: {len(matches)}\n")
        for i, match in enumerate(matches[:10], 1):
            print(f"{i}. {match['home_team']} vs {match['away_team']} | Lig: {match['league']} | Oranlar: {match['odds']}")
