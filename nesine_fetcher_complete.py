import aiohttp
from bs4 import BeautifulSoup

async def scrape_nesine_matches():
    """Nesine web sitesinden maç verilerini çek"""
    try:
        url = "https://www.nesine.com/iddaa"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    matches = []
                    # Maç elementlerini bul (bu selector'lar değişebilir)
                    match_elements = soup.find_all('div', class_=['match-row', 'event-item'])
                    
                    for element in match_elements:
                        match_data = extract_match_data(element)
                        if match_data:
                            matches.append(match_data)
                    
                    return matches
    except Exception as e:
        print(f"Scraping hatası: {e}")
        return []

def extract_match_data(element):
    """HTML elementinden maç verilerini çıkar"""
    try:
        # Bu selector'lar sitenin yapısına göre ayarlanmalı
        home_team = element.find('span', class_='home-team')
        away_team = element.find('span', class_='away-team')
        odds = element.find_all('span', class_='odd-value')
        
        if home_team and away_team:
            return {
                'home': home_team.text.strip(),
                'away': away_team.text.strip(),
                'odds': [odd.text.strip() for odd in odds] if odds else []
            }
    except:
        pass
    return None
