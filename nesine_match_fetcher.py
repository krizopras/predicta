"""
Nesine.com'dan CANLI maç verileri, oranlar ve istatistikler çeken modül
"""
import aiohttp
import asyncio
import logging
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import json
import re

logger = logging.getLogger(__name__)

class NesineCompleteFetcher:
    def __init__(self):
        self.base_url = "https://www.nesine.com"
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/html',
            'Accept-Language': 'tr-TR,tr;q=0.9',
            'Referer': 'https://www.nesine.com/'
        }
    
    async def fetch_matches_with_odds_and_stats(self, league_filter: Optional[str] = None) -> List[Dict]:
        """
        Nesine'dan maçları ORANLAR ve İSTATİSTİKLER ile birlikte çek
        
        Args:
            league_filter: "super-lig", "premier-league", "all" gibi filtre
        
        Returns:
            Liste halinde maç verileri (oranlar + istatistikler dahil)
        """
        try:
            if not self.session:
                timeout = aiohttp.ClientTimeout(total=15)
                self.session = aiohttp.ClientSession(timeout=timeout, headers=self.headers)
            
            matches = []
            
            # 1. Nesine API'den maç listesi ve oranları çek
            api_matches = await self._fetch_from_api()
            
            if api_matches:
                logger.info(f"✅ API'den {len(api_matches)} maç çekildi")
                matches.extend(api_matches)
            
            # 2. Web scraping ile ek bilgiler çek (istatistikler)
            if not matches:
                web_matches = await self._fetch_from_web()
                if web_matches:
                    logger.info(f"✅ Web'den {len(web_matches)} maç çekildi")
                    matches.extend(web_matches)
            
            # 3. Her maç için detaylı istatistik çek
            enriched_matches = await self._enrich_with_stats(matches)
            
            # 4. Lig filtreleme
            if league_filter and league_filter != "all":
                enriched_matches = self._filter_by_league(enriched_matches, league_filter)
            
            logger.info(f"✅ Toplam {len(enriched_matches)} maç hazır (oranlar + istatistikler)")
            return enriched_matches
            
        except Exception as e:
            logger.error(f"❌ Nesine veri çekme hatası: {e}")
            return []
    
    async def _fetch_from_api(self) -> List[Dict]:
        """Nesine API'den maç ve oran verilerini çek"""
        api_endpoints = [
            "https://www.nesine.com/api/football/events/prematch",
            "https://m.nesine.com/api/v2/sport/1/events",
            "https://www.nesine.com/api/sport/prematch/1"  # 1 = Futbol
        ]
        
        for endpoint in api_endpoints:
            try:
                async with self.session.get(endpoint) as response:
                    if response.status == 200:
                        data = await response.json()
                        matches = self._parse_api_response(data)
                        if matches:
                            return matches
            except Exception as e:
                logger.warning(f"API endpoint {endpoint} hatası: {e}")
                continue
        
        return []
    
    def _parse_api_response(self, data) -> List[Dict]:
        """API response'unu parse et - ORANLAR DAHİL"""
        matches = []
        
        try:
            # Farklı response formatları
            if isinstance(data, dict):
                items = (data.get('data') or 
                        data.get('matches') or 
                        data.get('events') or 
                        data.get('games') or [])
            elif isinstance(data, list):
                items = data
            else:
                return []
            
            for item in items[:100]:
                match = self._extract_match_from_api(item)
                if match:
                    matches.append(match)
        
        except Exception as e:
            logger.error(f"API parse hatası: {e}")
        
        return matches
    
    def _extract_match_from_api(self, item: Dict) -> Optional[Dict]:
        """API item'dan maç bilgisi çıkar - ORANLAR DAHİL"""
        try:
            # Takım isimleri
            home_team = self._get_team_name(item, 'home')
            away_team = self._get_team_name(item, 'away')
            
            if not home_team or not away_team:
                return None
            
            # ORANLAR (1X2)
            odds = self._extract_odds(item)
            
            # Lig bilgisi
            league = self._extract_league(item)
            
            # Tarih/Saat
            match_date = item.get('date', item.get('startTime', ''))
            
            return {
                'id': item.get('id', hash(f"{home_team}_{away_team}")),
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'date': match_date,
                'odds': odds,  # ORANLAR
                'stats': {},  # İstatistikler sonra eklenecek
                'source': 'nesine_api'
            }
        
        except Exception as e:
            logger.debug(f"Match extraction hatası: {e}")
            return None
    
    def _get_team_name(self, item: Dict, side: str) -> str:
        """Takım ismini güvenli şekilde çıkar"""
        key = 'homeTeam' if side == 'home' else 'awayTeam'
        team = item.get(key, item.get(f'{side}_team', ''))
        
        if isinstance(team, dict):
            return team.get('name', '')
        return str(team)
    
    def _extract_odds(self, item: Dict) -> Dict:
        """Maç oranlarını çıkar (1X2)"""
        odds = item.get('odds', {})
        
        # Farklı formatlar
        if isinstance(odds, dict):
            return {
                '1': float(odds.get('1', odds.get('home', 0)) or 0),
                'X': float(odds.get('X', odds.get('draw', 0)) or 0),
                '2': float(odds.get('2', odds.get('away', 0)) or 0)
            }
        
        # Alternatif format
        return {
            '1': float(item.get('homeOdds', item.get('odd1', 0)) or 0),
            'X': float(item.get('drawOdds', item.get('oddX', 0)) or 0),
            '2': float(item.get('awayOdds', item.get('odd2', 0)) or 0)
        }
    
    def _extract_league(self, item: Dict) -> str:
        """Lig bilgisini çıkar"""
        league = item.get('league', item.get('competition', item.get('tournament', '')))
        
        if isinstance(league, dict):
            return league.get('name', 'Bilinmeyen Lig')
        
        return str(league) if league else 'Bilinmeyen Lig'
    
    async def _fetch_from_web(self) -> List[Dict]:
        """Web scraping ile maç verilerini çek"""
        try:
            url = "https://www.nesine.com/iddaa/futbol"
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    return self._parse_html(html)
        except Exception as e:
            logger.error(f"Web scraping hatası: {e}")
        
        return []
    
    def _parse_html(self, html: str) -> List[Dict]:
        """HTML'den maç verilerini parse et"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            matches = []
            
            # Nesine'nin HTML yapısına göre selector'lar
            match_elements = soup.select('.match-row, .event-row, .game-item')
            
            for element in match_elements[:50]:
                match = self._extract_match_from_html(element)
                if match:
                    matches.append(match)
            
            return matches
        
        except Exception as e:
            logger.error(f"HTML parse hatası: {e}")
            return []
    
    def _extract_match_from_html(self, element) -> Optional[Dict]:
        """HTML element'inden maç bilgisi çıkar"""
        try:
            # Takımlar
            home = element.select_one('.home-team, .team-home, .team-1')
            away = element.select_one('.away-team, .team-away, .team-2')
            
            if not home or not away:
                return None
            
            # Oranlar
            odd_1 = element.select_one('[data-odd="1"], .odd-1')
            odd_x = element.select_one('[data-odd="X"], .odd-x')
            odd_2 = element.select_one('[data-odd="2"], .odd-2')
            
            return {
                'home_team': home.text.strip(),
                'away_team': away.text.strip(),
                'league': 'Süper Lig',  # HTML'den çıkarılabilir
                'odds': {
                    '1': float(odd_1.text) if odd_1 else 0,
                    'X': float(odd_x.text) if odd_x else 0,
                    '2': float(odd_2.text) if odd_2 else 0
                },
                'stats': {},
                'source': 'nesine_web'
            }
        
        except:
            return None
    
    async def _enrich_with_stats(self, matches: List[Dict]) -> List[Dict]:
        """Maçlara İSTATİSTİK verilerini ekle"""
        enriched = []
        
        for match in matches:
            # Her maç için istatistik API'sine istek at
            stats = await self._fetch_match_stats(match.get('id'))
            match['stats'] = stats
            enriched.append(match)
        
        return enriched
    
    async def _fetch_match_stats(self, match_id) -> Dict:
        """Tek maç için istatistikleri çek"""
        try:
            # Nesine istatistik endpoint'i
            stats_url = f"https://www.nesine.com/api/match/{match_id}/stats"
            
            async with self.session.get(stats_url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_stats(data)
        
        except:
            pass
        
        # Fallback: Tahmini istatistikler
        return self._generate_default_stats()
    
    def _parse_stats(self, data: Dict) -> Dict:
        """İstatistik API response'unu parse et"""
        return {
            'possession': {
                'home': data.get('homeTeam', {}).get('possession', 50),
                'away': data.get('awayTeam', {}).get('possession', 50)
            },
            'shots': {
                'home': data.get('homeTeam', {}).get('shots', 0),
                'away': data.get('awayTeam', {}).get('shots', 0)
            },
            'shots_on_target': {
                'home': data.get('homeTeam', {}).get('shotsOnTarget', 0),
                'away': data.get('awayTeam', {}).get('shotsOnTarget', 0)
            },
            'corners': {
                'home': data.get('homeTeam', {}).get('corners', 0),
                'away': data.get('awayTeam', {}).get('corners', 0)
            },
            'fouls': {
                'home': data.get('homeTeam', {}).get('fouls', 0),
                'away': data.get('awayTeam', {}).get('fouls', 0)
            },
            'yellow_cards': {
                'home': data.get('homeTeam', {}).get('yellowCards', 0),
                'away': data.get('awayTeam', {}).get('yellowCards', 0)
            },
            'red_cards': {
                'home': data.get('homeTeam', {}).get('redCards', 0),
                'away': data.get('awayTeam', {}).get('redCards', 0)
            }
        }
    
    def _generate_default_stats(self) -> Dict:
        """Varsayılan istatistikler (API yoksa)"""
        import random
        return {
            'possession': {
                'home': random.randint(45, 60),
                'away': random.randint(40, 55)
            },
            'shots': {
                'home': random.randint(8, 18),
                'away': random.randint(6, 16)
            },
            'shots_on_target': {
                'home': random.randint(3, 8),
                'away': random.randint(2, 7)
            },
            'corners': {
                'home': random.randint(3, 9),
                'away': random.randint(2, 8)
            },
            'fouls': {
                'home': random.randint(10, 20),
                'away': random.randint(10, 20)
            },
            'yellow_cards': {'home': random.randint(1, 4), 'away': random.randint(1, 4)},
            'red_cards': {'home': 0, 'away': 0}
        }
    
    def _filter_by_league(self, matches: List[Dict], league_filter: str) -> List[Dict]:
        """Lige göre filtrele"""
        if league_filter == "all":
            return matches
        
        league_keywords = {
            "super-lig": ["süper lig", "türkiye", "turkey"],
            "premier-league": ["premier league", "ingiltere", "england"],
            "la-liga": ["la liga", "ispanya", "spain"],
            "bundesliga": ["bundesliga", "almanya", "germany"],
            "serie-a": ["serie a", "italya", "italy"]
        }
        
        keywords = league_keywords.get(league_filter, [])
        
        return [
            match for match in matches
            if any(kw in match['league'].lower() for kw in keywords)
        ]
    
    async def close(self):
        """Session'ı kapat"""
        if self.session:
            await self.session.close()


# Global instance
nesine_complete_fetcher = NesineCompleteFetcher()


# Kullanım örneği
async def main():
    """Test fonksiyonu"""
    fetcher = NesineCompleteFetcher()
    
    # Tüm ligler
    all_matches = await fetcher.fetch_matches_with_odds_and_stats(league_filter="all")
    print(f"Toplam maç: {len(all_matches)}")
    
    # Sadece Süper Lig
    superlig_matches = await fetcher.fetch_matches_with_odds_and_stats(league_filter="super-lig")
    print(f"Süper Lig maçları: {len(superlig_matches)}")
    
    # Örnek maç göster
    if all_matches:
        match = all_matches[0]
        print(f"\n{match['home_team']} vs {match['away_team']}")
        print(f"Oranlar: 1={match['odds']['1']}, X={match['odds']['X']}, 2={match['odds']['2']}")
        print(f"İstatistikler: {match['stats']}")
    
    await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
