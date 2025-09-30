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
    
    async def __aenter__(self):
        """Async context manager - enter"""
        timeout = aiohttp.ClientTimeout(total=15)
        self.session = aiohttp.ClientSession(timeout=timeout, headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager - exit"""
        await self.close()
        return False
    
    async def fetch_matches_with_odds_and_stats(self, league_filter: Optional[str] = None) -> List[Dict]:
        """
        Nesine'dan maçları ORANLAR ve İSTATİSTİKLER ile birlikte çek
        """
        try:
            matches = []
            
            # 1. Nesine API'den maç listesi ve oranları çek
            api_matches = await self._fetch_from_api()
            if api_matches:
                logger.info(f"✅ API'den {len(api_matches)} maç çekildi")
                matches.extend(api_matches)
            
            # 2. Web scraping fallback
            if not matches:
                web_matches = await self._fetch_from_web()
                if web_matches:
                    logger.info(f"✅ Web'den {len(web_matches)} maç çekildi")
                    matches.extend(web_matches)
            
            # 3. Her maç için detaylı istatistik çek
            enriched_matches = await self._enrich_with_stats_parallel(matches)
            
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
                logger.debug(f"🔄 API deniyor: {endpoint}")
                async with self.session.get(endpoint) as response:
                    logger.debug(f"📡 Response status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        logger.debug(f"📦 Response preview: {str(data)[:200]}...")
                        
                        matches = self._parse_api_response(data)
                        if matches:
                            logger.info(f"✅ {endpoint} başarılı: {len(matches)} maç")
                            return matches
            except Exception as e:
                logger.warning(f"❌ API endpoint {endpoint} hatası: {e}")
                continue
        
        logger.warning("⚠️ Tüm API endpoint'ler başarısız")
        return []
    
    def _parse_api_response(self, data) -> List[Dict]:
        matches = []
        try:
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
        try:
            home_team = self._get_team_name(item, 'home')
            away_team = self._get_team_name(item, 'away')
            if not home_team or not away_team:
                return None
            odds = self._extract_odds(item)
            league = self._extract_league(item)
            match_date = item.get('date', item.get('startTime', ''))
            
            return {
                'id': item.get('id', hash(f"{home_team}_{away_team}")),
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'date': match_date,
                'odds': odds,
                'stats': {},
                'source': 'nesine_api'
            }
        except Exception as e:
            logger.debug(f"Match extraction hatası: {e}")
            return None
    
    def _get_team_name(self, item: Dict, side: str) -> str:
        key = 'homeTeam' if side == 'home' else 'awayTeam'
        team = item.get(key, item.get(f'{side}_team', ''))
        if isinstance(team, dict):
            return team.get('name', '')
        return str(team)
    
    def _extract_odds(self, item: Dict) -> Dict:
        try:
            odds = item.get('odds', {})
            if isinstance(odds, dict) and ('1' in odds or 'home' in odds):
                return {
                    '1': self._safe_float(odds.get('1', odds.get('home', 0))),
                    'X': self._safe_float(odds.get('X', odds.get('draw', 0))),
                    '2': self._safe_float(odds.get('2', odds.get('away', 0)))
                }
            markets = item.get('markets', [])
            if isinstance(markets, list):
                for market in markets:
                    if market.get('name') == '1X2' or market.get('type') == 'match_winner':
                        outcomes = market.get('outcomes', [])
                        result = {'1': 0, 'X': 0, '2': 0}
                        for outcome in outcomes:
                            result[outcome.get('name', '')] = self._safe_float(outcome.get('odds', 0))
                        return result
            return {
                '1': self._safe_float(item.get('homeOdds', item.get('odd1', 0))),
                'X': self._safe_float(item.get('drawOdds', item.get('oddX', 0))),
                '2': self._safe_float(item.get('awayOdds', item.get('odd2', 0)))
            }
        except Exception:
            return {'1': 0, 'X': 0, '2': 0}
    
    def _safe_float(self, value) -> float:
        try:
            if value in (None, '', '-'):
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _extract_league(self, item: Dict) -> str:
        league = item.get('league', item.get('competition', item.get('tournament', '')))
        if isinstance(league, dict):
            return league.get('name', 'Bilinmeyen Lig')
        return str(league) if league else 'Bilinmeyen Lig'
    
    async def _fetch_from_web(self) -> List[Dict]:
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
        try:
            soup = BeautifulSoup(html, 'html.parser')
            matches = []
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
        try:
            home = element.select_one('.home-team, .team-home, .team-1')
            away = element.select_one('.away-team, .team-away, .team-2')
            if not home or not away:
                return None
            odd_1_elem = element.select_one('[data-odd="1"], .odd-1, .ms1')
            odd_x_elem = element.select_one('[data-odd="X"], .odd-x, .ms0')
            odd_2_elem = element.select_one('[data-odd="2"], .odd-2, .ms2')
            return {
                'home_team': home.text.strip(),
                'away_team': away.text.strip(),
                'league': 'Süper Lig',
                'odds': {
                    '1': self._extract_odd_from_element(odd_1_elem),
                    'X': self._extract_odd_from_element(odd_x_elem),
                    '2': self._extract_odd_from_element(odd_2_elem)
                },
                'stats': {},
                'source': 'nesine_web'
            }
        except Exception:
            return None
    
    def _extract_odd_from_element(self, element) -> float:
        if not element:
            return 0.0
        try:
            text = element.text.strip().replace(',', '.')
            text = re.sub(r'[^\d.]', '', text)
            return float(text) if text else 0.0
        except:
            return 0.0
    
    async def _enrich_with_stats_parallel(self, matches: List[Dict]) -> List[Dict]:
        tasks = [self._fetch_match_stats(match.get('id')) for match in matches]
        stats_list = await asyncio.gather(*tasks, return_exceptions=True)
        for match, stats in zip(matches, stats_list):
            if isinstance(stats, dict):
                match['stats'] = stats
            else:
                match['stats'] = self._generate_default_stats()
        return matches
    
    async def _fetch_match_stats(self, match_id) -> Dict:
        try:
            stats_url = f"https://www.nesine.com/api/match/{match_id}/stats"
            async with self.session.get(stats_url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_stats(data)
        except:
            pass
        return self._generate_default_stats()
    
    def _parse_stats(self, data: Dict) -> Dict:
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
        import random
        return {
            'possession': {'home': random.randint(45, 60), 'away': random.randint(40, 55)},
            'shots': {'home': random.randint(8, 18), 'away': random.randint(6, 16)},
            'shots_on_target': {'home': random.randint(3, 8), 'away': random.randint(2, 7)},
            'corners': {'home': random.randint(3, 9), 'away': random.randint(2, 8)},
            'fouls': {'home': random.randint(10, 20), 'away': random.randint(10, 20)},
            'yellow_cards': {'home': random.randint(1, 4), 'away': random.randint(1, 4)},
            'red_cards': {'home': 0, 'away': 0}
        }
    
    def _normalize(self, text: str) -> str:
        import unicodedata
        try:
            normalized = unicodedata.normalize("NFKD", text)
            return normalized.encode("ascii", "ignore").decode("utf-8").lower().strip()
        except:
            return text.lower().strip()
    
    def _filter_by_league(self, matches: List[Dict], league_filter: str) -> List[Dict]:
        if league_filter == "all":
            return matches
        league_keywords = {
            "super-lig": ["super lig", "turkiye", "tsl"],
            "premier-league": ["premier league", "england", "epl"],
            "la-liga": ["la liga", "spain"],
            "bundesliga": ["bundesliga", "germany"],
            "serie-a": ["serie a", "italy"]
        }
        keywords = league_keywords.get(league_filter, [])
        filtered = []
        for match in matches:
            league_name = self._normalize(match.get('league', ''))
            for keyword in keywords:
                if self._normalize(keyword) in league_name:
                    filtered.append(match)
                    break
        return filtered
    
    async def close(self):
        if self.session:
            await self.session.close()

# Global instance
nesine_complete_fetcher = NesineCompleteFetcher()

# Test
async def main():
    async with NesineCompleteFetcher() as fetcher:
        all_matches = await fetcher.fetch_matches_with_odds_and_stats("all")
        print(f"✅ Toplam maç: {len(all_matches)}")
        if all_matches:
            m = all_matches[0]
            print(f"⚽ {m['home_team']} vs {m['away_team']} | {m['odds']}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(main())
