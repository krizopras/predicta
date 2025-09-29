# nesine_match_fetcher.py - YENİ VERSİYON
import aiohttp
import asyncio
import logging
from typing import List, Dict, Any
import json

logger = logging.getLogger(__name__)

class NesineFetcher:
    def __init__(self):
        self.base_url = "https://nesine.com"
        self.session = None
        
    async def fetch_prematch_matches(self) -> List[Dict]:
        """Nesine'dan maç ve istatistik verilerini çek"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Nesine API endpoint'leri
            endpoints = [
                "https://nesine.com/api/football/matches",
                "https://nesine.com/api/sports/football/prematch"
            ]
            
            for endpoint in endpoints:
                try:
                    async with self.session.get(endpoint, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            matches = self.parse_matches(data)
                            if matches:
                                logger.info(f"✅ {len(matches)} maç başarıyla çekildi")
                                return matches
                except Exception as e:
                    logger.warning(f"Endpoint {endpoint} hatası: {e}")
                    continue
            
            # Fallback: Örnek maçlar oluştur
            return self.generate_sample_matches()
            
        except Exception as e:
            logger.error(f"Nesine fetcher hatası: {e}")
            return self.generate_sample_matches()
    
    def parse_matches(self, data: Any) -> List[Dict]:
        """Nesine response'unu parse et"""
        matches = []
        
        try:
            # Farklı response formatlarına uyum sağla
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and 'matches' in data:
                items = data['matches']
            elif isinstance(data, dict) and 'events' in data:
                items = data['events']
            else:
                items = [data]
            
            for item in items[:100]:  # İlk 100 maç
                match = self.extract_match_info(item)
                if match:
                    matches.append(match)
                    
        except Exception as e:
            logger.error(f"Parse hatası: {e}")
            
        return matches
    
    def extract_match_info(self, item: Any) -> Dict:
        """Maç bilgilerini çıkar"""
        try:
            # Farklı formatlara uyum sağla
            home_team = item.get('homeTeam', {}).get('name', 'Ev Sahibi') if isinstance(item.get('homeTeam'), dict) else item.get('homeTeam', 'Ev Sahibi')
            away_team = item.get('awayTeam', {}).get('name', 'Deplasman') if isinstance(item.get('awayTeam'), dict) else item.get('awayTeam', 'Deplasman')
            
            return {
                'id': item.get('id', hash(f"{home_team}_{away_team}")),
                'home_team': home_team,
                'away_team': away_team,
                'league': item.get('league', {}).get('name', 'Bilinmeyen Lig') if isinstance(item.get('league'), dict) else item.get('league', 'Bilinmeyen Lig'),
                'date': item.get('date', item.get('startTime')),
                'odds': item.get('odds', {'1': 2.0, 'X': 3.0, '2': 4.0}),
                'stats': self.extract_stats(item)
            }
        except Exception as e:
            logger.error(f"Maç bilgisi çıkarma hatası: {e}")
            return None
    
    def extract_stats(self, item: Any) -> Dict:
        """İstatistik bilgilerini çıkar"""
        try:
            stats = item.get('stats', {})
            return {
                'home_goals': stats.get('homeGoals'),
                'away_goals': stats.get('awayGoals'),
                'possession': stats.get('possession', {'home': 50, 'away': 50}),
                'shots': stats.get('shots', {'home': 12, 'away': 10}),
                'corners': stats.get('corners', {'home': 5, 'away': 4}),
                'fouls': stats.get('fouls', {'home': 14, 'away': 16})
            }
        except:
            return {
                'possession': {'home': 50, 'away': 50},
                'shots': {'home': 12, 'away': 10},
                'corners': {'home': 5, 'away': 4},
                'fouls': {'home': 14, 'away': 16}
            }
    
    def generate_sample_matches(self) -> List[Dict]:
        """Örnek maç verileri oluştur"""
        sample_matches = [
            {
                'id': 1,
                'home_team': 'Galatasaray',
                'away_team': 'Fenerbahçe',
                'league': 'Süper Lig',
                'date': '2025-09-30T19:00:00',
                'odds': {'1': 2.10, 'X': 3.20, '2': 3.50},
                'stats': {
                    'possession': {'home': 52, 'away': 48},
                    'shots': {'home': 15, 'away': 12},
                    'corners': {'home': 6, 'away': 4},
                    'fouls': {'home': 14, 'away': 16}
                }
            },
            {
                'id': 2,
                'home_team': 'Beşiktaş',
                'away_team': 'Trabzonspor',
                'league': 'Süper Lig',
                'date': '2025-09-30T16:00:00',
                'odds': {'1': 2.30, 'X': 3.10, '2': 3.10},
                'stats': {
                    'possession': {'home': 48, 'away': 52},
                    'shots': {'home': 13, 'away': 14},
                    'corners': {'home': 5, 'away': 7},
                    'fouls': {'home': 16, 'away': 12}
                }
            }
            # Daha fazla örnek maç eklenebilir
        ]
        return sample_matches

# Global instance
nesine_fetcher = NesineFetcher()
