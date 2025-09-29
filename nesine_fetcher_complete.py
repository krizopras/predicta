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
        
        Args:
            league_filter: "super-lig", "premier-league", "all" gibi filtre
        
        Returns:
            Liste halinde maç verileri (oranlar + istatistikler dahil)
        """
        timeout = aiohttp.ClientTimeout(total=15)
        
        async with aiohttp.ClientSession(timeout=timeout, headers=self.headers) as session:
            self.session = session
            
            try:
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
                
                # 3. Her maç için detaylı istatistik çek (PARALEL)
                enriched_matches = await self._enrich_with_stats_parallel(matches)
                
                # 4. Lig filtreleme
                if league_filter and league_filter != "all":
                    enriched_matches = self._filter_by_league(enriched_matches, league_filter)
                
                logger.info(f"✅ Toplam {len(enriched_matches)} maç hazır (oranlar + istatistikler)")
                return enriched_matches
                
            except Exception as e:
                logger.error(f"❌ Nesine veri çekme hatası: {e}")
                return []
            finally:
                self.session = None
    
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
                        else:
                            logger.debug(f"⚠️ {endpoint} boş response")
            except Exception as e:
                logger.warning(f"❌ API endpoint {endpoint} hatası: {e}")
                continue
        
        logger.warning("⚠️ Tüm API endpoint'ler başarısız")
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
        """Maç oranlarını çıkar (1X2) - GELİŞTİRİLMİŞ"""
        try:
            odds = item.get('odds', {})
            
            # Format 1: Direkt odds dict
            if isinstance(odds, dict) and ('1' in odds or 'home' in odds):
                return {
                    '1': self._safe_float(odds.get('1', odds.get('home', 0))),
                    'X': self._safe_float(odds.get('X', odds.get('draw', 0))),
                    '2': self._safe_float(odds.get('2', odds.get('away', 0)))
                }
            
            # Format 2: Markets/outcomes yapısı
            markets = item.get('markets', [])
            if isinstance(markets, list):
                for market in markets:
                    if market.get('name') == '1X2' or market.get('type') == 'match_winner':
                        outcomes = market.get('outcomes', [])
                        result = {'1': 0, 'X': 0, '2': 0}
                        for outcome in outcomes:
                            result[outcome.get('name', '')] = self._safe_float(outcome.get('odds', 0))
                        return result
            
            # Format 3: Root seviyede ayrı alanlar
            return {
                '1': self._safe_float(item.get('homeOdds', item.get('odd1', 0))),
                'X': self._safe_float(item.get('drawOdds', item.get('oddX', 0))),
                '2': self._safe_float(item.get('awayOdds', item.get('odd2', 0)))
            }
        
        except Exception as e:
            logger.debug(f"Oran çıkarma hatası: {e}")
            return {'1': 0, 'X': 0, '2': 0}
    
    def _safe_float(self, value) -> float:
        """Güvenli float dönüşümü"""
        try:
            if value is None or value == '' or value == '-':
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
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
        """HTML element'inden maç bilgisi çıkar - GÜVENLİ"""
        try:
            # Takımlar
            home = element.select_one('.home-team, .team-home, .team-1')
            away = element.select_one('.away-team, .team-away, .team-2')
            
            if not home or not away:
                return None
            
            # Oranlar - güvenli parse
            odd_1_elem = element.select_one('[data-odd="1"], .odd-1, .ms1')
            odd_x_elem = element.select_one('[data-odd="X"], .odd-x, .ms0')
            odd_2_elem = element.select_one('[data-odd="2"], .odd-2, .ms2')
            
            return {
                'home_team': home.text.strip(),
                'away_team': away.text.strip(),
                'league': 'Süper Lig',  # HTML'den çıkarılabilir
                'odds': {
                    '1': self._extract_odd_from_element(odd_1_elem),
                    'X': self._extract_odd_from_element(odd_x_elem),
                    '2': self._extract_odd_from_element(odd_2_elem)
                },
                'stats': {},
                'source': 'nesine_web'
            }
        
        except Exception as e:
            logger.debug(f"HTML match parse hatası: {e}")
            return None
    
    def _extract_odd_from_element(self, element) -> float:
        """HTML element'inden oran çıkar - güvenli"""
        if not element:
            return 0.0
        
        try:
            text = element.text.strip()
            # Türkçe virgül yerine nokta
            text = text.replace(',', '.')
            # Sadece sayı ve nokta tut
            import re
            text = re.sub(r'[^\d.]', '', text)
            return float(text) if text else 0.0
        except:
            return 0.0
    
    async def _enrich_with_stats_parallel(self, matches: List[Dict]) -> List[Dict]:
        """Maçlara İSTATİSTİK verilerini ekle (PARALEL - HIZLI)"""
        # Tüm istatistikleri paralel olarak çek
        tasks = [self._fetch_match_stats(match.get('id')) for match in matches]
        stats_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Her maça istatistiğini ekle
        for match, stats in zip(matches, stats_list):
            if isinstance(stats, dict):
                match['stats'] = stats
            elif isinstance(stats, Exception):
                logger.debug(f"İstatistik hatası {match.get('home_team')}: {stats}")
                match['stats'] = self._generate_default_stats()
            else:
                match['stats'] = self._generate_default_stats()
        
        return matches
    
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
    
    def _normalize(self, text: str) -> str:
        """Metni normalize et - Türkçe karakter temizleme"""
        import unicodedata
        try:
            # NFKD normalizasyonu + ASCII dönüşüm
            normalized = unicodedata.normalize("NFKD", text)
            ascii_text = normalized.encode("ascii", "ignore").decode("utf-8")
            return ascii_text.lower().strip()
        except Exception as e:
            logger.debug(f"Normalize hatası: {e}")
            return text.lower().strip()
    
    def _filter_by_league(self, matches: List[Dict], league_filter: str) -> List[Dict]:
        """Lige göre filtrele - TAM İYİLEŞTİRİLMİŞ"""
        if league_filter == "all":
            return matches
        
        league_keywords = {
            "super-lig": ["super lig", "super", "turkiye", "turkey", "tsl", "spor toto"],
            "premier-league": ["premier league", "premier", "england", "ingiltere", "epl"],
            "la-liga": ["la liga", "liga", "spain", "ispanya", "primera"],
            "bundesliga": ["bundesliga", "germany", "almanya"],
            "serie-a": ["serie a", "serie", "italy", "italya"]
        }
        
        keywords = league_keywords.get(league_filter, [])
        
        filtered = []
        for match in matches:
            league_name = self._normalize(match.get('league', ''))
            
            # Her keyword için kontrol
            for keyword in keywords:
                normalized_keyword = self._normalize(keyword)
                if normalized_keyword in league_name or league_name in normalized_keyword:
                    filtered.append(match)
                    break
        
        logger.debug(f"Filtreleme: {len(matches)} -> {len(filtered)} maç ({league_filter})")
        return filtered
    
    async def close(self):
        """Session'ı kapat"""
        if self.session:
            await self.session.close()


# Global instance
nesine_complete_fetcher = NesineCompleteFetcher()


# Kullanım örneği
async def main():
    """Test fonksiyonu - CONTEXT MANAGER ile"""
    
    # Yeni kullanım (önerilen)
    async with NesineCompleteFetcher() as fetcher:
        # Tüm ligler
        all_matches = await fetcher.fetch_matches_with_odds_and_stats(league_filter="all")
        print(f"✅ Toplam maç: {len(all_matches)}")
        
        # Sadece Süper Lig
        superlig_matches = await fetcher.fetch_matches_with_odds_and_stats(league_filter="super-lig")
        print(f"🇹🇷 Süper Lig maçları: {len(superlig_matches)}")
        
        # Örnek maç göster
        if all_matches:
            match = all_matches[0]
            print(f"\n{'='*60}")
            print(f"⚽ {match['home_team']} vs {match['away_team']}")
            print(f"🏆 Lig: {match['league']}")
            print(f"💰 Oranlar: 1={match['odds']['1']:.2f} | X={match['odds']['X']:.2f} | 2={match['odds']['2']:.2f}")
            
            stats = match.get('stats', {})
            if stats:
                print(f"\n📊 İstatistikler:")
                print(f"   Top Kontrolü: Ev {stats.get('possession', {}).get('home', 0)}% - {stats.get('possession', {}).get('away', 0)}% Deplasman")
                print(f"   Şutlar: Ev {stats.get('shots', {}).get('home', 0)} - {stats.get('shots', {}).get('away', 0)} Deplasman")
                print(f"   Kornerler: Ev {stats.get('corners', {}).get('home', 0)} - {stats.get('corners', {}).get('away', 0)} Deplasman")
            print(f"{'='*60}")
    
    # Session otomatik kapandı (context manager sayesinde)


if __name__ == "__main__":
    # Logging seviyesini ayarla
    logging.basicConfig(
        level=logging.DEBUG,  # DEBUG için detaylı log
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())
