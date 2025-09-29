#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nesine.com'dan güncel maçları çeken modül
"""

import aiohttp
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class NesineMatchFetcher:
    """Nesine.com'dan maç verilerini çeken sınıf"""
    
    def __init__(self):
        self.base_url = "https://cdnbulten.nesine.com/api/bulten"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Referer': 'https://www.nesine.com/'
        }
    
    async def fetch_prematch_matches(self) -> List[Dict]:
        """Canlı olmayan (prematch) maçları çek"""
        try:
            url = f"{self.base_url}/getprebultenfull"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_nesine_data(data)
                    else:
                        logger.warning(f"Nesine API yanıt hatası: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Nesine veri çekme hatası: {e}")
            return []
    
    def _parse_nesine_data(self, data: Dict) -> List[Dict]:
        """Nesine API yanıtını parse et"""
        matches = []
        
        try:
            # Nesine API yapısı: sg -> EA (events array)
            events = data.get("sg", {}).get("EA", [])
            
            for event in events:
                # Sadece futbol (GT=1) maçlarını al
                if event.get("GT") != 1:
                    continue
                
                # Maç bilgilerini çıkar
                match_info = self._extract_match_info(event)
                
                if match_info and self._is_upcoming_match(match_info.get('date')):
                    matches.append(match_info)
            
            logger.info(f"Nesine'den {len(matches)} maç alındı")
            return matches[:50]  # İlk 50 maç
            
        except Exception as e:
            logger.error(f"Nesine data parsing hatası: {e}")
            return []
    
    def _extract_match_info(self, event: Dict) -> Optional[Dict]:
        """Tek bir maç verisini çıkar"""
        try:
            home_team = event.get("HN", "").strip()
            away_team = event.get("AN", "").strip()
            
            if not home_team or not away_team:
                return None
            
            # Oranları çıkar
            odds = self._extract_odds(event.get("MA", []))
            
            # Tarih ve saat
            match_date = event.get("D", "")
            match_time = event.get("T", "")
            
            return {
                'match_code': str(event.get("C", "")),
                'home_team': home_team,
                'away_team': away_team,
                'league': event.get("LC", "Bilinmeyen Lig"),
                'date': match_date,
                'time': match_time,
                'odds': odds,
                'match_status': event.get("S", 0)  # 0: bekliyor, 1: canlı vs.
            }
            
        except Exception as e:
            logger.debug(f"Maç bilgisi çıkarma hatası: {e}")
            return None
    
    def _extract_odds(self, markets: List[Dict]) -> Dict:
        """Maç oranlarını çıkar"""
        odds = {'1': 0, 'X': 0, '2': 0}
        
        try:
            for market in markets:
                # MTID=1 -> Maç Sonucu
                if market.get("MTID") == 1:
                    outcomes = market.get("OCA", [])
                    
                    if len(outcomes) >= 3:
                        odds['1'] = float(outcomes[0].get("O", 0))
                        odds['X'] = float(outcomes[1].get("O", 0))
                        odds['2'] = float(outcomes[2].get("O", 0))
                    break
                    
        except Exception as e:
            logger.debug(f"Oran çıkarma hatası: {e}")
        
        return odds
    
    def _is_upcoming_match(self, match_date: str) -> bool:
        """Maçın önümüzdeki günlerde mi olduğunu kontrol et"""
        try:
            if not match_date:
                return True
            
            # Format: YYYY-MM-DD
            match_dt = datetime.strptime(match_date, '%Y-%m-%d')
            today = datetime.now().date()
            match_date_obj = match_dt.date()
            
            # Bugünden itibaren 7 gün içindeki maçlar
            return today <= match_date_obj <= (today + timedelta(days=7))
            
        except Exception as e:
            logger.debug(f"Tarih kontrolü hatası: {e}")
            return True  # Hata durumunda dahil et

# Global instance
nesine_fetcher = NesineMatchFetcher()
