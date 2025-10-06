# txt_data_processor.py
import os
import pandas as pd
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
import aiofiles
import asyncio

logger = logging.getLogger(__name__)

class TXTDataProcessor:
    def __init__(self, raw_data_path: str = "data/raw"):
        self.raw_data_path = raw_data_path
        
    async def read_txt_file(self, file_path: str) -> List[str]:
        """TXT dosyasını async olarak oku"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()
            return content.splitlines()
        except Exception as e:
            logger.error(f"TXT okuma hatası {file_path}: {e}")
            return []
    
    def parse_football_data_line(self, line: str, country: str, league: str) -> Dict[str, Any]:
        """Futbol veri satırını parse et"""
        try:
            # Satırı temizle ve parçala
            parts = line.strip().split(',')
            if len(parts) < 8:
                return None
                
            # Temel maç bilgileri
            date = parts[0].strip()
            home_team = parts[1].strip()
            away_team = parts[2].strip()
            
            # Skorları al (FTHG = Full Time Home Goals, FTAG = Full Time Away Goals)
            fthg = int(parts[3]) if parts[3].strip().isdigit() else 0
            ftag = int(parts[4]) if parts[4].strip().isdigit() else 0
            
            # Maç sonucu
            if fthg > ftag:
                result = '1'
            elif fthg < ftag:
                result = '2'
            else:
                result = 'X'
                
            return {
                'date': date,
                'home_team': home_team,
                'away_team': away_team,
                'home_score': fthg,
                'away_score': ftag,
                'result': result,
                'country': country,
                'league': league,
                'full_data': line.strip()
            }
        except Exception as e:
            logger.warning(f"Satır parse hatası: {line[:50]}... - {e}")
            return None
    
    async def process_country_txt(self, country: str) -> List[Dict[str, Any]]:
        """Ülke TXT dosyalarını işle"""
        country_path = os.path.join(self.raw_data_path, country)
        if not os.path.exists(country_path):
            return []
            
        all_matches = []
        
        for file in os.listdir(country_path):
            if file.endswith('.txt'):
                file_path = os.path.join(country_path, file)
                league = file.replace('.txt', '')
                
                logger.info(f"📁 {country}/{file} işleniyor...")
                
                lines = await self.read_txt_file(file_path)
                matches = []
                
                for line in lines:
                    if line.strip() and not line.startswith('#'):  # Boş satır ve yorumları atla
                        match_data = self.parse_football_data_line(line, country, league)
                        if match_data:
                            matches.append(match_data)
                
                all_matches.extend(matches)
                logger.info(f"✅ {country}/{file}: {len(matches)} maç")
                
        return all_matches
    
    async def process_all_countries(self) -> Dict[str, Any]:
        """Tüm ülkelerin TXT dosyalarını işle"""
        if not os.path.exists(self.raw_data_path):
            logger.error(f"Raw data path bulunamadı: {self.raw_data_path}")
            return {"matches": [], "team_stats": {}}
        
        countries = [d for d in os.listdir(self.raw_data_path) 
                    if os.path.isdir(os.path.join(self.raw_data_path, d))]
        
        logger.info(f"🔄 {len(countries)} ülke işlenecek: {countries}")
        
        all_matches = []
        
        # Tüm ülkeleri paralel işle
        tasks = []
        for country in countries:
            task = self.process_country_txt(country)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        for country_matches in results:
            all_matches.extend(country_matches)
        
        # Takım istatistiklerini hesapla
        team_stats = self.calculate_team_stats(all_matches)
        
        # Sonuçları kaydet
        await self.save_processed_data(all_matches, team_stats)
        
        logger.info(f"🎯 İŞLEME TAMAMLANDI: {len(all_matches)} maç, {len(team_stats)} takım")
        
        return {
            "matches": all_matches,
            "team_stats": team_stats,
            "countries_processed": countries
        }
    
    def calculate_team_stats(self, matches: List[Dict]) -> Dict[str, Any]:
        """Takım istatistiklerini hesapla"""
        team_stats = {}
        
        for match in matches:
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Takımları başlat
            if home_team not in team_stats:
                team_stats[home_team] = self._init_team_stats()
            if away_team not in team_stats:
                team_stats[away_team] = self._init_team_stats()
            
            # İstatistikleri güncelle
            self._update_team_stats(team_stats[home_team], match, is_home=True)
            self._update_team_stats(team_stats[away_team], match, is_home=False)
        
        return team_stats
    
    def _init_team_stats(self) -> Dict[str, Any]:
        """Takım istatistikleri şablonu"""
        return {
            'matches_played': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_for': 0,
            'goals_against': 0,
            'home_matches': 0,
            'away_matches': 0,
            'home_wins': 0,
            'away_wins': 0,
            'last_5_results': [],
            'last_5_goals_scored': [],
            'last_5_goals_conceded': [],
            'leagues': set()
        }
    
    def _update_team_stats(self, stats: Dict, match: Dict, is_home: bool):
        """Takım istatistiklerini güncelle"""
        stats['matches_played'] += 1
        stats['leagues'].add(match['league'])
        
        if is_home:
            stats['home_matches'] += 1
            goals_for = match['home_score']
            goals_against = match['away_score']
        else:
            stats['away_matches'] += 1
            goals_for = match['away_score']
            goals_against = match['home_score']
        
        stats['goals_for'] += goals_for
        stats['goals_against'] += goals_against
        
        # Sonuç
        if goals_for > goals_against:
            stats['wins'] += 1
            if is_home:
                stats['home_wins'] += 1
            result = 'W'
        elif goals_for < goals_against:
            stats['losses'] += 1
            result = 'L'
        else:
            stats['draws'] += 1
            result = 'D'
        
        # Son 5 maç
        stats['last_5_results'].append(result)
        stats['last_5_goals_scored'].append(goals_for)
        stats['last_5_goals_conceded'].append(goals_against)
        
        # Son 5'i koru
        if len(stats['last_5_results']) > 5:
            stats['last_5_results'] = stats['last_5_results'][-5:]
            stats['last_5_goals_scored'] = stats['last_5_goals_scored'][-5:]
            stats['last_5_goals_conceded'] = stats['last_5_goals_conceded'][-5:]
    
    async def save_processed_data(self, matches: List[Dict], team_stats: Dict):
        """İşlenmiş verileri kaydet"""
        os.makedirs('data/processed', exist_ok=True)
        
        # Takım istatistiklerini JSON'a dönüştürülebilir yap
        serializable_stats = {}
        for team, stats in team_stats.items():
            serializable_stats[team] = stats
            serializable_stats[team]['leagues'] = list(stats['leagues'])
        
        # Async kaydet
        async with aiofiles.open('data/processed/matches.json', 'w', encoding='utf-8') as f:
            await f.write(json.dumps(matches, ensure_ascii=False, indent=2))
        
        async with aiofiles.open('data/processed/team_stats.json', 'w', encoding='utf-8') as f:
            await f.write(json.dumps(serializable_stats, ensure_ascii=False, indent=2))
        
        logger.info("💾 Veriler kaydedildi")
