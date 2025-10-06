# raw_data_parser.py oluşturalım
import os
import re
from datetime import datetime
from typing import List, Dict

class HistoricalDataParser:
    def __init__(self, raw_data_dir="./raw"):
        self.raw_data_dir = raw_data_dir
        self.historical_matches = []
    
    def parse_football_data_file(self, filepath: str) -> List[Dict]:
        """Football-Data.org formatındaki txt dosyalarını parse eder"""
        matches = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Maç satırlarını bul (tarih, takımlar, skor)
        match_pattern = r'(\w{3}\s+\w{3}/\d+(?:/\d+)?)\s+([\d:]+)\s+(.*?)\s+v\s+(.*?)\s+([\d-]+)\s*\(([\d-]+)\)'
        
        matches_found = re.findall(match_pattern, content)
        
        for match in matches_found:
            date_str, time_str, home_team, away_team, final_score, ht_score = match
            
            try:
                # Skoru parse et
                home_goals, away_goals = map(int, final_score.split('-'))
                
                match_data = {
                    'date': self._parse_date(date_str),
                    'time': time_str,
                    'home_team': home_team.strip(),
                    'away_team': away_team.strip(),
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    'result': '1' if home_goals > away_goals else ('X' if home_goals == away_goals else '2')
                }
                
                matches.append(match_data)
            except:
                continue
        
        return matches
    
    def _parse_date(self, date_str: str) -> str:
        """Tarihi standart formata çevir"""
        # Basitleştirilmiş - gerçek uygulamada daha kapsamlı olmalı
        return date_str  
    
    def load_all_historical_data(self) -> Dict[str, List[Dict]]:
        """Tüm raw verilerini yükle"""
        data_by_league = {}
        
        for country_folder in os.listdir(self.raw_data_dir):
            folder_path = os.path.join(self.raw_data_dir, country_folder)
            
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith('.txt'):
                        filepath = os.path.join(folder_path, filename)
                        matches = self.parse_football_data_file(filepath)
                        
                        league_name = f"{country_folder}_{filename.replace('.txt', '')}"
                        data_by_league[league_name] = matches
        
        return data_by_league
