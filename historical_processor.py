import os
import json
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class HistoricalDataProcessor:
    def __init__(self, raw_data_path: str = "data/raw"):
        self.raw_data_path = raw_data_path
        self.processed_data = {}
        
    def load_country_data(self, country: str) -> List[Dict]:
        """Ülke verilerini yükle"""
        country_path = os.path.join(self.raw_data_path, country)
        if not os.path.exists(country_path):
            return []
            
        matches = []
        # CSV dosyalarını oku
        for file in os.listdir(country_path):
            if file.endswith('.csv'):
                file_path = os.path.join(country_path, file)
                try:
                    df = pd.read_csv(file_path)
                    # Temel sütunları kontrol et
                    required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
                    if all(col in df.columns for col in required_cols):
                        for _, row in df.iterrows():
                            match = {
                                'date': row['Date'],
                                'home_team': row['HomeTeam'],
                                'away_team': row['AwayTeam'],
                                'home_score': row['FTHG'],
                                'away_score': row['FTAG'],
                                'country': country,
                                'league': file.replace('.csv', '')
                            }
                            matches.append(match)
                except Exception as e:
                    logger.warning(f"{file_path} okunamadı: {e}")
                    
        return matches
    
    def calculate_team_stats(self, matches: List[Dict]) -> Dict[str, Any]:
        """Takım istatistiklerini hesapla"""
        team_stats = {}
        
        for match in matches:
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Ev sahibi takım istatistikleri
            if home_team not in team_stats:
                team_stats[home_team] = self._init_team_stats()
                
            # Deplasman takım istatistikleri  
            if away_team not in team_stats:
                team_stats[away_team] = self._init_team_stats()
                
            # Maç sonucunu güncelle
            self._update_team_stats(team_stats[home_team], match, is_home=True)
            self._update_team_stats(team_stats[away_team], match, is_home=False)
            
        return team_stats
    
    def _init_team_stats(self) -> Dict:
        """Takım istatistikleri için boş şablon"""
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
            'last_5_goals_conceded': []
        }
    
    def _update_team_stats(self, stats: Dict, match: Dict, is_home: bool):
        """Takım istatistiklerini güncelle"""
        stats['matches_played'] += 1
        
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
    
    def process_all_countries(self):
        """Tüm ülke verilerini işle"""
        countries = [d for d in os.listdir(self.raw_data_path) 
                    if os.path.isdir(os.path.join(self.raw_data_path, d))]
        
        all_matches = []
        all_team_stats = {}
        
        for country in countries:
            logger.info(f"📊 {country} verileri işleniyor...")
            matches = self.load_country_data(country)
            all_matches.extend(matches)
            
            if matches:
                team_stats = self.calculate_team_stats(matches)
                all_team_stats.update(team_stats)
                
        # İşlenmiş verileri kaydet
        self._save_processed_data(all_matches, all_team_stats)
        
        logger.info(f"✅ {len(all_matches)} maç ve {len(all_team_stats)} takım işlendi")
        return all_matches, all_team_stats
    
    def _save_processed_data(self, matches: List[Dict], team_stats: Dict):
        """İşlenmiş verileri kaydet"""
        # Maçları kaydet
        with open('data/historical/processed_matches.json', 'w', encoding='utf-8') as f:
            json.dump(matches, f, ensure_ascii=False, indent=2)
            
        # Takım istatistiklerini kaydet
        with open('data/team_stats/team_statistics.json', 'w', encoding='utf-8') as f:
            json.dump(team_stats, f, ensure_ascii=False, indent=2)
