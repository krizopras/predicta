#!/usr/bin/env python3
"""
Geçmiş Maç Verilerini İşleme - GÜNCELLENMİŞ VERSİYON
CSV ve TXT dosyalarını destekler
"""

import os
import json
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class HistoricalDataProcessor:
    def __init__(self, raw_data_path: str = "data/raw"):
        self.raw_data_path = raw_data_path
        self.processed_data = {}
        
    def load_country_data(self, country: str) -> List[Dict]:
        """Ülke verilerini yükle (CSV veya TXT)"""
        country_path = os.path.join(self.raw_data_path, country)
        if not os.path.exists(country_path):
            logger.warning(f"❌ {country} klasörü bulunamadı: {country_path}")
            return []
        
        matches = []
        files = list(Path(country_path).glob("*"))
        logger.info(f"📂 {country}: {len(files)} dosya bulundu")
        
        # CSV dosyalarını oku
        for file in Path(country_path).glob("*.csv"):
            try:
                matches_from_csv = self._read_csv_file(file, country)
                matches.extend(matches_from_csv)
                logger.info(f"   ✅ {file.name}: {len(matches_from_csv)} maç")
            except Exception as e:
                logger.warning(f"   ❌ {file.name} okunamadı: {e}")
        
        # TXT dosyalarını oku
        for file in Path(country_path).glob("*.txt"):
            try:
                matches_from_txt = self._read_txt_file(file, country)
                matches.extend(matches_from_txt)
                logger.info(f"   ✅ {file.name}: {len(matches_from_txt)} maç")
            except Exception as e:
                logger.warning(f"   ❌ {file.name} okunamadı: {e}")
        
        return matches
    
    def _read_csv_file(self, file_path: Path, country: str) -> List[Dict]:
        """CSV dosyasını oku"""
        matches = []
        
        try:
            # Farklı encoding'leri dene
            encodings = ['utf-8', 'latin-1', 'iso-8859-9', 'cp1254']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except:
                    continue
            
            if df is None:
                raise Exception("Dosya okunamadı (encoding sorunu)")
            
            # Sütun isimlerini normalize et
            df.columns = df.columns.str.strip().str.lower()
            
            # Gerekli sütunları bul (flexible mapping)
            col_map = self._map_columns(df.columns)
            
            if not col_map:
                logger.warning(f"⚠️ {file_path.name}: Gerekli sütunlar bulunamadı")
                return []
            
            # Her satırı işle
            for _, row in df.iterrows():
                try:
                    home_team = str(row[col_map['home']]).strip()
                    away_team = str(row[col_map['away']]).strip()
                    
                    # Boş satırları atla
                    if not home_team or not away_team or home_team == 'nan' or away_team == 'nan':
                        continue
                    
                    home_score = self._safe_int(row[col_map['home_score']])
                    away_score = self._safe_int(row[col_map['away_score']])
                    
                    # Tarih
                    date_str = str(row[col_map['date']]) if col_map['date'] else ''
                    date = self._parse_date(date_str)
                    
                    # Lig ismi (dosya adından)
                    league = file_path.stem.replace('_', ' ').title()
                    
                    match = {
                        'date': date,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': home_score,
                        'away_score': away_score,
                        'country': country,
                        'league': league
                    }
                    matches.append(match)
                    
                except Exception as e:
                    continue
            
        except Exception as e:
            logger.error(f"CSV okuma hatası ({file_path.name}): {e}")
        
        return matches
    
    def _read_txt_file(self, file_path: Path, country: str) -> List[Dict]:
        """TXT dosyasını oku (CSV formatında olduğunu varsay)"""
        matches = []
        
        try:
            # TXT'yi CSV gibi oku
            encodings = ['utf-8', 'latin-1', 'iso-8859-9', 'cp1254']
            df = None
            
            for encoding in encodings:
                try:
                    # Virgül veya tab ile ayrılmış olabilir
                    df = pd.read_csv(file_path, encoding=encoding, sep=None, engine='python')
                    break
                except:
                    continue
            
            if df is None:
                # Manuel satır satır okuma dene
                return self._read_txt_manual(file_path, country)
            
            # Sütun isimlerini normalize et
            df.columns = df.columns.str.strip().str.lower()
            
            # CSV ile aynı işlemi yap
            col_map = self._map_columns(df.columns)
            
            if not col_map:
                return []
            
            for _, row in df.iterrows():
                try:
                    home_team = str(row[col_map['home']]).strip()
                    away_team = str(row[col_map['away']]).strip()
                    
                    if not home_team or not away_team or home_team == 'nan' or away_team == 'nan':
                        continue
                    
                    home_score = self._safe_int(row[col_map['home_score']])
                    away_score = self._safe_int(row[col_map['away_score']])
                    
                    date_str = str(row[col_map['date']]) if col_map['date'] else ''
                    date = self._parse_date(date_str)
                    
                    league = file_path.stem.replace('_', ' ').title()
                    
                    match = {
                        'date': date,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': home_score,
                        'away_score': away_score,
                        'country': country,
                        'league': league
                    }
                    matches.append(match)
                    
                except Exception as e:
                    continue
            
        except Exception as e:
            logger.error(f"TXT okuma hatası ({file_path.name}): {e}")
        
        return matches
    
    def _read_txt_manual(self, file_path: Path, country: str) -> List[Dict]:
        """TXT dosyasını manuel olarak satır satır oku"""
        matches = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # İlk satır başlık olabilir
            header = lines[0].strip().lower().split(',') if lines else []
            data_lines = lines[1:] if len(lines) > 1 else []
            
            col_map = self._map_columns(header)
            
            for line in data_lines:
                parts = [p.strip() for p in line.split(',')]
                
                if len(parts) < 4:
                    continue
                
                try:
                    # Basit format: Date,HomeTeam,AwayTeam,FTHG,FTAG veya
                    # HomeTeam,AwayTeam,Score (örn: 2-1)
                    if len(parts) >= 5:
                        date = self._parse_date(parts[0])
                        home = parts[1]
                        away = parts[2]
                        home_score = self._safe_int(parts[3])
                        away_score = self._safe_int(parts[4])
                    elif len(parts) >= 3 and '-' in parts[2]:
                        # Format: Home,Away,2-1
                        home = parts[0]
                        away = parts[1]
                        score_parts = parts[2].split('-')
                        home_score = self._safe_int(score_parts[0])
                        away_score = self._safe_int(score_parts[1])
                        date = ''
                    else:
                        continue
                    
                    league = file_path.stem.replace('_', ' ').title()
                    
                    match = {
                        'date': date,
                        'home_team': home,
                        'away_team': away,
                        'home_score': home_score,
                        'away_score': away_score,
                        'country': country,
                        'league': league
                    }
                    matches.append(match)
                    
                except:
                    continue
            
        except Exception as e:
            logger.error(f"Manuel TXT okuma hatası: {e}")
        
        return matches
    
    def _map_columns(self, columns: List[str]) -> Dict[str, str]:
        """Sütun isimlerini standart isimlere eşleştir"""
        col_map = {
            'date': None,
            'home': None,
            'away': None,
            'home_score': None,
            'away_score': None
        }
        
        # Olası isimler
        date_names = ['date', 'tarih', 'matchdate', 'gamedate']
        home_names = ['hometeam', 'home', 'home_team', 'ev', 'evsahibi']
        away_names = ['awayteam', 'away', 'away_team', 'dep', 'deplasman']
        home_score_names = ['fthg', 'hg', 'home_goals', 'homescore', 'home_score']
        away_score_names = ['ftag', 'ag', 'away_goals', 'awayscore', 'away_score']
        
        for col in columns:
            col_lower = str(col).lower().strip()
            
            if any(name in col_lower for name in date_names):
                col_map['date'] = col
            elif any(name in col_lower for name in home_names):
                col_map['home'] = col
            elif any(name in col_lower for name in away_names):
                col_map['away'] = col
            elif any(name in col_lower for name in home_score_names):
                col_map['home_score'] = col
            elif any(name in col_lower for name in away_score_names):
                col_map['away_score'] = col
        
        # Gerekli alanlar var mı?
        if col_map['home'] and col_map['away'] and col_map['home_score'] and col_map['away_score']:
            return col_map
        
        return None
    
    def _safe_int(self, value) -> int:
        """Güvenli integer dönüşümü"""
        try:
            if pd.isna(value):
                return 0
            return int(float(value))
        except:
            return 0
    
    def _parse_date(self, date_str: str) -> str:
        """Tarih string'ini standart formata çevir"""
        if not date_str or date_str == 'nan':
            return '2024-01-01'
        
        # Farklı tarih formatlarını dene
        formats = [
            '%d/%m/%Y',
            '%Y-%m-%d',
            '%d.%m.%Y',
            '%d-%m-%Y',
            '%Y/%m/%d',
            '%d/%m/%y',
            '%d.%m.%y'
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(str(date_str).strip(), fmt)
                return dt.strftime('%Y-%m-%d')
            except:
                continue
        
        return '2024-01-01'
    
    def calculate_team_stats(self, matches: List[Dict]) -> Dict[str, Any]:
        """Takım istatistiklerini hesapla"""
        team_stats = {}
        
        for match in matches:
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Ev sahibi takım
            if home_team not in team_stats:
                team_stats[home_team] = self._init_team_stats()
            
            # Deplasman takım
            if away_team not in team_stats:
                team_stats[away_team] = self._init_team_stats()
            
            # Maç sonucunu güncelle
            self._update_team_stats(team_stats[home_team], match, is_home=True)
            self._update_team_stats(team_stats[away_team], match, is_home=False)
        
        return team_stats
    
    def _init_team_stats(self) -> Dict:
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
            else:
                stats['away_wins'] += 1
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
        countries = [
            d for d in os.listdir(self.raw_data_path)
            if os.path.isdir(os.path.join(self.raw_data_path, d))
        ]
        
        logger.info(f"📊 {len(countries)} ülke bulundu: {', '.join(countries)}")
        
        all_matches = []
        all_team_stats = {}
        
        for country in countries:
            logger.info(f"🔄 {country} işleniyor...")
            matches = self.load_country_data(country)
            all_matches.extend(matches)
            
            if matches:
                team_stats = self.calculate_team_stats(matches)
                all_team_stats.update(team_stats)
        
        # İşlenmiş verileri kaydet
        self._save_processed_data(all_matches, all_team_stats)
        
        logger.info(f"✅ Toplam: {len(all_matches)} maç, {len(all_team_stats)} takım")
        return all_matches, all_team_stats
    
    def _save_processed_data(self, matches: List[Dict], team_stats: Dict):
        """İşlenmiş verileri kaydet"""
        try:
            # Dizinleri oluştur
            os.makedirs('data/historical', exist_ok=True)
            os.makedirs('data/team_stats', exist_ok=True)
            
            # Maçları kaydet
            with open('data/historical/processed_matches.json', 'w', encoding='utf-8') as f:
                json.dump(matches, f, ensure_ascii=False, indent=2)
            
            # Takım istatistiklerini kaydet
            with open('data/team_stats/team_statistics.json', 'w', encoding='utf-8') as f:
                json.dump(team_stats, f, ensure_ascii=False, indent=2)
            
            logger.info("💾 Veriler kaydedildi")
            
        except Exception as e:
            logger.error(f"Kaydetme hatası: {e}")


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    processor = HistoricalDataProcessor("data/raw")
    matches, stats = processor.process_all_countries()
    
    print(f"\n{'='*60}")
    print(f"✅ {len(matches)} maç işlendi")
    print(f"✅ {len(stats)} takım bulundu")
    print(f"{'='*60}")
    
    if matches:
        print("\n📋 İlk 3 maç:")
        for i, m in enumerate(matches[:3], 1):
            print(f"{i}. {m['home_team']} {m['home_score']}-{m['away_score']} {m['away_team']} ({m['league']})")