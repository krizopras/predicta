#!/usr/bin/env python3
"""
Geçmiş Maç Verilerini İşleme - FOOTBALL.DB FORMAT DESTEĞİ
CSV, TXT ve football.db formatlarını destekler
"""

import os
import json
import pandas as pd
from datetime import datetime
import logging
import re
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class HistoricalDataProcessor:
    def __init__(self, raw_data_path: str = "data/raw"):
        self.raw_data_path = raw_data_path
        self.processed_data = {}
        
    def load_country_data(self, country: str) -> List[Dict]:
        """Ülke verilerini yükle (CSV, TXT, football.db)"""
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
    
    def _read_txt_file(self, file_path: Path, country: str) -> List[Dict]:
        """TXT dosyasını oku (football.db ve diğer formatlar)"""
        matches = []
        
        try:
            # Encoding'leri dene
            content = None
            for encoding in ['utf-8', 'latin-1', 'iso-8859-9', 'cp1254']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except:
                    continue
            
            if not content:
                return []
            
            # Football.db formatını dene
            if self._is_football_db_format(content):
                matches = self._parse_football_db(content, file_path, country)
            else:
                # Standart CSV/TXT formatını dene
                matches = self._parse_standard_txt(file_path, country)
            
        except Exception as e:
            logger.error(f"TXT okuma hatası ({file_path.name}): {e}")
        
        return matches
    
    def _is_football_db_format(self, content: str) -> bool:
        """Football.db formatı mı kontrol et"""
        # = başlığı, # yorum satırları, » round işaretleri
        indicators = ['=', '»', 'Round', 'Date', 'Teams', 'Matches']
        return any(indicator in content[:500] for indicator in indicators)
    
    def _parse_football_db(self, content: str, file_path: Path, country: str) -> List[Dict]:
        """Football.db formatını parse et"""
        matches = []
        lines = content.split('\n')
        
        # Metadata
        league_name = None
        current_date = None
        
        # Regex patterns
        date_pattern = re.compile(r'^  (\w{3} \w{3}/\d{1,2}|\w{3} \w{3}/\d{1,2} \d{4})')
        time_pattern = re.compile(r'^\s+(\d{1,2}[:.]\d{2})')
        match_pattern = re.compile(
            r'^\s+\d{1,2}[:.]\d{2}\s+(.+?)\s+v\s+(.+?)\s+(\d+)-(\d+)'
        )
        
        for line in lines:
            # Başlık satırı (lig ismi)
            if line.startswith('='):
                league_name = line.strip('= ').strip()
                continue
            
            # Yorum satırları
            if line.startswith('#'):
                continue
            
            # Round başlığı
            if line.startswith('»'):
                continue
            
            # Tarih satırı (Tue Oct/8 2024)
            date_match = date_pattern.match(line)
            if date_match:
                date_str = date_match.group(1).strip()
                current_date = self._parse_football_db_date(date_str)
                continue
            
            # Maç satırı
            match_match = match_pattern.match(line)
            if match_match:
                home_team = match_match.group(1).strip()
                away_team = match_match.group(2).strip()
                home_score = int(match_match.group(3))
                away_score = int(match_match.group(4))
                
                # Takım isimlerini temizle (fazla boşlukları kaldır)
                home_team = ' '.join(home_team.split())
                away_team = ' '.join(away_team.split())
                
                # Lig ismini dosya adından çıkar
                if not league_name:
                    league_name = file_path.stem.replace('_', ' ').replace('-', ' ').title()
                
                match = {
                    'date': current_date or '2024-01-01',
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': home_score,
                    'away_score': away_score,
                    'country': country,
                    'league': league_name
                }
                matches.append(match)
        
        return matches
    
    def _parse_football_db_date(self, date_str: str) -> str:
        """Football.db tarih formatını parse et (Tue Oct/8 2024)"""
        try:
            # "Tue Oct/8 2024" veya "Wed Oct/9" gibi formatlar
            parts = date_str.split()
            
            if len(parts) >= 2:
                # Ay/Gün kısmını al
                month_day = parts[1]  # "Oct/8"
                
                if '/' in month_day:
                    month_str, day_str = month_day.split('/')
                    
                    # Yıl varsa al, yoksa şu anki yılı kullan
                    if len(parts) >= 3:
                        year = int(parts[2])
                    else:
                        # Yıl belirtilmemişse şu anki yılı kullan
                        from datetime import date
                        year = date.today().year
                    
                    # Ay isimlerini sayıya çevir
                    month_map = {
                        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
                        'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                        'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                    }
                    
                    month = month_map.get(month_str, 1)
                    day = int(day_str)
                    
                    return f"{year:04d}-{month:02d}-{day:02d}"
        except Exception as e:
            logger.debug(f"Tarih parse hatası: {date_str} -> {e}")
        
        # Varsayılan: Bugünün tarihi
        from datetime import date
        return date.today().strftime('%Y-%m-%d')
    
    def _parse_standard_txt(self, file_path: Path, country: str) -> List[Dict]:
        """Standart CSV/TXT formatını parse et"""
        matches = []
        
        try:
            # Pandas ile okumayı dene
            df = None
            for encoding in ['utf-8', 'latin-1', 'iso-8859-9', 'cp1254']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=None, engine='python')
                    break
                except:
                    continue
            
            if df is None:
                return []
            
            # Sütun isimlerini normalize et
            df.columns = df.columns.str.strip().str.lower()
            
            # Sütunları eşleştir
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
            logger.error(f"Standart TXT parse hatası: {e}")
        
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
            
            # Gerekli sütunları bul
            col_map = self._map_columns(df.columns)
            
            if not col_map:
                logger.warning(f"⚠️ {file_path.name}: Gerekli sütunlar bulunamadı")
                return []
            
            # Her satırı işle
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
            logger.error(f"CSV okuma hatası ({file_path.name}): {e}")
        
        return matches
    
    def _map_columns(self, columns: List[str]) -> Optional[Dict[str, str]]:
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
            # Varsayılan: Bugünün tarihi
            from datetime import date
            return date.today().strftime('%Y-%m-%d')
        
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
        
        # Hiçbir format uymazsa bugünün tarihi
        from datetime import date
        return date.today().strftime('%Y-%m-%d')
    
    def calculate_team_stats(self, matches: List[Dict]) -> Dict[str, Any]:
        """Takım istatistiklerini hesapla"""
        team_stats = {}
        
        for match in matches:
            home_team = match['home_team']
            away_team = match['away_team']
            
            if home_team not in team_stats:
                team_stats[home_team] = self._init_team_stats()
            
            if away_team not in team_stats:
                team_stats[away_team] = self._init_team_stats()
            
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
            os.makedirs('data/historical', exist_ok=True)
            os.makedirs('data/team_stats', exist_ok=True)
            
            with open('data/historical/processed_matches.json', 'w', encoding='utf-8') as f:
                json.dump(matches, f, ensure_ascii=False, indent=2)
            
            with open('data/team_stats/team_statistics.json', 'w', encoding='utf-8') as f:
                json.dump(team_stats, f, ensure_ascii=False, indent=2)
            
            logger.info("💾 Veriler kaydedildi")
            
        except Exception as e:
            logger.error(f"Kaydetme hatası: {e}")


# Test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    processor = HistoricalDataProcessor("data/raw")
    matches, stats = processor.process_all_countries()
    
    print(f"\n{'='*60}")
    print(f"✅ {len(matches)} maç işlendi")
    print(f"✅ {len(stats)} takım bulundu")
    print(f"{'='*60}")
    
    if matches:
        print("\n📋 İlk 5 maç:")
        for i, m in enumerate(matches[:5], 1):
            print(f"{i}. {m['date']} | {m['home_team']} {m['home_score']}-{m['away_score']} {m['away_team']} ({m['league']})")
        
        print(f"\n📋 Son 3 maç:")
        for i, m in enumerate(matches[-3:], len(matches)-2):
            print(f"{i}. {m['date']} | {m['home_team']} {m['home_score']}-{m['away_score']} {m['away_team']} ({m['league']})")