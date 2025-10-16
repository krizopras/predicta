#!/usr/bin/env python3
"""
Geçmiş Maç Verilerini İşleme + Football.DB Kulüp Desteği
----------------------------------------------------------
- CSV / TXT / football.db formatlarını destekler.
- Ayrıca data/clubs klasöründeki kulüp dosyalarından alias (eş anlamlı isim) verisi yükler.
"""

import os
import json
import pandas as pd
from datetime import datetime
import logging
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import glob

logger = logging.getLogger(__name__)

class HistoricalDataProcessor:
    def __init__(self, raw_data_dir: str = None, raw_data_path: str = "data/raw", clubs_path: str = "data/clubs"):
    """
    HistoricalDataProcessor (Backward Compatible)
    ---------------------------------------------
    Hem 'raw_data_dir' hem 'raw_data_path' parametrelerini destekler.
    """
    self.raw_data_path = raw_data_dir or raw_data_path or "data/raw"
    self.clubs_path = clubs_path or "data/clubs"
    self.club_aliases = self._load_club_aliases()
    self.processed_data = {}


    # ==========================================================
    # 1️⃣  KULÜP ALIAS YÜKLEME
    # ==========================================================
    def _load_club_aliases(self) -> Dict[str, str]:
        """data/clubs klasöründeki tüm ülke kulüp dosyalarını okuyarak alias haritası oluşturur."""
        aliases = {}
        if not os.path.exists(self.clubs_path):
            logger.warning(f"⚠️ Kulüp klasörü bulunamadı: {self.clubs_path}")
            return aliases

        files = glob.glob(os.path.join(self.clubs_path, "*.txt"))
        for file in files:
            country_code = os.path.basename(file).split(".")[0].upper()
            try:
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("#")]
                current_team = None
                for line in lines:
                    if not line.startswith("|"):
                        # Ana isim
                        current_team = line.split(",")[0].strip()
                        aliases[current_team.lower()] = current_team
                    else:
                        alias = line.strip("| ").split("#")[0].strip().lower()
                        if current_team:
                            aliases[alias] = current_team
                logger.info(f"✅ {country_code}: {len(lines)} satırdan alias yüklendi")
            except Exception as e:
                logger.warning(f"❌ {file} okunamadı: {e}")
        logger.info(f"🎯 Toplam {len(aliases)} kulüp alias yüklendi")
        return aliases

    # ==========================================================
    # 2️⃣  VERİ YÜKLEME (ÜLKE DÜZEYİNDE)
    # ==========================================================
    def load_country_data(self, country: str) -> List[Dict]:
        """Ülke verilerini yükle (CSV, TXT, football.db formatları desteklenir)"""
        country_path = os.path.join(self.raw_data_path, country)
        if not os.path.exists(country_path):
            logger.warning(f"❌ {country} klasörü bulunamadı: {country_path}")
            return []
        
        matches = []
        files = list(Path(country_path).glob("*"))
        logger.info(f"📂 {country}: {len(files)} dosya bulundu")
        
        # CSV dosyaları
        for file in Path(country_path).glob("*.csv"):
            matches += self._read_csv_file(file, country)
        
        # TXT dosyaları
        for file in Path(country_path).glob("*.txt"):
            matches += self._read_txt_file(file, country)
        
        return matches

    # ==========================================================
    # 3️⃣  TXT / FOOTBALL.DB FORMAT AYIRT ETME
    # ==========================================================
    def _read_txt_file(self, file_path: Path, country: str) -> List[Dict]:
        """TXT dosyasını oku ve football.db olup olmadığını tespit et"""
        matches = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if self._is_football_db_format(content):
                matches = self._parse_football_db(content, file_path, country)
            else:
                matches = self._parse_standard_txt(file_path, country)
        except Exception as e:
            logger.error(f"TXT okuma hatası ({file_path.name}): {e}")
        return matches

    def _is_football_db_format(self, content: str) -> bool:
        return any(k in content[:500] for k in ["Round", "»", "Date", "Matches", "="])

    # ==========================================================
    # 4️⃣  FOOTBALL.DB PARSER
    # ==========================================================
    def _parse_football_db(self, content: str, file_path: Path, country: str) -> List[Dict]:
        matches = []
        lines = content.splitlines()
        league_name, current_date = None, None

        date_pattern = re.compile(r'^\s*\w{3} \w{3}/\d{1,2}')
        match_pattern = re.compile(r'^\s*\d{1,2}[:.]\d{2}\s+(.+?)\s+v\s+(.+?)\s+(\d+)-(\d+)')

        for line in lines:
            if line.startswith('='):
                league_name = line.strip('= ').strip()
                continue
            if date_pattern.match(line):
                current_date = self._parse_football_db_date(line.strip())
                continue
            match = match_pattern.match(line)
            if match:
                home_team = self._normalize_team_name(match.group(1))
                away_team = self._normalize_team_name(match.group(2))
                hg, ag = int(match.group(3)), int(match.group(4))
                matches.append({
                    "date": current_date or "2024-01-01",
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": hg,
                    "away_score": ag,
                    "country": country,
                    "league": league_name or file_path.stem
                })
        return matches

    def _parse_football_db_date(self, line: str) -> str:
        try:
            parts = line.split()
            if len(parts) >= 2 and "/" in parts[1]:
                month, day = parts[1].split("/")
                from datetime import date
                year = date.today().year
                month_map = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
                             'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
                return f"{year}-{month_map.get(month,1):02d}-{int(day):02d}"
        except:
            pass
        return datetime.now().strftime("%Y-%m-%d")

    # ==========================================================
    # 5️⃣  STANDART TXT / CSV PARSER
    # ==========================================================
    def _parse_standard_txt(self, file_path: Path, country: str) -> List[Dict]:
        try:
            df = pd.read_csv(file_path, sep=None, engine="python", encoding="utf-8")
        except:
            return []
        return self._convert_df(df, file_path, country)

    def _read_csv_file(self, file_path: Path, country: str) -> List[Dict]:
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except:
            return []
        return self._convert_df(df, file_path, country)

    def _convert_df(self, df: pd.DataFrame, file_path: Path, country: str) -> List[Dict]:
        df.columns = [c.lower().strip() for c in df.columns]
        matches = []
        for _, row in df.iterrows():
            home = self._normalize_team_name(str(row.get("home_team", "")))
            away = self._normalize_team_name(str(row.get("away_team", "")))
            hg = int(row.get("home_score", 0))
            ag = int(row.get("away_score", 0))
            matches.append({
                "date": row.get("date", ""),
                "home_team": home,
                "away_team": away,
                "home_score": hg,
                "away_score": ag,
                "country": country,
                "league": file_path.stem
            })
        return matches

    # ==========================================================
    # 6️⃣  TAKIM ADI NORMALİZASYONU
    # ==========================================================
    def _normalize_team_name(self, name: str) -> str:
        if not name:
            return name
        key = name.lower().strip()
        return self.club_aliases.get(key, name)

# ==========================================================
# TEST (manuel kullanım için)
# ==========================================================
if __name__ == "__main__":
    proc = HistoricalDataProcessor()
    all_matches = proc.load_country_data("andorra")
    print(f"Toplam {len(all_matches)} maç yüklendi.")
