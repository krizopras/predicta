import pandas as pd
import logging
import json
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class AutoLeagueMapper:
    """
    Takım isimlerinden otomatik lig tespiti sistemi.
    merged_all.csv'deki geçmiş verilere dayanır.
    """
    def __init__(self, data_csv_path: str = "data/merged_all.csv", save_path: str = "data/team_league_map.json"):
        self.data_csv_path = data_csv_path
        self.save_path = save_path
        self.team_to_league: Dict[str, str] = {}
        self._build_map()

    def _build_map(self) -> None:
        """Geçmiş veriden takım → lig eşleştirmesi oluşturur"""
        # Önce kayıtlı map'i yükle
        if self._load_map():
            return
            
        if not os.path.exists(self.data_csv_path):
            logger.warning(f"⚠️ CSV bulunamadı: {self.data_csv_path}")
            return

        try:
            df = pd.read_csv(self.data_csv_path)
            df.columns = [c.lower().strip() for c in df.columns]

            # Sütunları bul
            league_col = next((c for c in df.columns if "lig" in c or "league" in c), None)
            home_col = next((c for c in df.columns if "home" in c or "ev" in c), None)
            away_col = next((c for c in df.columns if "away" in c or "dep" in c or "deplasman" in c), None)

            if not all([league_col, home_col, away_col]):
                logger.error("❌ Gerekli sütunlar bulunamadı (lig, ev, deplasman).")
                return

            # Takım-lig eşleştirmelerini oluştur
            team_league_counts = {}
            
            for _, row in df.iterrows():
                lig = str(row[league_col]).strip()
                if not lig or lig.lower() in ["nan", "unknown", ""]:
                    continue
                    
                home_team = str(row[home_col]).strip()
                away_team = str(row[away_col]).strip()
                
                if home_team and home_team.lower() != "nan":
                    key = home_team.lower()
                    if key not in team_league_counts:
                        team_league_counts[key] = {}
                    team_league_counts[key][lig] = team_league_counts[key].get(lig, 0) + 1
                
                if away_team and away_team.lower() != "nan":
                    key = away_team.lower()
                    if key not in team_league_counts:
                        team_league_counts[key] = {}
                    team_league_counts[key][lig] = team_league_counts[key].get(lig, 0) + 1

            # En çok görülen ligi seç
            for team, leagues in team_league_counts.items():
                most_common_league = max(leagues.items(), key=lambda x: x[1])[0]
                self.team_to_league[team] = most_common_league

            logger.info(f"✅ {len(self.team_to_league)} takım için lig eşleştirmesi oluşturuldu.")
            self._save_map()
            
        except Exception as e:
            logger.error(f"❌ Takım-lig haritası oluşturma hatası: {e}")

    def _load_map(self) -> bool:
        """Kayıtlı map'i yükle"""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, "r", encoding="utf-8") as f:
                    self.team_to_league = json.load(f)
                logger.info(f"✅ Takım-lig haritası yüklendi: {len(self.team_to_league)} takım")
                return True
            except Exception as e:
                logger.warning(f"⚠️ Harita yükleme hatası: {e}")
        return False

    def _save_map(self) -> None:
        """Eşleştirmeyi JSON olarak kaydet"""
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump(self.team_to_league, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Takım-lig haritası kaydedildi: {self.save_path}")
        except Exception as e:
            logger.error(f"❌ Harita kaydetme hatası: {e}")

    def detect_league(self, home_team: str, away_team: str) -> str:
        """
        Verilen takımlara göre lig tahmin eder
        
        Args:
            home_team: Ev sahibi takım
            away_team: Deplasman takımı
            
        Returns:
            Tespit edilen lig veya "Bilinmeyen Lig"
        """
        if not self.team_to_league:
            self._load_map()

        home = home_team.lower().strip()
        away = away_team.lower().strip()

        # Önce ev sahibi takıma bak
        if home in self.team_to_league:
            league = self.team_to_league[home]
            logger.debug(f"🔍 Lig tespit edildi: '{home_team}' → '{league}'")
            return league
        
        # Sonra deplasman takımına bak
        if away in self.team_to_league:
            league = self.team_to_league[away]
            logger.debug(f"🔍 Lig tespit edildi: '{away_team}' → '{league}'")
            return league
        
        # Kısmi eşleşme kontrolü
        for team_name, league in self.team_to_league.items():
            if team_name in home or home in team_name:
                logger.debug(f"🔍 Kısmi eşleşme: '{home_team}' → '{league}'")
                return league
            if team_name in away or away in team_name:
                logger.debug(f"🔍 Kısmi eşleşme: '{away_team}' → '{league}'")
                return league
        
        logger.warning(f"⚠️ Lig tespit edilemedi: '{home_team}' vs '{away_team}'")
        return "Bilinmeyen Lig"

    def refresh_map(self) -> None:
        """Haritayı manuel olarak yenile"""
        logger.info("🔄 Takım-lig haritası yenileniyor...")
        self.team_to_league = {}
        self._build_map()
        logger.info(f"✅ Harita güncellendi: {len(self.team_to_league)} takım")

    def get_team_info(self, team_name: str) -> Dict[str, str]:
        """Takım bilgilerini döndürür"""
        team_lower = team_name.lower().strip()
        league = self.team_to_league.get(team_lower, "Bilinmeyen Lig")
        
        return {
            "team": team_name,
            "league": league,
            "is_known": league != "Bilinmeyen Lig"
        }


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("🧪 AUTO LEAGUE MAPPER TEST")
    print("=" * 50)
    
    mapper = AutoLeagueMapper("data/merged_all.csv")
    
    test_cases = [
        ("Manchester City", "Liverpool"),
        ("Barcelona", "Real Madrid"),
        ("Grorud IL", "Follo"),
        ("Unknown Team", "Another Unknown"),
        ("", "Bayern Munich")
    ]
    
    for home, away in test_cases:
        league = mapper.detect_league(home, away)
        print(f"🏠 {home or 'N/A'} vs ✈️ {away or 'N/A'} → 🏆 {league}")