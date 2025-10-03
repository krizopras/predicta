#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CollectAPI Entegrasyon Servisi
Gerçek futbol verilerini çeker
"""

import os
import logging
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class CollectAPIService:
    """
    CollectAPI'den futbol verisi çeker
    Ücretsiz plan: Günde 100 istek
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("COLLECTAPI_KEY")
        self.base_url = "https://api.collectapi.com"
        
        if not self.api_key:
            logger.warning("CollectAPI anahtarı bulunamadı. Servis kullanılamayacak.")
        
        self.headers = {
            'content-type': "application/json",
            'authorization': f"apikey {self.api_key}" if self.api_key else ""
        }
    
    def is_available(self) -> bool:
        """API kullanılabilir mi kontrol et"""
        return bool(self.api_key)
    
    def get_leagues(self) -> List[Dict]:
        """
        Mevcut ligleri listeler
        Endpoint: GET /football/leaguesList
        """
        if not self.is_available():
            logger.warning("CollectAPI anahtarı yok, boş liste döndürülüyor")
            return []
        
        try:
            response = requests.get(
                f"{self.base_url}/football/leaguesList",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                leagues = data.get('result', [])
                logger.info(f"✅ {len(leagues)} lig bulundu")
                return leagues
            else:
                logger.error(f"❌ Lig listesi hatası: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"❌ CollectAPI istek hatası: {e}")
            return []
    
    def get_league_matches(self, league: str) -> List[Dict]:
        """
        Belirli bir ligin maçlarını getirir
        
        Args:
            league: Lig kodu (örn: "super-lig", "premier-league")
        
        Endpoint: GET /football/results?data.league={league}
        """
        if not self.is_available():
            return []
        
        try:
            response = requests.get(
                f"{self.base_url}/football/results",
                headers=self.headers,
                params={"data.league": league},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                matches = data.get('result', [])
                logger.info(f"✅ {league}: {len(matches)} maç bulundu")
                return matches
            else:
                logger.error(f"❌ Maç listesi hatası ({league}): {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"❌ CollectAPI istek hatası: {e}")
            return []
    
    def get_team_stats(self, team_name: str, league: str) -> Optional[Dict]:
        """
        Takım istatistiklerini getirir
        
        NOT: CollectAPI'nin ücretsiz planında takım detay endpoint'i
        olmayabilir. Bu durumda maç sonuçlarından çıkarım yapılır.
        """
        if not self.is_available():
            return None
        
        # Maçları çek
        matches = self.get_league_matches(league)
        
        if not matches:
            return None
        
        # Takımın maçlarını filtrele
        team_matches = [
            m for m in matches 
            if team_name.lower() in m.get('home', '').lower() or 
               team_name.lower() in m.get('away', '').lower()
        ]
        
        if not team_matches:
            logger.warning(f"⚠️ {team_name} için maç bulunamadı")
            return None
        
        # İstatistikleri hesapla
        stats = self._calculate_team_stats(team_name, team_matches)
        return stats
    
    def _calculate_team_stats(self, team_name: str, matches: List[Dict]) -> Dict:
        """
        Maç sonuçlarından takım istatistikleri hesaplar
        """
        wins = 0
        draws = 0
        losses = 0
        goals_for = 0
        goals_against = 0
        recent_form = []
        
        for match in matches[-10:]:  # Son 10 maç
            home = match.get('home', '')
            away = match.get('away', '')
            score = match.get('score', '0-0')
            
            try:
                home_goals, away_goals = map(int, score.split('-'))
            except:
                continue
            
            is_home = team_name.lower() in home.lower()
            
            if is_home:
                goals_for += home_goals
                goals_against += away_goals
                
                if home_goals > away_goals:
                    wins += 1
                    recent_form.append('G')
                elif home_goals < away_goals:
                    losses += 1
                    recent_form.append('M')
                else:
                    draws += 1
                    recent_form.append('B')
            else:
                goals_for += away_goals
                goals_against += home_goals
                
                if away_goals > home_goals:
                    wins += 1
                    recent_form.append('G')
                elif away_goals < home_goals:
                    losses += 1
                    recent_form.append('M')
                else:
                    draws += 1
                    recent_form.append('B')
        
        matches_played = wins + draws + losses
        
        return {
            'team_name': team_name,
            'matches_played': matches_played,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'recent_form': recent_form[-5:],  # Son 5 maç
            'points': wins * 3 + draws,
            'position': 0,  # Hesaplanamıyor
            'total_teams': 20
        }
    
    def get_match_odds(self, home_team: str, away_team: str) -> Optional[Dict]:
        """
        Maç oranlarını getirir
        
        NOT: CollectAPI ücretsiz planda bahis oranı vermeyebilir.
        Bu durumda None döner.
        """
        # CollectAPI'de odds endpoint'i yoksa None döner
        logger.info(f"⚠️ CollectAPI'de oran servisi bulunamadı: {home_team} vs {away_team}")
        return None


# Test fonksiyonu
def test_collectapi():
    """CollectAPI'yi test eder"""
    import json
    
    # API anahtarını environment variable'dan al
    api_key = os.getenv("COLLECTAPI_KEY")
    
    if not api_key:
        print("❌ COLLECTAPI_KEY environment variable tanımlanmamış")
        print("Kullanım: export COLLECTAPI_KEY='apikey YOUR_KEY_HERE'")
        return
    
    service = CollectAPIService(api_key)
    
    # 1. Ligleri listele
    print("\n📋 Mevcut Ligler:")
    leagues = service.get_leagues()
    
    if leagues:
        for i, league in enumerate(leagues[:5], 1):
            print(f"{i}. {league.get('league', 'Bilinmeyen')}")
    
    # 2. Süper Lig maçları
    print("\n⚽ Süper Lig Maçları:")
    matches = service.get_league_matches("super-lig")
    
    if matches:
        for i, match in enumerate(matches[:3], 1):
            print(f"{i}. {match.get('home')} {match.get('score')} {match.get('away')}")
    
    # 3. Takım istatistikleri
    print("\n📊 Galatasaray İstatistikleri:")
    stats = service.get_team_stats("Galatasaray", "super-lig")
    
    if stats:
        print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    test_collectapi()
