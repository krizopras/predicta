#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nesine.com veri çekme modülü - AI Entegrasyonlu
"""

import requests
import json
import logging
import time
import re
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import trafilatura
from typing import List, Dict, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

def get_website_text_content(url: str) -> str:
    """
    Web sitesinden ana metin içeriğini çıkarır
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        return text if text else ""
    except Exception as e:
        logger.error(f"Web scraping hatası: {e}")
        return ""

class NesineDataScraper:
    
    def __init__(self):
        self.session = requests.Session()
        
        # Retry strategy - AI için optimize edilmiş
        retry_strategy = Retry(
            total=3,
            connect=2,
            read=2,
            status=2,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=1,
            raise_on_status=False
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # AI için optimize edilmiş headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'tr-TR,tr;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        self.default_timeout = (10, 30)
    
    def fetch_matches(self) -> List[Dict]:
        """Nesine'den maç verilerini çek - AI için zenginleştirilmiş"""
        try:
            # Önce API'yi dene
            api_matches = self.fetch_from_api()
            if api_matches:
                logger.info(f"🤖 API'den {len(api_matches)} maç alındı")
                return self.enrich_matches_for_ai(api_matches)
            
            # API başarısızsa web scraping dene
            web_matches = self.fetch_from_website()
            if web_matches:
                logger.info(f"🤖 Web'den {len(web_matches)} maç alındı")
                return self.enrich_matches_for_ai(web_matches)
            
            # Fallback: örnek veriler
            logger.warning("🤖 Her iki kaynak başarısız, örnek veriler kullanılıyor")
            return self.enrich_matches_for_ai(self.generate_sample_matches())
            
        except Exception as e:
            logger.error(f"❌ Veri çekme hatası: {e}")
            return self.enrich_matches_for_ai(self.generate_sample_matches())
    
    def enrich_matches_for_ai(self, matches: List[Dict]) -> List[Dict]:
        """AI için maç verilerini zenginleştir"""
        enriched_matches = []
        
        for match in matches:
            try:
                # AI için ekstra context bilgileri
                enriched_match = match.copy()
                
                # Takım istatistikleri için placeholder'lar
                enriched_match['home_stats'] = self.generate_ai_team_stats(
                    match['home_team'], match['league'], is_home=True
                )
                enriched_match['away_stats'] = self.generate_ai_team_stats(
                    match['away_team'], match['league'], is_home=False
                )
                
                # AI için context bilgileri
                enriched_match['context'] = {
                    'importance': self.calculate_match_importance(match),
                    'pressure': self.estimate_pressure_level(match),
                    'motivation': self.estimate_motivation(match),
                    'weather_impact': 1.0,  # Varsayılan
                    'venue_impact': 1.0     # Varsayılan
                }
                
                # AI feature extraction için raw data
                enriched_match['raw_features'] = self.extract_raw_features(match)
                
                enriched_matches.append(enriched_match)
                
            except Exception as e:
                logger.warning(f"⚠️ Maç zenginleştirme hatası: {e}")
                enriched_matches.append(match)  # Orijinal maçı ekle
        
        logger.info(f"🎯 AI için {len(enriched_matches)} maç zenginleştirildi")
        return enriched_matches
    
    def generate_ai_team_stats(self, team_name: str, league: str, is_home: bool) -> Dict:
        """AI için takım istatistikleri oluştur"""
        # Gerçek veri yoksa simüle edilmiş istatistikler
        strong_teams = ['Galatasaray', 'Fenerbahçe', 'Beşiktaş', 'Trabzonspor', 
                       'Real Madrid', 'Barcelona', 'Manchester City', 'Liverpool']
        
        is_strong = any(strong in team_name for strong in strong_teams)
        
        if is_strong:
            base_stats = {
                'position': max(1, min(5, hash(team_name) % 5 + 1)),
                'points': max(25, min(40, hash(team_name) % 15 + 25)),
                'matches_played': 20,
                'wins': max(8, min(15, hash(team_name) % 8 + 8)),
                'draws': max(2, min(6, hash(team_name) % 5 + 2)),
                'losses': max(1, min(5, hash(team_name) % 5 + 1)),
                'goals_for': max(20, min(35, hash(team_name) % 16 + 20)),
                'goals_against': max(8, min(18, hash(team_name) % 11 + 8)),
                'recent_form': self.generate_recent_form(is_strong),
                'total_teams': 20
            }
        else:
            base_stats = {
                'position': max(8, min(18, hash(team_name) % 11 + 8)),
                'points': max(10, min(25, hash(team_name) % 16 + 10)),
                'matches_played': 20,
                'wins': max(3, min(9, hash(team_name) % 7 + 3)),
                'draws': max(4, min(8, hash(team_name) % 5 + 4)),
                'losses': max(5, min(12, hash(team_name) % 8 + 5)),
                'goals_for': max(10, min(22, hash(team_name) % 13 + 10)),
                'goals_against': max(18, min(30, hash(team_name) % 13 + 18)),
                'recent_form': self.generate_recent_form(is_strong),
                'total_teams': 20
            }
        
        # AI için ekstra özellikler
        base_stats.update({
            'name': team_name,
            'league': league,
            'advanced_strength': self.calculate_advanced_strength(base_stats, is_home),
            'form_momentum': self.calculate_form_momentum(base_stats['recent_form']),
            'attack_power': base_stats['goals_for'] / max(base_stats['matches_played'], 1),
            'defense_strength': 1.0 / (1.0 + base_stats['goals_against'] / max(base_stats['matches_played'], 1))
        })
        
        return base_stats
    
    def generate_recent_form(self, is_strong_team: bool) -> List[str]:
        """Takım formu oluştur"""
        if is_strong_team:
            forms = [['G', 'G', 'B', 'G', 'M'], ['B', 'G', 'G', 'B', 'G'], 
                    ['G', 'M', 'G', 'G', 'B']]
        else:
            forms = [['M', 'B', 'G', 'M', 'B'], ['B', 'M', 'B', 'G', 'M'],
                    ['G', 'B', 'M', 'M', 'B']]
        
        import random
        return random.choice(forms)
    
    def calculate_advanced_strength(self, stats: Dict, is_home: bool) -> float:
        """Gelişmiş takım gücü hesapla"""
        try:
            mp = max(stats['matches_played'], 1)
            win_rate = stats['wins'] / mp
            points_per_game = stats['points'] / mp / 3.0
            goal_diff = (stats['goals_for'] - stats['goals_against']) / mp
            
            strength = (win_rate * 0.4 + points_per_game * 0.3 + goal_diff * 0.3)
            
            if is_home:
                strength *= 1.1  # Ev sahibi avantajı
            
            return max(0.1, min(1.0, strength))
        except:
            return 0.5
    
    def calculate_form_momentum(self, recent_form: List[str]) -> float:
        """Form momentum hesapla"""
        if not recent_form:
            return 0.5
        
        weights = [0.4, 0.3, 0.2, 0.07, 0.03]
        score = 0.0
        
        for i, result in enumerate(recent_form[:5]):
            weight = weights[i] if i < len(weights) else 0.01
            if result == 'G': score += 1.0 * weight
            elif result == 'B': score += 0.5 * weight
        
        return score
    
    def calculate_match_importance(self, match: Dict) -> float:
        """Maç önem derecesi"""
        big_teams = ['Galatasaray', 'Fenerbahçe', 'Beşiktaş', 'Trabzonspor']
        home_big = any(team in match['home_team'] for team in big_teams)
        away_big = any(team in match['away_team'] for team in big_teams)
        
        if home_big and away_big:
            return 1.0  # Büyük derbi
        elif home_big or away_big:
            return 0.7  # Büyük takım maçı
        else:
            return 0.4  # Normal maç
    
    def estimate_pressure_level(self, match: Dict) -> float:
        """Basınç seviyesi tahmini"""
        # Büyük takımlar ve derbilerde basınç yüksek
        derby_teams = [('Galatasaray', 'Fenerbahçe'), ('Beşiktaş', 'Fenerbahçe'), 
                      ('Galatasaray', 'Beşiktaş')]
        
        for team1, team2 in derby_teams:
            if team1 in match['home_team'] and team2 in match['away_team']:
                return 0.9
        
        return 0.5
    
    def estimate_motivation(self, match: Dict) -> float:
        """Motivasyon seviyesi tahmini"""
        # Lig liderliği, küme düşme gibi faktörler
        return 0.6  # Varsayılan
    
    def extract_raw_features(self, match: Dict) -> Dict:
        """AI için ham özellikleri çıkar"""
        return {
            'home_team_length': len(match['home_team']),
            'away_team_length': len(match['away_team']),
            'league_complexity': len(match['league']),
            'odds_variance': max(match['odds'].values()) - min(match['odds'].values()),
            'is_weekend': self.is_weekend_match(match),
            'time_category': self.categorize_match_time(match['time'])
        }
    
    def is_weekend_match(self, match: Dict) -> bool:
        """Hafta sonu maçı mı?"""
        try:
            match_date = datetime.strptime(match['date'], '%Y-%m-%d')
            return match_date.weekday() >= 5  # Cumartesi veya Pazar
        except:
            return False
    
    def categorize_match_time(self, match_time: str) -> str:
        """Maç saatini kategorize et"""
        try:
            hour = int(match_time.split(':')[0])
            if 18 <= hour <= 23:
                return 'prime_time'
            elif 12 <= hour < 18:
                return 'afternoon'
            else:
                return 'morning'
        except:
            return 'unknown'
    
    # AŞAĞIDAKİ FONKSİYONLAR ORİJİNAL HALİYLE KALACAK
    # Sadece log mesajları AI temalı yapıldı
    
    def fetch_from_api(self):
        """Nesine API'den veri çek - gelişmiş error handling ile"""
        try:
            url = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"
            
            logger.debug(f"🤖 API isteği gönderiliyor: {url}")
            response = self.session.get(url, timeout=self.default_timeout)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    return self.process_api_data(data)
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON parse hatası: {e}")
                    return []
            elif response.status_code == 429:
                logger.warning("⏳ Rate limit aşıldı, 5 saniye bekleniyor...")
                time.sleep(5)
                return []
            else:
                logger.warning(f"⚠️ API başarısız: {response.status_code}")
                return []
                
        except requests.exceptions.Timeout:
            logger.error("⏰ API timeout hatası")
        except requests.exceptions.ConnectionError:
            logger.error("🔌 API bağlantı hatası")
        except Exception as e:
            logger.error(f"❌ Beklenmedik API hatası: {e}")
        
        return []
    
    def process_api_data(self, data):
        """API verisini işle"""
        matches = []
        
        try:
            events = data.get("sg", {}).get("EA", [])
            
            for event in events[:50]:
                if event.get("GT") != 1:  # Sadece futbol
                    continue
                
                match_date = event.get('D', '')
                if not self.is_upcoming_match(match_date):
                    continue
                
                match_info = {
                    'match_code': str(event.get('C', '')),
                    'home_team': self.clean_team_name(event.get('HN', '')),
                    'away_team': self.clean_team_name(event.get('AN', '')),
                    'league': self.clean_league_name(event.get('LC', '')),
                    'date': match_date,
                    'time': event.get('T', ''),
                    'odds': self.extract_odds_from_api(event.get('MA', []))
                }
                
                if match_info['home_team'] and match_info['away_team']:
                    matches.append(match_info)
            
        except Exception as e:
            logger.error(f"❌ API veri işleme hatası: {e}")
        
        return matches
    
    def fetch_from_website(self):
        """Web sitesinden veri çek"""
        matches = []
        
        try:
            url = "https://www.nesine.com/futbol"
            response = self.session.get(url, timeout=self.default_timeout)
            
            if response.status_code == 200:
                # Trafilatura ile temiz metin çıkar
                text_content = get_website_text_content(url)
                if text_content:
                    matches = self.parse_text_content(text_content)
                
                # BeautifulSoup fallback
                if not matches:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    matches = self.parse_html_content(soup)
                
        except Exception as e:
            logger.error(f"❌ Web scraping hatası: {e}")
        
        return matches
    
    def parse_text_content(self, text_content):
        """Metin içeriğinden maç verisi çıkar"""
        matches = []
        
        try:
            match_patterns = [
                r'([A-ZÇĞIİÖŞÜ][a-zçğııöşü]+(?:\s+[A-ZÇĞIİÖŞÜ][a-zçğııöşü]+)*)\s+vs\s+([A-ZÇĞIİÖŞÜ][a-zçğııöşü]+(?:\s+[A-ZÇĞIİÖŞÜ][a-zçğııöşü]+)*)',
                r'([A-ZÇĞIİÖŞÜ][a-zçğııöşü]+(?:\s+[A-ZÇĞIİÖŞÜ][a-zçğııöşü]+)*)\s+-\s+([A-ZÇĞIİÖŞÜ][a-zçğııöşü]+(?:\s+[A-ZÇĞIİÖŞÜ][a-zçğııöşü]+)*)',
            ]
            
            for pattern in match_patterns:
                found_matches = re.findall(pattern, text_content, re.IGNORECASE)
                for home, away in found_matches[:20]:
                    if len(home) > 2 and len(away) > 2:
                        match_info = {
                            'match_code': f"WEB_{len(matches)}",
                            'home_team': self.clean_team_name(home),
                            'away_team': self.clean_team_name(away),
                            'league': 'Web Scraping',
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'time': '20:00',
                            'odds': {'1': 2.0, 'X': 3.0, '2': 3.5}
                        }
                        matches.append(match_info)
                if matches:
                    break
            
        except Exception as e:
            logger.error(f"❌ Metin parsing hatası: {e}")
        
        return matches
    
    def parse_html_content(self, soup):
        """HTML içeriğinden maç verisi çıkar"""
        matches = []
        
        try:
            selectors = ['.match-item', '.game-item', '.event-item', '[data-match-id]', '.fixture']
            
            for selector in selectors:
                match_elements = soup.select(selector)
                if match_elements:
                    for i, element in enumerate(match_elements[:20]):
                        match_text = element.get_text(strip=True)
                        teams = self.extract_teams_from_text(match_text)
                        if teams:
                            match_info = {
                                'match_code': f"HTML_{i}",
                                'home_team': teams[0],
                                'away_team': teams[1],
                                'league': 'HTML Scraping',
                                'date': datetime.now().strftime('%Y-%m-%d'),
                                'time': '20:00',
                                'odds': {'1': 2.0, 'X': 3.0, '2': 3.5}
                            }
                            matches.append(match_info)
                    if matches:
                        break
            
        except Exception as e:
            logger.error(f"❌ HTML parsing hatası: {e}")
        
        return matches
    
    def extract_teams_from_text(self, text):
        """Metinden takım isimlerini çıkar"""
        separators = [' vs ', ' - ', ' – ', ' x ', ' V ', ' v ']
        
        for sep in separators:
            if sep in text:
                parts = text.split(sep)
                if len(parts) >= 2:
                    home = self.clean_team_name(parts[0])
                    away = self.clean_team_name(parts[1])
                    if home and away:
                        return [home, away]
        return None
    
    def is_upcoming_match(self, match_date):
        """Gelecek maç mı kontrol et"""
        try:
            if not match_date:
                return True
            match_dt = datetime.strptime(match_date, '%Y-%m-%d')
            today = datetime.now().date()
            match_date_obj = match_dt.date()
            return today <= match_date_obj <= (today + timedelta(days=7))
        except:
            return True
    
    def clean_team_name(self, name):
        """Takım adını temizle"""
        if not name:
            return ""
        clean_name = re.sub(r'[^a-zA-Z0-9\s çğıöşüÇĞIİÖŞÜ-]', '', str(name).strip())
        clean_name = re.sub(r'\s+', ' ', clean_name)
        
        replacements = {
            'Fenerbahce': 'Fenerbahçe', 'Besiktas': 'Beşiktaş',
            'Galatasaray': 'Galatasaray', 'Trabzonspor': 'Trabzonspor'
        }
        
        for old, new in replacements.items():
            if old.lower() in clean_name.lower():
                clean_name = new
                break
        
        return clean_name[:30]
    
    def clean_league_name(self, league):
        """Lig adını temizle"""
        if not league:
            return "Bilinmeyen Lig"
        clean_league = re.sub(r'[^a-zA-Z0-9\s çğıöşüÇĞIİÖŞÜ-]', '', str(league).strip())
        return clean_league[:50]
    
    def extract_odds_from_api(self, ma_data):
        """API'den oranları çıkar"""
        odds = {'1': 2.0, 'X': 3.0, '2': 3.5}
        try:
            for market in ma_data:
                if market.get('MTID') == 1:
                    outcomes = market.get('OCA', [])
                    if len(outcomes) >= 3:
                        odds['1'] = float(outcomes[0].get('O', 2.0))
                        odds['X'] = float(outcomes[1].get('O', 3.0))
                        odds['2'] = float(outcomes[2].get('O', 3.5))
                        break
        except Exception as e:
            logger.warning(f"⚠️ Oran çıkarma hatası: {e}")
        return odds
    
    def generate_sample_matches(self):
        """Örnek maç verileri oluştur"""
        sample_matches = [
            {
                'match_code': 'SAMPLE_001',
                'home_team': 'Fenerbahçe',
                'away_team': 'Galatasaray',
                'league': 'Süper Lig',
                'date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'time': '19:00',
                'odds': {'1': 2.1, 'X': 3.4, '2': 3.2}
            },
            {
                'match_code': 'SAMPLE_002', 
                'home_team': 'Beşiktaş',
                'away_team': 'Trabzonspor',
                'league': 'Süper Lig',
                'date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'time': '16:00',
                'odds': {'1': 1.8, 'X': 3.6, '2': 4.2}
            }
        ]
        return sample_matches
