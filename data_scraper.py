#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nesine.com veri çekme modülü
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

logger = logging.getLogger(__name__)

class NesineDataScraper:
    
    def __init__(self):
        self.session = requests.Session()
        
        # Retry strategy - daha agresif retry mekanizması
        retry_strategy = Retry(
            total=3,  # Toplam retry sayısı
            connect=2,  # Bağlantı hataları için retry
            read=2,     # Read hataları için retry
            status=2,   # HTTP status hataları için retry
            status_forcelist=[429, 500, 502, 503, 504],  # Bu kodlarda retry yap
            backoff_factor=1,  # 1, 2, 4 saniye bekle
            raise_on_status=False
        )
        
        # HTTP Adapter ile retry mekanizması
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'tr-TR,tr;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Improved timeouts
        self.default_timeout = (10, 30)  # connect: 10s, read: 30s
    
    def fetch_matches(self):
        """Nesine'den maç verilerini çek - gelişmiş fallback logic ile"""
        api_matches = []
        web_matches = []
        
        try:
            # Önce API'yi dene
            logger.info("Nesine API'den veri çekiliyor...")
            api_matches = self.fetch_from_api()
            
            if api_matches:
                logger.info(f"API'den {len(api_matches)} maç alındı")
                return api_matches
            else:
                logger.warning("API'den maç verisi alınamadı")
            
        except Exception as e:
            logger.error(f"API veri çekme hatası: {e}")
        
        try:
            # API başarısızsa web scraping dene
            logger.info("Web scraping ile veri çekiliyor...")
            web_matches = self.fetch_from_website()
            
            if web_matches:
                logger.info(f"Web scraping'den {len(web_matches)} maç alındı")
                return web_matches
            else:
                logger.warning("Web scraping'den maç verisi alınamadı")
                
        except Exception as e:
            logger.error(f"Web scraping hatası: {e}")
        
        # Sadece her iki kaynak da boş liste döndürdüğünde örnek verilere geç
        if not api_matches and not web_matches:
            logger.warning("Her iki kaynak da başarısız oldu, örnek veriler kullanılıyor")
            return self.generate_sample_matches()
        
        # Bu noktaya ulaşılmamalı, ama güvenlik için
        return []
    
    def fetch_from_api(self):
        """Nesine API'den veri çek - gelişmiş error handling ile"""
        try:
            url = "https://cdnbulten.nesine.com/api/bulten/getprebultenfull"
            
            logger.debug(f"API isteği gönderiliyor: {url}")
            response = self.session.get(url, timeout=self.default_timeout)
            
            logger.debug(f"API yanıt durumu: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    logger.debug(f"API veri boyutu: {len(str(data))} karakter")
                    
                    # Debug için ilk birkaç EA entry'yi logla
                    if 'sg' in data and 'EA' in data['sg']:
                        ea_entries = data['sg']['EA']
                        logger.debug(f"Toplam EA entry sayısı: {len(ea_entries)}")
                        
                        # İlk 3 entry'yi debug için logla
                        for i, entry in enumerate(ea_entries[:3]):
                            logger.debug(f"EA entry {i+1}: {entry.get('HN', 'N/A')} vs {entry.get('AN', 'N/A')} - {entry.get('LC', 'N/A')}")
                    
                    return self.process_api_data(data)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse hatası: {e}")
                    return []
                    
            elif response.status_code == 429:
                logger.warning("Rate limit aşıldı, 5 saniye bekleniyor...")
                time.sleep(5)
                return []  # Retry mekanizması otomatik devreye girecek
                
            else:
                logger.warning(f"API başarısız yanıt kodu: {response.status_code}, Response: {response.text[:200]}")
                return []
                
        except requests.exceptions.Timeout as e:
            logger.error(f"API timeout hatası: {e}")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"API bağlantı hatası: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"API istek hatası: {e}")
        except Exception as e:
            logger.error(f"Beklenmedik API hatası: {e}")
        
        return []
    
    def process_api_data(self, data):
        """API verisini işle"""
        matches = []
        
        try:
            events = data.get("sg", {}).get("EA", [])
            
            for event in events[:50]:  # İlk 50 maç
                if event.get("GT") != 1:  # Sadece futbol
                    continue
                
                match_date = event.get('D', '')
                match_time = event.get('T', '')
                
                # Bugün ve yarının maçlarını al
                if not self.is_upcoming_match(match_date):
                    continue
                
                match_info = {
                    'match_code': str(event.get('C', '')),
                    'home_team': self.clean_team_name(event.get('HN', '')),
                    'away_team': self.clean_team_name(event.get('AN', '')),
                    'league': self.clean_league_name(event.get('LC', '')),
                    'date': match_date,
                    'time': match_time,
                    'odds': self.extract_odds_from_api(event.get('MA', []))
                }
                
                if match_info['home_team'] and match_info['away_team']:
                    matches.append(match_info)
            
            logger.info(f"API'den {len(matches)} maç alındı")
            
        except Exception as e:
            logger.error(f"API veri işleme hatası: {e}")
        
        return matches
    
    def fetch_from_website(self):
        """Web sitesinden veri çek - gelişmiş error handling ile"""
        matches = []
        
        try:
            url = "https://www.nesine.com/futbol"
            
            logger.debug(f"Web scraping isteği gönderiliyor: {url}")
            response = self.session.get(url, timeout=self.default_timeout)
            
            logger.debug(f"Web scraping yanıt durumu: {response.status_code}")
            
            if response.status_code == 200:
                logger.debug(f"Web sayfası içerik boyutu: {len(response.text)} karakter")
                
                # Trafilatura ile temiz metin çıkar
                text_content = get_website_text_content(url)
                
                if text_content:
                    logger.debug(f"Trafilatura ile çıkarılan metin boyutu: {len(text_content)}")
                    matches = self.parse_text_content(text_content)
                
                # Eğer trafilatura başarısızsa BeautifulSoup dene
                if not matches:
                    logger.debug("Trafilatura başarısız, BeautifulSoup deneniyor...")
                    soup = BeautifulSoup(response.content, 'html.parser')
                    matches = self.parse_html_content(soup)
                
                logger.info(f"Web sitesinden {len(matches)} maç alındı")
                
            elif response.status_code == 403:
                logger.error("Web sitesi erişimi reddetti (403 Forbidden)")
            elif response.status_code == 429:
                logger.warning("Web sitesi rate limit (429), biraz bekle...")
                time.sleep(3)
            else:
                logger.warning(f"Web sitesi başarısız yanıt: {response.status_code}")
            
        except requests.exceptions.Timeout as e:
            logger.error(f"Web scraping timeout: {e}")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Web scraping bağlantı hatası: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Web scraping istek hatası: {e}")
        except Exception as e:
            logger.error(f"Beklenmedik web scraping hatası: {e}")
        
        return matches
    
    def parse_text_content(self, text_content):
        """Metin içeriğinden maç verisi çıkar"""
        matches = []
        
        try:
            # Gelişmiş regex ile maç formatlarını bul
            # Nesine'ye özel formatlar da dahil
            match_patterns = [
                r'([A-ZÇĞIİÖŞÜ][a-zçğııöşü]+(?:\s+[A-ZÇĞIİÖŞÜ][a-zçğııöşü]+)*)\s+vs\s+([A-ZÇĞIİÖŞÜ][a-zçğııöşü]+(?:\s+[A-ZÇĞIİÖŞÜ][a-zçğııöşü]+)*)',
                r'([A-ZÇĞIİÖŞÜ][a-zçğııöşü]+(?:\s+[A-ZÇĞIİÖŞÜ][a-zçğııöşü]+)*)\s+-\s+([A-ZÇĞIİÖŞÜ][a-zçğııöşü]+(?:\s+[A-ZÇĞIİÖŞÜ][a-zçğııöşü]+)*)',
                r'([A-ZÇĞIİÖŞÜ][a-zçğııöşü]+(?:\s+[A-ZÇĞIİÖŞÜ][a-zçğııöşü]+)*)\s+–\s+([A-ZÇĞIİÖŞÜ][a-zçğııöşü]+(?:\s+[A-ZÇĞIİÖŞÜ][a-zçğııöşü]+)*)',
                r'(\w+(?:\s+\w+)*)\s+vs\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+-\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+–\s+(\w+(?:\s+\w+)*)'
            ]
            
            for pattern in match_patterns:
                found_matches = re.findall(pattern, text_content, re.IGNORECASE)
                
                for home, away in found_matches[:20]:  # İlk 20 maç
                    if len(home) > 2 and len(away) > 2:  # Geçerli takım ismi
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
            logger.error(f"Metin parsing hatası: {e}")
        
        return matches
    
    def parse_html_content(self, soup):
        """HTML içeriğinden maç verisi çıkar"""
        matches = []
        
        try:
            # Çeşitli HTML selectors dene
            selectors = [
                '.match-item',
                '.game-item',
                '.event-item',
                '[data-match-id]',
                '.fixture'
            ]
            
            for selector in selectors:
                match_elements = soup.select(selector)
                
                if match_elements:
                    for i, element in enumerate(match_elements[:20]):
                        match_text = element.get_text(strip=True)
                        
                        # Takım isimlerini çıkarmaya çalış
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
            logger.error(f"HTML parsing hatası: {e}")
        
        return matches
    
    def extract_teams_from_text(self, text):
        """Metinden takım isimlerini çıkar"""
        # Çeşitli ayırıcıları dene
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
            
            # Bugün ve gelecek 7 gün
            return today <= match_date_obj <= (today + timedelta(days=7))
            
        except:
            return True
    
    def clean_team_name(self, name):
        """Takım adını temizle"""
        if not name:
            return ""
        
        # Özel karakterleri ve sayıları temizle
        clean_name = re.sub(r'[^a-zA-Z0-9\s çğıöşüÇĞIİÖŞÜ-]', '', str(name).strip())
        
        # Çoklu boşlukları tek boşluk yap
        clean_name = re.sub(r'\s+', ' ', clean_name)
        
        # Türkçe takım isim düzeltmeleri
        replacements = {
            'Fenerbahce': 'Fenerbahçe',
            'Besiktas': 'Beşiktaş',
            'Galatasaray': 'Galatasaray',
            'Trabzonspor': 'Trabzonspor',
            'Basaksehir': 'Başakşehir'
        }
        
        for old, new in replacements.items():
            if old.lower() in clean_name.lower():
                clean_name = new
                break
        
        return clean_name[:30]  # Maksimum 30 karakter
    
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
                if market.get('MTID') == 1:  # Maç sonucu marketi
                    outcomes = market.get('OCA', [])
                    if len(outcomes) >= 3:
                        odds['1'] = float(outcomes[0].get('O', 2.0))
                        odds['X'] = float(outcomes[1].get('O', 3.0))
                        odds['2'] = float(outcomes[2].get('O', 3.5))
                        break
        except Exception as e:
            logger.warning(f"Oran çıkarma hatası: {e}")
        
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
            },
            {
                'match_code': 'SAMPLE_003',
                'home_team': 'Manchester City',
                'away_team': 'Liverpool',
                'league': 'Premier League',
                'date': (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d'),
                'time': '21:00',
                'odds': {'1': 2.3, 'X': 3.1, '2': 3.0}
            },
            {
                'match_code': 'SAMPLE_004',
                'home_team': 'Real Madrid',
                'away_team': 'Barcelona',
                'league': 'La Liga',
                'date': (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d'),
                'time': '22:00',
                'odds': {'1': 2.0, 'X': 3.2, '2': 3.8}
            },
            {
                'match_code': 'SAMPLE_005',
                'home_team': 'Bayern Munich',
                'away_team': 'Dortmund',
                'league': 'Bundesliga',
                'date': (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d'),
                'time': '18:30',
                'odds': {'1': 1.9, 'X': 3.5, '2': 4.0}
            }
        ]
        
        logger.info("Örnek maç verileri oluşturuldu")
        return sample_matches
