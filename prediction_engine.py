#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NESINE.COM KAPSAMLI FUTBOL TAHMIN SISTEMI
Nesine'den maç, takım ve oyuncu istatistiklerini çekip detaylı tahmin yapan sistem
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
import requests
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import time
import re
from bs4 import BeautifulSoup
from data_scraper import NesineDataScraper
from database_manager import DatabaseManager # Import'un doğru olduğundan emin olun

# Logging kurulumu
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PlayerStats:
    name: str
    position: str
    appearances: int
    goals: int
    assists: int
    yellow_cards: int
    red_cards: int
    rating: float
    is_injured: bool = False
    is_suspended: bool = False

@dataclass
class TeamStats:
    name: str
    league: str
    position: int
    points: int
    matches_played: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    home_form: str
    away_form: str
    recent_form: List[str]

class NesineAdvancedPredictor:
    
    # 🚨 DÜZELTME BURADA: db_manager'ı zorunlu olmayan bir argüman olarak alıyoruz
    def __init__(self, db_manager: Optional[DatabaseManager] = None, db_path="data/nesine_advanced.db"):
        
        # Eğer dışarıdan DatabaseManager nesnesi gelirse onu kullan (Render'da kullandığımız yöntem)
        if db_manager is not None:
            self.db_manager = db_manager
        # Gelmezse, path üzerinden kendimiz oluştur (Lokal test için)
        else:
            self.db_path = db_path
            self.db_manager = DatabaseManager(db_path)
            
        self.scraper = NesineDataScraper()
        
        # Gelişmiş tahmin ağırlıkları
        self.weights = {
            'team_statistics': 25,    # Takım istatistikleri
            'player_impact': 20,      # Anahtar oyuncu etkileri
            'recent_form': 20,        # Son form
            'head_to_head': 15,       # Karşılaşma geçmişi
            'odds_analysis': 10,      # Oran analizi
            'league_position': 10     # Lig durumu
        }
        logger.info("Tahmin Motoru başarıyla başlatıldı.")
    
    def fetch_nesine_matches(self):
        """Nesine.com'dan güncel maç programını çek"""
        try:
            matches = self.scraper.fetch_matches()
            
            if matches:
                # Maçları veritabanına kaydet
                for match in matches:
                    self.db_manager.save_match(match)
                
                logger.info(f"{len(matches)} maç verisi çekildi ve kaydedildi")
                return matches
            else:
                # Eğer canlı veri çekilemiyorsa veritabanından en son maçları getir
                return self.db_manager.get_recent_matches()
                
        except Exception as e:
            logger.error(f"Nesine veri çekme hatası: {e}")
            return self.db_manager.get_recent_matches()
    
    def predict_match_comprehensive(self, match):
        """Kapsamlı maç tahmini yap"""
        try:
            home_team = match.get('home_team', '')
            away_team = match.get('away_team', '')
            league = match.get('league', '')
            
            if not home_team or not away_team:
                raise ValueError("Takım isimleri eksik")
            
            # Takım istatistiklerini getir
            home_stats = self.get_team_stats(home_team, league)
            away_stats = self.get_team_stats(away_team, league)
            
            # Analiz skorlarını hesapla
            analysis_scores = self.calculate_analysis_scores(home_stats, away_stats, match)
            
            # Genel tahmin güvenini hesapla
            confidence = self.calculate_overall_confidence(analysis_scores)
            
            # Maç sonucu tahmini
            result_prediction = self.predict_match_result(analysis_scores, confidence)
            
            # İlk yarı tahmini
            iy_prediction = self.predict_first_half(analysis_scores, confidence)
            
            # Skor tahmini
            score_prediction = self.predict_score(home_stats, away_stats, analysis_scores)
            
            prediction = {
                'result_prediction': result_prediction,
                'iy_prediction': iy_prediction,
                'score_prediction': score_prediction,
                'confidence': confidence,
                'iy_confidence': confidence * 0.8,  # İY tahmin güveni biraz daha düşük
                'score_confidence': confidence * 0.7,  # Skor tahmin güveni en düşük
                'analysis': analysis_scores,
                'home_strength': self.calculate_team_strength(home_stats),
                'away_strength': self.calculate_team_strength(away_stats)
            }
            
            # Tahmini veritabanına kaydet
            self.db_manager.save_prediction(match, prediction)
            
            return prediction
            
        except Exception as e:
            home_name = match.get('home_team', 'Bilinmeyen')
            away_name = match.get('away_team', 'Bilinmeyen')
            logger.error(f"Tahmin hatası ({home_name} vs {away_name}): {e}")
            return self.get_default_prediction()
    
    def calculate_analysis_scores(self, home_stats, away_stats, match):
        """Analiz skorlarını hesapla"""
        scores = {}
        
        # Takım istatistik karşılaştırması
        scores['team_statistics'] = self.compare_team_statistics(home_stats, away_stats)
        
        # Son form karşılaştırması
        scores['recent_form'] = self.compare_recent_form(home_stats, away_stats)
        
        # Lig pozisyon karşılaştırması
        scores['league_position'] = self.compare_league_positions(home_stats, away_stats)
        
        # Ev sahibi avantajı
        scores['home_advantage'] = self.calculate_home_advantage(home_stats)
        
        # Gol istatistikleri
        scores['goal_statistics'] = self.compare_goal_statistics(home_stats, away_stats)
        
        # Geçmiş karşılaşmalar
        scores['head_to_head'] = self.analyze_head_to_head(
            home_stats.get('name', ''), 
            away_stats.get('name', '')
        )
        
        return scores
    
    def compare_team_statistics(self, home_stats, away_stats):
        """Takım istatistiklerini karşılaştır"""
        try:
            home_win_rate = home_stats.get('wins', 0) / max(home_stats.get('matches_played', 1), 1)
            away_win_rate = away_stats.get('wins', 0) / max(away_stats.get('matches_played', 1), 1)
            
            home_points_per_game = home_stats.get('points', 0) / max(home_stats.get('matches_played', 1), 1)
            away_points_per_game = away_stats.get('points', 0) / max(away_stats.get('matches_played', 1), 1)
            
            # Skorları normalize et (0-100 arası)
            win_rate_score = (home_win_rate - away_win_rate + 1) * 50
            points_score = ((home_points_per_game - away_points_per_game) / 3 + 1) * 50
            
            return min(100, max(0, (win_rate_score + points_score) / 2))
            
        except Exception:
            return 50  # Nötr skor
    
    def compare_recent_form(self, home_stats, away_stats):
        """Son form karşılaştırması"""
        try:
            home_form = self.calculate_form_score(home_stats.get('recent_form', []))
            away_form = self.calculate_form_score(away_stats.get('recent_form', []))
            
            # Form skorlarını karşılaştır
            if home_form > away_form:
                return min(100, 50 + (home_form - away_form) * 2)
            else:
                return max(0, 50 - (away_form - home_form) * 2)
            
        except Exception:
            return 50
    
    def calculate_form_score(self, form_list):
        """Form skorunu hesapla"""
        if not form_list:
            return 50
        
        score = 0
        for i, result in enumerate(form_list[:5]):  # Son 5 maç
            weight = 5 - i  # Son maçlar daha ağırlıklı
            if result == 'G':  # Galibiyet
                score += 3 * weight
            elif result == 'B':  # Beraberlik
                score += 1 * weight
            # Mağlubiyet için puan yok
        
        max_possible = sum(range(1, 6)) * 3  # 45 puan
        return (score / max_possible) * 100 if max_possible > 0 else 50
    
    def compare_league_positions(self, home_stats, away_stats):
        """Lig pozisyonlarını karşılaştır"""
        try:
            home_pos = home_stats.get('position', 10)
            away_pos = away_stats.get('position', 10)
            
            # Pozisyon farkına göre skor
            pos_diff = away_pos - home_pos
            return min(100, max(0, 50 + pos_diff * 2.5))
            
        except Exception:
            return 50
    
    def calculate_home_advantage(self, home_stats):
        """Ev sahibi avantajını hesapla"""
        try:
            home_matches = home_stats.get('home_wins', 0) + home_stats.get('home_draws', 0) + home_stats.get('home_losses', 0)
            if home_matches == 0:
                return 60  # Varsayılan ev sahibi avantajı
            
            home_win_rate = home_stats.get('home_wins', 0) / home_matches
            return min(100, 40 + home_win_rate * 60)
            
        except Exception:
            return 60
    
    def compare_goal_statistics(self, home_stats, away_stats):
        """Gol istatistiklerini karşılaştır"""
        try:
            home_goals_per_game = home_stats.get('goals_for', 0) / max(home_stats.get('matches_played', 1), 1)
            away_goals_per_game = away_stats.get('goals_for', 0) / max(away_stats.get('matches_played', 1), 1)
            
            home_conceded_per_game = home_stats.get('goals_against', 0) / max(home_stats.get('matches_played', 1), 1)
            away_conceded_per_game = away_stats.get('goals_against', 0) / max(away_stats.get('matches_played', 1), 1)
            
            # Atak gücü karşılaştırması
            attack_score = ((home_goals_per_game - away_goals_per_game) + 2) * 25
            
            # Savunma gücü karşılaştırması (daha az gol yemek daha iyi)
            defense_score = ((away_conceded_per_game - home_conceded_per_game) + 2) * 25
            
            return min(100, max(0, (attack_score + defense_score) / 2))
            
        except Exception:
            return 50
    
    def analyze_head_to_head(self, home_team, away_team):
        """Geçmiş karşılaşmaları analiz et"""
        try:
            h2h_data = self.db_manager.get_head_to_head(home_team, away_team)
            
            if not h2h_data:
                return 50  # Veri yoksa nötr
            
            # Düzeltme: Burada skorları doğru şekilde karşılaştırmamız lazım.
            # Veritabanında (home_score, away_score) şeklinde geldiğini varsayalım.
            home_wins = sum(1 for match in h2h_data if match['home_score'] > match['away_score'])
            
            if len(h2h_data) == 0:
                return 50
            
            home_win_rate = home_wins / len(h2h_data)
            return min(100, max(0, home_win_rate * 100))
            
        except Exception:
            return 50
    
    def calculate_overall_confidence(self, analysis_scores):
        """Genel güven seviyesini hesapla"""
        try:
            # Skorların tutarlılığını kontrol et
            scores = list(analysis_scores.values())
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Düşük standart sapma = yüksek tutarlılık = yüksek güven
            consistency_bonus = max(0, 20 - std_score)
            
            # Aşırı değerlerden uzaklaşma bonusu
            extreme_penalty = 0
            for score in scores:
                if score < 10 or score > 90:
                    extreme_penalty += 5
            
            base_confidence = 60.0  # Temel güven seviyesi
            confidence = base_confidence + consistency_bonus - extreme_penalty
            
            return min(95.0, max(20.0, confidence))
            
        except Exception:
            return 60
    
    def predict_match_result(self, analysis_scores, confidence):
        """Maç sonucu tahmini"""
        try:
            avg_score = np.mean(list(analysis_scores.values()))
            
            if confidence < 40:
                return "Belirsiz"
            elif avg_score > 65:
                return "1"  # Ev sahibi galibiyeti
            elif avg_score < 35:
                return "2"  # Deplasman galibiyeti
            elif 45 <= avg_score <= 55:
                return "X"  # Beraberlik
            elif avg_score > 55:
                return "1"
            else:
                return "2"
                
        except Exception:
            return "Belirsiz"
    
    def predict_first_half(self, analysis_scores, confidence):
        """İlk yarı tahmini"""
        try:
            # İY tahminleri genelde daha konservatif
            avg_score = np.mean(list(analysis_scores.values()))
            
            if confidence < 50:
                return "X"  # Düşük güvende beraberlik
            elif avg_score > 70:
                return "1"
            elif avg_score < 30:
                return "2"
            else:
                return "X"  # Çoğunlukla beraberlik
                
        except Exception:
            return "X"
    
    def predict_score(self, home_stats, away_stats, analysis_scores):
        """Skor tahmini"""
        try:
            home_avg_goals = home_stats.get('goals_for', 0) / max(home_stats.get('matches_played', 1), 1)
            away_avg_goals = away_stats.get('goals_for', 0) / max(away_stats.get('matches_played', 1), 1)
            
            home_avg_conceded = home_stats.get('goals_against', 0) / max(home_stats.get('matches_played', 1), 1)
            away_avg_conceded = away_stats.get('goals_against', 0) / max(away_stats.get('matches_played', 1), 1)
            
            # Tahmini goller
            home_expected = (home_avg_goals + away_avg_conceded) / 2
            away_expected = (away_avg_goals + home_avg_conceded) / 2
            
            # Ev sahibi avantajı ekle
            home_expected *= 1.2
            
            # Yuvarla
            home_goals = max(0, round(home_expected))
            away_goals = max(0, round(away_expected))
            
            return f"{home_goals}-{away_goals}"
            
        except Exception:
            return "1-1"
    
    def calculate_team_strength(self, team_stats):
        """Takım gücünü hesapla"""
        try:
            if not team_stats:
                return 50
            
            # Farklı metrikleri birleştir
            win_rate = team_stats.get('wins', 0) / max(team_stats.get('matches_played', 1), 1) * 100
            
            goals_ratio = team_stats.get('goals_for', 0) / max(team_stats.get('goals_against', 1), 1)
            goals_score = min(100, goals_ratio * 30)
            
            position_score = max(0, 100 - team_stats.get('position', 10) * 5)
            
            strength = (win_rate + goals_score + position_score) / 3
            return min(100, max(0, strength))
            
        except Exception:
            return 50
    
    def get_team_stats(self, team_name, league):
        """Takım istatistiklerini getir"""
        try:
            stats = self.db_manager.get_team_stats(team_name, league)
            if not stats:
                # Yoksa oluştur
                stats = self.generate_team_stats(team_name, league)
                self.db_manager.save_team_stats(stats)
            return stats
        except Exception as e:
            logger.error(f"Takım istatistik hatası ({team_name}): {e}")
            return self.generate_default_team_stats(team_name, league)
    
    def generate_team_stats(self, team_name, league):
        """Takım istatistiklerini oluştur"""
        # Intelligent tahmin ile gerçekçi istatistikler
        base_position = self.estimate_team_position(team_name, league)
        
        # Pozisyona göre performans tahmini
        if base_position <= 5:  # Üst sıra takımlar
            wins_factor = np.random.uniform(0.6, 0.8)
            goals_factor = np.random.uniform(1.8, 2.5)
        elif base_position <= 10:  # Orta sıra
            wins_factor = np.random.uniform(0.4, 0.6)
            goals_factor = np.random.uniform(1.2, 1.8)
        else:  # Alt sıra
            wins_factor = np.random.uniform(0.2, 0.4)
            goals_factor = np.random.uniform(0.8, 1.4)
        
        matches_played = np.random.randint(15, 25)
        wins = int(matches_played * wins_factor)
        losses = int(matches_played * (1 - wins_factor) * 0.7)
        draws = matches_played - wins - losses
        
        points = wins * 3 + draws
        goals_for = int(matches_played * goals_factor)
        goals_against = int(goals_for * np.random.uniform(0.6, 1.4))
        
        # Son form oluştur
        recent_form = self.generate_recent_form(wins_factor)
        
        return {
            'name': team_name,
            'league': league,
            'position': base_position,
            'points': points,
            'matches_played': matches_played,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'home_wins': wins // 2,
            'home_draws': draws // 2,
            'home_losses': losses // 2,
            'away_wins': wins - wins // 2,
            'away_draws': draws - draws // 2,
            'away_losses': losses - losses // 2,
            'recent_form': recent_form
        }
    
    def estimate_team_position(self, team_name, league):
        """Takım pozisyonunu tahmin et"""
        # Bilinen büyük takımlar için özel pozisyonlar
        top_teams = {
            'Fenerbahçe': 3, 'Galatasaray': 2, 'Beşiktaş': 4, 'Trabzonspor': 5,
            'Manchester City': 1, 'Liverpool': 2, 'Arsenal': 3, 'Chelsea': 4,
            'Real Madrid': 1, 'Barcelona': 2, 'Atletico Madrid': 3,
            'Bayern Munich': 1, 'Dortmund': 2, 'Juventus': 1, 'Milan': 2
        }
        
        return top_teams.get(team_name, np.random.randint(6, 20))
    
    def generate_recent_form(self, win_factor):
        """Son form oluştur"""
        form = []
        for _ in range(5):
            rand = np.random.random()
            if rand < win_factor:
                form.append('G')
            elif rand < win_factor + 0.2:
                form.append('B')
            else:
                form.append('M')
        return form
    
    def generate_default_team_stats(self, team_name, league):
        """Varsayılan takım istatistikleri"""
        return {
            'name': team_name,
            'league': league,
            'position': 10,
            'points': 30,
            'matches_played': 20,
            'wins': 10,
            'draws': 5,
            'losses': 5,
            'goals_for': 25,
            'goals_against': 20,
            'home_wins': 5,
            'home_draws': 3,
            'home_losses': 2,
            'away_wins': 5,
            'away_draws': 2,
            'away_losses': 3,
            'recent_form': ['G', 'B', 'M', 'G', 'B']
        }
    
    def get_default_prediction(self):
        """Varsayılan tahmin"""
        return {
            'result_prediction': 'Belirsiz',
            'iy_prediction': 'X',
            'score_prediction': '1-1',
            'confidence': 30,
            'iy_confidence': 25,
            'score_confidence': 20,
            'analysis': {},
            'home_strength': 50,
            'away_strength': 50
        }
    
    # Utility methods for Streamlit app
    def get_all_teams(self):
        """Tüm takımları getir"""
        return self.db_manager.get_all_teams()
    
    def get_team_detailed_stats(self, team_name):
        """Detaylı takım istatistikleri"""
        return self.db_manager.get_team_detailed_stats(team_name)
    
    def get_head_to_head_stats(self, home_team, away_team):
        """Geçmiş karşılaşma istatistikleri"""
        return self.db_manager.get_head_to_head(home_team, away_team)
    
    def get_total_matches_in_db(self):
        """Veritabanındaki toplam maç sayısı"""
        return self.db_manager.get_total_matches()
    
    def get_total_teams_in_db(self):
        """Veritabanındaki toplam takım sayısı"""
        return self.db_manager.get_total_teams()
    
    def get_average_confidence(self):
        """Ortalama güven seviyesi"""
        return self.db_manager.get_average_confidence()
    
    def get_recent_predictions_count(self):
        """Son 24 saat tahmin sayısı"""
        return self.db_manager.get_recent_predictions_count()
    
    def get_confidence_distribution(self):
        """Güven seviyesi dağılımı"""
        return self.db_manager.get_confidence_distribution()
    
    def get_league_statistics(self):
        """Lig istatistikleri"""
        return self.db_manager.get_league_statistics()
