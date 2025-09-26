#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Veritabanı yönetim modülü
"""

import sqlite3
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class DatabaseManager:
    
    def __init__(self, db_path="nesine_advanced.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Veritabanı yapısını oluştur"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Maçlar tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY,
                match_code TEXT UNIQUE,
                home_team TEXT,
                away_team TEXT,
                league TEXT,
                match_date TEXT,
                match_time TEXT,
                odds_1 REAL,
                odds_x REAL,
                odds_2 REAL,
                prediction_ms TEXT,
                prediction_iy TEXT,
                predicted_score TEXT,
                confidence REAL,
                home_strength REAL,
                away_strength REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Takım istatistikleri
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS team_stats (
                id INTEGER PRIMARY KEY,
                team_name TEXT,
                league TEXT,
                position INTEGER,
                points INTEGER,
                matches_played INTEGER,
                wins INTEGER,
                draws INTEGER,
                losses INTEGER,
                goals_for INTEGER,
                goals_against INTEGER,
                home_wins INTEGER DEFAULT 0,
                home_draws INTEGER DEFAULT 0,
                home_losses INTEGER DEFAULT 0,
                away_wins INTEGER DEFAULT 0,
                away_draws INTEGER DEFAULT 0,
                away_losses INTEGER DEFAULT 0,
                recent_form TEXT,
                avg_goals_scored REAL DEFAULT 0,
                avg_goals_conceded REAL DEFAULT 0,
                clean_sheets INTEGER DEFAULT 0,
                failed_to_score INTEGER DEFAULT 0,
                strength_rating REAL DEFAULT 50,
                attack_rating REAL DEFAULT 50,
                defense_rating REAL DEFAULT 50,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(team_name, league)
            )
        ''')
        
        # Oyuncu istatistikleri
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_stats (
                id INTEGER PRIMARY KEY,
                player_name TEXT,
                team_name TEXT,
                league TEXT,
                position TEXT,
                appearances INTEGER DEFAULT 0,
                goals INTEGER DEFAULT 0,
                assists INTEGER DEFAULT 0,
                yellow_cards INTEGER DEFAULT 0,
                red_cards INTEGER DEFAULT 0,
                rating REAL DEFAULT 0,
                market_value REAL DEFAULT 0,
                is_injured BOOLEAN DEFAULT 0,
                is_suspended BOOLEAN DEFAULT 0,
                importance_score REAL DEFAULT 5,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(player_name, team_name, league)
            )
        ''')
        
        # Geçmiş karşılaşmalar
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS head_to_head (
                id INTEGER PRIMARY KEY,
                home_team TEXT,
                away_team TEXT,
                match_date TEXT,
                home_score INTEGER,
                away_score INTEGER,
                league TEXT,
                season TEXT,
                venue TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tahmin sonuçları
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                match_code TEXT,
                home_team TEXT,
                away_team TEXT,
                prediction_result TEXT,
                prediction_iy TEXT,
                predicted_score TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
        logger.info("Veritabanı yapısı kuruldu")
    
    def save_match(self, match):
        """Maçı veritabanına kaydet"""
        try:
            cursor = self.conn.cursor()
            odds = match.get('odds', {})
            
            cursor.execute('''
                INSERT OR REPLACE INTO matches 
                (match_code, home_team, away_team, league, match_date, match_time, 
                 odds_1, odds_x, odds_2, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                match.get('match_code', ''),
                match.get('home_team', ''),
                match.get('away_team', ''),
                match.get('league', ''),
                match.get('date', ''),
                match.get('time', ''),
                odds.get('1', 0),
                odds.get('X', 0),
                odds.get('2', 0),
                datetime.now()
            ))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Maç kaydetme hatası: {e}")
    
    def save_team_stats(self, stats):
        """Takım istatistiklerini kaydet"""
        try:
            cursor = self.conn.cursor()
            
            recent_form_str = json.dumps(stats.get('recent_form', []))
            
            cursor.execute('''
                INSERT OR REPLACE INTO team_stats 
                (team_name, league, position, points, matches_played, wins, draws, losses,
                 goals_for, goals_against, home_wins, home_draws, home_losses,
                 away_wins, away_draws, away_losses, recent_form, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                stats.get('name', ''),
                stats.get('league', ''),
                stats.get('position', 0),
                stats.get('points', 0),
                stats.get('matches_played', 0),
                stats.get('wins', 0),
                stats.get('draws', 0),
                stats.get('losses', 0),
                stats.get('goals_for', 0),
                stats.get('goals_against', 0),
                stats.get('home_wins', 0),
                stats.get('home_draws', 0),
                stats.get('home_losses', 0),
                stats.get('away_wins', 0),
                stats.get('away_draws', 0),
                stats.get('away_losses', 0),
                recent_form_str,
                datetime.now()
            ))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Takım istatistik kaydetme hatası: {e}")
    
    def save_prediction(self, match, prediction):
        """Tahmin sonucunu kaydet"""
        try:
            cursor = self.conn.cursor()
            
            # Maç tablosunu güncelle
            cursor.execute('''
                UPDATE matches SET 
                prediction_ms = ?, prediction_iy = ?, predicted_score = ?, 
                confidence = ?, home_strength = ?, away_strength = ?
                WHERE match_code = ?
            ''', (
                prediction.get('result_prediction', ''),
                prediction.get('iy_prediction', ''),
                prediction.get('score_prediction', ''),
                prediction.get('confidence', 0),
                prediction.get('home_strength', 0),
                prediction.get('away_strength', 0),
                match.get('match_code', '')
            ))
            
            # Tahmin tablosuna da kaydet
            cursor.execute('''
                INSERT INTO predictions 
                (match_code, home_team, away_team, prediction_result, prediction_iy, 
                 predicted_score, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                match.get('match_code', ''),
                match.get('home_team', ''),
                match.get('away_team', ''),
                prediction.get('result_prediction', ''),
                prediction.get('iy_prediction', ''),
                prediction.get('score_prediction', ''),
                prediction.get('confidence', 0),
                datetime.now()
            ))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Tahmin kaydetme hatası: {e}")
    
    def get_team_stats(self, team_name, league):
        """Takım istatistiklerini getir"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT * FROM team_stats 
                WHERE team_name = ? AND league = ?
                ORDER BY updated_at DESC LIMIT 1
            ''', (team_name, league))
            
            result = cursor.fetchone()
            
            if result:
                columns = [description[0] for description in cursor.description]
                stats = dict(zip(columns, result))
                
                # JSON string'i liste'ye çevir
                if stats.get('recent_form'):
                    try:
                        stats['recent_form'] = json.loads(stats['recent_form'])
                    except:
                        stats['recent_form'] = []
                
                return stats
            
            return None
            
        except Exception as e:
            logger.error(f"Takım istatistik getirme hatası: {e}")
            return None
    
    def get_head_to_head(self, home_team, away_team, limit=10):
        """Geçmiş karşılaşmaları getir"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT * FROM head_to_head 
                WHERE (home_team = ? AND away_team = ?) 
                   OR (home_team = ? AND away_team = ?)
                ORDER BY match_date DESC LIMIT ?
            ''', (home_team, away_team, away_team, home_team, limit))
            
            results = cursor.fetchall()
            
            if results:
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in results]
            
            return []
            
        except Exception as e:
            logger.error(f"Geçmiş karşılaşma getirme hatası: {e}")
            return []
    
    def get_recent_matches(self, days=7):
        """Son maçları getir"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT * FROM matches 
                WHERE created_at > datetime('now', '-{} days')
                ORDER BY created_at DESC
            '''.format(days))
            
            results = cursor.fetchall()
            
            if results:
                columns = [description[0] for description in cursor.description]
                matches = []
                
                for row in results:
                    match_dict = dict(zip(columns, row))
                    
                    # Oranları düzenle
                    match_dict['odds'] = {
                        '1': match_dict.get('odds_1', 0),
                        'X': match_dict.get('odds_x', 0),
                        '2': match_dict.get('odds_2', 0)
                    }
                    
                    # Tarih ve saat düzenle
                    match_dict['date'] = match_dict.get('match_date', '')
                    match_dict['time'] = match_dict.get('match_time', '')
                    
                    matches.append(match_dict)
                
                return matches
            
            return []
            
        except Exception as e:
            logger.error(f"Son maç getirme hatası: {e}")
            return []
    
    def get_all_teams(self):
        """Tüm takımları getir"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT DISTINCT team_name FROM team_stats ORDER BY team_name')
            results = cursor.fetchall()
            return [row[0] for row in results if row[0]]
            
        except Exception as e:
            logger.error(f"Takım listesi getirme hatası: {e}")
            return []
    
    def get_team_detailed_stats(self, team_name):
        """Detaylı takım istatistikleri getir"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT * FROM team_stats 
                WHERE team_name = ? 
                ORDER BY updated_at DESC LIMIT 1
            ''', (team_name,))
            
            result = cursor.fetchone()
            
            if result:
                columns = [description[0] for description in cursor.description]
                stats = dict(zip(columns, result))
                
                # recent_form'u parse et
                if stats.get('recent_form'):
                    try:
                        stats['recent_form'] = json.loads(stats['recent_form'])
                    except:
                        stats['recent_form'] = []
                
                # Takım adını name alanına kopyala
                stats['name'] = stats.get('team_name', '')
                
                return stats
            
            return None
            
        except Exception as e:
            logger.error(f"Detaylı takım istatistik hatası: {e}")
            return None
    
    def get_total_matches(self):
        """Toplam maç sayısı"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM matches')
            return cursor.fetchone()[0]
        except:
            return 0
    
    def get_total_teams(self):
        """Toplam takım sayısı"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT COUNT(DISTINCT team_name) FROM team_stats')
            return cursor.fetchone()[0]
        except:
            return 0
    
    def get_average_confidence(self):
        """Ortalama güven seviyesi"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT AVG(confidence) FROM matches WHERE confidence > 0')
            result = cursor.fetchone()[0]
            return result if result else 50.0
        except:
            return 50.0
    
    def get_recent_predictions_count(self):
        """Son 24 saat tahmin sayısı"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM predictions 
                WHERE created_at > datetime('now', '-1 day')
            ''')
            return cursor.fetchone()[0]
        except:
            return 0
    
    def get_confidence_distribution(self):
        """Güven seviyesi dağılımı"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT confidence FROM matches WHERE confidence > 0')
            results = cursor.fetchall()
            return [row[0] for row in results if row[0]]
        except:
            return []
    
    def get_league_statistics(self):
        """Lig istatistikleri"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT 
                    league,
                    COUNT(*) as team_count,
                    AVG(points) as avg_points,
                    AVG(goals_for) as avg_goals_for,
                    AVG(goals_against) as avg_goals_against
                FROM team_stats 
                GROUP BY league
                ORDER BY team_count DESC
            ''')
            
            results = cursor.fetchall()
            
            if results:
                columns = ['Lig', 'Takım Sayısı', 'Ort. Puan', 'Ort. Gol Attı', 'Ort. Gol Yedi']
                return [dict(zip(columns, row)) for row in results]
            
            return []
            
        except Exception as e:
            logger.error(f"Lig istatistik hatası: {e}")
            return []
    
    def get_latest_predictions_from_db(self, min_confidence=60.0):
        """Tahminleri veritabanından getir"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT * FROM predictions 
                WHERE confidence >= ? 
                ORDER BY created_at DESC
                LIMIT 20
            ''', (min_confidence,))
            
            results = cursor.fetchall()
            
            # Sütun isimlerini al
            columns = [description[0] for description in cursor.description]
            
            # Dictionary formatına çevir
            predictions = []
            for row in results:
                prediction_dict = dict(zip(columns, row))
                predictions.append(prediction_dict)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Tahmin getirme hatası: {e}")
            return []
    
    def get_average_confidence_predictions(self):
        """Tahminler tablosundan ortalama güven seviyesi"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT AVG(confidence) FROM predictions')
            result = cursor.fetchone()
            return result[0] or 0.0
        except:
            return 0.0
    
    def close(self):
        """Veritabanı bağlantısını kapat"""
        if hasattr(self, 'conn'):
            self.conn.close()
