#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Entegre Veritabanı Yöneticisi - Gelişmiş Futbol Tahmin Sistemi
"""

import sqlite3
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class AIDatabaseManager:
    def __init__(self, db_path: str = "data/nesine_advanced.db"):
        self.db_path = db_path
        self._ensure_directories()
        self._init_database()
    
    def _ensure_directories(self):
        """Gerekli dizinleri oluştur"""
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/ai_models_v2", exist_ok=True)
    
    def _init_database(self):
        """Veritabanını başlat ve tabloları oluştur"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # matches tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    league TEXT NOT NULL,
                    match_date TEXT,
                    home_score INTEGER,
                    away_score INTEGER,
                    result TEXT,
                    odds_1 REAL,
                    odds_x REAL,
                    odds_2 REAL,
                    possession_home INTEGER,
                    possession_away INTEGER,
                    shots_home INTEGER,
                    shots_away INTEGER,
                    corners_home INTEGER,
                    corners_away INTEGER,
                    fouls_home INTEGER,
                    fouls_away INTEGER,
                    source TEXT DEFAULT 'basic',
                    certainty_index REAL DEFAULT 0.5,
                    risk_factors TEXT,
                    ai_model_version TEXT DEFAULT '1.0',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # predictions tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id INTEGER,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    league TEXT NOT NULL,
                    prediction_result TEXT,
                    confidence REAL,
                    home_win_prob REAL,
                    draw_prob REAL,
                    away_win_prob REAL,
                    score_prediction TEXT,
                    analysis TEXT,
                    source TEXT DEFAULT 'basic',
                    certainty_index REAL DEFAULT 0.5,
                    risk_factors TEXT,
                    ai_powered BOOLEAN DEFAULT 0,
                    probabilities TEXT,
                    is_correct BOOLEAN,
                    actual_result TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (match_id) REFERENCES matches (id)
                )
            ''')
            
            # team_stats tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS team_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_name TEXT NOT NULL,
                    league TEXT NOT NULL,
                    position INTEGER,
                    points INTEGER,
                    matches_played INTEGER,
                    wins INTEGER,
                    draws INTEGER,
                    losses INTEGER,
                    goals_for INTEGER,
                    goals_against INTEGER,
                    goal_difference INTEGER,
                    recent_form TEXT,
                    xG_for REAL,
                    xG_against REAL,
                    advanced_strength REAL DEFAULT 0.5,
                    form_momentum REAL DEFAULT 0.5,
                    consistency_score REAL DEFAULT 0.5,
                    performance_trend REAL DEFAULT 0.5,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(team_name, league)
                )
            ''')
            
            # ai_model_performance tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT,
                    accuracy REAL,
                    training_samples INTEGER,
                    feature_count INTEGER,
                    cross_val_score REAL,
                    training_duration REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    model_stability REAL,
                    adaptation_status TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # ai_feature_importance tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_feature_importance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_name TEXT,
                    importance_score REAL,
                    model_version TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Indexler
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_matches_league_date 
                ON matches(league, match_date)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_predictions_source 
                ON predictions(source)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_predictions_created 
                ON predictions(created_at)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_team_stats_league 
                ON team_stats(league, position)
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ Veritabanı tabloları başarıyla oluşturuldu")
            
        except Exception as e:
            logger.error(f"❌ Veritabanı başlatma hatası: {e}")
    
    def save_match_prediction(self, data: Dict[str, Any]) -> bool:
        """Maç tahminini veritabanına kaydet"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Önce maçı kaydet
            match_data = {
                'home_team': data.get('home_team', ''),
                'away_team': data.get('away_team', ''),
                'league': data.get('league', ''),
                'match_date': data.get('match_date', datetime.now().isoformat()),
                'odds_1': data.get('odds', {}).get('1', 0),
                'odds_x': data.get('odds', {}).get('X', 0),
                'odds_2': data.get('odds', {}).get('2', 0),
                'source': data.get('source', 'basic'),
                'certainty_index': data.get('certainty_index', 0.5),
                'risk_factors': json.dumps(data.get('risk_factors', {})) if data.get('risk_factors') else None,
                'ai_model_version': data.get('ai_model_version', '1.0')
            }
            
            cursor.execute('''
                INSERT OR REPLACE INTO matches 
                (home_team, away_team, league, match_date, odds_1, odds_x, odds_2, 
                 source, certainty_index, risk_factors, ai_model_version, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                match_data['home_team'], match_data['away_team'], match_data['league'],
                match_data['match_date'], match_data['odds_1'], match_data['odds_x'],
                match_data['odds_2'], match_data['source'], match_data['certainty_index'],
                match_data['risk_factors'], match_data['ai_model_version']
            ))
            
            match_id = cursor.lastrowid
            
            # Tahmini kaydet
            prediction_data = data.get('ai_prediction', {})
            probabilities = {
                '1': prediction_data.get('home_win_prob', 0),
                'X': prediction_data.get('draw_prob', 0),
                '2': prediction_data.get('away_win_prob', 0)
            }
            
            cursor.execute('''
                INSERT INTO predictions 
                (match_id, home_team, away_team, league, prediction_result, confidence,
                 home_win_prob, draw_prob, away_win_prob, score_prediction, analysis,
                 source, certainty_index, risk_factors, ai_powered, probabilities)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                match_id, data['home_team'], data['away_team'], data['league'],
                prediction_data.get('prediction', ''),
                prediction_data.get('confidence', 0),
                prediction_data.get('home_win_prob', 0),
                prediction_data.get('draw_prob', 0),
                prediction_data.get('away_win_prob', 0),
                prediction_data.get('score_prediction', ''),
                prediction_data.get('analysis', ''),
                data.get('source', 'basic'),
                data.get('certainty_index', 0.5),
                json.dumps(data.get('risk_factors', {})) if data.get('risk_factors') else None,
                data.get('ai_powered', False),
                json.dumps(probabilities)
            ))
            
            conn.commit()
            conn.close()
            logger.debug(f"✅ Tahmin kaydedildi: {data['home_team']} vs {data['away_team']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Tahmin kaydetme hatası: {e}")
            return False
    
    def get_team_stats(self, team_name: str, league: str) -> Optional[Dict]:
        """Takım istatistiklerini getir"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM team_stats 
                WHERE team_name = ? AND league = ?
                ORDER BY last_updated DESC LIMIT 1
            ''', (team_name, league))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, result))
            
            return self._generate_fallback_stats(team_name, league)
            
        except Exception as e:
            logger.error(f"❌ Takım istatistikleri getirme hatası: {e}")
            return self._generate_fallback_stats(team_name, league)
    
    def _generate_fallback_stats(self, team_name: str, league: str) -> Dict:
        """Fallback takım istatistikleri oluştur"""
        return {
            'team_name': team_name,
            'league': league,
            'position': np.random.randint(1, 20),
            'points': np.random.randint(10, 60),
            'matches_played': np.random.randint(10, 30),
            'wins': np.random.randint(3, 20),
            'draws': np.random.randint(3, 15),
            'losses': np.random.randint(3, 15),
            'goals_for': np.random.randint(10, 50),
            'goals_against': np.random.randint(10, 40),
            'goal_difference': 0,
            'recent_form': json.dumps(['G', 'B', 'M', 'G', 'B'][:np.random.randint(3, 6)]),
            'xG_for': round(np.random.uniform(20, 45), 1),
            'xG_against': round(np.random.uniform(15, 35), 1),
            'advanced_strength': np.random.uniform(0.3, 0.8),
            'form_momentum': np.random.uniform(0.2, 0.9),
            'consistency_score': np.random.uniform(0.4, 0.8),
            'performance_trend': np.random.uniform(0.3, 0.7)
        }
    
    def get_recent_matches(self, league: str, limit: int = 10) -> List[Dict]:
        """Son maçları getir"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT m.*, p.prediction_result, p.confidence, p.ai_powered
                FROM matches m
                LEFT JOIN predictions p ON m.id = p.match_id
                WHERE m.league = ?
                ORDER BY m.match_date DESC, m.created_at DESC
                LIMIT ?
            ''', (league, limit))
            
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            matches = []
            for result in results:
                match_dict = dict(zip(columns, result))
                
                # Oranları dict formatına çevir
                match_dict['odds'] = {
                    '1': match_dict.get('odds_1', 0),
                    'X': match_dict.get('odds_x', 0),
                    '2': match_dict.get('odds_2', 0)
                }
                
                # İstatistikleri dict formatına çevir
                match_dict['stats'] = {
                    'possession': {
                        'home': match_dict.get('possession_home', 50),
                        'away': match_dict.get('possession_away', 50)
                    },
                    'shots': {
                        'home': match_dict.get('shots_home', 0),
                        'away': match_dict.get('shots_away', 0)
                    },
                    'corners': {
                        'home': match_dict.get('corners_home', 0),
                        'away': match_dict.get('corners_away', 0)
                    }
                }
                
                matches.append(match_dict)
            
            conn.close()
            return matches
            
        except Exception as e:
            logger.error(f"❌ Maç listeleme hatası: {e}")
            return []
    
    def save_ai_performance(self, performance_data: Dict) -> bool:
        """AI performans metriklerini kaydet"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ai_model_performance 
                (model_name, accuracy, training_samples, feature_count, cross_val_score,
                 training_duration, precision, recall, f1_score, model_stability, adaptation_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                performance_data.get('model_name', 'EnhancedSuperLearningAI'),
                performance_data.get('accuracy', 0),
                performance_data.get('training_samples', 0),
                performance_data.get('feature_count', 0),
                performance_data.get('cross_val_score', 0),
                performance_data.get('training_duration', 0),
                performance_data.get('precision', 0),
                performance_data.get('recall', 0),
                performance_data.get('f1_score', 0),
                performance_data.get('model_stability', 0),
                performance_data.get('adaptation_status', 'unknown')
            ))
            
            conn.commit()
            conn.close()
            logger.info("✅ AI performans metrikleri kaydedildi")
            return True
            
        except Exception as e:
            logger.error(f"❌ AI performans kaydetme hatası: {e}")
            return False
    
    def save_feature_importance(self, features: Dict[str, float], model_version: str = "1.0"):
        """Özellik önem skorlarını kaydet"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for feature_name, importance_score in features.items():
                cursor.execute('''
                    INSERT INTO ai_feature_importance 
                    (feature_name, importance_score, model_version)
                    VALUES (?, ?, ?)
                ''', (feature_name, importance_score, model_version))
            
            conn.commit()
            conn.close()
            logger.info(f"✅ {len(features)} özellik önem skoru kaydedildi")
            
        except Exception as e:
            logger.error(f"❌ Özellik önem kaydetme hatası: {e}")
    
    def get_training_data(self, limit: int = 1000) -> pd.DataFrame:
        """AI eğitimi için verileri getir"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT 
                    m.home_team, m.away_team, m.league,
                    m.odds_1, m.odds_x, m.odds_2,
                    m.possession_home, m.possession_away,
                    m.shots_home, m.shots_away,
                    m.corners_home, m.corners_away,
                    m.fouls_home, m.fouls_away,
                    m.home_score, m.away_score,
                    m.result,
                    p.confidence,
                    ts_home.advanced_strength as home_strength,
                    ts_away.advanced_strength as away_strength,
                    ts_home.form_momentum as home_momentum,
                    ts_away.form_momentum as away_momentum
                FROM matches m
                LEFT JOIN predictions p ON m.id = p.match_id
                LEFT JOIN team_stats ts_home ON m.home_team = ts_home.team_name AND m.league = ts_home.league
                LEFT JOIN team_stats ts_away ON m.away_team = ts_away.team_name AND m.league = ts_away.league
                WHERE m.result IS NOT NULL
                ORDER BY m.match_date DESC
                LIMIT ?
            '''
            
            df = pd.read_sql_query(query, conn, params=(limit,))
            conn.close()
            
            logger.info(f"✅ {len(df)} eğitim verisi yüklendi")
            return df
            
        except Exception as e:
            logger.error(f"❌ Eğitim verisi yükleme hatası: {e}")
            return pd.DataFrame()
    
    def update_team_stats(self, team_data: Dict) -> bool:
        """Takım istatistiklerini güncelle"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO team_stats 
                (team_name, league, position, points, matches_played, wins, draws, losses,
                 goals_for, goals_against, goal_difference, recent_form, xG_for, xG_against,
                 advanced_strength, form_momentum, consistency_score, performance_trend, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                team_data['team_name'],
                team_data['league'],
                team_data.get('position', 0),
                team_data.get('points', 0),
                team_data.get('matches_played', 0),
                team_data.get('wins', 0),
                team_data.get('draws', 0),
                team_data.get('losses', 0),
                team_data.get('goals_for', 0),
                team_data.get('goals_against', 0),
                team_data.get('goal_difference', 0),
                json.dumps(team_data.get('recent_form', [])),
                team_data.get('xG_for', 0),
                team_data.get('xG_against', 0),
                team_data.get('advanced_strength', 0.5),
                team_data.get('form_momentum', 0.5),
                team_data.get('consistency_score', 0.5),
                team_data.get('performance_trend', 0.5)
            ))
            
            conn.commit()
            conn.close()
            logger.debug(f"✅ Takım istatistikleri güncellendi: {team_data['team_name']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Takım istatistikleri güncelleme hatası: {e}")
            return False
    
    def get_ai_performance_history(self, days: int = 30) -> List[Dict]:
        """AI performans geçmişini getir"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM ai_model_performance 
                WHERE timestamp >= datetime('now', ?)
                ORDER BY timestamp DESC
            ''', (f'-{days} days',))
            
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            performance_data = []
            for result in results:
                performance_data.append(dict(zip(columns, result)))
            
            conn.close()
            return performance_data
            
        except Exception as e:
            logger.error(f"❌ AI performans geçmişi getirme hatası: {e}")
            return []
    
    def cleanup_old_data(self, days: int = 90):
        """Eski verileri temizle"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Eski maçları sil
            cursor.execute('''
                DELETE FROM matches 
                WHERE created_at < datetime('now', ?)
            ''', (f'-{days} days',))
            
            # Eski tahminleri sil
            cursor.execute('''
                DELETE FROM predictions 
                WHERE created_at < datetime('now', ?)
            ''', (f'-{days} days',))
            
            conn.commit()
            conn.close()
            logger.info(f"✅ {days} günden eski veriler temizlendi")
            
        except Exception as e:
            logger.error(f"❌ Veri temizleme hatası: {e}")
