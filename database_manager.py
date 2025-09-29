#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Entegre Veritabanı Yöneticisi - Gelişmiş Futbol Tahmin Sistemi
"""

import sqlite3  # Python built-in
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
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
            
            # Sadece temel tablolar
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    league TEXT NOT NULL,
                    match_date TEXT,
                    odds_1 REAL,
                    odds_x REAL,
                    odds_2 REAL,
                    source TEXT DEFAULT 'basic',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    league TEXT NOT NULL,
                    prediction_result TEXT,
                    confidence REAL,
                    ai_powered BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    accuracy REAL,
                    training_samples INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
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
            
            # Maçı kaydet
            cursor.execute('''
                INSERT INTO matches 
                (home_team, away_team, league, match_date, odds_1, odds_x, odds_2, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.get('home_team', ''),
                data.get('away_team', ''),
                data.get('league', ''),
                data.get('match_date', datetime.now().isoformat()),
                data.get('odds', {}).get('1', 0),
                data.get('odds', {}).get('X', 0),
                data.get('odds', {}).get('2', 0),
                data.get('source', 'basic')
            ))
            
            # Tahmini kaydet
            prediction_data = data.get('ai_prediction', {})
            cursor.execute('''
                INSERT INTO predictions 
                (home_team, away_team, league, prediction_result, confidence, ai_powered)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                data['home_team'],
                data['away_team'],
                data['league'],
                prediction_data.get('prediction', ''),
                prediction_data.get('confidence', 0),
                data.get('ai_powered', False)
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
        # Basit fallback istatistikler
        return {
            'team_name': team_name,
            'league': league,
            'position': np.random.randint(1, 20),
            'points': np.random.randint(10, 60),
            'goals_for': np.random.randint(10, 50),
            'goals_against': np.random.randint(10, 40),
        }
    
    def get_recent_matches(self, league: str, limit: int = 10) -> List[Dict]:
        """Son maçları getir"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT m.*, p.prediction_result, p.confidence, p.ai_powered
                FROM matches m
                LEFT JOIN predictions p ON m.home_team = p.home_team AND m.away_team = p.away_team
                WHERE m.league = ?
                ORDER BY m.created_at DESC
                LIMIT ?
            ''', (league, limit))
            
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            matches = []
            for result in results:
                match_dict = dict(zip(columns, result))
                match_dict['odds'] = {
                    '1': match_dict.get('odds_1', 0),
                    'X': match_dict.get('odds_x', 0),
                    '2': match_dict.get('odds_2', 0)
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
                (accuracy, training_samples)
                VALUES (?, ?)
            ''', (
                performance_data.get('accuracy', 0),
                performance_data.get('training_samples', 0)
            ))
            
            conn.commit()
            conn.close()
            logger.info("✅ AI performans metrikleri kaydedildi")
            return True
            
        except Exception as e:
            logger.error(f"❌ AI performans kaydetme hatası: {e}")
            return False
    
    def get_training_data(self, limit: int = 1000) -> pd.DataFrame:
        """AI eğitimi için verileri getir"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT 
                    m.home_team, m.away_team, m.league,
                    m.odds_1, m.odds_x, m.odds_2,
                    p.confidence
                FROM matches m
                LEFT JOIN predictions p ON m.home_team = p.home_team AND m.away_team = p.away_team
                LIMIT ?
            '''
            
            df = pd.read_sql_query(query, conn, params=(limit,))
            conn.close()
            
            logger.info(f"✅ {len(df)} eğitim verisi yüklendi")
            return df
            
        except Exception as e:
            logger.error(f"❌ Eğitim verisi yükleme hatası: {e}")
            return pd.DataFrame()
