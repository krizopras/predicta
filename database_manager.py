#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Veritabanı yönetim modülü - AI Entegrasyonlu
"""

import sqlite3
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class DatabaseManager:
    
    def __init__(self, db_path="data/nesine_advanced.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Veritabanı yapısını oluştur - AI özellikleri eklendi"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Maçlar tablosu (AI alanları eklendi)
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
                -- YENİ: AI özellikleri
                source TEXT DEFAULT 'basic',  -- 'basic' veya 'ai_enhanced'
                certainty_index REAL DEFAULT 0.5,
                risk_factors TEXT,  -- JSON string olarak risk faktörleri
                ai_model_version TEXT DEFAULT '1.0',
                features_used INTEGER DEFAULT 0,
                blended_weights TEXT,  -- JSON string olarak ağırlıklar
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Takım istatistikleri (AI için geliştirilmiş)
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
                -- YENİ: AI özellikleri
                advanced_strength REAL DEFAULT 0.5,
                form_momentum REAL DEFAULT 0.5,
                consistency_score REAL DEFAULT 0.5,
                performance_trend REAL DEFAULT 0.5,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(team_name, league)
            )
        ''')
        
        # AI Model Performans Tablosu (YENİ)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_model_performance (
                id INTEGER PRIMARY KEY,
                model_name TEXT,
                accuracy REAL,
                training_samples INTEGER,
                feature_count INTEGER,
                cross_val_score REAL,
                training_duration REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # AI Feature Importance Tablosu (YENİ)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_feature_importance (
                id INTEGER PRIMARY KEY,
                feature_name TEXT,
                importance_score REAL,
                model_version TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tahmin sonuçları (AI alanları eklendi)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                match_code TEXT,
                home_team TEXT,
                away_team TEXT,
                league TEXT,
                prediction_result TEXT,
                prediction_iy TEXT,
                predicted_score TEXT,
                confidence REAL,
                -- YENİ: AI özellikleri
                source TEXT DEFAULT 'basic',
                certainty_index REAL DEFAULT 0.5,
                risk_factors TEXT,
                ai_powered BOOLEAN DEFAULT 0,
                model_version TEXT DEFAULT '1.0',
                probabilities TEXT,  -- JSON string olarak olasılıklar
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Geçmiş karşılaşmalar (aynı)
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
        
        # Oyuncu istatistikleri (aynı)
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
        
        self.conn.commit()
        logger.info("🤖 AI entegre veritabanı yapısı kuruldu")
    
    def save_prediction(self, match_data: Dict, prediction_data: Dict, source: str = "basic"):
        """Tahmin sonucunu kaydet - AI desteği eklendi"""
        try:
            cursor = self.conn.cursor()
            
            # Match code oluştur
            match_code = f"{match_data.get('home_team', '')}_{match_data.get('away_team', '')}_{match_data.get('date', '')}"
            
            # Risk faktörlerini JSON'a çevir
            risk_factors_json = json.dumps(prediction_data.get('risk_factors', {})) if prediction_data.get('risk_factors') else None
            
            # Olasılıkları JSON'a çevir
            probabilities_json = json.dumps(prediction_data.get('probabilities', {})) if prediction_data.get('probabilities') else None
            
            # Blended weights JSON'a çevir
            blended_weights_json = json.dumps(prediction_data.get('blended_weights', {})) if prediction_data.get('blended_weights') else None
            
            # Maç tablosunu güncelle (AI alanları eklendi)
            cursor.execute('''
                INSERT OR REPLACE INTO matches 
                (match_code, home_team, away_team, league, match_date, match_time, 
                 odds_1, odds_x, odds_2, prediction_ms, prediction_iy, predicted_score, 
                 confidence, source, certainty_index, risk_factors, ai_model_version,
                 features_used, blended_weights, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                match_code,
                match_data.get('home_team', ''),
                match_data.get('away_team', ''),
                match_data.get('league', ''),
                match_data.get('date', ''),
                match_data.get('time', ''),
                match_data.get('odds', {}).get('1', 0),
                match_data.get('odds', {}).get('X', 0),
                match_data.get('odds', {}).get('2', 0),
                prediction_data.get('result_prediction', ''),
                prediction_data.get('iy_prediction', ''),
                prediction_data.get('score_prediction', ''),
                prediction_data.get('confidence', 0),
                source,
                prediction_data.get('certainty_index', 0.5),
                risk_factors_json,
                prediction_data.get('model_version', '1.0'),
                prediction_data.get('features_used', 0),
                blended_weights_json,
                datetime.now()
            ))
            
            # Tahmin tablosuna da kaydet (AI alanları eklendi)
            cursor.execute('''
                INSERT INTO predictions 
                (match_code, home_team, away_team, league, prediction_result, prediction_iy, 
                 predicted_score, confidence, source, certainty_index, risk_factors,
                 ai_powered, model_version, probabilities, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                match_code,
                match_data.get('home_team', ''),
                match_data.get('away_team', ''),
                match_data.get('league', ''),
                prediction_data.get('result_prediction', ''),
                prediction_data.get('iy_prediction', ''),
                prediction_data.get('score_prediction', ''),
                prediction_data.get('confidence', 0),
                source,
                prediction_data.get('certainty_index', 0.5),
                risk_factors_json,
                prediction_data.get('ai_powered', False),
                prediction_data.get('model_version', '1.0'),
                probabilities_json,
                datetime.now()
            ))
            
            self.conn.commit()
            logger.debug(f"✅ Tahmin kaydedildi: {match_code} (Source: {source})")
            
        except Exception as e:
            logger.error(f"❌ Tahmin kaydetme hatası: {e}")
    
    def save_ai_model_performance(self, model_name: str, accuracy: float, training_samples: int, 
                                feature_count: int, cross_val_score: float, training_duration: float):
        """AI model performansını kaydet"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                INSERT INTO ai_model_performance 
                (model_name, accuracy, training_samples, feature_count, cross_val_score, training_duration)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                model_name,
                accuracy,
                training_samples,
                feature_count,
                cross_val_score,
                training_duration
            ))
            
            self.conn.commit()
            logger.info(f"📊 AI model performansı kaydedildi: {model_name} - Doğruluk: {accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"AI model performans kaydetme hatası: {e}")
    
    def save_feature_importance(self, feature_name: str, importance_score: float, model_version: str = "1.0"):
        """Özellik önem skorlarını kaydet"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                INSERT INTO ai_feature_importance 
                (feature_name, importance_score, model_version)
                VALUES (?, ?, ?)
            ''', (feature_name, importance_score, model_version))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Özellik önem kaydetme hatası: {e}")
    
    def get_ai_predictions(self, min_confidence: float = 60.0, limit: int = 20):
        """Sadece AI tarafından yapılan tahminleri getir"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT * FROM predictions 
                WHERE source = 'ai_enhanced' AND confidence >= ? 
                ORDER BY created_at DESC
                LIMIT ?
            ''', (min_confidence, limit))
            
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            predictions = []
            for row in results:
                pred_dict = dict(zip(columns, row))
                
                # JSON alanlarını parse et
                if pred_dict.get('risk_factors'):
                    try:
                        pred_dict['risk_factors'] = json.loads(pred_dict['risk_factors'])
                    except:
                        pred_dict['risk_factors'] = {}
                
                if pred_dict.get('probabilities'):
                    try:
                        pred_dict['probabilities'] = json.loads(pred_dict['probabilities'])
                    except:
                        pred_dict['probabilities'] = {}
                
                predictions.append(pred_dict)
            
            return predictions
            
        except Exception as e:
            logger.error(f"AI tahmin getirme hatası: {e}")
            return []
    
    def get_ai_model_performance(self, days: int = 30):
        """AI model performans geçmişini getir"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT * FROM ai_model_performance 
                WHERE timestamp > datetime('now', '-{} days')
                ORDER BY timestamp DESC
            '''.format(days))
            
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            logger.error(f"AI performans getirme hatası: {e}")
            return []
    
    def get_feature_importance(self, model_version: str = "1.0", limit: int = 10):
        """Özellik önem skorlarını getir"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT feature_name, importance_score 
                FROM ai_feature_importance 
                WHERE model_version = ?
                ORDER BY importance_score DESC
                LIMIT ?
            ''', (model_version, limit))
            
            results = cursor.fetchall()
            return [{'feature': row[0], 'importance': row[1]} for row in results]
            
        except Exception as e:
            logger.error(f"Özellik önem getirme hatası: {e}")
            return []
    
    def get_ai_statistics(self):
        """AI istatistiklerini getir"""
        try:
            cursor = self.conn.cursor()
            
            # Toplam AI tahmin sayısı
            cursor.execute('SELECT COUNT(*) FROM predictions WHERE source = "ai_enhanced"')
            total_ai_predictions = cursor.fetchone()[0]
            
            # Ortalama AI güven seviyesi
            cursor.execute('SELECT AVG(confidence) FROM predictions WHERE source = "ai_enhanced"')
            avg_ai_confidence = cursor.fetchone()[0] or 0.0
            
            # Son 7 gün AI tahmin sayısı
            cursor.execute('''
                SELECT COUNT(*) FROM predictions 
                WHERE source = "ai_enhanced" AND created_at > datetime('now', '-7 days')
            ''')
            recent_ai_predictions = cursor.fetchone()[0]
            
            # En yüksek güvenli AI tahmin
            cursor.execute('''
                SELECT confidence, home_team, away_team 
                FROM predictions 
                WHERE source = "ai_enhanced" 
                ORDER BY confidence DESC LIMIT 1
            ''')
            best_ai_result = cursor.fetchone()
            best_confidence = best_ai_result[0] if best_ai_result else 0.0
            best_match = f"{best_ai_result[1]} vs {best_ai_result[2]}" if best_ai_result else "Yok"
            
            return {
                'total_ai_predictions': total_ai_predictions,
                'avg_ai_confidence': round(avg_ai_confidence, 2),
                'recent_ai_predictions': recent_ai_predictions,
                'best_ai_confidence': round(best_confidence, 2),
                'best_ai_match': best_match,
                'ai_success_rate': self.calculate_ai_success_rate()
            }
            
        except Exception as e:
            logger.error(f"AI istatistik getirme hatası: {e}")
            return {}
    
    def calculate_ai_success_rate(self):
        """AI başarı oranını hesapla (basit implementasyon)"""
        try:
            cursor = self.conn.cursor()
            
            # Burada gerçek maç sonuçlarıyla karşılaştırma yapılabilir
            # Şimdilik basit bir hesaplama
            cursor.execute('''
                SELECT COUNT(*) FROM predictions 
                WHERE source = "ai_enhanced" AND confidence > 70
            ''')
            high_confidence_count = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT COUNT(*) FROM predictions 
                WHERE source = "ai_enhanced"
            ''')
            total_count = cursor.fetchone()[0]
            
            if total_count > 0:
                return round((high_confidence_count / total_count) * 100, 2)
            return 0.0
            
        except Exception as e:
            logger.error(f"AI başarı oranı hesaplama hatası: {e}")
            return 0.0
    
    def get_latest_predictions_from_db(self, min_confidence=60.0, limit=20):
        """Tahminleri veritabanından getir - AI desteği eklendi"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT * FROM predictions 
                WHERE confidence >= ? 
                ORDER BY created_at DESC
                LIMIT ?
            ''', (min_confidence, limit))
            
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            predictions = []
            for row in results:
                prediction_dict = dict(zip(columns, row))
                
                # JSON alanlarını parse et
                if prediction_dict.get('risk_factors'):
                    try:
                        prediction_dict['risk_factors'] = json.loads(prediction_dict['risk_factors'])
                    except:
                        prediction_dict['risk_factors'] = {}
                
                if prediction_dict.get('probabilities'):
                    try:
                        prediction_dict['probabilities'] = json.loads(prediction_dict['probabilities'])
                    except:
                        prediction_dict['probabilities'] = {}
                
                predictions.append(prediction_dict)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Tahmin getirme hatası: {e}")
            return []
    
    # Aşağıdaki mevcut fonksiyonlar aynı kalacak, sadece küçük iyileştirmeler
    
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
        """Takım istatistiklerini kaydet - AI alanları eklendi"""
        try:
            cursor = self.conn.cursor()
            
            recent_form_str = json.dumps(stats.get('recent_form', []))
            
            cursor.execute('''
                INSERT OR REPLACE INTO team_stats 
                (team_name, league, position, points, matches_played, wins, draws, losses,
                 goals_for, goals_against, home_wins, home_draws, home_losses,
                 away_wins, away_draws, away_losses, recent_form, 
                 advanced_strength, form_momentum, consistency_score, performance_trend, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                stats.get('advanced_strength', 0.5),
                stats.get('form_momentum', 0.5),
                stats.get('consistency_score', 0.5),
                stats.get('performance_trend', 0.5),
                datetime.now()
            ))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Takım istatistik kaydetme hatası: {e}")
    
    # Diğer mevcut fonksiyonlar aynı kalacak...
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
                    
                    match_dict['odds'] = {
                        '1': match_dict.get('odds_1', 0),
                        'X': match_dict.get('odds_x', 0),
                        '2': match_dict.get('odds_2', 0)
                    }
                    
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
                
                if stats.get('recent_form'):
                    try:
                        stats['recent_form'] = json.loads(stats['recent_form'])
                    except:
                        stats['recent_form'] = []
                
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
# Compatibility alias for main.py
class AIDatabaseManager(DatabaseManager):
    """AI Database Manager - DatabaseManager'ın alias'ı"""
    
    def __init__(self, db_path="data/predicta_ai.db"):
        super().__init__(db_path)
        logger.info("AIDatabaseManager başlatıldı (DatabaseManager alias)")
    
    def save_match_prediction(self, data: Dict):
        """Maç tahminini kaydet - main.py uyumluluğu için"""
        try:
            # Main.py'den gelen veri formatını DatabaseManager formatına çevir
            match_data = {
                'home_team': data.get('home_team', ''),
                'away_team': data.get('away_team', ''),
                'league': data.get('league', ''),
                'date': data.get('match_date', ''),
                'odds': data.get('odds', {})
            }
            
            prediction_data = {
                'result_prediction': data.get('ai_prediction', {}).get('prediction', ''),
                'confidence': data.get('ai_prediction', {}).get('confidence', 0),
                'model_version': '3.0'
            }
            
            # DatabaseManager'ın save_prediction metodunu kullan
            self.save_prediction(match_data, prediction_data, source="ai_enhanced")
            
        except Exception as e:
            logger.error(f"Match prediction save error: {e}")
    
    def get_recent_matches(self, league: str, limit: int = 10) -> List[Dict]:
        """Son maçları getir - main.py formatında"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT home_team, away_team, league, match_date, 
                       odds_1, odds_x, odds_2, prediction_ms, confidence
                FROM matches 
                WHERE league = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (league, limit))
            
            matches = []
            for row in cursor.fetchall():
                matches.append({
                    'home_team': row[0],
                    'away_team': row[1],
                    'league': row[2],
                    'match_date': row[3],
                    'odds': {'1': row[4], 'X': row[5], '2': row[6]},
                    'prediction': row[7],
                    'confidence': row[8]
                })
            
            return matches
            
        except Exception as e:
            logger.error(f"Recent matches error: {e}")
            return []            
