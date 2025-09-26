# update_database.py
#!/usr/bin/env python3
import sqlite3
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_database_schema():
    """Mevcut veritabanını AI özellikleri ile günceller"""
    
    db_path = "data/nesine_advanced.db"
    
    # Veritabanı dizinini oluştur
    os.makedirs("data", exist_ok=True)
    
    # Database yoksa oluştur
    if not os.path.exists(db_path):
        logger.info("📋 Database bulunamadı, yeni oluşturuluyor...")
        from database_manager import DatabaseManager
        db_manager = DatabaseManager(db_path)
        logger.info("✅ Yeni database oluşturuldu")
        return True
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Mevcut tabloları kontrol et
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [table[0] for table in cursor.fetchall()]
        logger.info(f"📊 Mevcut tablolar: {existing_tables}")
        
        # Yeni AI tablolarını ekle
        if 'ai_model_performance' not in existing_tables:
            cursor.execute('''
                CREATE TABLE ai_model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT,
                    accuracy REAL,
                    training_samples INTEGER,
                    feature_count INTEGER,
                    cross_val_score REAL,
                    training_duration REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            logger.info("✅ AI model performance tablosu eklendi")
        
        if 'ai_feature_importance' not in existing_tables:
            cursor.execute('''
                CREATE TABLE ai_feature_importance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_name TEXT,
                    importance_score REAL,
                    model_version TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            logger.info("✅ AI feature importance tablosu eklendi")
        
        # Mevcut tablolara AI sütunlarını ekle
        table_columns = {}
        for table in ['matches', 'predictions', 'team_stats']:
            cursor.execute(f"PRAGMA table_info({table})")
            table_columns[table] = [column[1] for column in cursor.fetchall()]
        
        # matches tablosuna AI sütunları ekle
        if 'source' not in table_columns.get('matches', []):
            cursor.execute("ALTER TABLE matches ADD COLUMN source TEXT DEFAULT 'basic'")
            logger.info("✅ matches tablosuna source sütunu eklendi")
        
        if 'certainty_index' not in table_columns.get('matches', []):
            cursor.execute("ALTER TABLE matches ADD COLUMN certainty_index REAL DEFAULT 0.5")
            logger.info("✅ matches tablosuna certainty_index sütunu eklendi")
        
        if 'risk_factors' not in table_columns.get('matches', []):
            cursor.execute("ALTER TABLE matches ADD COLUMN risk_factors TEXT")
            logger.info("✅ matches tablosuna risk_factors sütunu eklendi")
        
        if 'ai_model_version' not in table_columns.get('matches', []):
            cursor.execute("ALTER TABLE matches ADD COLUMN ai_model_version TEXT DEFAULT '1.0'")
            logger.info("✅ matches tablosuna ai_model_version sütunu eklendi")
        
        # predictions tablosuna AI sütunları ekle
        if 'source' not in table_columns.get('predictions', []):
            cursor.execute("ALTER TABLE predictions ADD COLUMN source TEXT DEFAULT 'basic'")
            logger.info("✅ predictions tablosuna source sütunu eklendi")
        
        if 'certainty_index' not in table_columns.get('predictions', []):
            cursor.execute("ALTER TABLE predictions ADD COLUMN certainty_index REAL DEFAULT 0.5")
            logger.info("✅ predictions tablosuna certainty_index sütunu eklendi")
        
        if 'risk_factors' not in table_columns.get('predictions', []):
            cursor.execute("ALTER TABLE predictions ADD COLUMN risk_factors TEXT")
            logger.info("✅ predictions tablosuna risk_factors sütunu eklendi")
        
        if 'ai_powered' not in table_columns.get('predictions', []):
            cursor.execute("ALTER TABLE predictions ADD COLUMN ai_powered BOOLEAN DEFAULT 0")
            logger.info("✅ predictions tablosuna ai_powered sütunu eklendi")
        
        if 'probabilities' not in table_columns.get('predictions', []):
            cursor.execute("ALTER TABLE predictions ADD COLUMN probabilities TEXT")
            logger.info("✅ predictions tablosuna probabilities sütunu eklendi")
        
        # team_stats tablosuna AI sütunları ekle
        ai_columns = ['advanced_strength', 'form_momentum', 'consistency_score', 'performance_trend']
        for column in ai_columns:
            if column not in table_columns.get('team_stats', []):
                cursor.execute(f"ALTER TABLE team_stats ADD COLUMN {column} REAL DEFAULT 0.5")
                logger.info(f"✅ team_stats tablosuna {column} sütunu eklendi")
        
        # Indexleri ekle
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        existing_indexes = [index[0] for index in cursor.fetchall()]
        
        if 'idx_predictions_source' not in existing_indexes:
            cursor.execute("CREATE INDEX idx_predictions_source ON predictions(source)")
            logger.info("✅ predictions source index eklendi")
        
        if 'idx_predictions_created' not in existing_indexes:
            cursor.execute("CREATE INDEX idx_predictions_created ON predictions(created_at)")
            logger.info("✅ predictions created_at index eklendi")
        
        if 'idx_ai_performance_timestamp' not in existing_indexes:
            cursor.execute("CREATE INDEX idx_ai_performance_timestamp ON ai_model_performance(timestamp)")
            logger.info("✅ AI performance timestamp index eklendi")
        
        conn.commit()
        conn.close()
        logger.info("✅ Veritabanı schema güncellemesi tamamlandı")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Veritabanı güncelleme hatası: {e}")
        return False

def check_database_status():
    """Database durumunu kontrol et"""
    db_path = "data/nesine_advanced.db"
    
    if not os.path.exists(db_path):
        print("❌ Database dosyası bulunamadı!")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Tabloları kontrol et
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in cursor.fetchall()]
        
        # AI tablolarını kontrol et
        ai_tables = ['ai_model_performance', 'ai_feature_importance']
        missing_ai_tables = [table for table in ai_tables if table not in tables]
        
        print("📋 Database Durumu:")
        print(f"📍 Konum: {os.path.abspath(db_path)}")
        print(f"📊 Dosya boyutu: {os.path.getsize(db_path)} bytes")
        print(f"📈 Toplam tablo: {len(tables)}")
        
        if missing_ai_tables:
            print(f"❌ Eksik AI tabloları: {missing_ai_tables}")
        else:
            print("✅ Tüm AI tabloları mevcut")
        
        conn.close()
        return len(missing_ai_tables) == 0
        
    except Exception as e:
        print(f"❌ Kontrol hatası: {e}")
        return False

if __name__ == "__main__":
    print("🔄 Database güncelleme başlatılıyor...")
    
    if update_database_schema():
        print("✅ Güncelleme başarılı!")
        print("\n🔍 Son durum kontrolü:")
        check_database_status()
    else:
        print("❌ Güncelleme başarısız!")