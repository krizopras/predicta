#!/usr/bin/env python3
"""
Prediction Tracker & Validator - FIXED VERSION
-------------------------------
JSON parse hataları düzeltildi
"""

import os
import json
import csv
import pickle
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class PredictionTracker:
    """Tahminleri kaydet ve doğruluk analizi yap"""
    
    def __init__(self, storage_dir: str = "data/predictions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Günlük tahmin dosyası
        self.today_file = self.storage_dir / f"predictions_{date.today()}.json"
        
        # Geçmiş tahminler
        self.history_file = self.storage_dir / "prediction_history.json"
        
        # İstatistik özeti
        self.stats_file = self.storage_dir / "accuracy_stats.json"
        
        # 🆕 İLK AÇILIŞTA BOZUK DOSYALARI TEMİZLE
        self._initial_cleanup()
    
    def _initial_cleanup(self):
        """🆕 İlk açılışta bozuk JSON dosyalarını tespit et ve temizle"""
        try:
            logger.info("🔍 JSON dosyaları kontrol ediliyor...")
            
            for json_file in self.storage_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # Liste değilse düzelt
                    if not isinstance(data, list):
                        logger.warning(f"⚠️ Bozuk format tespit edildi: {json_file}")
                        backup = json_file.with_suffix('.json.backup')
                        json_file.rename(backup)
                        logger.info(f"💾 Backup oluşturuldu: {backup}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON parse hatası: {json_file} - {e}")
                    # Bozuk dosyayı taşı
                    corrupted = json_file.with_suffix('.json.corrupted')
                    json_file.rename(corrupted)
                    logger.warning(f"🗑️ Bozuk dosya taşındı: {corrupted}")
                    
                except Exception as e:
                    logger.error(f"❌ Dosya kontrolü hatası: {json_file} - {e}")
            
            logger.info("✅ JSON temizlik tamamlandı")
            
        except Exception as e:
            logger.error(f"❌ Initial cleanup hatası: {e}")
    
    # ============================================
    # 🔧 DÜZELTİLMİŞ METODLAR
    # ============================================
    
    def _append_to_file(self, file_path: Path, record: Dict):
        """🔧 JSON dosyasına ekle - GÜVENLİ VERSİYON"""
        try:
            # Dosya varsa oku
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        
                        # Liste değilse düzelt
                        if not isinstance(data, list):
                            logger.warning(f"⚠️ Bozuk format: {file_path}, yeni liste oluşturuluyor")
                            # Backup oluştur
                            backup_path = file_path.with_suffix('.json.backup')
                            import shutil
                            shutil.copy2(file_path, backup_path)
                            logger.info(f"💾 Backup: {backup_path}")
                            data = []
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ JSON parse hatası: {e}")
                        # Backup oluştur
                        backup_path = file_path.with_suffix(f'.json.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
                        import shutil
                        shutil.copy2(file_path, backup_path)
                        logger.info(f"💾 Bozuk dosya backup: {backup_path}")
                        data = []
            else:
                data = []
            
            # Yeni kaydı ekle
            data.append(record)
            
            # 🆕 ATOMIC WRITE - Önce geçici dosyaya yaz
            temp_path = file_path.with_suffix('.json.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Başarılı yazma sonrası asıl dosyayı değiştir
            temp_path.replace(file_path)
            
            logger.debug(f"✅ Kayıt eklendi: {file_path.name}")
            
        except Exception as e:
            logger.error(f"❌ Dosya yazma hatası: {file_path} - {e}")
            raise
    
    def _append_to_history(self, record: Dict):
        """🔧 History dosyasına ekle - GÜVENLİ VERSİYON"""
        try:
            # Dosya varsa oku
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    try:
                        history = json.load(f)
                        
                        if not isinstance(history, list):
                            logger.warning("⚠️ History bozuk format, yeniden oluşturuluyor")
                            # Backup
                            backup = self.history_file.with_suffix(f'.json.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
                            import shutil
                            shutil.copy2(self.history_file, backup)
                            logger.info(f"💾 History backup: {backup}")
                            history = []
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ History JSON hatası: {e}")
                        # Backup ve sıfırlama
                        backup = self.history_file.with_suffix(f'.json.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
                        import shutil
                        shutil.copy2(self.history_file, backup)
                        logger.info(f"💾 History backup: {backup}")
                        history = []
            else:
                history = []
            
            # Aynı ID'ye sahip eski kaydı sil (duplicate önleme)
            prediction_id = record.get('prediction_id')
            if prediction_id:
                history = [h for h in history if h.get('prediction_id') != prediction_id]
            
            history.append(record)
            
            # 🆕 ATOMIC WRITE
            temp_path = self.history_file.with_suffix('.json.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
            temp_path.replace(self.history_file)
            
            logger.debug("✅ History güncellendi")
            
        except Exception as e:
            logger.error(f"❌ History yazma hatası: {e}")
            # History hatası critical değil, işlemi durdurma
    
    def _update_history(self, prediction_id: str, actual_result: Dict, accuracy: Dict):
        """🔧 History dosyasını güncelle - GÜVENLİ VERSİYON"""
        if not self.history_file.exists():
            return
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                try:
                    history = json.load(f)
                    
                    if not isinstance(history, list):
                        logger.warning("⚠️ History bozuk format, güncelleme atlanıyor")
                        return
                        
                except json.JSONDecodeError as e:
                    logger.error(f"❌ History parse hatası: {e}")
                    return
            
            # İlgili kaydı bul ve güncelle
            updated = False
            for pred in history:
                if pred.get('prediction_id') == prediction_id:
                    pred['actual_result'] = actual_result
                    pred['accuracy'] = accuracy
                    pred['status'] = 'validated'
                    updated = True
                    break
            
            if updated:
                # 🆕 ATOMIC WRITE
                temp_path = self.history_file.with_suffix('.json.tmp')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)
                
                temp_path.replace(self.history_file)
                logger.debug(f"✅ History güncellendi: {prediction_id}")
            else:
                logger.warning(f"⚠️ History'de kayıt bulunamadı: {prediction_id}")
                
        except Exception as e:
            logger.warning(f"⚠️ History güncelleme hatası: {e}")
    
    def _load_recent_predictions(self, days: int) -> List[Dict]:
        """🔧 Son N günün tahminlerini yükle - GÜVENLİ VERSİYON"""
        all_preds = []
        
        for i in range(days):
            target_date = date.today() - timedelta(days=i)
            pred_file = self.storage_dir / f"predictions_{target_date}.json"
            
            if pred_file.exists():
                try:
                    with open(pred_file, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            
                            # Liste kontrolü
                            if isinstance(data, list):
                                all_preds.extend(data)
                            else:
                                logger.warning(f"⚠️ Bozuk format: {pred_file.name}, atlandı")
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"❌ JSON parse hatası: {pred_file.name} - {e}")
                            continue
                            
                except Exception as e:
                    logger.warning(f"⚠️ {pred_file.name} okunamadı: {e}")
                    continue
        
        return all_preds
    
    # ============================================
    # 🆕 YENİ METODLAR
    # ============================================
    
    def cleanup_corrupted_files(self):
        """🆕 Bozuk JSON dosyalarını temizle"""
        logger.info("🧹 Bozuk dosyalar temizleniyor...")
        
        cleaned = 0
        for json_file in self.storage_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Liste değilse taşı
                if not isinstance(data, list):
                    corrupted = json_file.with_suffix('.json.corrupted')
                    json_file.rename(corrupted)
                    logger.warning(f"🗑️ Bozuk dosya taşındı: {corrupted}")
                    cleaned += 1
                    
            except json.JSONDecodeError:
                corrupted = json_file.with_suffix('.json.corrupted')
                json_file.rename(corrupted)
                logger.warning(f"🗑️ Parse hatası, dosya taşındı: {corrupted}")
                cleaned += 1
                
            except Exception as e:
                logger.error(f"❌ Dosya kontrolü hatası: {json_file} - {e}")
        
        logger.info(f"✅ Temizlik tamamlandı: {cleaned} dosya taşındı")
    
    def repair_all_files(self):
        """🆕 Tüm JSON dosyalarını onar"""
        logger.info("🔧 JSON dosyaları onarılıyor...")
        
        repaired = 0
        for json_file in self.storage_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        
                        # Liste değilse düzelt
                        if not isinstance(data, list):
                            logger.info(f"🔧 Onarılıyor: {json_file.name}")
                            
                            # Backup
                            backup = json_file.with_suffix('.json.backup')
                            import shutil
                            shutil.copy2(json_file, backup)
                            
                            # Yeni liste oluştur
                            with open(json_file, 'w', encoding='utf-8') as fw:
                                json.dump([], fw, indent=2, ensure_ascii=False)
                            
                            repaired += 1
                            logger.info(f"✅ Onarıldı: {json_file.name}")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Parse hatası: {json_file.name}")
                        
                        # Backup ve sıfırlama
                        backup = json_file.with_suffix('.json.corrupted')
                        import shutil
                        shutil.copy2(json_file, backup)
                        
                        # Yeni dosya
                        with open(json_file, 'w', encoding='utf-8') as fw:
                            json.dump([], fw, indent=2, ensure_ascii=False)
                        
                        repaired += 1
                        logger.info(f"✅ Sıfırlandı: {json_file.name}")
                        
            except Exception as e:
                logger.error(f"❌ Onarım hatası: {json_file} - {e}")
        
        logger.info(f"✅ Onarım tamamlandı: {repaired} dosya işlendi")
    
    # ============================================
    # ESKİ METODLAR (DEĞİŞMEDİ)
    # ============================================
    
    def save_prediction(
        self, 
        match_data: Dict, 
        prediction: Dict,
        prediction_date: Optional[str] = None
    ) -> bool:
        """Tek bir tahmin kaydet"""
        try:
            pred_date = prediction_date or str(date.today())
            
            prediction_id = self._generate_id(match_data)
            
            record = {
                "prediction_id": prediction_id,
                "prediction_date": pred_date,
                "prediction_timestamp": datetime.now().isoformat(),
                "match": {
                    "home_team": match_data.get("home_team"),
                    "away_team": match_data.get("away_team"),
                    "league": match_data.get("league"),
                    "match_time": match_data.get("time"),
                    "match_date": match_data.get("match_date"),
                    "odds": match_data.get("odds", {})
                },
                "prediction": {
                    "ms_prediction": prediction.get("prediction"),
                    "confidence": prediction.get("confidence", 0.0),
                    "probabilities": prediction.get("probabilities", {}),
                    "score_prediction": prediction.get("score_prediction"),
                    "alternative_scores": prediction.get("alternative_scores", []),
                    "value_index": prediction.get("value_bet", {}).get("value_index", 0),
                    "risk": prediction.get("value_bet", {}).get("risk", "Unknown"),
                    "recommendation": prediction.get("value_bet", {}).get("recommendation", "")
                },
                "actual_result": None,
                "status": "pending"
            }
            
            # Bugünkü dosyaya ekle (DÜZELTİLMİŞ metod kullanılıyor)
            self._append_to_file(self.today_file, record)
            
            # History dosyasına da ekle (DÜZELTİLMİŞ metod kullanılıyor)
            self._append_to_history(record)
            
            logger.info(f"✅ Tahmin kaydedildi: {match_data['home_team']} - {match_data['away_team']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Tahmin kaydetme hatası: {e}")
            return False
    
    # Diğer tüm metodlar aynı kalacak...
    # (update_actual_result, get_accuracy_report, export_to_csv, vs.)
    
    def _generate_id(self, match_data: Dict) -> str:
        """Benzersiz tahmin ID'si oluştur"""
        home = match_data.get('home_team', '').lower().replace(' ', '_')
        away = match_data.get('away_team', '').lower().replace(' ', '_')
        league = match_data.get('league', '').lower().replace(' ', '_')[:10]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{league}_{home}_vs_{away}_{timestamp}"


# ==============================================
# 🆕 MANUEL ONARIM SCRIPT'İ
# ==============================================
if __name__ == "__main__":
    import sys
    
    tracker = PredictionTracker()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "cleanup":
            print("🧹 Bozuk dosyalar temizleniyor...")
            tracker.cleanup_corrupted_files()
            
        elif command == "repair":
            print("🔧 Dosyalar onarılıyor...")
            tracker.repair_all_files()
            
        elif command == "check":
            print("🔍 Dosyalar kontrol ediliyor...")
            tracker._initial_cleanup()
            
        else:
            print("❌ Bilinmeyen komut!")
            print("Kullanım: python prediction_tracker.py [cleanup|repair|check]")
    else:
        print("✅ Prediction Tracker yüklendi")
        print("\nKomutlar:")
        print("  python prediction_tracker.py cleanup  - Bozuk dosyaları temizle")
        print("  python prediction_tracker.py repair   - Dosyaları onar")
        print("  python prediction_tracker.py check    - Dosyaları kontrol et")
