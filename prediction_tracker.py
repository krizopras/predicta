#!/usr/bin/env python3
"""
Prediction Tracker & Validator
-------------------------------
Tahminleri kaydeder ve gerçek sonuçlarla karşılaştırır
"""

import os
import json
import pickle
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

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
    
    def save_prediction(
        self, 
        match_data: Dict, 
        prediction: Dict,
        prediction_date: Optional[str] = None
    ) -> bool:
        """
        Tek bir tahmin kaydet
        
        Args:
            match_data: {"home_team", "away_team", "league", "odds", "time"}
            prediction: MLPredictionEngine.predict_match() çıktısı
            prediction_date: ISO format tarih (None ise bugün)
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            pred_date = prediction_date or str(date.today())
            
            record = {
                "prediction_id": self._generate_id(match_data),
                "prediction_date": pred_date,
                "prediction_timestamp": datetime.now().isoformat(),
                "match": {
                    "home_team": match_data.get("home_team"),
                    "away_team": match_data.get("away_team"),
                    "league": match_data.get("league"),
                    "match_time": match_data.get("time"),
                    "odds": match_data.get("odds", {})
                },
                "prediction": {
                    "ms_prediction": prediction.get("prediction"),
                    "confidence": prediction.get("confidence"),
                    "probabilities": prediction.get("probabilities", {}),
                    "score_prediction": prediction.get("score_prediction"),
                    "alternative_scores": prediction.get("alternative_scores", []),
                    "value_index": prediction.get("value_bet", {}).get("value_index", 0),
                    "risk": prediction.get("value_bet", {}).get("risk", "Unknown"),
                    "recommendation": prediction.get("value_bet", {}).get("recommendation", "")
                },
                "actual_result": None,  # Sonradan doldurulacak
                "status": "pending"  # pending, validated, expired
            }
            
            # Bugünkü dosyaya ekle
            self._append_to_file(self.today_file, record)
            
            logger.info(f"✅ Tahmin kaydedildi: {match_data['home_team']} - {match_data['away_team']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Tahmin kaydetme hatası: {e}")
            return False
    
    def save_batch_predictions(
        self, 
        predictions: List[Dict],
        prediction_date: Optional[str] = None
    ) -> int:
        """
        Toplu tahmin kaydet
        
        Args:
            predictions: [{"match": {...}, "prediction": {...}}, ...]
            prediction_date: ISO format tarih
        
        Returns:
            int: Kaydedilen tahmin sayısı
        """
        count = 0
        for item in predictions:
            match_data = item.get("match") or item
            prediction = item.get("prediction") or item
            
            if self.save_prediction(match_data, prediction, prediction_date):
                count += 1
        
        logger.info(f"📊 Toplam {count}/{len(predictions)} tahmin kaydedildi")
        return count
    
    def update_actual_result(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        match_date: Optional[str] = None
    ) -> bool:
        """
        Gerçek sonucu güncelle
        
        Args:
            home_team, away_team: Takım isimleri
            home_score, away_score: Gerçek skorlar
            match_date: Maç tarihi (None ise bugün)
        
        Returns:
            bool: Güncelleme başarılı mı?
        """
        try:
            match_date = match_date or str(date.today())
            pred_file = self.storage_dir / f"predictions_{match_date}.json"
            
            if not pred_file.exists():
                logger.warning(f"⚠️ {match_date} için tahmin dosyası bulunamadı")
                return False
            
            # Dosyayı oku
            with open(pred_file, 'r', encoding='utf-8') as f:
                predictions = json.load(f)
            
            # İlgili tahmini bul
            updated = False
            for pred in predictions:
                if (pred['match']['home_team'].lower() == home_team.lower() and
                    pred['match']['away_team'].lower() == away_team.lower()):
                    
                    # Gerçek sonucu ekle
                    pred['actual_result'] = {
                        "home_score": home_score,
                        "away_score": away_score,
                        "result": self._calculate_ms(home_score, away_score),
                        "score": f"{home_score}-{away_score}",
                        "updated_at": datetime.now().isoformat()
                    }
                    pred['status'] = 'validated'
                    
                    # Doğruluk hesapla
                    pred['accuracy'] = self._calculate_accuracy(pred)
                    
                    updated = True
                    logger.info(f"✅ Sonuç güncellendi: {home_team} {home_score}-{away_score} {away_team}")
                    break
            
            if updated:
                # Dosyayı kaydet
                with open(pred_file, 'w', encoding='utf-8') as f:
                    json.dump(predictions, f, indent=2, ensure_ascii=False)
                
                # İstatistikleri güncelle
                self._update_stats()
                
                return True
            else:
                logger.warning(f"⚠️ Tahmin bulunamadı: {home_team} - {away_team}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Sonuç güncelleme hatası: {e}")
            return False
    
    def get_accuracy_report(self, days: int = 7) -> Dict[str, Any]:
        """
        Son N gün için doğruluk raporu
        
        Args:
            days: Kaç günlük veri analiz edilsin
        
        Returns:
            Dict: Detaylı istatistikler
        """
        try:
            all_predictions = self._load_recent_predictions(days)
            
            if not all_predictions:
                return {
                    "status": "no_data",
                    "message": f"Son {days} günde veri yok"
                }
            
            validated = [p for p in all_predictions if p.get('status') == 'validated']
            
            if not validated:
                return {
                    "status": "no_validated",
                    "message": "Henüz doğrulanmış tahmin yok",
                    "total_predictions": len(all_predictions)
                }
            
            # MS doğruluğu
            ms_correct = sum(1 for p in validated 
                           if p['prediction']['ms_prediction'] == p['actual_result']['result'])
            ms_accuracy = (ms_correct / len(validated)) * 100
            
            # Skor doğruluğu (tam isabet)
            score_correct = sum(1 for p in validated 
                              if p['prediction']['score_prediction'] == p['actual_result']['score'])
            score_accuracy = (score_correct / len(validated)) * 100
            
            # Gol farkı doğruluğu (±1 gol)
            goal_diff_close = sum(1 for p in validated 
                                 if self._is_close_score(
                                     p['prediction']['score_prediction'],
                                     p['actual_result']['score']
                                 ))
            goal_diff_accuracy = (goal_diff_close / len(validated)) * 100
            
            # Güven aralığı analizi
            confidence_analysis = self._analyze_confidence(validated)
            
            # Risk kategorisi başarısı
            risk_analysis = self._analyze_risk(validated)
            
            return {
                "status": "success",
                "period": f"Last {days} days",
                "total_predictions": len(all_predictions),
                "validated_predictions": len(validated),
                "pending_predictions": len([p for p in all_predictions if p['status'] == 'pending']),
                "ms_accuracy": {
                    "correct": ms_correct,
                    "total": len(validated),
                    "percentage": round(ms_accuracy, 2)
                },
                "score_accuracy": {
                    "exact_match": score_correct,
                    "total": len(validated),
                    "percentage": round(score_accuracy, 2)
                },
                "goal_diff_accuracy": {
                    "within_1_goal": goal_diff_close,
                    "total": len(validated),
                    "percentage": round(goal_diff_accuracy, 2)
                },
                "confidence_analysis": confidence_analysis,
                "risk_analysis": risk_analysis,
                "best_predictions": self._get_best_predictions(validated)[:5],
                "worst_predictions": self._get_worst_predictions(validated)[:5]
            }
            
        except Exception as e:
            logger.error(f"❌ Rapor oluşturma hatası: {e}")
            return {"status": "error", "error": str(e)}
    
    def export_to_csv(self, days: int = 30, output_file: str = None) -> str:
        """CSV formatında dışa aktar"""
        try:
            import csv
            
            predictions = self._load_recent_predictions(days)
            validated = [p for p in predictions if p['status'] == 'validated']
            
            if not validated:
                return None
            
            output_file = output_file or str(self.storage_dir / f"export_{date.today()}.csv")
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow([
                    'Date', 'Home Team', 'Away Team', 'League',
                    'Predicted MS', 'Actual MS', 'MS Correct',
                    'Predicted Score', 'Actual Score', 'Score Correct',
                    'Confidence', 'Value Index', 'Risk'
                ])
                
                # Rows
                for p in validated:
                    writer.writerow([
                        p['prediction_date'],
                        p['match']['home_team'],
                        p['match']['away_team'],
                        p['match']['league'],
                        p['prediction']['ms_prediction'],
                        p['actual_result']['result'],
                        'Yes' if p['prediction']['ms_prediction'] == p['actual_result']['result'] else 'No',
                        p['prediction']['score_prediction'],
                        p['actual_result']['score'],
                        'Yes' if p['prediction']['score_prediction'] == p['actual_result']['score'] else 'No',
                        p['prediction']['confidence'],
                        p['prediction']['value_index'],
                        p['prediction']['risk']
                    ])
            
            logger.info(f"✅ CSV export: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"❌ CSV export hatası: {e}")
            return None
    
    # ============================================
    # HELPER METHODS
    # ============================================
    
    def _generate_id(self, match_data: Dict) -> str:
        """Benzersiz tahmin ID'si oluştur"""
        home = match_data.get('home_team', '').lower().replace(' ', '_')
        away = match_data.get('away_team', '').lower().replace(' ', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{home}_vs_{away}_{timestamp}"
    
    def _append_to_file(self, file_path: Path, record: Dict):
        """JSON dosyasına ekle"""
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = []
        
        data.append(record)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _calculate_ms(self, home_score: int, away_score: int) -> str:
        """Skor üzerinden MS hesapla"""
        if home_score > away_score:
            return "1"
        elif home_score < away_score:
            return "2"
        else:
            return "X"
    
    def _calculate_accuracy(self, prediction: Dict) -> Dict:
        """Tek tahmin için doğruluk metrikleri"""
        pred = prediction['prediction']
        actual = prediction['actual_result']
        
        ms_correct = pred['ms_prediction'] == actual['result']
        score_correct = pred['score_prediction'] == actual['score']
        
        return {
            "ms_correct": ms_correct,
            "score_correct": score_correct,
            "goal_diff_close": self._is_close_score(pred['score_prediction'], actual['score'])
        }
    
    def _is_close_score(self, predicted: str, actual: str) -> bool:
        """Skor tahmini ±1 gol içinde mi?"""
        try:
            ph, pa = map(int, predicted.split('-'))
            ah, aa = map(int, actual.split('-'))
            return abs(ph - ah) <= 1 and abs(pa - aa) <= 1
        except:
            return False
    
    def _load_recent_predictions(self, days: int) -> List[Dict]:
        """Son N günün tahminlerini yükle"""
        from datetime import timedelta
        all_preds = []
        
        for i in range(days):
            target_date = date.today() - timedelta(days=i)
            pred_file = self.storage_dir / f"predictions_{target_date}.json"
            
            if pred_file.exists():
                try:
                    with open(pred_file, 'r', encoding='utf-8') as f:
                        all_preds.extend(json.load(f))
                except:
                    continue
        
        return all_preds
    
    def _analyze_confidence(self, validated: List[Dict]) -> Dict:
        """Güven aralıklarına göre başarı analizi"""
        ranges = {
            "70-100%": {"correct": 0, "total": 0},
            "60-70%": {"correct": 0, "total": 0},
            "50-60%": {"correct": 0, "total": 0},
            "0-50%": {"correct": 0, "total": 0}
        }
        
        for p in validated:
            conf = p['prediction']['confidence']
            correct = p['accuracy']['ms_correct']
            
            if conf >= 70:
                ranges["70-100%"]["total"] += 1
                if correct:
                    ranges["70-100%"]["correct"] += 1
            elif conf >= 60:
                ranges["60-70%"]["total"] += 1
                if correct:
                    ranges["60-70%"]["correct"] += 1
            elif conf >= 50:
                ranges["50-60%"]["total"] += 1
                if correct:
                    ranges["50-60%"]["correct"] += 1
            else:
                ranges["0-50%"]["total"] += 1
                if correct:
                    ranges["0-50%"]["correct"] += 1
        
        # Yüzdeleri hesapla
        for key in ranges:
            if ranges[key]["total"] > 0:
                ranges[key]["accuracy"] = round(
                    (ranges[key]["correct"] / ranges[key]["total"]) * 100, 2
                )
            else:
                ranges[key]["accuracy"] = 0
        
        return ranges
    
    def _analyze_risk(self, validated: List[Dict]) -> Dict:
        """Risk kategorilerine göre başarı"""
        risks = {}
        
        for p in validated:
            risk = p['prediction']['risk']
            if risk not in risks:
                risks[risk] = {"correct": 0, "total": 0}
            
            risks[risk]["total"] += 1
            if p['accuracy']['ms_correct']:
                risks[risk]["correct"] += 1
        
        # Yüzdeleri hesapla
        for risk in risks:
            if risks[risk]["total"] > 0:
                risks[risk]["accuracy"] = round(
                    (risks[risk]["correct"] / risks[risk]["total"]) * 100, 2
                )
        
        return risks
    
    def _get_best_predictions(self, validated: List[Dict]) -> List[Dict]:
        """En başarılı tahminler"""
        correct = [p for p in validated if p['accuracy']['ms_correct']]
        sorted_preds = sorted(
            correct,
            key=lambda x: x['prediction']['confidence'],
            reverse=True
        )
        
        return [
            {
                "match": f"{p['match']['home_team']} - {p['match']['away_team']}",
                "predicted": f"{p['prediction']['ms_prediction']} ({p['prediction']['score_prediction']})",
                "actual": f"{p['actual_result']['result']} ({p['actual_result']['score']})",
                "confidence": p['prediction']['confidence'],
                "value_index": p['prediction']['value_index']
            }
            for p in sorted_preds
        ]
    
    def _get_worst_predictions(self, validated: List[Dict]) -> List[Dict]:
        """En kötü tahminler"""
        incorrect = [p for p in validated if not p['accuracy']['ms_correct']]
        sorted_preds = sorted(
            incorrect,
            key=lambda x: x['prediction']['confidence'],
            reverse=True
        )
        
        return [
            {
                "match": f"{p['match']['home_team']} - {p['match']['away_team']}",
                "predicted": f"{p['prediction']['ms_prediction']} ({p['prediction']['score_prediction']})",
                "actual": f"{p['actual_result']['result']} ({p['actual_result']['score']})",
                "confidence": p['prediction']['confidence'],
                "value_index": p['prediction']['value_index']
            }
            for p in sorted_preds
        ]
    
    def _update_stats(self):
        """Genel istatistikleri güncelle"""
        try:
            report = self.get_accuracy_report(days=30)
            
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.warning(f"⚠️ İstatistik güncelleme hatası: {e}")


# ==============================================
# QUICK TEST
# ==============================================
if __name__ == "__main__":
    import pandas as pd
    
    tracker = PredictionTracker()
    
    # Test prediction
    test_match = {
        "home_team": "Barcelona",
        "away_team": "Real Madrid",
        "league": "La Liga",
        "time": "20:00",
        "odds": {"1": 2.10, "X": 3.40, "2": 3.20}
    }
    
    test_prediction = {
        "prediction": "1",
        "confidence": 65.5,
        "probabilities": {"1": 65.5, "X": 20.0, "2": 14.5},
        "score_prediction": "2-1",
        "alternative_scores": [
            {"score": "2-1", "prob": 0.35},
            {"score": "3-1", "prob": 0.25}
        ],
        "value_bet": {
            "value_index": 0.12,
            "risk": "Medium Risk",
            "recommendation": "Good Bet"
        }
    }
    
    # Kaydet
    tracker.save_prediction(test_match, test_prediction)
    print("✅ Tahmin kaydedildi")
    
    # Sonucu güncelle
    tracker.update_actual_result("Barcelona", "Real Madrid", 3, 1)
    print("✅ Sonuç güncellendi")
    
    # Rapor
    report = tracker.get_accuracy_report(days=7)
    print(f"\n📊 Rapor:")
    print(json.dumps(report, indent=2, ensure_ascii=False))
