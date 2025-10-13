#!/usr/bin/env python3
"""
Prediction Tracker & Validator
-------------------------------
Tahminleri kaydeder ve gerçek sonuçlarla karşılaştırır
Geliştirilmiş sürüm: Toplu güncelleme, temizlik, gelişmiş istatistikler
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
            
            # Tahmin ID'si oluştur
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
                "actual_result": None,  # Sonradan doldurulacak
                "status": "pending"  # pending, validated, expired
            }
            
            # Bugünkü dosyaya ekle
            self._append_to_file(self.today_file, record)
            
            # History dosyasına da ekle
            self._append_to_history(record)
            
            logger.info(f"✅ Tahmin kaydedildi: {match_data['home_team']} - {match_data['away_team']} (ID: {prediction_id})")
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
                
                # History dosyasını da güncelle
                self._update_history(pred['prediction_id'], pred['actual_result'], pred['accuracy'])
                
                # İstatistikleri güncelle
                self._update_stats()
                
                return True
            else:
                logger.warning(f"⚠️ Tahmin bulunamadı: {home_team} - {away_team}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Sonuç güncelleme hatası: {e}")
            return False
    
    def update_batch_results(self, results: List[Dict]) -> int:
        """
        Toplu sonuç güncelleme
        
        Args:
            results: [{"home_team": "", "away_team": "", "home_score": 0, "away_score": 0, "match_date": ""}]
        
        Returns:
            int: Başarılı güncelleme sayısı
        """
        success_count = 0
        for result in results:
            if self.update_actual_result(
                home_team=result['home_team'],
                away_team=result['away_team'],
                home_score=result['home_score'],
                away_score=result['away_score'],
                match_date=result.get('match_date')
            ):
                success_count += 1
        
        logger.info(f"✅ Toplu sonuç güncelleme: {success_count}/{len(results)}")
        return success_count
    
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
            pending = [p for p in all_predictions if p.get('status') == 'pending']
            
            if not validated:
                return {
                    "status": "no_validated",
                    "message": "Henüz doğrulanmış tahmin yok",
                    "total_predictions": len(all_predictions),
                    "pending_predictions": len(pending)
                }
            
            # Temel doğruluk metrikleri
            ms_accuracy = self._calculate_ms_accuracy(validated)
            score_accuracy = self._calculate_score_accuracy(validated)
            goal_diff_accuracy = self._calculate_goal_diff_accuracy(validated)
            
            # Gelişmiş analizler
            confidence_analysis = self._analyze_confidence(validated)
            risk_analysis = self._analyze_risk(validated)
            league_analysis = self._analyze_league_performance(validated)
            daily_trends = self._calculate_daily_trends(validated, days)
            
            # Value bet performansı
            value_analysis = self._analyze_value_performance(validated)
            
            return {
                "status": "success",
                "period": f"Son {days} gün",
                "summary": {
                    "total_predictions": len(all_predictions),
                    "validated_predictions": len(validated),
                    "pending_predictions": len(pending),
                    "validation_rate": round((len(validated) / len(all_predictions)) * 100, 2)
                },
                "accuracy_metrics": {
                    "ms_accuracy": ms_accuracy,
                    "score_accuracy": score_accuracy,
                    "goal_diff_accuracy": goal_diff_accuracy
                },
                "advanced_analysis": {
                    "confidence_analysis": confidence_analysis,
                    "risk_analysis": risk_analysis,
                    "league_analysis": league_analysis,
                    "value_analysis": value_analysis,
                    "daily_trends": daily_trends
                },
                "best_predictions": self._get_best_predictions(validated)[:5],
                "worst_predictions": self._get_worst_predictions(validated)[:5],
                "most_profitable": self._get_most_profitable_predictions(validated)[:5]
            }
            
        except Exception as e:
            logger.error(f"❌ Rapor oluşturma hatası: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_advanced_stats(self, days: int = 30) -> Dict:
        """Gelişmiş istatistikler ve trend analizi"""
        predictions = self._load_recent_predictions(days)
        validated = [p for p in predictions if p.get('status') == 'validated']
        
        if not validated:
            return {"status": "no_data"}
        
        # Lig bazlı performans
        league_stats = self._analyze_league_performance(validated)
        
        # Zaman içinde performans trendi
        daily_trends = self._calculate_daily_trends(validated, days)
        
        # Takım bazlı analiz
        team_analysis = self._analyze_team_performance(validated)
        
        # Confidence dağılımı
        confidence_distribution = self._get_confidence_distribution(validated)
        
        return {
            "period": f"Son {days} gün",
            "league_performance": league_stats,
            "daily_trends": daily_trends,
            "team_analysis": team_analysis,
            "confidence_distribution": confidence_distribution,
            "performance_insights": self._generate_insights(validated)
        }
    
    def export_to_csv(self, days: int = 30, output_file: str = None) -> str:
        """CSV formatında dışa aktar"""
        try:
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
                    'Confidence', 'Value Index', 'Risk', 'Recommendation'
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
                        p['prediction']['risk'],
                        p['prediction']['recommendation']
                    ])
            
            logger.info(f"✅ CSV export: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"❌ CSV export hatası: {e}")
            return None
    
    def cleanup_old_predictions(self, days_to_keep: int = 90):
        """Eski tahmin dosyalarını temizle"""
        cutoff_date = date.today() - timedelta(days=days_to_keep)
        
        for pred_file in self.storage_dir.glob("predictions_*.json"):
            try:
                file_date_str = pred_file.stem.split('_')[1]
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d").date()
                if file_date < cutoff_date:
                    pred_file.unlink()
                    logger.info(f"🗑️ Eski dosya silindi: {pred_file}")
            except (ValueError, IndexError):
                continue
        
        logger.info(f"🧹 {days_to_keep} günden eski dosyalar temizlendi")
    
    def get_prediction_by_id(self, prediction_id: str) -> Optional[Dict]:
        """ID'ye göre tahmin bul"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                
                for pred in history:
                    if pred.get('prediction_id') == prediction_id:
                        return pred
        except Exception as e:
            logger.error(f"❌ Tahmin bulma hatası: {e}")
        
        return None
    
    # ============================================
    # HELPER METHODS
    # ============================================
    
    def _generate_id(self, match_data: Dict) -> str:
        """Benzersiz tahmin ID'si oluştur"""
        home = match_data.get('home_team', '').lower().replace(' ', '_')
        away = match_data.get('away_team', '').lower().replace(' ', '_')
        league = match_data.get('league', '').lower().replace(' ', '_')[:10]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{league}_{home}_vs_{away}_{timestamp}"
    
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
    
    def _append_to_history(self, record: Dict):
        """History dosyasına ekle"""
        if self.history_file.exists():
            with open(self.history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []
        
        # Aynı ID'ye sahip eski kaydı sil
        history = [h for h in history if h.get('prediction_id') != record['prediction_id']]
        history.append(record)
        
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    def _update_history(self, prediction_id: str, actual_result: Dict, accuracy: Dict):
        """History dosyasını güncelle"""
        if not self.history_file.exists():
            return
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            for pred in history:
                if pred.get('prediction_id') == prediction_id:
                    pred['actual_result'] = actual_result
                    pred['accuracy'] = accuracy
                    pred['status'] = 'validated'
                    break
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"⚠️ History güncelleme hatası: {e}")
    
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
        all_preds = []
        
        for i in range(days):
            target_date = date.today() - timedelta(days=i)
            pred_file = self.storage_dir / f"predictions_{target_date}.json"
            
            if pred_file.exists():
                try:
                    with open(pred_file, 'r', encoding='utf-8') as f:
                        all_preds.extend(json.load(f))
                except Exception as e:
                    logger.warning(f"⚠️ {pred_file} okunamadı: {e}")
                    continue
        
        return all_preds
    
    def _calculate_ms_accuracy(self, validated: List[Dict]) -> Dict:
        """MS doğruluğunu hesapla"""
        correct = sum(1 for p in validated 
                     if p['prediction']['ms_prediction'] == p['actual_result']['result'])
        total = len(validated)
        
        return {
            "correct": correct,
            "total": total,
            "percentage": round((correct / total) * 100, 2) if total > 0 else 0
        }
    
    def _calculate_score_accuracy(self, validated: List[Dict]) -> Dict:
        """Skor doğruluğunu hesapla"""
        correct = sum(1 for p in validated 
                     if p['prediction']['score_prediction'] == p['actual_result']['score'])
        total = len(validated)
        
        return {
            "correct": correct,
            "total": total,
            "percentage": round((correct / total) * 100, 2) if total > 0 else 0
        }
    
    def _calculate_goal_diff_accuracy(self, validated: List[Dict]) -> Dict:
        """Gol farkı doğruluğunu hesapla"""
        correct = sum(1 for p in validated 
                     if self._is_close_score(
                         p['prediction']['score_prediction'],
                         p['actual_result']['score']
                     ))
        total = len(validated)
        
        return {
            "correct": correct,
            "total": total,
            "percentage": round((correct / total) * 100, 2) if total > 0 else 0
        }
    
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
    
    def _analyze_league_performance(self, validated: List[Dict]) -> Dict:
        """Lig bazlı performans analizi"""
        leagues = {}
        
        for p in validated:
            league = p['match']['league']
            if league not in leagues:
                leagues[league] = {"correct": 0, "total": 0}
            
            leagues[league]["total"] += 1
            if p['accuracy']['ms_correct']:
                leagues[league]["correct"] += 1
        
        # Yüzdeleri hesapla ve sırala
        for league in leagues:
            if leagues[league]["total"] > 0:
                leagues[league]["accuracy"] = round(
                    (leagues[league]["correct"] / leagues[league]["total"]) * 100, 2
                )
        
        # Accuracy'e göre sırala
        sorted_leagues = dict(sorted(
            leagues.items(),
            key=lambda x: x[1].get('accuracy', 0),
            reverse=True
        ))
        
        return sorted_leagues
    
    def _analyze_value_performance(self, validated: List[Dict]) -> Dict:
        """Value bet performans analizi"""
        value_ranges = {
            "High (>=0.15)": {"correct": 0, "total": 0},
            "Medium (0.05-0.15)": {"correct": 0, "total": 0},
            "Low (0-0.05)": {"correct": 0, "total": 0},
            "Negative (<0)": {"correct": 0, "total": 0}
        }
        
        for p in validated:
            value_index = p['prediction']['value_index']
            correct = p['accuracy']['ms_correct']
            
            if value_index >= 0.15:
                value_ranges["High (>=0.15)"]["total"] += 1
                if correct:
                    value_ranges["High (>=0.15)"]["correct"] += 1
            elif value_index >= 0.05:
                value_ranges["Medium (0.05-0.15)"]["total"] += 1
                if correct:
                    value_ranges["Medium (0.05-0.15)"]["correct"] += 1
            elif value_index >= 0:
                value_ranges["Low (0-0.05)"]["total"] += 1
                if correct:
                    value_ranges["Low (0-0.05)"]["correct"] += 1
            else:
                value_ranges["Negative (<0)"]["total"] += 1
                if correct:
                    value_ranges["Negative (<0)"]["correct"] += 1
        
        # Yüzdeleri hesapla
        for key in value_ranges:
            if value_ranges[key]["total"] > 0:
                value_ranges[key]["accuracy"] = round(
                    (value_ranges[key]["correct"] / value_ranges[key]["total"]) * 100, 2
                )
            else:
                value_ranges[key]["accuracy"] = 0
        
        return value_ranges
    
    def _calculate_daily_trends(self, validated: List[Dict], days: int) -> Dict:
        """Günlük performans trendleri"""
        daily_stats = {}
        
        for i in range(days):
            target_date = date.today() - timedelta(days=i)
            date_str = str(target_date)
            
            daily_predictions = [p for p in validated if p['prediction_date'] == date_str]
            
            if daily_predictions:
                correct = sum(1 for p in daily_predictions if p['accuracy']['ms_correct'])
                total = len(daily_predictions)
                
                daily_stats[date_str] = {
                    "correct": correct,
                    "total": total,
                    "accuracy": round((correct / total) * 100, 2) if total > 0 else 0
                }
        
        return daily_stats
    
    def _analyze_team_performance(self, validated: List[Dict]) -> Dict:
        """Takım bazlı performans analizi"""
        team_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for p in validated:
            home_team = p['match']['home_team']
            away_team = p['match']['away_team']
            
            # Home team performance
            team_stats[home_team]["total"] += 1
            if p['accuracy']['ms_correct'] and p['prediction']['ms_prediction'] == "1":
                team_stats[home_team]["correct"] += 1
            
            # Away team performance  
            team_stats[away_team]["total"] += 1
            if p['accuracy']['ms_correct'] and p['prediction']['ms_prediction'] == "2":
                team_stats[away_team]["correct"] += 1
        
        # Yüzdeleri hesapla
        for team in team_stats:
            if team_stats[team]["total"] > 0:
                team_stats[team]["accuracy"] = round(
                    (team_stats[team]["correct"] / team_stats[team]["total"]) * 100, 2
                )
        
        # En başarılı 10 takım
        best_teams = dict(sorted(
            team_stats.items(),
            key=lambda x: x[1].get('accuracy', 0),
            reverse=True
        )[:10])
        
        return best_teams
    
    def _get_confidence_distribution(self, validated: List[Dict]) -> Dict:
        """Güven dağılımı analizi"""
        confidences = [p['prediction']['confidence'] for p in validated]
        
        if not confidences:
            return {}
        
        return {
            "mean": round(sum(confidences) / len(confidences), 2),
            "min": round(min(confidences), 2),
            "max": round(max(confidences), 2),
            "std": round((sum((x - (sum(confidences) / len(confidences))) ** 2 for x in confidences) / len(confidences)) ** 0.5, 2)
        }
    
    def _generate_insights(self, validated: List[Dict]) -> List[str]:
        """Otomatik insight'lar oluştur"""
        insights = []
        
        if not validated:
            return insights
        
        # Confidence accuracy correlation
        high_conf = [p for p in validated if p['prediction']['confidence'] >= 70]
        if high_conf:
            high_conf_accuracy = sum(1 for p in high_conf if p['accuracy']['ms_correct']) / len(high_conf)
            if high_conf_accuracy >= 0.7:
                insights.append("Yüksek güvenilirlikli tahminlerde doğruluk oranı %70'in üzerinde")
            elif high_conf_accuracy <= 0.5:
                insights.append("Yüksek güvenilirlikli tahminler beklenenden düşük performans gösteriyor")
        
        # Risk analysis
        low_risk = [p for p in validated if p['prediction']['risk'] == 'Low Risk']
        if low_risk:
            low_risk_accuracy = sum(1 for p in low_risk if p['accuracy']['ms_correct']) / len(low_risk)
            insights.append(f"Düşük riskli bahislerde doğruluk oranı: %{low_risk_accuracy*100:.1f}")
        
        # Best performing league
        league_stats = self._analyze_league_performance(validated)
        if league_stats:
            best_league = next(iter(league_stats.items()))
            insights.append(f"En iyi performans: {best_league[0]} (%{best_league[1].get('accuracy', 0)})")
        
        return insights
    
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
                "league": p['match']['league'],
                "predicted": f"{p['prediction']['ms_prediction']} ({p['prediction']['score_prediction']})",
                "actual": f"{p['actual_result']['result']} ({p['actual_result']['score']})",
                "confidence": p['prediction']['confidence'],
                "value_index": p['prediction']['value_index'],
                "risk": p['prediction']['risk']
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
                "league": p['match']['league'],
                "predicted": f"{p['prediction']['ms_prediction']} ({p['prediction']['score_prediction']})",
                "actual": f"{p['actual_result']['result']} ({p['actual_result']['score']})",
                "confidence": p['prediction']['confidence'],
                "value_index": p['prediction']['value_index'],
                "risk": p['prediction']['risk']
            }
            for p in sorted_preds
        ]
    
    def _get_most_profitable_predictions(self, validated: List[Dict]) -> List[Dict]:
        """En karlı tahminler (yüksek value + doğru tahmin)"""
        correct_high_value = [
            p for p in validated 
            if p['accuracy']['ms_correct'] and p['prediction']['value_index'] >= 0.1
        ]
        
        sorted_preds = sorted(
            correct_high_value,
            key=lambda x: x['prediction']['value_index'],
            reverse=True
        )
        
        return [
            {
                "match": f"{p['match']['home_team']} - {p['match']['away_team']}",
                "league": p['match']['league'],
                "predicted": f"{p['prediction']['ms_prediction']} ({p['prediction']['score_prediction']})",
                "actual": f"{p['actual_result']['result']} ({p['actual_result']['score']})",
                "confidence": p['prediction']['confidence'],
                "value_index": p['prediction']['value_index'],
                "risk": p['prediction']['risk']
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
# KULLANIM ÖRNEKLERİ
# ==============================================
if __name__ == "__main__":
    import pandas as pd
    
    # Tracker oluştur
    tracker = PredictionTracker()
    
    # Test tahmini
    test_match = {
        "home_team": "Barcelona",
        "away_team": "Real Madrid",
        "league": "La Liga",
        "time": "20:00",
        "match_date": str(date.today()),
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
    
    # Tahmin kaydet
    tracker.save_prediction(test_match, test_prediction)
    print("✅ Tahmin kaydedildi")
    
    # Sonuç güncelle
    tracker.update_actual_result("Barcelona", "Real Madrid", 3, 1)
    print("✅ Sonuç güncellendi")
    
    # Detaylı rapor
    report = tracker.get_accuracy_report(days=7)
    print(f"\n📊 Temel Rapor:")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    
    # Gelişmiş istatistikler
    advanced_stats = tracker.get_advanced_stats(days=30)
    print(f"\n📈 Gelişmiş İstatistikler:")
    print(json.dumps(advanced_stats, indent=2, ensure_ascii=False))
    
    # CSV export
    csv_file = tracker.export_to_csv(days=7)
    if csv_file:
        print(f"\n💾 CSV dosyası: {csv_file}")
    
    # Temizlik (opsiyonel)
    # tracker.cleanup_old_predictions(days_to_keep=30)
