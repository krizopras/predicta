#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yardımcı fonksiyonlar - AI Entegrasyonlu Güncelleme
"""

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

def format_odds(odds):
    """Oranları formatla - AI risk faktörleri için geliştirilmiş"""
    try:
        if isinstance(odds, dict):
            return {k: f"{float(v):.2f}" for k, v in odds.items()}
        return f"{float(odds):.2f}"
    except:
        return "N/A"

def format_confidence(confidence):
    """Güven seviyesini formatla - AI certainty index entegrasyonu"""
    try:
        return f"%{float(confidence):.1f}"
    except:
        return "%0.0"

def format_certainty_index(certainty):
    """Kesinlik indeksini formatla"""
    try:
        certainty_float = float(certainty)
        if certainty_float >= 0.8:
            return "🔒 Çok Yüksek"
        elif certainty_float >= 0.6:
            return "🎯 Yüksek"
        elif certainty_float >= 0.4:
            return "⚖️ Orta"
        elif certainty_float >= 0.2:
            return "⚠️ Düşük"
        else:
            return "❓ Belirsiz"
    except:
        return "❓ Belirsiz"

def get_prediction_color(confidence):
    """Güven seviyesine göre renk - AI özellikli"""
    try:
        conf = float(confidence)
        if conf >= 80:
            return "very-high"
        elif conf >= 70:
            return "high"
        elif conf >= 60:
            return "medium-high"
        elif conf >= 50:
            return "medium"
        elif conf >= 40:
            return "medium-low"
        else:
            return "low"
    except:
        return "low"

def format_team_name(name, max_length=20):
    """Takım adını formatla - AI istatistikleri için optimize"""
    if not name:
        return "Bilinmeyen"
    
    # Önemli takımları kısaltma
    important_teams = {
        'Fenerbahçe': 'FB',
        'Galatasaray': 'GS',
        'Beşiktaş': 'BJK',
        'Trabzonspor': 'TS',
        'Başakşehir': 'İBFK'
    }
    
    if name in important_teams:
        return important_teams[name]
    
    if len(name) > max_length:
        return name[:max_length-3] + "..."
    
    return name

def format_league_name(league):
    """Lig adını formatla - AI model versiyonu için"""
    league_map = {
        'Süper Lig': '🇹🇷 Süper Lig',
        'Premier League': '🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League',
        'La Liga': '🇪🇸 La Liga',
        'Bundesliga': '🇩🇪 Bundesliga',
        'Serie A': '🇮🇹 Serie A',
        'Ligue 1': '🇫🇷 Ligue 1',
        'Özel Maç': '🌟 Özel Maç'
    }
    
    return league_map.get(league, f"⚽ {league}")

def format_ai_source(source):
    """AI kaynağını formatla"""
    source_map = {
        'basic': '📊 Temel',
        'ai_enhanced': '🤖 AI Gelişmiş',
        'hybrid': '🔗 Hibrit'
    }
    return source_map.get(source, f"❓ {source}")

def format_risk_factors(risk_factors):
    """Risk faktörlerini formatla"""
    if not risk_factors or not isinstance(risk_factors, dict):
        return "📊 Risk analizi yok"
    
    risk_messages = []
    
    if risk_factors.get('close_probabilities'):
        risk_messages.append("⚖️ Yakın olasılıklar")
    
    if risk_factors.get('high_importance'):
        risk_messages.append("🔥 Önemli maç")
    
    if risk_factors.get('unstable_form'):
        risk_messages.append("📉 Dengesiz form")
    
    if risk_factors.get('market_inefficiency'):
        risk_messages.append("💸 Piyasa anomalisi")
    
    overall_risk = risk_factors.get('overall_risk', 0)
    if overall_risk >= 0.7:
        risk_level = "🔴 Yüksek Risk"
    elif overall_risk >= 0.4:
        risk_level = "🟡 Orta Risk"
    else:
        risk_level = "🟢 Düşük Risk"
    
    if risk_messages:
        return f"{risk_level}: {', '.join(risk_messages)}"
    
    return f"{risk_level}"

def format_probabilities(probabilities):
    """Olasılıkları formatla"""
    if not probabilities or not isinstance(probabilities, dict):
        return "1: 33% | X: 34% | 2: 33%"
    
    try:
        home = probabilities.get('1', 33.3)
        draw = probabilities.get('X', 33.3)
        away = probabilities.get('2', 33.3)
        
        return f"1: {home:.1f}% | X: {draw:.1f}% | 2: {away:.1f}%"
    except:
        return "1: 33% | X: 34% | 2: 33%"

def calculate_goal_difference(goals_for, goals_against):
    """Gol averajını hesapla - AI için optimize"""
    try:
        return int(goals_for) - int(goals_against)
    except:
        return 0

def format_form_display(form_list):
    """Form listesini görsel olarak formatla - AI momentum için"""
    if not form_list:
        return "📊 Veri yok"
    
    form_icons = {
        'G': '🟢',  # Galibiyet
        'B': '🔵',  # Beraberlik
        'M': '🔴',   # Mağlubiyet
        'W': '🟢',  # Win (İngilizce)
        'D': '🔵',  # Draw (İngilizce)
        'L': '🔴'   # Loss (İngilizce)
    }
    
    display = ""
    for result in form_list[:5]:  # Son 5 maç
        display += form_icons.get(result, '⚪')
    
    return display

def calculate_form_score(form_list):
    """Form skoru hesapla - AI için"""
    if not form_list:
        return 0.5
    
    score_map = {'G': 1.0, 'W': 1.0, 'B': 0.5, 'D': 0.5, 'M': 0.0, 'L': 0.0}
    total_score = 0
    valid_matches = 0
    
    for i, result in enumerate(form_list[:5]):  # Son 5 maç, ağırlıklı
        weight = 1.0 - (i * 0.15)  # En son maçlar daha ağırlıklı
        if result in score_map:
            total_score += score_map[result] * weight
            valid_matches += weight
    
    if valid_matches > 0:
        return total_score / valid_matches
    
    return 0.5

def get_team_strength_category(strength):
    """Takım gücü kategorisi - AI advanced strength için"""
    try:
        strength = float(strength)
        if strength >= 0.8:
            return "🔥 Çok Güçlü"
        elif strength >= 0.65:
            return "💪 Güçlü"
        elif strength >= 0.5:
            return "⚖️ Orta"
        elif strength >= 0.35:
            return "📉 Zayıf"
        else:
            return "🆘 Çok Zayıf"
    except:
        return "❓ Bilinmeyen"

def format_percentage(value, decimal_places=1):
    """Yüzde formatla - AI probability için"""
    try:
        return f"%{float(value):.{decimal_places}f}"
    except:
        return "%0.0"

def safe_divide(numerator, denominator, default=0):
    """Güvenli bölme işlemi - AI feature engineering için"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default

def get_match_importance(league, teams, context=None):
    """Maç önemini değerlendir - AI context integration"""
    important_leagues = ['Süper Lig', 'Premier League', 'La Liga', 'Bundesliga', 'Serie A']
    big_teams = ['Fenerbahçe', 'Galatasaray', 'Beşiktaş', 'Trabzonspor', 'Başakşehir',
                 'Manchester City', 'Liverpool', 'Real Madrid', 'Barcelona', 'Bayern Munich', 'Juventus']
    
    importance = 1.0
    
    # Context'ten importance al
    if context and 'importance' in context:
        importance *= float(context['importance'])
    
    # Lig önemine göre
    if league in important_leagues:
        importance *= 1.5
    
    # Büyük takım varsa
    for team in teams:
        if team in big_teams:
            importance *= 1.2
    
    return min(3.0, importance)  # Maksimum 3x önem

def validate_prediction_data(prediction):
    """Tahmin verisini doğrula - AI alanları eklendi"""
    required_fields = ['result_prediction', 'confidence']
    
    for field in required_fields:
        if field not in prediction:
            return False
    
    try:
        confidence = float(prediction['confidence'])
        if not (0 <= confidence <= 100):
            return False
    except:
        return False
    
    valid_results = ['1', 'X', '2', 'Belirsiz']
    if prediction['result_prediction'] not in valid_results:
        return False
    
    # AI spesifik validasyonlar
    if 'ai_powered' in prediction and prediction['ai_powered']:
        if 'certainty_index' not in prediction:
            return False
        certainty = float(prediction.get('certainty_index', 0))
        if not (0 <= certainty <= 1):
            return False
    
    return True

def generate_match_summary(match, prediction):
    """Maç özeti oluştur - AI özellikleri entegre"""
    home = match.get('home_team', 'Ev Sahibi')
    away = match.get('away_team', 'Deplasman')
    league = match.get('league', 'Bilinmeyen Lig')
    confidence = prediction.get('confidence', 0)
    result = prediction.get('result_prediction', 'Belirsiz')
    
    summary = f"📊 {home} vs {away}\n"
    summary += f"🏆 {format_league_name(league)}\n"
    summary += f"🎯 Tahmin: {result}\n"
    summary += f"📈 Güven: {format_confidence(confidence)}\n"
    
    # AI spesifik bilgiler
    if prediction.get('ai_powered'):
        summary += f"🤖 Kaynak: {format_ai_source(prediction.get('source', 'ai_enhanced'))}\n"
        summary += f"🔒 Kesinlik: {format_certainty_index(prediction.get('certainty_index', 0))}\n"
    
    if prediction.get('score_prediction'):
        summary += f"⚽ Skor: {prediction['score_prediction']}\n"
    
    if prediction.get('probabilities'):
        summary += f"📊 Olasılıklar: {format_probabilities(prediction['probabilities'])}\n"
    
    if prediction.get('risk_factors'):
        summary += f"⚠️ Risk: {format_risk_factors(prediction['risk_factors'])}\n"
    
    return summary

def export_predictions_to_text(matches_with_predictions):
    """Tahminleri metin formatında dışa aktar - AI raporu"""
    from datetime import datetime
    
    output = "PREDICTA AI FUTBOL TAHMİN RAPORU\n"
    output += "=" * 60 + "\n"
    output += f"Oluşturulma Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    output += f"AI Model Versiyonu: 3.0-advanced\n"
    output += f"Toplam Maç: {len(matches_with_predictions)}\n\n"
    
    # AI tahminlerini öne al
    ai_predictions = [mp for mp in matches_with_predictions 
                     if mp['prediction'].get('ai_powered', False)]
    basic_predictions = [mp for mp in matches_with_predictions 
                        if not mp['prediction'].get('ai_powered', False)]
    
    # AI tahminleri
    if ai_predictions:
        output += "🤖 AI TAHMİNLERİ (Gelişmiş)\n"
        output += "-" * 40 + "\n"
        for i, data in enumerate(ai_predictions, 1):
            output += _format_prediction_for_export(data, i)
    
    # Temel tahminler
    if basic_predictions:
        output += "\n📊 TEMEL TAHMİNLER\n"
        output += "-" * 40 + "\n"
        start_index = len(ai_predictions) + 1
        for i, data in enumerate(basic_predictions, start_index):
            output += _format_prediction_for_export(data, i)
    
    # İstatistikler
    output += "\n📈 İSTATİSTİKLER\n"
    output += "-" * 40 + "\n"
    total_confidences = [mp['prediction'].get('confidence', 0) for mp in matches_with_predictions]
    ai_confidences = [mp['prediction'].get('confidence', 0) for mp in matches_with_predictions 
                     if mp['prediction'].get('ai_powered', False)]
    
    output += f"Ortalama Güven: {np.mean(total_confidences):.1f}%\n"
    if ai_confidences:
        output += f"AI Ortalama Güven: {np.mean(ai_confidences):.1f}%\n"
    output += f"AI Tahmin Oranı: {len(ai_predictions)}/{len(matches_with_predictions)} "
    output += f"(%{(len(ai_predictions)/len(matches_with_predictions)*100):.1f})\n"
    
    return output

def _format_prediction_for_export(data, index):
    """İhracat için tahmin formatlama"""
    match = data['match']
    prediction = data['prediction']
    
    output = f"{index}. {match.get('home_team', '')} vs {match.get('away_team', '')}\n"
    output += f"   Lig: {match.get('league', '')}\n"
    output += f"   Tarih: {match.get('date', '')} {match.get('time', '')}\n"
    output += f"   Tahmin: {prediction.get('result_prediction', '')}\n"
    output += f"   Güven: {format_confidence(prediction.get('confidence', 0))}\n"
    
    if prediction.get('ai_powered'):
        output += f"   🤖 Kaynak: AI Gelişmiş\n"
        output += f"   🔒 Kesinlik: {format_certainty_index(prediction.get('certainty_index', 0))}\n"
    
    output += f"   Skor: {prediction.get('score_prediction', '')}\n"
    
    if prediction.get('probabilities'):
        output += f"   Olasılıklar: {format_probabilities(prediction['probabilities'])}\n"
    
    if prediction.get('risk_factors'):
        output += f"   Risk: {format_risk_factors(prediction['risk_factors'])}\n"
    
    output += "-" * 30 + "\n"
    return output

def format_ai_performance(performance_data):
    """AI performans verilerini formatla"""
    if not performance_data:
        return "🤖 AI performans verisi yok"
    
    output = "🤖 AI PERFORMANS RAPORU\n"
    output += "=" * 40 + "\n"
    
    output += f"Model Durumu: {performance_data.get('adaptation_status', 'Bilinmiyor')}\n"
    output += f"Doğruluk: {performance_data.get('current_accuracy', 0)*100:.1f}%\n"
    output += f"Eğitim Örnekleri: {performance_data.get('training_samples', 0)}\n"
    output += f"Özellik Sayısı: {performance_data.get('feature_count', 0)}\n"
    output += f"Çapraz Doğrulama: {performance_data.get('cross_validation_score', 0)*100:.1f}%\n"
    output += f"Model Kararlılığı: {performance_data.get('model_stability', 0)*100:.1f}%\n"
    output += f"Son Eğitim: {performance_data.get('last_training', 'Bilinmiyor')}\n"
    
    return output

def calculate_expected_points(home_goals, away_goals):
    """Beklenen puanları hesapla - AI için"""
    try:
        home_goals = float(home_goals)
        away_goals = float(away_goals)
        
        if home_goals > away_goals:
            return 3.0, 0.0
        elif home_goals == away_goals:
            return 1.0, 1.0
        else:
            return 0.0, 3.0
    except:
        return 1.0, 1.0

def normalize_feature_value(value, min_val, max_val):
    """Özellik değerini normalize et - AI feature engineering için"""
    try:
        if max_val == min_val:
            return 0.5
        return (float(value) - min_val) / (max_val - min_val)
    except:
        return 0.5

def detect_anomalies(values, threshold=2.0):
    """Anomali tespiti - AI risk assessment için"""
    if not values or len(values) < 2:
        return []
    
    try:
        values_array = np.array(values)
        mean_val = np.mean(values_array)
        std_val = np.std(values_array)
        
        if std_val == 0:
            return []
        
        z_scores = np.abs((values_array - mean_val) / std_val)
        anomalies = np.where(z_scores > threshold)[0]
        
        return anomalies.tolist()
    except:
        return []
