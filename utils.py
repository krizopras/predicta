#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yardımcı fonksiyonlar
"""

def format_odds(odds):
    """Oranları formatla"""
    try:
        return f"{float(odds):.2f}"
    except:
        return "N/A"

def format_confidence(confidence):
    """Güven seviyesini formatla"""
    try:
        return f"%{float(confidence):.1f}"
    except:
        return "%0.0"

def get_prediction_color(confidence):
    """Güven seviyesine göre renk"""
    try:
        conf = float(confidence)
        if conf >= 70:
            return "high"
        elif conf >= 50:
            return "medium"
        else:
            return "low"
    except:
        return "low"

def format_team_name(name, max_length=20):
    """Takım adını formatla"""
    if not name:
        return "Bilinmeyen"
    
    if len(name) > max_length:
        return name[:max_length-3] + "..."
    
    return name

def format_league_name(league):
    """Lig adını formatla"""
    league_map = {
        'Süper Lig': '🇹🇷 Süper Lig',
        'Premier League': '🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League',
        'La Liga': '🇪🇸 La Liga',
        'Bundesliga': '🇩🇪 Bundesliga',
        'Serie A': '🇮🇹 Serie A',
        'Ligue 1': '🇫🇷 Ligue 1'
    }
    
    return league_map.get(league, f"⚽ {league}")

def calculate_goal_difference(goals_for, goals_against):
    """Gol averajını hesapla"""
    try:
        return int(goals_for) - int(goals_against)
    except:
        return 0

def format_form_display(form_list):
    """Form listesini görsel olarak formatla"""
    if not form_list:
        return "📊 Veri yok"
    
    form_icons = {
        'G': '🟢',  # Galibiyet
        'B': '🔵',  # Beraberlik
        'M': '🔴'   # Mağlubiyet
    }
    
    display = ""
    for result in form_list[:5]:  # Son 5 maç
        display += form_icons.get(result, '⚪')
    
    return display

def get_team_strength_category(strength):
    """Takım gücü kategorisi"""
    try:
        strength = float(strength)
        if strength >= 80:
            return "🔥 Çok Güçlü"
        elif strength >= 65:
            return "💪 Güçlü"
        elif strength >= 50:
            return "⚖️ Orta"
        elif strength >= 35:
            return "📉 Zayıf"
        else:
            return "🆘 Çok Zayıf"
    except:
        return "❓ Bilinmeyen"

def format_percentage(value, decimal_places=1):
    """Yüzde formatla"""
    try:
        return f"%{float(value):.{decimal_places}f}"
    except:
        return "%0.0"

def safe_divide(numerator, denominator, default=0):
    """Güvenli bölme işlemi"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default

def get_match_importance(league, teams):
    """Maç önemini değerlendir"""
    important_leagues = ['Süper Lig', 'Premier League', 'La Liga', 'Bundesliga', 'Serie A']
    big_teams = ['Fenerbahçe', 'Galatasaray', 'Beşiktaş', 'Manchester City', 'Liverpool', 
                 'Real Madrid', 'Barcelona', 'Bayern Munich', 'Juventus']
    
    importance = 1.0
    
    # Lig önemine göre
    if league in important_leagues:
        importance *= 1.5
    
    # Büyük takım varsa
    for team in teams:
        if team in big_teams:
            importance *= 1.2
    
    return min(3.0, importance)  # Maksimum 3x önem

def validate_prediction_data(prediction):
    """Tahmin verisini doğrula"""
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
    
    return True

def generate_match_summary(match, prediction):
    """Maç özeti oluştur"""
    home = match.get('home_team', 'Ev Sahibi')
    away = match.get('away_team', 'Deplasman')
    league = match.get('league', 'Bilinmeyen Lig')
    confidence = prediction.get('confidence', 0)
    result = prediction.get('result_prediction', 'Belirsiz')
    
    summary = f"📊 {home} vs {away}\n"
    summary += f"🏆 {league}\n"
    summary += f"🎯 Tahmin: {result}\n"
    summary += f"📈 Güven: {format_confidence(confidence)}\n"
    
    if prediction.get('score_prediction'):
        summary += f"⚽ Skor: {prediction['score_prediction']}\n"
    
    return summary

def export_predictions_to_text(matches_with_predictions):
    """Tahminleri metin formatında dışa aktar"""
    from datetime import datetime
    
    output = "NESINE FUTBOL TAHMİN RAPORU\n"
    output += "=" * 50 + "\n"
    output += f"Oluşturulma Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for i, data in enumerate(matches_with_predictions, 1):
        match = data['match']
        prediction = data['prediction']
        
        output += f"{i}. {match.get('home_team', '')} vs {match.get('away_team', '')}\n"
        output += f"   Lig: {match.get('league', '')}\n"
        output += f"   Tarih: {match.get('date', '')} {match.get('time', '')}\n"
        output += f"   Tahmin: {prediction.get('result_prediction', '')}\n"
        output += f"   Güven: {format_confidence(prediction.get('confidence', 0))}\n"
        output += f"   Skor: {prediction.get('score_prediction', '')}\n"
        output += "-" * 30 + "\n"
    
    return output
