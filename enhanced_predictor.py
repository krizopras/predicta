# enhanced_predictor.py
from advanced_feature_engineer import AdvancedFeatureEngineer
from improved_prediction_engine import ProfessionalPredictionEngine
from raw_data_parser import HistoricalDataParser
import logging

logger = logging.getLogger(__name__)

class EnhancedPredictor:
    def __init__(self):
        self.feature_engineer = AdvancedFeatureEngineer()
        self.prediction_engine = ProfessionalPredictionEngine()
        self.data_parser = HistoricalDataParser()
        
        # Geçmiş verileri yükle
        self._load_historical_data()
    
    def _load_historical_data(self):
        """Geçmiş maç verilerini feature engineer'a yükle"""
        try:
            historical_data = self.data_parser.load_all_historical_data()
            
            for league, matches in historical_data.items():
                for match in matches:
                    # Takım geçmişini güncelle
                    self._update_team_stats(match['home_team'], match, is_home=True)
                    self._update_team_stats(match['away_team'], match, is_home=False)
                    
                    # H2H geçmişini güncelle
                    self.feature_engineer.update_h2h_history(
                        match['home_team'],
                        match['away_team'],
                        {
                            'result': match['result'],
                            'home_goals': match['home_goals'],
                            'away_goals': match['away_goals']
                        }
                    )
            
            logger.info(f"✅ {len(historical_data)} lig için geçmiş veriler yüklendi")
            
        except Exception as e:
            logger.error(f"❌ Geçmiş veri yükleme hatası: {e}")
    
    def _update_team_stats(self, team: str, match: Dict, is_home: bool):
        """Takım istatistiklerini güncelle"""
        if is_home:
            result = match['result']
            goals_for = match['home_goals']
            goals_against = match['away_goals']
        else:
            result = '2' if match['result'] == '1' else ('1' if match['result'] == '2' else 'X')
            goals_for = match['away_goals']
            goals_against = match['home_goals']
        
        match_data = {
            'result': 'W' if (result == '1' and is_home) or (result == '2' and not is_home) else 
                     ('D' if result == 'X' else 'L'),
            'goals_for': goals_for,
            'goals_against': goals_against,
            'date': match['date'],
            'venue': 'home' if is_home else 'away'
        }
        
        self.feature_engineer.update_team_history(team, match_data)
    
    def predict_match(self, home_team: str, away_team: str, odds: Dict, league: str):
        """Gelişmiş tahmin - geçmiş verilerle"""
        
        # Feature extraction (geçmiş verilerle zenginleştirilmiş)
        match_data = {
            'home_team': home_team,
            'away_team': away_team,
            'league': league,
            'odds': odds,
            'date': datetime.now().isoformat()
        }
        
        features = self.feature_engineer.extract_features(match_data)
        
        # Temel tahmin motoru
        base_prediction = self.prediction_engine.predict_match(
            home_team, away_team, odds, league
        )
        
        # Feature'ları kullanarak tahmini güçlendir
        enhanced_prediction = self._enhance_with_features(base_prediction, features)
        
        return enhanced_prediction
    
    def _enhance_with_features(self, base_prediction: Dict, features) -> Dict:
        """Feature'ları kullanarak tahmini iyileştir"""
        
        # Form faktörü
        home_form = features[0]  # home_win_rate
        away_form = features[4]  # away_win_rate
        
        # H2H faktörü
        h2h_home_win = features[8]  # h2h_home_win_rate
        
        # Güven seviyesini ayarla
        confidence = base_prediction['confidence']
        
        # Form farkı büyükse güveni artır
        form_diff = abs(home_form - away_form)
        if form_diff > 0.3:
            confidence *= 1.1
        
        # H2H tarihi net ise güveni artır
        if h2h_home_win > 0.7 or h2h_home_win < 0.3:
            confidence *= 1.05
        
        base_prediction['confidence'] = min(confidence, 95)  # Max %95
        base_prediction['features_used'] = True
        base_prediction['form_factor'] = {
            'home_form': round(home_form, 2),
            'away_form': round(away_form, 2),
            'h2h_home_advantage': round(h2h_home_win, 2)
        }
        
        return base_prediction
