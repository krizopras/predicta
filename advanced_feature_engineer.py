#!/usr/bin/env python3
"""
Advanced Feature Engineering for Football Match Prediction v2.0
Improvements:
- Fixed momentum calculation normalization
- Enhanced rest days with log transformation
- Added missing features (fatigue, expected goals diff, form difference)
- League-specific team ratings with fallback
- Feature name constants for reproducibility
- Better datetime handling
"""
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
import json
from pathlib import Path

# Feature name constants for reproducibility
FEATURE_NAMES = [
    # Team Form (8 features)
    'home_win_rate', 'home_goals_per_match', 'home_goals_conceded_per_match', 'home_clean_sheet_rate',
    'away_win_rate', 'away_goals_per_match', 'away_goals_conceded_per_match', 'away_clean_sheet_rate',
    
    # Head-to-Head (4 features)
    'h2h_home_win_rate', 'h2h_avg_goals_home', 'h2h_avg_goals_away', 'h2h_btts_rate',
    
    # Odds-Based (4 features)
    'implied_prob_home', 'implied_prob_draw', 'implied_prob_away', 'bookmaker_margin',
    
    # League Stats (3 features)
    'league_avg_goals', 'league_home_win_rate', 'league_btts_rate',
    
    # Advanced Metrics (6 features)
    'home_momentum', 'away_momentum', 'home_rest_days_log', 'away_rest_days_log',
    'home_team_rating', 'away_team_rating',
    
    # NEW FEATURES (9 features)
    'home_advantage_score', 'away_disadvantage_score',
    'home_fatigue_index', 'away_fatigue_index',
    'home_xg_diff', 'away_xg_diff',
    'league_variance_factor', 'form_difference', 'rating_difference'
]

class AdvancedFeatureEngineer:
    def __init__(self, team_ratings_path: str = None):
        self.team_history = defaultdict(list)
        self.h2h_history = defaultdict(list)
        self.league_results = defaultdict(list)  # For variance calculation
        
        # Load team ratings if available
        self.team_ratings = self._load_team_ratings(team_ratings_path)
        
    def _load_team_ratings(self, path: str) -> Dict:
        """Load team ratings from JSON file or use defaults"""
        if path and Path(path).exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Default ratings
        return {
            'manchester city': 94, 'real madrid': 93, 'bayern munich': 95,
            'liverpool': 91, 'barcelona': 90, 'psg': 90, 'inter': 88,
            'galatasaray': 82, 'fenerbahce': 80, 'besiktas': 78, 'trabzonspor': 76,
            'arsenal': 89, 'chelsea': 86, 'tottenham': 81, 'manchester united': 84,
            'atletico madrid': 87, 'sevilla': 78, 'valencia': 75, 'villarreal': 77,
            'juventus': 85, 'milan': 86, 'napoli': 84, 'roma': 80, 'atalanta': 82,
            'dortmund': 86, 'leipzig': 83, 'leverkusen': 84, 'union berlin': 77,
            'marseille': 78, 'lyon': 77, 'monaco': 79, 'lille': 78,
            'legia warsaw': 72, 'lech poznan': 70, 'rakow': 69,
            'fcsb': 68, 'cfr cluj': 67, 'rapid bucuresti': 65
        }
    
    def get_feature_count(self) -> int:
        """Return total number of features"""
        return len(FEATURE_NAMES)
    
    def extract_features(self, match_data: Dict) -> np.ndarray:
        """
        Comprehensive feature extraction with all improvements
        
        Args:
            match_data: Dictionary containing match information
                Required keys: 'home_team', 'away_team', 'league', 'odds', 'date'
        
        Returns:
            numpy array of shape (34,) containing all features
        """
        features = []
        
        # 1. TEAM FORM FEATURES (8 features)
        home_form = self._calculate_form(match_data['home_team'], is_home=True)
        away_form = self._calculate_form(match_data['away_team'], is_home=False)
        features.extend([
            home_form['win_rate'],
            home_form['goals_per_match'],
            home_form['goals_conceded_per_match'],
            home_form['clean_sheet_rate'],
            away_form['win_rate'],
            away_form['goals_per_match'],
            away_form['goals_conceded_per_match'],
            away_form['clean_sheet_rate']
        ])
        
        # 2. HEAD-TO-HEAD FEATURES (4 features)
        h2h = self._get_h2h_stats(match_data['home_team'], match_data['away_team'])
        features.extend([
            h2h['home_win_rate'],
            h2h['avg_goals_home'],
            h2h['avg_goals_away'],
            h2h['btts_rate']
        ])
        
        # 3. ODDS-BASED FEATURES (4 features)
        odds = match_data.get('odds', {'1': 2.5, 'X': 3.2, '2': 3.0})
        implied_probs = self._calculate_implied_probabilities(odds)
        features.extend([
            implied_probs['home'],
            implied_probs['draw'],
            implied_probs['away'],
            implied_probs['margin']
        ])
        
        # 4. LEAGUE STRENGTH & HOME ADVANTAGE (3 features)
        league_stats = self._get_league_stats(match_data['league'])
        features.extend([
            league_stats['avg_goals'],
            league_stats['home_win_rate'],
            league_stats['btts_rate']
        ])
        
        # 5. ADVANCED METRICS (6 features) - FIXED MOMENTUM
        home_momentum = self._calculate_momentum(match_data['home_team'])
        away_momentum = self._calculate_momentum(match_data['away_team'])
        home_rest = self._calculate_rest_days_log(match_data['home_team'], match_data.get('date'))
        away_rest = self._calculate_rest_days_log(match_data['away_team'], match_data.get('date'))
        home_rating = self._get_team_value_rating(match_data['home_team'], match_data['league'])
        away_rating = self._get_team_value_rating(match_data['away_team'], match_data['league'])
        
        features.extend([
            home_momentum,
            away_momentum,
            home_rest,
            away_rest,
            home_rating,
            away_rating
        ])
        
        # 6. NEW FEATURES (9 features)
        # Home advantage score
        home_adv = self._calculate_home_advantage(match_data['home_team'])
        away_disadv = self._calculate_away_disadvantage(match_data['away_team'])
        
        # Fatigue index
        home_fatigue = self._calculate_fatigue_index(match_data['home_team'], match_data.get('date'))
        away_fatigue = self._calculate_fatigue_index(match_data['away_team'], match_data.get('date'))
        
        # Expected goals difference (approximation)
        home_xg_diff = home_form['goals_per_match'] - home_form['goals_conceded_per_match']
        away_xg_diff = away_form['goals_per_match'] - away_form['goals_conceded_per_match']
        
        # League variance factor
        league_var = self._calculate_league_variance(match_data['league'])
        
        # Form difference
        form_diff = (home_form['win_rate'] * 3 + home_form['goals_per_match']) - \
                    (away_form['win_rate'] * 3 + away_form['goals_per_match'])
        
        # Rating difference
        rating_diff = home_rating - away_rating
        
        features.extend([
            home_adv,
            away_disadv,
            home_fatigue,
            away_fatigue,
            home_xg_diff,
            away_xg_diff,
            league_var,
            form_diff,
            rating_diff
        ])
        
        # Ensure consistent output
        assert len(features) == len(FEATURE_NAMES), \
            f"Feature count mismatch: {len(features)} != {len(FEATURE_NAMES)}"
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_form(self, team: str, is_home: bool, n_matches: int = 5) -> Dict:
        """Calculate team form from last N matches"""
        recent = self.team_history[team][-n_matches:]
        
        if not recent:
            return {
                'win_rate': 0.5, 
                'goals_per_match': 1.5,
                'goals_conceded_per_match': 1.2, 
                'clean_sheet_rate': 0.3
            }
        
        wins = sum(1 for m in recent if m['result'] == 'W')
        goals_scored = sum(m['goals_for'] for m in recent)
        goals_conceded = sum(m['goals_against'] for m in recent)
        clean_sheets = sum(1 for m in recent if m['goals_against'] == 0)
        
        return {
            'win_rate': wins / len(recent),
            'goals_per_match': goals_scored / len(recent),
            'goals_conceded_per_match': goals_conceded / len(recent),
            'clean_sheet_rate': clean_sheets / len(recent)
        }
    
    def _get_h2h_stats(self, home: str, away: str, n_matches: int = 10) -> Dict:
        """Head-to-head statistics"""
        key = f"{home}_vs_{away}"
        recent_h2h = self.h2h_history[key][-n_matches:]
        
        if not recent_h2h:
            return {
                'home_win_rate': 0.4, 
                'avg_goals_home': 1.5,
                'avg_goals_away': 1.3, 
                'btts_rate': 0.5
            }
        
        home_wins = sum(1 for m in recent_h2h if m['result'] == 'H')
        total_goals_h = sum(m['home_goals'] for m in recent_h2h)
        total_goals_a = sum(m['away_goals'] for m in recent_h2h)
        btts = sum(1 for m in recent_h2h if m['home_goals'] > 0 and m['away_goals'] > 0)
        
        return {
            'home_win_rate': home_wins / len(recent_h2h),
            'avg_goals_home': total_goals_h / len(recent_h2h),
            'avg_goals_away': total_goals_a / len(recent_h2h),
            'btts_rate': btts / len(recent_h2h)
        }
    
    def _calculate_implied_probabilities(self, odds: Dict) -> Dict:
        """Convert odds to probabilities and calculate margin"""
        implied = {}
        for key in ['1', 'X', '2']:
            implied[{'1': 'home', 'X': 'draw', '2': 'away'}[key]] = 1 / float(odds[key])
        
        total = sum(implied.values())
        margin = (total - 1) * 100
        
        # Normalize to true probabilities
        for key in ['home', 'draw', 'away']:
            implied[key] = implied[key] / total
        
        implied['margin'] = margin
        return implied
    
    def _get_league_stats(self, league: str) -> Dict:
        """League-specific statistics"""
        league_profiles = {
            'super lig': {'avg_goals': 2.68, 'home_win_rate': 0.45, 'btts_rate': 0.52},
            'premier league': {'avg_goals': 2.82, 'home_win_rate': 0.44, 'btts_rate': 0.54},
            'la liga': {'avg_goals': 2.58, 'home_win_rate': 0.46, 'btts_rate': 0.51},
            'bundesliga': {'avg_goals': 3.15, 'home_win_rate': 0.43, 'btts_rate': 0.58},
            'serie a': {'avg_goals': 2.75, 'home_win_rate': 0.44, 'btts_rate': 0.50},
            'ligue 1': {'avg_goals': 2.63, 'home_win_rate': 0.45, 'btts_rate': 0.49},
            'ekstraklasa': {'avg_goals': 2.55, 'home_win_rate': 0.47, 'btts_rate': 0.48},
            'liga 1': {'avg_goals': 2.45, 'home_win_rate': 0.48, 'btts_rate': 0.46},
            '1. lig': {'avg_goals': 2.50, 'home_win_rate': 0.46, 'btts_rate': 0.50}
        }
        
        league_lower = league.lower()
        for known_league, stats in league_profiles.items():
            if known_league in league_lower:
                return stats
        
        return {'avg_goals': 2.65, 'home_win_rate': 0.45, 'btts_rate': 0.52}
    
    def _calculate_momentum(self, team: str) -> float:
        """
        Calculate team momentum (weighted recent form)
        FIXED: Proper normalization
        """
        recent = self.team_history[team][-5:]
        if not recent:
            return 0.5
        
        weights = [0.1, 0.15, 0.2, 0.25, 0.3]
        weighted_points = sum(
            w * (3 if m['result'] == 'W' else 1 if m['result'] == 'D' else 0)
            for w, m in zip(weights, recent)
        )
        
        # Fixed: Normalize by sum of weights and max points
        return weighted_points / (sum(weights) * 3)
    
    def _calculate_rest_days_log(self, team: str, match_date: str = None) -> float:
        """
        Days since last match with log transformation
        FIXED: No artificial cap, uses log scale
        """
        if not self.team_history[team]:
            return np.log1p(7)  # Default 7 days
        
        last_match = self.team_history[team][-1]
        
        # Use provided date or current date
        if match_date:
            current_date = datetime.fromisoformat(match_date)
        else:
            current_date = datetime.now()
        
        last_date = datetime.fromisoformat(last_match.get('date', current_date.isoformat()))
        days = max(0, (current_date - last_date).days)
        
        # Log transformation to handle wide range
        return np.log1p(days)
    
    def _get_team_value_rating(self, team: str, league: str) -> float:
        """
        Team quality rating (0-1) with league-specific fallback
        IMPROVED: Uses league averages as fallback
        """
        # League average ratings as fallback
        league_avg_rating = {
            'super lig': 0.78,
            'premier league': 0.90,
            'la liga': 0.88,
            'bundesliga': 0.89,
            'serie a': 0.87,
            'ligue 1': 0.83,
            'ekstraklasa': 0.72,
            'liga 1': 0.68,
            '1. lig': 0.73
        }
        
        team_lower = team.lower()
        for known_team, rating in self.team_ratings.items():
            if known_team in team_lower or team_lower in known_team:
                return rating / 100
        
        # Use league average as fallback
        league_lower = league.lower()
        for known_league, avg_rating in league_avg_rating.items():
            if known_league in league_lower:
                return avg_rating
        
        return 0.70  # Global fallback
    
    # NEW FEATURE METHODS
    
    def _calculate_home_advantage(self, team: str, n_matches: int = 10) -> float:
        """Calculate average points won at home"""
        home_matches = [m for m in self.team_history[team][-n_matches:] if m.get('venue') == 'home']
        
        if not home_matches:
            return 1.5  # League average ~1.5 points/home match
        
        points = sum(3 if m['result'] == 'W' else 1 if m['result'] == 'D' else 0 
                     for m in home_matches)
        return points / len(home_matches)
    
    def _calculate_away_disadvantage(self, team: str, n_matches: int = 10) -> float:
        """Calculate average points won away"""
        away_matches = [m for m in self.team_history[team][-n_matches:] if m.get('venue') == 'away']
        
        if not away_matches:
            return 1.0  # League average ~1.0 points/away match
        
        points = sum(3 if m['result'] == 'W' else 1 if m['result'] == 'D' else 0 
                     for m in away_matches)
        return points / len(away_matches)
    
    def _calculate_fatigue_index(self, team: str, match_date: str = None, days: int = 10) -> int:
        """Number of matches played in last N days"""
        if not self.team_history[team]:
            return 0
        
        if match_date:
            current_date = datetime.fromisoformat(match_date)
        else:
            current_date = datetime.now()
        
        cutoff_date = current_date - timedelta(days=days)
        
        recent_matches = [
            m for m in self.team_history[team]
            if datetime.fromisoformat(m.get('date', '2024-01-01')) >= cutoff_date
        ]
        
        return len(recent_matches)
    
    def _calculate_league_variance(self, league: str) -> float:
        """
        Calculate result distribution variance in league
        Higher variance = more unpredictable league
        """
        if league not in self.league_results or len(self.league_results[league]) < 20:
            # Default moderate variance
            return 0.5
        
        results = self.league_results[league]
        home_wins = sum(1 for r in results if r == 'H') / len(results)
        draws = sum(1 for r in results if r == 'D') / len(results)
        away_wins = sum(1 for r in results if r == 'A') / len(results)
        
        # Calculate variance of result distribution
        mean = 1/3
        variance = ((home_wins - mean)**2 + (draws - mean)**2 + (away_wins - mean)**2) / 3
        
        # Normalize to 0-1 scale
        return min(variance * 10, 1.0)
    
    def update_team_history(self, team: str, match_result: Dict):
        """Update team history with new match"""
        self.team_history[team].append(match_result)
        
        # Keep only last 20 matches
        if len(self.team_history[team]) > 20:
            self.team_history[team] = self.team_history[team][-20:]
    
    def update_h2h_history(self, home: str, away: str, match_result: Dict):
        """Update head-to-head history"""
        key = f"{home}_vs_{away}"
        self.h2h_history[key].append(match_result)
        
        if len(self.h2h_history[key]) > 15:
            self.h2h_history[key] = self.h2h_history[key][-15:]
    
    def update_league_results(self, league: str, result: str):
        """Update league result distribution"""
        self.league_results[league].append(result)
        
        if len(self.league_results[league]) > 500:
            self.league_results[league] = self.league_results[league][-500:]


# Usage example
if __name__ == "__main__":
    engineer = AdvancedFeatureEngineer()
    
    print(f"Total features: {engineer.get_feature_count()}")
    print(f"\nFeature names:\n{FEATURE_NAMES}")
    
    # Example match
    match = {
        'home_team': 'Galatasaray',
        'away_team': 'Fenerbahce',
        'league': 'Super Lig',
        'date': '2024-03-15',
        'odds': {'1': 2.1, 'X': 3.4, '2': 3.6}
    }
    
    features = engineer.extract_features(match)
    print(f"\nExtracted features shape: {features.shape}")
    print(f"Sample features: {features[:10]}")
