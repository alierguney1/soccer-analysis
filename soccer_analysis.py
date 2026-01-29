#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detailed Soccer Analysis System
Analyzes player ratings with 6-axis rating system and voter reliability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

class SoccerAnalysis:
    """Comprehensive soccer player rating and voter analysis system"""
    
    # Reliability scoring thresholds and penalties
    HIGH_VARIANCE_THRESHOLD = 2.5  # Standard deviation indicating inconsistent scoring
    LOW_VARIANCE_THRESHOLD = 1.0   # Standard deviation indicating lack of differentiation
    GENEROUS_DEVIATION = 1.5       # Mean deviation indicating too generous scoring
    HARSH_DEVIATION = -1.5         # Mean deviation indicating too harsh scoring
    SKILL_INCONSISTENCY_THRESHOLD = 3.0  # Std deviation for similar skills inconsistency
    
    # Reliability score penalties
    PENALTY_HIGH_VARIANCE = 20
    PENALTY_LOW_VARIANCE = 10
    PENALTY_GENEROUS = 25
    PENALTY_HARSH = 25
    PENALTY_MISSING_DATA = 15
    
    # Filtering thresholds
    RELIABILITY_FILTER_THRESHOLD = 75  # Voters below this reliability are excluded
    
    def __init__(self, csv_file):
        """Initialize with CSV file"""
        self.csv_file = csv_file
        self.df = None
        self.players = []
        self.voters = []
        self.skill_categories = {}
        self.player_ratings = {}
        self.voter_analysis = {}
        
    def load_data(self):
        """Load and parse the CSV data"""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        
        # Read CSV with semicolon separator
        self.df = pd.read_csv(self.csv_file, sep=';', encoding='utf-8-sig')
        
        # Clean column names
        self.df.columns = self.df.columns.str.strip()
        
        print(f"Data loaded successfully!")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Extract player names
        self.players = self.df['Name'].unique().tolist()
        print(f"\nNumber of players: {len(self.players)}")
        print(f"Players: {self.players}")
        
        # Extract voter columns
        self.voters = [col for col in self.df.columns if col.startswith('Voter')]
        print(f"\nNumber of voters: {len(self.voters)}")
        
        # Categorize skills
        self._categorize_skills()
        
    def _categorize_skills(self):
        """Categorize skills into groups"""
        self.skill_categories = {
            'technical_ball_control': [
                'Technical >> Controlling',
                'Technical >> Dribbling',
                'Technical >> Passing'
            ],
            'shooting_finishing': [
                'Technical >> Shooting',
                'Technical >> Finishing'
            ],
            'offensive_play': [
                'Technical >> Offensive Play'
            ],
            'defensive_play': [
                'Technical >> Defensive Play'
            ],
            'tactical_psychological': [
                'Tactical/Psychological/Physical >> Disciplined',
                'Tactical/Psychological/Physical >> Decision making',
                'Tactical/Psychological/Physical >> Teamwork'
            ],
            'physical_condition': [
                'Tactical/Psychological/Physical >> Condition'
            ]
        }
        
    def calculate_6_axis_ratings(self):
        """Calculate 6-axis player rating system"""
        print("\n" + "=" * 80)
        print("6-AXIS PLAYER RATING SYSTEM CALCULATION")
        print("=" * 80)
        
        print("\nAxis definitions:")
        print("1. Technical Ball Control: Controlling, Dribbling, Passing")
        print("2. Shooting & Finishing: Shooting, Finishing")
        print("3. Offensive Play: Offensive Play")
        print("4. Defensive Play: Defensive Play")
        print("5. Tactical/Psychological: Disciplined, Decision making, Teamwork")
        print("6. Physical/Condition: Condition")
        
        self.player_ratings = {}
        
        for player in self.players:
            player_data = self.df[self.df['Name'] == player]
            
            ratings = {
                'player_name': player,
                'axis_ratings': {},
                'overall_rating': 0,
                'skill_details': {}
            }
            
            # Calculate each axis
            for axis_name, skills in self.skill_categories.items():
                axis_scores = []
                
                for skill in skills:
                    skill_data = player_data[player_data['Skill'] == skill]
                    if not skill_data.empty:
                        # Get all voter scores (excluding empty values)
                        scores = []
                        for voter in self.voters:
                            try:
                                score = skill_data[voter].values[0]
                                if pd.notna(score) and str(score).strip():
                                    scores.append(float(score))
                            except:
                                continue
                        
                        if scores:
                            avg_score = np.mean(scores)
                            axis_scores.append(avg_score)
                            ratings['skill_details'][skill] = {
                                'scores': scores,
                                'mean': avg_score,
                                'std': np.std(scores),
                                'median': np.median(scores),
                                'min': min(scores),
                                'max': max(scores)
                            }
                
                # Calculate axis average
                if axis_scores:
                    ratings['axis_ratings'][axis_name] = {
                        'score': np.mean(axis_scores),
                        'std': np.std(axis_scores)
                    }
            
            # Calculate overall rating (weighted average of all axes)
            # Using equal weights for simplicity, but can be adjusted
            axis_weights = {
                'technical_ball_control': 0.20,
                'shooting_finishing': 0.15,
                'offensive_play': 0.15,
                'defensive_play': 0.15,
                'tactical_psychological': 0.20,
                'physical_condition': 0.15
            }
            
            overall = 0
            total_weight = 0
            for axis_name, weight in axis_weights.items():
                if axis_name in ratings['axis_ratings']:
                    overall += ratings['axis_ratings'][axis_name]['score'] * weight
                    total_weight += weight
            
            if total_weight > 0:
                ratings['overall_rating'] = overall / total_weight
            
            self.player_ratings[player] = ratings
        
        # Display results
        self._display_player_ratings()
        
    def _display_player_ratings(self):
        """Display player ratings in detail"""
        print("\n" + "-" * 80)
        print("PLAYER RATINGS SUMMARY")
        print("-" * 80)
        
        # Create DataFrame for better display
        summary_data = []
        for player, data in self.player_ratings.items():
            row = {'Player': player, 'Overall': round(data['overall_rating'], 2)}
            for axis_name, axis_data in data['axis_ratings'].items():
                axis_display = axis_name.replace('_', ' ').title()
                row[axis_display] = round(axis_data['score'], 2)
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Overall', ascending=False)
        print(summary_df.to_string(index=False))
        
        # Detailed breakdown for top 3 players
        print("\n" + "-" * 80)
        print("DETAILED BREAKDOWN - TOP 3 PLAYERS")
        print("-" * 80)
        
        top_players = summary_df.head(3)['Player'].tolist()
        for player in top_players:
            data = self.player_ratings[player]
            print(f"\n{player} - Overall Rating: {data['overall_rating']:.2f}")
            print("-" * 60)
            for axis_name, axis_data in data['axis_ratings'].items():
                print(f"{axis_name.replace('_', ' ').title():30s}: {axis_data['score']:.2f} ± {axis_data['std']:.2f}")
        
    def analyze_voters(self):
        """Comprehensive voter analysis"""
        print("\n" + "=" * 80)
        print("VOTER RELIABILITY AND BIAS ANALYSIS")
        print("=" * 80)
        
        self.voter_analysis = {}
        
        for voter_idx, voter in enumerate(self.voters):
            voter_num = voter_idx + 1
            
            analysis = {
                'voter_id': voter,
                'voter_number': voter_num,
                'statistics': {},
                'reliability_score': 0,
                'bias_indicators': {},
                'anomalies': []
            }
            
            # Collect all scores from this voter
            all_scores = []
            for _, row in self.df.iterrows():
                try:
                    score = row[voter]
                    if pd.notna(score) and str(score).strip():
                        all_scores.append(float(score))
                except:
                    continue
            
            if all_scores:
                # Basic statistics
                analysis['statistics'] = {
                    'mean': np.mean(all_scores),
                    'median': np.median(all_scores),
                    'std': np.std(all_scores),
                    'min': min(all_scores),
                    'max': max(all_scores),
                    'range': max(all_scores) - min(all_scores),
                    'count': len(all_scores),
                    'missing_count': len(self.df) - len(all_scores)
                }
                
                # Variance analysis
                analysis['bias_indicators']['high_variance'] = analysis['statistics']['std'] > self.HIGH_VARIANCE_THRESHOLD
                analysis['bias_indicators']['low_variance'] = analysis['statistics']['std'] < self.LOW_VARIANCE_THRESHOLD
                
                # Mean deviation from overall mean
                all_voters_scores = []
                for v in self.voters:
                    for _, row in self.df.iterrows():
                        try:
                            score = row[v]
                            if pd.notna(score) and str(score).strip():
                                all_voters_scores.append(float(score))
                        except:
                            continue
                
                overall_mean = np.mean(all_voters_scores)
                deviation = analysis['statistics']['mean'] - overall_mean
                analysis['bias_indicators']['mean_deviation'] = deviation
                analysis['bias_indicators']['too_generous'] = deviation > self.GENEROUS_DEVIATION
                analysis['bias_indicators']['too_harsh'] = deviation < self.HARSH_DEVIATION
                
                # NOTE: Self-rating analysis removed because voter order is random
                # and we don't have voter-to-player mapping data
                
                # Analyze consistency
                self._analyze_consistency(voter, analysis)
                
                # Calculate reliability score (0-100)
                reliability = 100
                if analysis['bias_indicators']['high_variance']:
                    reliability -= self.PENALTY_HIGH_VARIANCE
                if analysis['bias_indicators']['low_variance']:
                    reliability -= self.PENALTY_LOW_VARIANCE
                if analysis['bias_indicators']['too_generous']:
                    reliability -= self.PENALTY_GENEROUS
                if analysis['bias_indicators']['too_harsh']:
                    reliability -= self.PENALTY_HARSH
                if analysis['statistics']['missing_count'] > 10:
                    reliability -= self.PENALTY_MISSING_DATA
                
                analysis['reliability_score'] = max(0, reliability)
            
            self.voter_analysis[voter] = analysis
        
        # Display voter analysis
        self._display_voter_analysis()
        
    # NOTE: _analyze_self_rating method removed because voter order is random
    # and we don't have reliable voter-to-player mapping data
    # Self-bias cannot be detected without knowing which voter is which player
    
    def _analyze_consistency(self, voter, analysis):
        """Analyze voter consistency across similar skills"""
        # Check if voter gives similar scores for similar skill categories
        inconsistencies = []
        
        for player in self.players:
            player_data = self.df[self.df['Name'] == player]
            
            # Check technical skills consistency
            tech_scores = []
            for skill in ['Technical >> Controlling', 'Technical >> Dribbling', 'Technical >> Passing']:
                skill_data = player_data[player_data['Skill'] == skill]
                if not skill_data.empty:
                    try:
                        score = skill_data[voter].values[0]
                        if pd.notna(score) and str(score).strip():
                            tech_scores.append(float(score))
                    except:
                        continue
            
            if len(tech_scores) >= 2:
                std = np.std(tech_scores)
                if std > self.SKILL_INCONSISTENCY_THRESHOLD:  # High variation in similar skills
                    inconsistencies.append({
                        'player': player,
                        'category': 'Technical',
                        'scores': tech_scores,
                        'std': std
                    })
        
        analysis['bias_indicators']['inconsistencies'] = inconsistencies
        analysis['bias_indicators']['has_inconsistencies'] = len(inconsistencies) > 3
    
    def _display_voter_analysis(self):
        """Display voter analysis results"""
        print("\n" + "-" * 80)
        print("VOTER STATISTICS SUMMARY")
        print("-" * 80)
        
        voter_summary = []
        for voter, data in self.voter_analysis.items():
            if 'statistics' in data and data['statistics']:
                row = {
                    'Voter': data['voter_number'],
                    'Reliability': data['reliability_score'],
                    'Mean': round(data['statistics']['mean'], 2),
                    'Std': round(data['statistics']['std'], 2),
                    'Range': round(data['statistics']['range'], 2),
                    'Missing': data['statistics']['missing_count']
                }
                voter_summary.append(row)
        
        summary_df = pd.DataFrame(voter_summary)
        summary_df = summary_df.sort_values('Reliability', ascending=False)
        print(summary_df.to_string(index=False))
        
        # Detailed analysis
        print("\n" + "-" * 80)
        print("DETAILED VOTER BIAS ANALYSIS")
        print("-" * 80)
        
        for voter, data in sorted(self.voter_analysis.items(), 
                                 key=lambda x: x[1]['reliability_score']):
            if 'statistics' not in data or not data['statistics']:
                continue
                
            print(f"\nVoter {data['voter_number']} - Reliability Score: {data['reliability_score']}/100")
            print("-" * 60)
            
            # Display flags
            flags = []
            if data['bias_indicators'].get('too_generous'):
                flags.append(f"⚠️  TOO GENEROUS (mean {data['statistics']['mean']:.2f} vs overall)")
            if data['bias_indicators'].get('too_harsh'):
                flags.append(f"⚠️  TOO HARSH (mean {data['statistics']['mean']:.2f} vs overall)")
            if data['bias_indicators'].get('high_variance'):
                flags.append(f"⚠️  HIGH VARIANCE (std {data['statistics']['std']:.2f})")
            if data['bias_indicators'].get('low_variance'):
                flags.append(f"⚠️  LOW VARIANCE (std {data['statistics']['std']:.2f}) - possibly not differentiating")
            if data['bias_indicators'].get('has_inconsistencies'):
                flags.append(f"⚠️  INCONSISTENT RATINGS")
            
            if flags:
                for flag in flags:
                    print(flag)
            else:
                print("✓ No major bias indicators detected")
        
        # Summary of problematic voters
        print("\n" + "=" * 80)
        print("SUMMARY: VOTERS WITH POTENTIAL ISSUES")
        print("=" * 80)
        
        problematic = []
        for voter, data in self.voter_analysis.items():
            issues = []
            if data['bias_indicators'].get('too_generous'):
                issues.append('Too generous')
            if data['bias_indicators'].get('too_harsh'):
                issues.append('Too harsh')
            if data['bias_indicators'].get('has_inconsistencies'):
                issues.append('Inconsistent')
            
            if issues:
                problematic.append({
                    'Voter': data['voter_number'],
                    'Reliability': data['reliability_score'],
                    'Issues': ', '.join(issues)
                })
        
        if problematic:
            prob_df = pd.DataFrame(problematic)
            prob_df = prob_df.sort_values('Reliability')
            print(prob_df.to_string(index=False))
        else:
            print("No voters with significant issues detected.")
    
    def analyze_player_perception(self):
        """Analyze how different voters perceive each player"""
        print("\n" + "=" * 80)
        print("PLAYER PERCEPTION VARIANCE ANALYSIS")
        print("=" * 80)
        print("Shows which players are rated most inconsistently by different voters")
        
        perception_data = []
        
        for player in self.players:
            overall_scores = []
            
            # Get overall ratings from all voters
            player_overall = self.df[(self.df['Name'] == player) & 
                                     (self.df['Skill'] == 'Overall  Player Rating')]
            
            if not player_overall.empty:
                for voter in self.voters:
                    try:
                        score = player_overall[voter].values[0]
                        if pd.notna(score) and str(score).strip():
                            overall_scores.append(float(score))
                    except:
                        continue
            
            if overall_scores:
                perception_data.append({
                    'Player': player,
                    'Mean': round(np.mean(overall_scores), 2),
                    'Std': round(np.std(overall_scores), 2),
                    'Min': min(overall_scores),
                    'Max': max(overall_scores),
                    'Range': max(overall_scores) - min(overall_scores),
                    'Voters': len(overall_scores)
                })
        
        perception_df = pd.DataFrame(perception_data)
        perception_df = perception_df.sort_values('Std', ascending=False)
        
        print("\nPlayers with MOST INCONSISTENT ratings (high disagreement):")
        print(perception_df.head(5).to_string(index=False))
        
        print("\nPlayers with MOST CONSISTENT ratings (high agreement):")
        print(perception_df.tail(5).to_string(index=False))
    
    def generate_recommendations(self):
        """Generate recommendations based on analysis"""
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS AND CONCLUSIONS")
        print("=" * 80)
        
        # Data quality assessment
        print("\n1. DATA QUALITY ASSESSMENT:")
        reliable_voters = sum(1 for v in self.voter_analysis.values() 
                            if v['reliability_score'] >= 70)
        total_voters = len(self.voter_analysis)
        
        print(f"   - {reliable_voters}/{total_voters} voters have reliability score >= 70%")
        
        if reliable_voters / total_voters >= 0.8:
            print("   ✓ Overall data quality is GOOD")
        elif reliable_voters / total_voters >= 0.6:
            print("   ⚠️  Overall data quality is MODERATE")
        else:
            print("   ❌ Overall data quality is QUESTIONABLE")
        
        # Self-bias assessment
        print("\n2. SELF-BIAS ASSESSMENT:")
        self_biased = sum(1 for v in self.voter_analysis.values() 
                         if v['bias_indicators'].get('self_bias', False))
        print(f"   - {self_biased} voters show self-bias tendencies")
        
        if self_biased > 0:
            print("   ⚠️  Some voters rated themselves higher than others")
            for voter, data in self.voter_analysis.items():
                if data['bias_indicators'].get('self_bias', False):
                    player = data['bias_indicators'].get('player_name', 'Unknown')
                    diff = data['bias_indicators'].get('self_vs_others_diff', 0)
                    print(f"      - Voter {data['voter_number']} ({player}): +{diff:.2f} points self-inflation")
        
        # Generosity/Harshness
        print("\n3. RATING TENDENCY:")
        generous = sum(1 for v in self.voter_analysis.values() 
                      if v['bias_indicators'].get('too_generous', False))
        harsh = sum(1 for v in self.voter_analysis.values() 
                   if v['bias_indicators'].get('too_harsh', False))
        
        print(f"   - {generous} voters are too generous")
        print(f"   - {harsh} voters are too harsh")
        
        # Top players recommendation
        print("\n4. TOP PLAYER RECOMMENDATIONS:")
        summary_data = []
        for player, data in self.player_ratings.items():
            summary_data.append({
                'Player': player,
                'Overall': data['overall_rating']
            })
        summary_df = pd.DataFrame(summary_data).sort_values('Overall', ascending=False)
        
        print("   Based on 6-axis analysis, top 5 players are:")
        for idx, row in summary_df.head(5).iterrows():
            print(f"      {row['Player']}: {row['Overall']:.2f}")
        
        print("\n" + "=" * 80)
    
    def run_complete_analysis(self):
        """Run all analyses"""
        self.load_data()
        self.calculate_6_axis_ratings()
        self.analyze_voters()
        self.analyze_player_perception()
        self.generate_recommendations()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
    
    def get_filtered_voters(self):
        """Get list of reliable voters (not harsh, reliability >= threshold)"""
        filtered = []
        excluded = []
        
        for voter, data in self.voter_analysis.items():
            reliability = data['reliability_score']
            is_harsh = data['bias_indicators'].get('too_harsh', False)
            
            if reliability >= self.RELIABILITY_FILTER_THRESHOLD and not is_harsh:
                filtered.append(voter)
            else:
                reason = []
                if reliability < self.RELIABILITY_FILTER_THRESHOLD:
                    reason.append(f"reliability {reliability}% < {self.RELIABILITY_FILTER_THRESHOLD}%")
                if is_harsh:
                    reason.append("too harsh")
                excluded.append({
                    'voter': voter,
                    'voter_num': data['voter_number'],
                    'reliability': reliability,
                    'reason': ', '.join(reason)
                })
        
        return filtered, excluded
    
    def calculate_filtered_ratings(self):
        """Calculate player ratings using only filtered voters"""
        print("\n" + "=" * 80)
        print("FILTERED ANALYSIS (Excluding Too Harsh & Low Reliability Voters)")
        print("=" * 80)
        
        filtered_voters, excluded = self.get_filtered_voters()
        
        print(f"\n📊 Filtering Criteria:")
        print(f"   - Minimum reliability: {self.RELIABILITY_FILTER_THRESHOLD}%")
        print(f"   - Exclude 'too harsh' voters")
        print(f"\n✓ Included voters: {len(filtered_voters)}")
        print(f"✗ Excluded voters: {len(excluded)}")
        
        if excluded:
            print("\nExcluded voters:")
            for ex in excluded:
                print(f"   - Voter {ex['voter_num']}: {ex['reason']}")
        
        self.filtered_player_ratings = {}
        
        for player in self.players:
            player_data = self.df[self.df['Name'] == player]
            
            ratings = {
                'player_name': player,
                'axis_ratings': {},
                'overall_rating': 0,
                'skill_details': {}
            }
            
            # Calculate each axis using only filtered voters
            for axis_name, skills in self.skill_categories.items():
                axis_scores = []
                
                for skill in skills:
                    skill_data = player_data[player_data['Skill'] == skill]
                    if not skill_data.empty:
                        scores = []
                        for voter in filtered_voters:
                            try:
                                score = skill_data[voter].values[0]
                                if pd.notna(score) and str(score).strip():
                                    scores.append(float(score))
                            except:
                                continue
                        
                        if scores:
                            avg_score = np.mean(scores)
                            axis_scores.append(avg_score)
                            ratings['skill_details'][skill] = {
                                'scores': scores,
                                'mean': avg_score,
                                'std': np.std(scores),
                                'median': np.median(scores),
                                'min': min(scores),
                                'max': max(scores)
                            }
                
                if axis_scores:
                    ratings['axis_ratings'][axis_name] = {
                        'score': np.mean(axis_scores),
                        'std': np.std(axis_scores)
                    }
            
            # Calculate overall rating
            axis_weights = {
                'technical_ball_control': 0.20,
                'shooting_finishing': 0.15,
                'offensive_play': 0.15,
                'defensive_play': 0.15,
                'tactical_psychological': 0.20,
                'physical_condition': 0.15
            }
            
            overall = 0
            total_weight = 0
            for axis_name, weight in axis_weights.items():
                if axis_name in ratings['axis_ratings']:
                    overall += ratings['axis_ratings'][axis_name]['score'] * weight
                    total_weight += weight
            
            if total_weight > 0:
                ratings['overall_rating'] = overall / total_weight
            
            self.filtered_player_ratings[player] = ratings
        
        # Display comparison
        self._display_filtered_comparison()
        
        return self.filtered_player_ratings
    
    def _display_filtered_comparison(self):
        """Display comparison between all voters and filtered voters"""
        print("\n" + "-" * 80)
        print("COMPARISON: ALL VOTERS vs FILTERED VOTERS")
        print("-" * 80)
        
        comparison_data = []
        for player in self.players:
            all_rating = self.player_ratings[player]['overall_rating']
            filtered_rating = self.filtered_player_ratings[player]['overall_rating']
            diff = filtered_rating - all_rating
            
            comparison_data.append({
                'Player': player,
                'All Voters': round(all_rating, 2),
                'Filtered': round(filtered_rating, 2),
                'Difference': round(diff, 2)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Filtered', ascending=False)
        print(comparison_df.to_string(index=False))


def main():
    """Main function"""
    print("=" * 80)
    print("SOCCER PLAYER ANALYSIS SYSTEM")
    print("Detailed 6-Axis Rating & Voter Reliability Analysis")
    print("=" * 80)
    
    analyzer = SoccerAnalysis('upk_halisaha.csv')
    analyzer.run_complete_analysis()
    
    # Calculate filtered ratings (excluding harsh and low reliability voters)
    analyzer.calculate_filtered_ratings()
    
    # Create visualizations
    try:
        from visualizer import SoccerVisualizer
        visualizer = SoccerVisualizer(analyzer)
        visualizer.create_all_visualizations()
    except ImportError as e:
        print(f"\n⚠️  Could not import visualizer: {e}")
    except Exception as e:
        print(f"\n⚠️  Error creating visualizations: {e}")
    
    # Generate text summary report
    try:
        from report_generator import generate_text_report
        generate_text_report(analyzer)
    except ImportError as e:
        print(f"\n⚠️  Could not import report generator: {e}")
    except Exception as e:
        print(f"\n⚠️  Error generating text report: {e}")


if __name__ == '__main__':
    main()
