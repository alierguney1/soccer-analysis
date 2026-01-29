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
        self.reliable_100_player_ratings = {}
        self.top8_player_ratings = {}
        
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
            if self.df is None:
                continue
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
            if self.df is None:
                continue
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
                    'missing_count': (len(self.df) if self.df is not None else 0) - len(all_scores)
                }
                
                # Variance analysis
                analysis['bias_indicators']['high_variance'] = analysis['statistics']['std'] > self.HIGH_VARIANCE_THRESHOLD
                analysis['bias_indicators']['low_variance'] = analysis['statistics']['std'] < self.LOW_VARIANCE_THRESHOLD
                
                # Mean deviation from overall mean
                all_voters_scores = []
                if self.df is not None:
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
        
        if self.df is None:
            return
        
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
        
        if self.df is None:
            return
        
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
    
    
    def advanced_voter_bias_analysis(self):
        """
        Advanced voter bias analysis using statistical techniques
        Note: Voter numbers are randomly assigned and do not correspond to player identities
        """
        print("\n" + "=" * 80)
        print("ADVANCED VOTER BIAS ANALYSIS")
        print("=" * 80)
        print("NOTE: Voter numbers are randomly assigned for anonymity")
        print("=" * 80)
        
        # 1. Voter Correlation Analysis
        self._analyze_voter_correlations()
        
        # 2. Voting Pattern Clustering
        self._analyze_voting_patterns()
        
        # 3. Statistical Significance Tests
        self._statistical_significance_tests()
        
        # 4. Bias Type Classification
        self._classify_bias_types()
    
    def _analyze_voter_correlations(self):
        """Analyze correlations between voters to detect similar voting patterns"""
        print("\n--- VOTER CORRELATION ANALYSIS ---")
        print("Identifying voters with similar rating patterns...")
        
        if self.df is None:
            return
        
        # Create a matrix of voter scores
        voter_scores = {}
        for voter in self.voters:
            scores = []
            for _, row in self.df.iterrows():
                try:
                    score = row[voter]
                    if pd.notna(score) and str(score).strip():
                        scores.append(float(score))
                    else:
                        scores.append(np.nan)
                except:
                    scores.append(np.nan)
            voter_scores[voter] = scores
        
        # Calculate correlation matrix
        voter_df = pd.DataFrame(voter_scores)
        correlation_matrix = voter_df.corr()
        
        # Store correlation data
        self.voter_correlations = correlation_matrix
        
        # Find highly correlated voter pairs (may indicate collusion or similar perspectives)
        print("\nHighly Correlated Voters (correlation > 0.8):")
        found_correlations = False
        for i in range(len(self.voters)):
            for j in range(i+1, len(self.voters)):
                corr = correlation_matrix.iloc[i, j]
                if corr > 0.8 and not np.isnan(corr):
                    voter1_num = self.voter_analysis[self.voters[i]]['voter_number']
                    voter2_num = self.voter_analysis[self.voters[j]]['voter_number']
                    print(f"  - Voter {voter1_num} & Voter {voter2_num}: {corr:.3f}")
                    found_correlations = True
        
        if not found_correlations:
            print("  No highly correlated voter pairs found (good - indicates independent judgments)")
        
        # Find negatively correlated voters (opposite rating tendencies)
        print("\nNegatively Correlated Voters (correlation < -0.5):")
        found_neg_correlations = False
        for i in range(len(self.voters)):
            for j in range(i+1, len(self.voters)):
                corr = correlation_matrix.iloc[i, j]
                if corr < -0.5 and not np.isnan(corr):
                    voter1_num = self.voter_analysis[self.voters[i]]['voter_number']
                    voter2_num = self.voter_analysis[self.voters[j]]['voter_number']
                    print(f"  - Voter {voter1_num} & Voter {voter2_num}: {corr:.3f}")
                    found_neg_correlations = True
        
        if not found_neg_correlations:
            print("  No strongly negatively correlated voters found")
    
    def _analyze_voting_patterns(self):
        """Cluster voters by their voting patterns"""
        print("\n--- VOTING PATTERN ANALYSIS ---")
        print("Analyzing systematic biases in voting behavior...")
        
        # Categorize voters by their bias patterns
        strict_voters = []
        lenient_voters = []
        balanced_voters = []
        inconsistent_voters = []
        
        for voter, data in self.voter_analysis.items():
            voter_num = data['voter_number']
            mean = data['statistics']['mean']
            std = data['statistics']['std']
            
            # Classification logic
            if data['bias_indicators'].get('too_harsh', False):
                strict_voters.append(voter_num)
            elif data['bias_indicators'].get('too_generous', False):
                lenient_voters.append(voter_num)
            elif data['bias_indicators'].get('high_variance', False):
                inconsistent_voters.append(voter_num)
            else:
                balanced_voters.append(voter_num)
        
        print(f"\nVoter Pattern Distribution:")
        print(f"  - Strict/Harsh voters: {len(strict_voters)} - {strict_voters if strict_voters else 'None'}")
        print(f"  - Lenient/Generous voters: {len(lenient_voters)} - {lenient_voters if lenient_voters else 'None'}")
        print(f"  - Inconsistent voters: {len(inconsistent_voters)} - {inconsistent_voters if inconsistent_voters else 'None'}")
        print(f"  - Balanced voters: {len(balanced_voters)} - {balanced_voters if balanced_voters else 'None'}")
        
        # Store pattern classifications
        self.voter_patterns = {
            'strict': strict_voters,
            'lenient': lenient_voters,
            'inconsistent': inconsistent_voters,
            'balanced': balanced_voters
        }
    
    def _statistical_significance_tests(self):
        """Perform statistical tests to validate bias indicators"""
        print("\n--- STATISTICAL SIGNIFICANCE TESTS ---")
        print("Testing if voter biases are statistically significant...")
        
        if self.df is None:
            return
        
        # Collect all voter means
        voter_means = []
        for voter, data in self.voter_analysis.items():
            if data['statistics']['count'] > 0:
                voter_means.append(data['statistics']['mean'])
        
        overall_mean = np.mean(voter_means)
        overall_std = np.std(voter_means)
        
        print(f"\nOverall Statistics:")
        print(f"  - Mean rating across all voters: {overall_mean:.2f}")
        print(f"  - Std deviation of voter means: {overall_std:.2f}")
        
        # Test each voter for significant deviation
        print("\nVoters with Statistically Significant Biases (p < 0.05):")
        significant_biases = []
        
        for voter, data in self.voter_analysis.items():
            if data['statistics']['count'] < 3:  # Need minimum sample size
                continue
            
            voter_num = data['voter_number']
            voter_mean = data['statistics']['mean']
            voter_std = data['statistics']['std']
            n = data['statistics']['count']
            
            # One-sample t-test against overall mean
            # Calculate t-statistic
            if voter_std > 0:
                t_stat = (voter_mean - overall_mean) / (voter_std / np.sqrt(n))
                # Two-tailed p-value (approximate)
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
                
                if p_value < 0.05:
                    bias_type = "harsh" if voter_mean < overall_mean else "generous"
                    significant_biases.append({
                        'voter_num': voter_num,
                        'mean': voter_mean,
                        'bias_type': bias_type,
                        'p_value': p_value
                    })
        
        if significant_biases:
            for bias in sorted(significant_biases, key=lambda x: x['p_value']):
                print(f"  - Voter {bias['voter_num']}: {bias['bias_type'].upper()} "
                      f"(mean={bias['mean']:.2f}, p={bias['p_value']:.4f})")
        else:
            print("  No statistically significant biases detected")
        
        self.significant_biases = significant_biases
    
    def _classify_bias_types(self):
        """Classify and summarize different types of biases"""
        print("\n--- BIAS TYPE CLASSIFICATION ---")
        print("Categorizing voters by dominant bias characteristics...")
        
        bias_summary = {
            'Scale Compression (Low Variance)': [],
            'Scale Expansion (High Variance)': [],
            'Systematic Overrating (Generous)': [],
            'Systematic Underrating (Harsh)': [],
            'Inconsistent Rating (High Std in Similar Skills)': [],
            'Missing Data (Incomplete Ratings)': [],
            'Well-Calibrated (Minimal Bias)': []
        }
        
        for voter, data in self.voter_analysis.items():
            voter_num = data['voter_number']
            biases = data['bias_indicators']
            
            # Classify by dominant bias
            if biases.get('low_variance', False):
                bias_summary['Scale Compression (Low Variance)'].append(voter_num)
            elif biases.get('high_variance', False):
                bias_summary['Scale Expansion (High Variance)'].append(voter_num)
            elif biases.get('too_generous', False):
                bias_summary['Systematic Overrating (Generous)'].append(voter_num)
            elif biases.get('too_harsh', False):
                bias_summary['Systematic Underrating (Harsh)'].append(voter_num)
            elif data['statistics'].get('missing_count', 0) > len(self.df) * 0.3:
                bias_summary['Missing Data (Incomplete Ratings)'].append(voter_num)
            else:
                bias_summary['Well-Calibrated (Minimal Bias)'].append(voter_num)
        
        print("\nBias Type Summary:")
        for bias_type, voters in bias_summary.items():
            if voters:
                print(f"  - {bias_type}: {len(voters)} voter(s) - {voters}")
        
        self.bias_type_summary = bias_summary
        
        # Overall assessment
        total_voters = len(self.voter_analysis)
        well_calibrated = len(bias_summary['Well-Calibrated (Minimal Bias)'])
        
        print(f"\n✓ Overall Bias Assessment:")
        print(f"  - {well_calibrated}/{total_voters} ({100*well_calibrated/total_voters:.1f}%) voters are well-calibrated")
        print(f"  - {total_voters - well_calibrated}/{total_voters} ({100*(total_voters-well_calibrated)/total_voters:.1f}%) voters show some form of bias")
    
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
        self.advanced_voter_bias_analysis()  # New: Advanced bias analysis
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
        
        if self.df is None:
            return
        
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
    
    def get_100_percent_reliable_voters(self):
        """Get list of 100% reliable voters only"""
        reliable_100 = []
        excluded = []
        
        for voter, data in self.voter_analysis.items():
            reliability = data['reliability_score']
            
            if reliability == 100:
                reliable_100.append(voter)
            else:
                excluded.append({
                    'voter': voter,
                    'voter_num': data['voter_number'],
                    'reliability': reliability,
                    'reason': f'reliability {reliability}% < 100%'
                })
        
        return reliable_100, excluded
    
    def calculate_100_percent_reliable_ratings(self):
        """Calculate player ratings using only 100% reliable voters"""
        print("\n" + "=" * 80)
        print("100% RELIABLE DATA ANALYSIS")
        print("=" * 80)
        
        reliable_voters, excluded = self.get_100_percent_reliable_voters()
        
        print(f"\n📊 Filtering Criteria:")
        print(f"   - Only 100% reliability voters included")
        print(f"\n✓ Included voters: {len(reliable_voters)}")
        print(f"✗ Excluded voters: {len(excluded)}")
        
        if excluded:
            print("\nExcluded voters:")
            for ex in excluded:
                print(f"   - Voter {ex['voter_num']}: {ex['reason']}")
        
        self.reliable_100_player_ratings = {}
        
        if self.df is None:
            return
        
        for player in self.players:
            player_data = self.df[self.df['Name'] == player]
            
            ratings = {
                'player_name': player,
                'axis_ratings': {},
                'overall_rating': 0,
                'skill_details': {}
            }
            
            # Calculate each axis using only 100% reliable voters
            for axis_name, skills in self.skill_categories.items():
                axis_scores = []
                
                for skill in skills:
                    skill_data = player_data[player_data['Skill'] == skill]
                    if not skill_data.empty:
                        scores = []
                        for voter in reliable_voters:
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
            
            self.reliable_100_player_ratings[player] = ratings
        
        # Display comparison
        self._display_100_reliable_comparison()
        
        return self.reliable_100_player_ratings
    
    def _display_100_reliable_comparison(self):
        """Display comparison between all voters and 100% reliable voters"""
        print("\n" + "-" * 80)
        print("COMPARISON: ALL VOTERS vs 100% RELIABLE VOTERS")
        print("-" * 80)
        
        comparison_data = []
        for player in self.players:
            all_rating = self.player_ratings[player]['overall_rating']
            reliable_rating = self.reliable_100_player_ratings[player]['overall_rating']
            diff = reliable_rating - all_rating
            
            comparison_data.append({
                'Player': player,
                'All Voters': round(all_rating, 2),
                '100% Reliable': round(reliable_rating, 2),
                'Difference': round(diff, 2)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('100% Reliable', ascending=False)
        print(comparison_df.to_string(index=False))
    
    def calculate_top8_features_rating(self):
        """Calculate rating based on average of top 8 rated features for each player"""
        print("\n" + "=" * 80)
        print("TOP 8 FEATURES RATING ANALYSIS")
        print("=" * 80)
        
        self.top8_player_ratings = {}
        
        for player in self.players:
            player_data = self.player_ratings[player]
            
            # Collect all individual skill ratings
            all_skill_ratings = []
            for skill, details in player_data['skill_details'].items():
                all_skill_ratings.append({
                    'skill': skill,
                    'rating': details['mean']
                })
            
            # Sort by rating descending and take top 8
            all_skill_ratings.sort(key=lambda x: x['rating'], reverse=True)
            top_8_skills = all_skill_ratings[:8]
            
            # Calculate average of top 8
            if top_8_skills:
                top8_avg = np.mean([s['rating'] for s in top_8_skills])
            else:
                top8_avg = 0
            
            self.top8_player_ratings[player] = {
                'player_name': player,
                'top8_rating': top8_avg,
                'top8_skills': top_8_skills,
                'num_skills': len(all_skill_ratings)
            }
        
        # Display results
        self._display_top8_ratings()
        
        return self.top8_player_ratings
    
    def _display_top8_ratings(self):
        """Display top 8 features rating results"""
        print("\n" + "-" * 80)
        print("TOP 8 FEATURES RATING SUMMARY")
        print("-" * 80)
        
        # Sort by top8 rating
        sorted_players = sorted(self.top8_player_ratings.items(), 
                               key=lambda x: x[1]['top8_rating'], 
                               reverse=True)
        
        comparison_data = []
        for player, data in sorted_players:
            overall_rating = self.player_ratings[player]['overall_rating']
            top8_rating = data['top8_rating']
            diff = top8_rating - overall_rating
            
            comparison_data.append({
                'Player': player,
                'Overall Rating': round(overall_rating, 2),
                'Top 8 Avg': round(top8_rating, 2),
                'Difference': round(diff, 2)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Show top 5 with their best skills
        print("\n" + "-" * 80)
        print("TOP 5 PLAYERS - BEST SKILLS")
        print("-" * 80)
        
        for i, (player, data) in enumerate(sorted_players[:5], 1):
            print(f"\n{i}. {player} - Top 8 Avg: {data['top8_rating']:.2f}")
            print(f"   Best 8 Skills:")
            for j, skill_info in enumerate(data['top8_skills'], 1):
                skill_name = skill_info['skill'].replace('Technical >> ', '').replace('Tactical/Psychological/Physical >> ', '')
                print(f"      {j}. {skill_name}: {skill_info['rating']:.2f}")
    
    def analyze_voter_player_bias(self):
        """Deep analysis of voter bias towards specific players"""
        print("\n" + "=" * 80)
        print("DEEP VOTER-PLAYER BIAS ANALYSIS")
        print("=" * 80)
        
        # For each voter, analyze if they have systematic bias towards specific players
        voter_player_bias = {}
        
        for voter in self.voters:
            voter_num = int(voter.split()[-1])
            voter_bias = {
                'voter_number': voter_num,
                'player_deviations': {},
                'most_favorable': [],
                'most_critical': []
            }
            
            # Calculate deviation from mean for each player
            all_deviations = []
            for player in self.players:
                player_data = self.df[self.df['Name'] == player]
                voter_scores = []
                all_voter_scores = []
                
                for skill in player_data['Skill'].unique():
                    skill_row = player_data[player_data['Skill'] == skill]
                    if not skill_row.empty:
                        # This voter's score
                        try:
                            voter_score = skill_row[voter].values[0]
                            if pd.notna(voter_score) and str(voter_score).strip():
                                voter_scores.append(float(voter_score))
                        except:
                            pass
                        
                        # All voters' scores for comparison
                        for v in self.voters:
                            try:
                                v_score = skill_row[v].values[0]
                                if pd.notna(v_score) and str(v_score).strip():
                                    all_voter_scores.append(float(v_score))
                            except:
                                pass
                
                if voter_scores and all_voter_scores:
                    voter_mean = np.mean(voter_scores)
                    overall_mean = np.mean(all_voter_scores)
                    deviation = voter_mean - overall_mean
                    
                    voter_bias['player_deviations'][player] = {
                        'voter_mean': voter_mean,
                        'overall_mean': overall_mean,
                        'deviation': deviation,
                        'num_ratings': len(voter_scores)
                    }
                    all_deviations.append((player, deviation))
            
            # Sort to find most favorable and most critical
            all_deviations.sort(key=lambda x: x[1], reverse=True)
            voter_bias['most_favorable'] = all_deviations[:3]  # Top 3 positive deviations
            voter_bias['most_critical'] = all_deviations[-3:]  # Top 3 negative deviations
            
            voter_player_bias[voter] = voter_bias
        
        self.voter_player_bias = voter_player_bias
        
        # Display analysis
        self._display_voter_player_bias()
        
        return voter_player_bias
    
    def _display_voter_player_bias(self):
        """Display voter-player bias analysis"""
        print("\n" + "-" * 80)
        print("VOTER BIAS TOWARDS SPECIFIC PLAYERS")
        print("-" * 80)
        
        # Only show voters with significant bias
        for voter, bias_data in self.voter_player_bias.items():
            voter_num = bias_data['voter_number']
            
            # Check if there's significant bias (deviation > 1.0)
            has_significant_bias = False
            for player, dev in bias_data['most_favorable']:
                if abs(dev) > 1.0:
                    has_significant_bias = True
                    break
            for player, dev in bias_data['most_critical']:
                if abs(dev) > 1.0:
                    has_significant_bias = True
                    break
            
            if has_significant_bias:
                print(f"\nVoter {voter_num}:")
                
                # Show most favorable
                print("  Most Favorable Players:")
                for player, deviation in bias_data['most_favorable']:
                    if deviation > 0.5:  # Only show significant positive bias
                        details = bias_data['player_deviations'][player]
                        print(f"    • {player}: {deviation:+.2f} (voter avg: {details['voter_mean']:.2f} vs overall: {details['overall_mean']:.2f})")
                
                # Show most critical
                print("  Most Critical Towards:")
                for player, deviation in bias_data['most_critical']:
                    if deviation < -0.5:  # Only show significant negative bias
                        details = bias_data['player_deviations'][player]
                        print(f"    • {player}: {deviation:+.2f} (voter avg: {details['voter_mean']:.2f} vs overall: {details['overall_mean']:.2f})")


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
    
    # Calculate 100% reliable voter ratings
    analyzer.calculate_100_percent_reliable_ratings()
    
    # Calculate top 8 features rating
    analyzer.calculate_top8_features_rating()
    
    # Analyze voter-player bias
    analyzer.analyze_voter_player_bias()
    
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
