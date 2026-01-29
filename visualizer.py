#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Module for Soccer Analysis
Creates detailed charts and graphs for the analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class SoccerVisualizer:
    """Create visualizations for soccer analysis"""
    
    def __init__(self, analyzer):
        """Initialize with analyzer instance"""
        self.analyzer = analyzer
        self.output_dir = 'analysis_output'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_all_visualizations(self):
        """Create all visualizations"""
        print("\n" + "=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)
        
        # Create individual plots
        self.plot_player_rankings()
        self.plot_6axis_radar_all()  # For ALL players
        self.plot_voter_reliability()
        self.plot_player_heatmap()
        self.plot_voter_distributions()
        self.plot_skill_comparison_all()  # For ALL players
        
        # Advanced bias visualizations
        if hasattr(self.analyzer, 'voter_correlations'):
            self.plot_voter_correlation_heatmap()
        if hasattr(self.analyzer, 'voter_patterns'):
            self.plot_bias_pattern_distribution()
        
        # Create filtered analysis visualizations
        if hasattr(self.analyzer, 'filtered_player_ratings'):
            self.plot_filtered_comparison()
            self.plot_filtered_rankings()
            self.plot_filtered_radar_all()
        
        # Create 100% reliable analysis visualizations
        if hasattr(self.analyzer, 'reliable_100_player_ratings'):
            self.plot_100_reliable_comparison()
            self.plot_100_reliable_radar_all()
        
        # Create top 8 features visualizations
        if hasattr(self.analyzer, 'top8_player_ratings'):
            self.plot_top8_comparison()
        
        # Create voter-player bias visualizations
        if hasattr(self.analyzer, 'voter_player_bias'):
            self.plot_voter_player_bias_heatmap()
        
        # Create comprehensive PDF report
        self.create_pdf_report()
        
        print(f"\n✓ All visualizations saved to '{self.output_dir}/' directory")
    
    def plot_player_rankings(self):
        """Create player rankings bar chart"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Prepare data
        players = []
        ratings = []
        for player, data in self.analyzer.player_ratings.items():
            players.append(player)
            ratings.append(data['overall_rating'])
        
        # Sort by rating
        sorted_data = sorted(zip(players, ratings), key=lambda x: x[1], reverse=True)
        players, ratings = zip(*sorted_data)
        
        # Create color gradient
        colors = plt.colormaps.get_cmap('RdYlGn')(np.array(ratings) / 10)
        
        # Create bar chart
        bars = ax.barh(players, ratings, color=colors, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for i, (player, rating) in enumerate(zip(players, ratings)):
            ax.text(rating + 0.1, i, f'{rating:.2f}', va='center', fontweight='bold')
        
        ax.set_xlabel('Overall Rating (0-10)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Player', fontweight='bold', fontsize=12)
        ax.set_title('Player Overall Rankings - 6-Axis Rating System', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xlim(0, 10)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_player_rankings.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created player rankings chart")
    
    def plot_6axis_radar(self):
        """Create radar charts for top players (legacy - for backward compatibility)"""
        self.plot_6axis_radar_all()
    
    def plot_6axis_radar_all(self):
        """Create radar charts for ALL players"""
        players = list(self.analyzer.player_ratings.keys())
        num_players = len(players)
        
        # Calculate grid size
        cols = 4
        rows = (num_players + cols - 1) // cols
        
        # Create subplot grid
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows), subplot_kw=dict(projection='polar'))
        axes = axes.flatten() if num_players > 1 else [axes]
        
        # Axis categories
        categories = [
            'Technical\nBall Control',
            'Shooting &\nFinishing',
            'Offensive\nPlay',
            'Defensive\nPlay',
            'Tactical/\nPsychological',
            'Physical/\nCondition'
        ]
        
        axis_map = {
            'technical_ball_control': 0,
            'shooting_finishing': 1,
            'offensive_play': 2,
            'defensive_play': 3,
            'tactical_psychological': 4,
            'physical_condition': 5
        }
        
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        
        # Sort players by overall rating
        sorted_players = sorted(players, 
            key=lambda p: self.analyzer.player_ratings[p]['overall_rating'], 
            reverse=True)
        
        for idx, player in enumerate(sorted_players):
            ax = axes[idx]
            data = self.analyzer.player_ratings[player]
            
            # Get axis values
            values = [0] * N
            for axis_name, pos in axis_map.items():
                if axis_name in data['axis_ratings']:
                    values[pos] = data['axis_ratings'][axis_name]['score']
            
            values += values[:1]
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, label=player, color='#2E86AB')
            ax.fill(angles, values, alpha=0.25, color='#2E86AB')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=8)
            ax.set_ylim(0, 10)
            ax.set_yticks([2, 4, 6, 8, 10])
            ax.set_yticklabels(['2', '4', '6', '8', '10'], size=7)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{player}\nOverall: {data["overall_rating"]:.2f}', 
                        fontweight='bold', size=10, pad=10)
        
        # Hide empty subplots
        for idx in range(num_players, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('ALL Players - 6-Axis Radar Analysis', 
                    fontweight='bold', fontsize=16, y=1.01)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/02_6axis_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created 6-axis radar charts for ALL players")
    
    def plot_voter_reliability(self):
        """Create voter reliability chart"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data
        voters = []
        reliability = []
        for voter, data in self.analyzer.voter_analysis.items():
            voters.append(f"Voter {data['voter_number']}")
            reliability.append(data['reliability_score'])
        
        # Sort by reliability
        sorted_data = sorted(zip(voters, reliability), key=lambda x: x[1])
        voters, reliability = zip(*sorted_data)
        
        # Create color coding
        colors = ['#d32f2f' if r < 50 else '#f57c00' if r < 70 else '#388e3c' 
                 for r in reliability]
        
        # Create bar chart
        bars = ax.barh(voters, reliability, color=colors, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for i, (voter, rel) in enumerate(zip(voters, reliability)):
            ax.text(rel + 1, i, f'{rel:.0f}%', va='center', fontweight='bold')
        
        # Add reliability zones
        ax.axvline(70, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='70% Threshold')
        ax.axvline(50, color='red', linestyle='--', alpha=0.5, linewidth=2, label='50% Threshold')
        
        ax.set_xlabel('Reliability Score (%)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Voter', fontweight='bold', fontsize=12)
        ax.set_title('Voter Reliability Scores', fontweight='bold', fontsize=14, pad=20)
        ax.set_xlim(0, 110)
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/03_voter_reliability.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created voter reliability chart")
    
    # NOTE: plot_self_bias method removed because voter order is random
    # and we don't have reliable voter-to-player mapping data
    
    def plot_player_heatmap(self):
        """Create heatmap of player skills"""
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Prepare data matrix
        players = list(self.analyzer.player_ratings.keys())
        skills = ['Controlling', 'Dribbling', 'Passing', 'Shooting', 'Finishing',
                 'Offensive Play', 'Defensive Play', 'Disciplined', 'Decision making',
                 'Teamwork', 'Condition']
        
        skill_mapping = {
            'Controlling': 'Technical >> Controlling',
            'Dribbling': 'Technical >> Dribbling',
            'Passing': 'Technical >> Passing',
            'Shooting': 'Technical >> Shooting',
            'Finishing': 'Technical >> Finishing',
            'Offensive Play': 'Technical >> Offensive Play',
            'Defensive Play': 'Technical >> Defensive Play',
            'Disciplined': 'Tactical/Psychological/Physical >> Disciplined',
            'Decision making': 'Tactical/Psychological/Physical >> Decision making',
            'Teamwork': 'Tactical/Psychological/Physical >> Teamwork',
            'Condition': 'Tactical/Psychological/Physical >> Condition'
        }
        
        # Create matrix
        data_matrix = []
        for player in players:
            row = []
            for skill in skills:
                full_skill = skill_mapping[skill]
                if full_skill in self.analyzer.player_ratings[player]['skill_details']:
                    row.append(self.analyzer.player_ratings[player]['skill_details'][full_skill]['mean'])
                else:
                    row.append(0)
            data_matrix.append(row)
        
        df_heatmap = pd.DataFrame(data_matrix, index=players, columns=skills)
        
        # Sort by overall rating
        overall_ratings = {p: self.analyzer.player_ratings[p]['overall_rating'] for p in players}
        df_heatmap['_overall'] = df_heatmap.index.map(overall_ratings)
        df_heatmap = df_heatmap.sort_values('_overall', ascending=False)
        df_heatmap = df_heatmap.drop('_overall', axis=1)
        
        # Create heatmap
        sns.heatmap(df_heatmap, annot=True, fmt='.1f', cmap='RdYlGn', 
                   vmin=0, vmax=10, center=5, cbar_kws={'label': 'Rating (0-10)'},
                   linewidths=0.5, linecolor='gray', ax=ax)
        
        ax.set_title('Player Skills Heatmap - All Players & Skills', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xlabel('Skill Category', fontweight='bold', fontsize=12)
        ax.set_ylabel('Player (sorted by overall rating)', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/05_player_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created player skills heatmap")
    
    def plot_voter_distributions(self):
        """Create voter score distribution plots"""
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        axes = axes.flatten()
        
        for idx, voter in enumerate(self.analyzer.voters):
            ax = axes[idx]
            
            # Collect scores
            scores = []
            for _, row in self.analyzer.df.iterrows():
                try:
                    score = row[voter]
                    if pd.notna(score) and str(score).strip():
                        scores.append(float(score))
                except:
                    continue
            
            if scores:
                voter_num = idx + 1
                reliability = self.analyzer.voter_analysis[voter]['reliability_score']
                
                # Create histogram
                ax.hist(scores, bins=range(1, 12), edgecolor='black', 
                       color='skyblue', alpha=0.7)
                
                # Add mean line
                mean = np.mean(scores)
                ax.axvline(mean, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {mean:.2f}')
                
                ax.set_title(f'Voter {voter_num}\nReliability: {reliability}%', 
                           fontweight='bold', fontsize=10)
                ax.set_xlabel('Score', fontsize=9)
                ax.set_ylabel('Frequency', fontsize=9)
                ax.set_xlim(0, 11)
                ax.legend(fontsize=8)
                ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Voter Score Distributions', fontweight='bold', fontsize=16, y=0.995)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/06_voter_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created voter distribution charts")
    
    def plot_skill_comparison(self):
        """Create skill comparison for ALL players (legacy - calls new function)"""
        self.plot_skill_comparison_all()
    
    def plot_skill_comparison_all(self):
        """Create skill comparison across ALL players"""
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # Get ALL players sorted by overall rating
        summary_data = []
        for player, data in self.analyzer.player_ratings.items():
            summary_data.append({
                'Player': player,
                'Overall': data['overall_rating']
            })
        summary_df = pd.DataFrame(summary_data).sort_values('Overall', ascending=False)
        all_players = summary_df['Player'].tolist()
        
        # Axis categories
        categories = ['Technical\nBall Control', 'Shooting &\nFinishing', 
                     'Offensive\nPlay', 'Defensive\nPlay', 
                     'Tactical/\nPsychological', 'Physical/\nCondition']
        
        axis_map = {
            'technical_ball_control': 0,
            'shooting_finishing': 1,
            'offensive_play': 2,
            'defensive_play': 3,
            'tactical_psychological': 4,
            'physical_condition': 5
        }
        
        x = np.arange(len(categories))
        width = 0.8 / len(all_players)  # Adjust width based on number of players
        
        colors = plt.colormaps.get_cmap('tab20')(np.linspace(0, 1, len(all_players)))
        
        for i, player in enumerate(all_players):
            data = self.analyzer.player_ratings[player]
            values = [0] * len(categories)
            
            for axis_name, pos in axis_map.items():
                if axis_name in data['axis_ratings']:
                    values[pos] = data['axis_ratings'][axis_name]['score']
            
            offset = width * (i - len(all_players)/2)
            ax.bar(x + offset, values, width, label=player[:12], 
                  color=colors[i], edgecolor='black', linewidth=0.3)
        
        ax.set_xlabel('Skill Category', fontweight='bold', fontsize=12)
        ax.set_ylabel('Rating (0-10)', fontweight='bold', fontsize=12)
        ax.set_title('ALL Players - Skill Category Comparison', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, ncol=1)
        ax.set_ylim(0, 10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/04_skill_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created skill comparison chart for ALL players")
    
    def plot_filtered_comparison(self):
        """Create comparison chart between all voters and filtered voters"""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        players = list(self.analyzer.player_ratings.keys())
        
        # Sort by filtered rating
        sorted_players = sorted(players,
            key=lambda p: self.analyzer.filtered_player_ratings[p]['overall_rating'],
            reverse=True)
        
        x = np.arange(len(sorted_players))
        width = 0.35
        
        all_ratings = [self.analyzer.player_ratings[p]['overall_rating'] for p in sorted_players]
        filtered_ratings = [self.analyzer.filtered_player_ratings[p]['overall_rating'] for p in sorted_players]
        
        bars1 = ax.bar(x - width/2, all_ratings, width, label='All Voters', 
                      color='#64b5f6', edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, filtered_ratings, width, label='Filtered (No Harsh, ≥75% Reliability)', 
                      color='#81c784', edgecolor='black', linewidth=0.5)
        
        # Add difference annotations
        for i, (all_r, filt_r) in enumerate(zip(all_ratings, filtered_ratings)):
            diff = filt_r - all_r
            color = 'green' if diff > 0 else 'red' if diff < 0 else 'black'
            ax.annotate(f'{diff:+.2f}', xy=(i, max(all_r, filt_r) + 0.2),
                       ha='center', fontsize=8, color=color, fontweight='bold')
        
        ax.set_xlabel('Player', fontweight='bold', fontsize=12)
        ax.set_ylabel('Overall Rating (0-10)', fontweight='bold', fontsize=12)
        ax.set_title('Rating Comparison: All Voters vs Filtered Voters\n(Filtered = Excluding Too Harsh & Low Reliability <75%)', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_players, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper right')
        ax.set_ylim(0, 10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/07_filtered_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created filtered comparison chart")
    
    def plot_filtered_rankings(self):
        """Create filtered player rankings bar chart"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Prepare data
        players = []
        ratings = []
        for player, data in self.analyzer.filtered_player_ratings.items():
            players.append(player)
            ratings.append(data['overall_rating'])
        
        # Sort by rating
        sorted_data = sorted(zip(players, ratings), key=lambda x: x[1], reverse=True)
        players, ratings = zip(*sorted_data)
        
        # Create color gradient
        colors = plt.colormaps.get_cmap('RdYlGn')(np.array(ratings) / 10)
        
        # Create bar chart
        bars = ax.barh(players, ratings, color=colors, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for i, (player, rating) in enumerate(zip(players, ratings)):
            ax.text(rating + 0.1, i, f'{rating:.2f}', va='center', fontweight='bold')
        
        ax.set_xlabel('Overall Rating (0-10)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Player', fontweight='bold', fontsize=12)
        ax.set_title('FILTERED Player Rankings\n(Excluding Too Harsh & Low Reliability <75% Voters)', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xlim(0, 10)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/08_filtered_rankings.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created filtered player rankings chart")
    
    def plot_filtered_radar_all(self):
        """Create radar charts for ALL players using filtered data"""
        players = list(self.analyzer.filtered_player_ratings.keys())
        num_players = len(players)
        
        # Calculate grid size
        cols = 4
        rows = (num_players + cols - 1) // cols
        
        # Create subplot grid
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows), subplot_kw=dict(projection='polar'))
        axes = axes.flatten() if num_players > 1 else [axes]
        
        # Axis categories
        categories = [
            'Technical\nBall Control',
            'Shooting &\nFinishing',
            'Offensive\nPlay',
            'Defensive\nPlay',
            'Tactical/\nPsychological',
            'Physical/\nCondition'
        ]
        
        axis_map = {
            'technical_ball_control': 0,
            'shooting_finishing': 1,
            'offensive_play': 2,
            'defensive_play': 3,
            'tactical_psychological': 4,
            'physical_condition': 5
        }
        
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        
        # Sort players by overall rating
        sorted_players = sorted(players, 
            key=lambda p: self.analyzer.filtered_player_ratings[p]['overall_rating'], 
            reverse=True)
        
        for idx, player in enumerate(sorted_players):
            ax = axes[idx]
            data = self.analyzer.filtered_player_ratings[player]
            
            # Get axis values
            values = [0] * N
            for axis_name, pos in axis_map.items():
                if axis_name in data['axis_ratings']:
                    values[pos] = data['axis_ratings'][axis_name]['score']
            
            values += values[:1]
            
            # Plot with different color for filtered
            ax.plot(angles, values, 'o-', linewidth=2, label=player, color='#4CAF50')
            ax.fill(angles, values, alpha=0.25, color='#4CAF50')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=8)
            ax.set_ylim(0, 10)
            ax.set_yticks([2, 4, 6, 8, 10])
            ax.set_yticklabels(['2', '4', '6', '8', '10'], size=7)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{player}\nFiltered: {data["overall_rating"]:.2f}', 
                        fontweight='bold', size=10, pad=10)
        
        # Hide empty subplots
        for idx in range(num_players, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('ALL Players - FILTERED 6-Axis Radar Analysis\n(Excluding Too Harsh & Low Reliability <75% Voters)', 
                    fontweight='bold', fontsize=16, y=1.01)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/09_filtered_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created FILTERED 6-axis radar charts for ALL players")
    
    def plot_100_reliable_comparison(self):
        """Create comparison chart for 100% reliable voters"""
        if not hasattr(self.analyzer, 'reliable_100_player_ratings'):
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        players = []
        all_voters = []
        reliable_100 = []
        
        for player in self.analyzer.players:
            all_rating = self.analyzer.player_ratings[player]['overall_rating']
            reliable_rating = self.analyzer.reliable_100_player_ratings[player]['overall_rating']
            
            players.append(player)
            all_voters.append(all_rating)
            reliable_100.append(reliable_rating)
        
        # Sort by 100% reliable rating
        sorted_indices = np.argsort(reliable_100)[::-1]
        players = [players[i] for i in sorted_indices]
        all_voters = [all_voters[i] for i in sorted_indices]
        reliable_100 = [reliable_100[i] for i in sorted_indices]
        
        x = np.arange(len(players))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, all_voters, width, label='All Voters', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x + width/2, reliable_100, width, label='100% Reliable', alpha=0.8, color='gold')
        
        ax.set_xlabel('Players', fontweight='bold', fontsize=12)
        ax.set_ylabel('Rating', fontweight='bold', fontsize=12)
        ax.set_title('Player Ratings: All Voters vs 100% Reliable Voters', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(players, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/12_100reliable_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created 100% reliable comparison chart")
    
    def plot_100_reliable_radar_all(self):
        """Create radar charts for all players using 100% reliable data"""
        if not hasattr(self.analyzer, 'reliable_100_player_ratings'):
            return
        
        # Create grid for all players
        n_players = len(self.analyzer.players)
        cols = 4
        rows = (n_players + cols - 1) // cols
        
        fig = plt.figure(figsize=(20, 5 * rows))
        
        # Axis labels
        axis_labels = [
            'Technical\nBall Control',
            'Shooting\nFinishing',
            'Offensive\nPlay',
            'Defensive\nPlay',
            'Tactical\nPsychological',
            'Physical\nCondition'
        ]
        
        axis_keys = [
            'technical_ball_control',
            'shooting_finishing',
            'offensive_play',
            'defensive_play',
            'tactical_psychological',
            'physical_condition'
        ]
        
        # Sort players by their 100% reliable rating
        sorted_players = sorted(self.analyzer.reliable_100_player_ratings.items(),
                               key=lambda x: x[1]['overall_rating'],
                               reverse=True)
        
        for idx, (player, data) in enumerate(sorted_players, 1):
            ax = fig.add_subplot(rows, cols, idx, projection='polar')
            
            # Get values for this player
            values = []
            for key in axis_keys:
                if key in data['axis_ratings']:
                    values.append(data['axis_ratings'][key]['score'])
                else:
                    values.append(0)
            
            # Complete the circle
            values += values[:1]
            
            # Angles
            angles = np.linspace(0, 2 * np.pi, len(axis_labels), endpoint=False).tolist()
            angles += angles[:1]
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, label=player, color='gold')
            ax.fill(angles, values, alpha=0.25, color='gold')
            ax.set_ylim(0, 10)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(axis_labels, size=8)
            ax.set_yticks([2, 4, 6, 8, 10])
            ax.grid(True)
            
            # Title
            overall = data['overall_rating']
            ax.set_title(f"{player}\n(100% Reliable: {overall:.2f})", 
                        fontweight='bold', size=10, pad=20)
        
        plt.suptitle('ALL Players - 100% RELIABLE Voters 6-Axis Radar Analysis', 
                    fontweight='bold', fontsize=16, y=1.01)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/13_100reliable_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created 100% reliable 6-axis radar charts for ALL players")
    
    def plot_top8_comparison(self):
        """Create comparison chart for top 8 features rating"""
        if not hasattr(self.analyzer, 'top8_player_ratings'):
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        players = []
        overall_ratings = []
        top8_ratings = []
        
        # Sort by top 8 rating
        sorted_players = sorted(self.analyzer.top8_player_ratings.items(),
                               key=lambda x: x[1]['top8_rating'],
                               reverse=True)
        
        for player, data in sorted_players:
            overall_rating = self.analyzer.player_ratings[player]['overall_rating']
            
            players.append(player)
            overall_ratings.append(overall_rating)
            top8_ratings.append(data['top8_rating'])
        
        x = np.arange(len(players))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, overall_ratings, width, label='Overall Rating', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x + width/2, top8_ratings, width, label='Top 8 Features Avg', alpha=0.8, color='coral')
        
        ax.set_xlabel('Players', fontweight='bold', fontsize=12)
        ax.set_ylabel('Rating', fontweight='bold', fontsize=12)
        ax.set_title('Player Ratings: Overall vs Top 8 Features Average', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(players, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/14_top8_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created top 8 features comparison chart")
    
    def plot_voter_player_bias_heatmap(self):
        """Create heatmap showing voter bias towards specific players"""
        if not hasattr(self.analyzer, 'voter_player_bias'):
            return
        
        # Create matrix of deviations
        voters = []
        players = self.analyzer.players
        deviations = []
        
        for voter, bias_data in sorted(self.analyzer.voter_player_bias.items(), 
                                      key=lambda x: x[1]['voter_number']):
            voters.append(f"Voter {bias_data['voter_number']}")
            voter_devs = []
            for player in players:
                if player in bias_data['player_deviations']:
                    voter_devs.append(bias_data['player_deviations'][player]['deviation'])
                else:
                    voter_devs.append(0)
            deviations.append(voter_devs)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, 10))
        
        im = ax.imshow(deviations, cmap='RdYlGn', aspect='auto', vmin=-2, vmax=2)
        
        # Set ticks
        ax.set_xticks(np.arange(len(players)))
        ax.set_yticks(np.arange(len(voters)))
        ax.set_xticklabels(players, rotation=45, ha='right')
        ax.set_yticklabels(voters)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Deviation from Overall Mean', rotation=270, labelpad=20)
        
        # Add values to cells
        for i in range(len(voters)):
            for j in range(len(players)):
                text = ax.text(j, i, f'{deviations[i][j]:.1f}',
                             ha="center", va="center", color="black", fontsize=7)
        
        ax.set_title('Voter Bias Towards Specific Players\n(Positive = More Favorable, Negative = More Critical)', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xlabel('Players', fontweight='bold', fontsize=12)
        ax.set_ylabel('Voters', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/15_voter_player_bias.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created voter-player bias heatmap")
    
    def create_pdf_report(self):
        """Create comprehensive PDF report with high quality"""
        pdf_path = f'{self.output_dir}/Soccer_Analysis_Report.pdf'
        
        with PdfPages(pdf_path) as pdf:
            # Set PDF metadata
            d = pdf.infodict()
            d['Title'] = 'Soccer Analysis Report - 6-Axis Rating System'
            d['Author'] = 'Soccer Analysis System'
            d['Subject'] = 'Player Ratings and Voter Reliability Analysis'
            
            # Page 1: Player Rankings (All Voters)
            self._create_pdf_player_rankings(pdf)
            
            # Page 2: Radar Charts (All Voters)
            self._create_pdf_radar_charts(pdf)
            
            # Page 3: Voter Reliability
            self._create_pdf_voter_reliability(pdf)
            
            # Page 4: Skill Comparison
            self._create_pdf_skill_comparison(pdf)
            
            # Page 5: Heatmap
            self._create_pdf_heatmap(pdf)
            
            # Page 6: Distributions
            self._create_pdf_distributions(pdf)
            
            # Advanced bias analysis pages
            if hasattr(self.analyzer, 'voter_correlations'):
                self._create_pdf_voter_correlation(pdf)
            if hasattr(self.analyzer, 'bias_type_summary'):
                self._create_pdf_bias_patterns(pdf)
            
            # Page 7-9: Filtered results if available
            if hasattr(self.analyzer, 'filtered_player_ratings'):
                self._create_pdf_filtered_comparison(pdf)
                self._create_pdf_filtered_rankings(pdf)
                self._create_pdf_filtered_radar(pdf)
            
            # NEW: Top 8 features rating analysis
            if hasattr(self.analyzer, 'top8_player_ratings'):
                self._create_pdf_top8_comparison(pdf)
            
            # NEW: Voter-player bias analysis
            if hasattr(self.analyzer, 'voter_player_bias'):
                self._create_pdf_voter_player_bias(pdf)
            
            # NEW: 100% reliable voter analysis (at the end as requested)
            if hasattr(self.analyzer, 'reliable_100_player_ratings'):
                self._create_pdf_100reliable_comparison(pdf)
                self._create_pdf_100reliable_radar(pdf)
        
        print(f"\n✓ High quality PDF report created: {pdf_path}")
    
    def _create_pdf_player_rankings(self, pdf):
        """Create player rankings page for PDF"""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        
        players = []
        ratings = []
        for player, data in self.analyzer.player_ratings.items():
            players.append(player)
            ratings.append(data['overall_rating'])
        
        sorted_data = sorted(zip(players, ratings), key=lambda x: x[1], reverse=True)
        players, ratings = zip(*sorted_data)
        
        colors = plt.colormaps.get_cmap('RdYlGn')(np.array(ratings) / 10)
        bars = ax.barh(players, ratings, color=colors, edgecolor='black', linewidth=0.5)
        
        for i, (player, rating) in enumerate(zip(players, ratings)):
            ax.text(rating + 0.1, i, f'{rating:.2f}', va='center', fontweight='bold')
        
        ax.set_xlabel('Overall Rating (0-10)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Player', fontweight='bold', fontsize=12)
        ax.set_title('Player Overall Rankings - 6-Axis Rating System', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xlim(0, 10)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_pdf_radar_charts(self, pdf):
        """Create radar charts page for PDF"""
        players = list(self.analyzer.player_ratings.keys())
        num_players = len(players)
        
        cols = 3
        rows = (num_players + cols - 1) // cols
        
        fig = plt.figure(figsize=(15, rows * 4))
        
        categories = ['Technical\nBall Control', 'Shooting &\nFinishing', 
                     'Offensive\nPlay', 'Defensive\nPlay',
                     'Tactical/\nPsychological', 'Physical/\nCondition']
        
        for idx, player in enumerate(players):
            ax = fig.add_subplot(rows, cols, idx + 1, projection='polar')
            data = self.analyzer.player_ratings[player]
            
            values = [
                data['axis_ratings']['technical_ball_control']['score'],
                data['axis_ratings']['shooting_finishing']['score'],
                data['axis_ratings']['offensive_play']['score'],
                data['axis_ratings']['defensive_play']['score'],
                data['axis_ratings']['tactical_psychological']['score'],
                data['axis_ratings']['physical_condition']['score']
            ]
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=player)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=8)
            ax.set_ylim(0, 10)
            ax.set_title(f'{player}\n({data["overall_rating"]:.2f}/10)', 
                        fontweight='bold', size=10, pad=20)
            ax.grid(True)
        
        plt.suptitle('6-Axis Skill Radar Charts - All Players', 
                    fontweight='bold', fontsize=16, y=0.995)
        plt.tight_layout()
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_pdf_voter_reliability(self, pdf):
        """Create voter reliability page for PDF"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        voters = []
        reliability_scores = []
        for voter_id, analysis in self.analyzer.voter_analysis.items():
            voters.append(f"V{analysis['voter_number']}")
            reliability_scores.append(analysis['reliability_score'])
        
        sorted_data = sorted(zip(voters, reliability_scores), key=lambda x: x[1], reverse=True)
        voters, reliability_scores = zip(*sorted_data)
        
        colors = ['green' if score >= 75 else 'orange' if score >= 50 else 'red' 
                 for score in reliability_scores]
        
        bars = ax1.barh(voters, reliability_scores, color=colors, edgecolor='black', linewidth=0.5)
        
        for i, (voter, score) in enumerate(zip(voters, reliability_scores)):
            ax1.text(score + 1, i, f'{score:.0f}%', va='center', fontweight='bold')
        
        ax1.axvline(x=75, color='red', linestyle='--', linewidth=2, label='Reliability Threshold (75%)')
        ax1.set_xlabel('Reliability Score (%)', fontweight='bold')
        ax1.set_ylabel('Voter', fontweight='bold')
        ax1.set_title('Voter Reliability Scores', fontweight='bold', pad=10)
        ax1.set_xlim(0, 105)
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)
        
        # Reliability distribution
        bins = [0, 50, 75, 90, 100]
        hist, _ = np.histogram(reliability_scores, bins=bins)
        bin_labels = ['<50%', '50-75%', '75-90%', '90-100%']
        colors2 = ['red', 'orange', 'lightgreen', 'green']
        
        ax2.bar(bin_labels, hist, color=colors2, edgecolor='black', linewidth=1)
        ax2.set_xlabel('Reliability Range', fontweight='bold')
        ax2.set_ylabel('Number of Voters', fontweight='bold')
        ax2.set_title('Reliability Score Distribution', fontweight='bold', pad=10)
        ax2.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(hist):
            ax2.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_pdf_skill_comparison(self, pdf):
        """Create skill comparison page for PDF"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        all_players = list(self.analyzer.player_ratings.keys())
        
        categories = {
            'Technical\nBall Control': 'technical_ball_control',
            'Shooting &\nFinishing': 'shooting_finishing',
            'Offensive\nPlay': 'offensive_play',
            'Defensive\nPlay': 'defensive_play',
            'Tactical/\nPsych': 'tactical_psychological',
            'Physical/\nCondition': 'physical_condition'
        }
        
        x = np.arange(len(categories))
        width = 0.8 / len(all_players)
        
        colors = plt.colormaps.get_cmap('tab20')(np.linspace(0, 1, len(all_players)))
        
        for i, player in enumerate(all_players):
            data = self.analyzer.player_ratings[player]
            values = [0] * len(categories)
            
            for j, (cat_name, cat_key) in enumerate(categories.items()):
                if cat_key in data['axis_ratings']:
                    values[j] = data['axis_ratings'][cat_key]['score']
            
            offset = (i - len(all_players)/2) * width
            ax.bar(x + offset, values, width, label=player, color=colors[i], 
                  edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Skill Category', fontweight='bold', fontsize=12)
        ax.set_ylabel('Rating (0-10)', fontweight='bold', fontsize=12)
        ax.set_title('Skill Comparison Across All Players', fontweight='bold', fontsize=14, pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(categories.keys())
        ax.set_ylim(0, 10)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_pdf_heatmap(self, pdf):
        """Create player heatmap page for PDF"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        skill_names = []
        for cat_key, skills in self.analyzer.skill_categories.items():
            skill_names.extend(skills)
        
        players = list(self.analyzer.player_ratings.keys())
        heatmap_data = []
        
        for player in players:
            player_row = []
            for skill in skill_names:
                if self.analyzer.df is not None:
                    skill_data = self.analyzer.df[
                        (self.analyzer.df['Name'] == player) & 
                        (self.analyzer.df['Skill'] == skill)
                    ]
                    if not skill_data.empty:
                        scores = []
                        for voter in self.analyzer.voters:
                            try:
                                score = skill_data[voter].values[0]
                                if pd.notna(score):
                                    scores.append(float(score))
                            except:
                                continue
                        player_row.append(np.mean(scores) if scores else 0)
                    else:
                        player_row.append(0)
                else:
                    player_row.append(0)
            heatmap_data.append(player_row)
        
        heatmap_array = np.array(heatmap_data)
        
        im = ax.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=10)
        
        ax.set_xticks(np.arange(len(skill_names)))
        ax.set_yticks(np.arange(len(players)))
        ax.set_xticklabels([s.split(' >> ')[-1] for s in skill_names], rotation=45, ha='right')
        ax.set_yticklabels(players)
        
        plt.colorbar(im, ax=ax, label='Rating (0-10)')
        
        ax.set_title('Player Skills Heatmap', fontweight='bold', fontsize=14, pad=15)
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_pdf_distributions(self, pdf):
        """Create voter distributions page for PDF"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, (voter_id, analysis) in enumerate(list(self.analyzer.voter_analysis.items())[:9]):
            if idx >= 9:
                break
                
            ax = axes[idx]
            
            all_scores = []
            if self.analyzer.df is not None:
                for _, row in self.analyzer.df.iterrows():
                    try:
                        score = row[voter_id]
                        if pd.notna(score) and str(score).strip():
                            all_scores.append(float(score))
                    except:
                        continue
            
            if all_scores:
                ax.hist(all_scores, bins=range(1, 12), edgecolor='black', alpha=0.7)
                ax.axvline(analysis['statistics']['mean'], color='red', 
                          linestyle='--', linewidth=2, label=f"Mean: {analysis['statistics']['mean']:.2f}")
                ax.set_xlabel('Rating')
                ax.set_ylabel('Frequency')
                ax.set_title(f"Voter {analysis['voter_number']} (Reliability: {analysis['reliability_score']:.0f}%)", 
                           fontsize=10)
                ax.legend(fontsize=8)
                ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(self.analyzer.voter_analysis), 9):
            axes[idx].axis('off')
        
        plt.suptitle('Voter Score Distributions', fontweight='bold', fontsize=16)
        plt.tight_layout()
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_pdf_filtered_comparison(self, pdf):
        """Create filtered comparison page for PDF"""
        if not hasattr(self.analyzer, 'filtered_player_ratings'):
            return
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        players = []
        all_ratings = []
        filtered_ratings = []
        
        for player in self.analyzer.players:
            if player in self.analyzer.player_ratings and player in self.analyzer.filtered_player_ratings:
                players.append(player)
                all_ratings.append(self.analyzer.player_ratings[player]['overall_rating'])
                filtered_ratings.append(self.analyzer.filtered_player_ratings[player]['overall_rating'])
        
        x = np.arange(len(players))
        width = 0.35
        
        ax.bar(x - width/2, all_ratings, width, label='All Voters', 
              color='skyblue', edgecolor='black', linewidth=0.5)
        ax.bar(x + width/2, filtered_ratings, width, label='Filtered (Reliable Only)', 
              color='lightgreen', edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Player', fontweight='bold', fontsize=12)
        ax.set_ylabel('Overall Rating', fontweight='bold', fontsize=12)
        ax.set_title('Rating Comparison: All Voters vs Filtered (Reliable) Voters', 
                    fontweight='bold', fontsize=14, pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(players, rotation=45, ha='right')
        ax.set_ylim(0, 10)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_pdf_filtered_rankings(self, pdf):
        """Create filtered rankings page for PDF"""
        if not hasattr(self.analyzer, 'filtered_player_ratings'):
            return
            
        fig, ax = plt.subplots(figsize=(14, 10))
        
        players = []
        ratings = []
        for player, data in self.analyzer.filtered_player_ratings.items():
            players.append(player)
            ratings.append(data['overall_rating'])
        
        sorted_data = sorted(zip(players, ratings), key=lambda x: x[1], reverse=True)
        players, ratings = zip(*sorted_data)
        
        colors = plt.colormaps.get_cmap('RdYlGn')(np.array(ratings) / 10)
        bars = ax.barh(players, ratings, color=colors, edgecolor='black', linewidth=0.5)
        
        for i, (player, rating) in enumerate(zip(players, ratings)):
            ax.text(rating + 0.1, i, f'{rating:.2f}', va='center', fontweight='bold')
        
        ax.set_xlabel('Overall Rating (0-10)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Player', fontweight='bold', fontsize=12)
        ax.set_title('FILTERED Player Rankings\n(Excluding Too Harsh & Low Reliability <75% Voters)', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xlim(0, 10)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_pdf_filtered_radar(self, pdf):
        """Create filtered radar charts page for PDF"""
        if not hasattr(self.analyzer, 'filtered_player_ratings'):
            return
            
        players = list(self.analyzer.filtered_player_ratings.keys())
        num_players = len(players)
        
        cols = 3
        rows = (num_players + cols - 1) // cols
        
        fig = plt.figure(figsize=(15, rows * 4))
        
        categories = ['Technical\nBall Control', 'Shooting &\nFinishing', 
                     'Offensive\nPlay', 'Defensive\nPlay',
                     'Tactical/\nPsychological', 'Physical/\nCondition']
        
        for idx, player in enumerate(players):
            ax = fig.add_subplot(rows, cols, idx + 1, projection='polar')
            data = self.analyzer.filtered_player_ratings[player]
            
            values = [
                data['axis_ratings']['technical_ball_control']['score'],
                data['axis_ratings']['shooting_finishing']['score'],
                data['axis_ratings']['offensive_play']['score'],
                data['axis_ratings']['defensive_play']['score'],
                data['axis_ratings']['tactical_psychological']['score'],
                data['axis_ratings']['physical_condition']['score']
            ]
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=player)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=8)
            ax.set_ylim(0, 10)
            ax.set_title(f'{player}\n({data["overall_rating"]:.2f}/10)', 
                        fontweight='bold', size=10, pad=20)
            ax.grid(True)
        
        plt.suptitle('6-Axis Skill Radar Charts - Filtered (Reliable Voters Only)', 
                    fontweight='bold', fontsize=16, y=0.995)
        plt.tight_layout()
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_voter_correlation_heatmap(self):
        """Create correlation heatmap for voter ratings"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get voter numbers for labels
        voter_labels = []
        for voter in self.analyzer.voters:
            voter_num = self.analyzer.voter_analysis[voter]['voter_number']
            voter_labels.append(f'V{voter_num}')
        
        # Create heatmap
        correlation_matrix = self.analyzer.voter_correlations
        
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        ax.set_xticks(np.arange(len(voter_labels)))
        ax.set_yticks(np.arange(len(voter_labels)))
        ax.set_xticklabels(voter_labels, rotation=45, ha='right')
        ax.set_yticklabels(voter_labels)
        
        # Add correlation values
        for i in range(len(voter_labels)):
            for j in range(len(voter_labels)):
                val = correlation_matrix.iloc[i, j]
                if not np.isnan(val):
                    text_color = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                           color=text_color, fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Correlation Coefficient')
        
        ax.set_title('Voter Rating Correlation Matrix\n(High correlation may indicate similar perspectives or collusion)', 
                    fontweight='bold', fontsize=14, pad=15)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/10_voter_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created voter correlation heatmap")
    
    def plot_bias_pattern_distribution(self):
        """Create visualization of bias pattern distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart of voter patterns
        patterns = self.analyzer.voter_patterns
        labels = []
        sizes = []
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
        
        for pattern_name, voters in patterns.items():
            if voters:
                labels.append(f'{pattern_name}\n({len(voters)})')
                sizes.append(len(voters))
        
        if sizes:
            ax1.pie(sizes, labels=labels, colors=colors[:len(sizes)], autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 10})
            ax1.set_title('Voter Pattern Distribution', fontweight='bold', fontsize=12)
        
        # Bar chart of bias type summary
        if hasattr(self.analyzer, 'bias_type_summary'):
            bias_types = []
            counts = []
            
            for bias_type, voters in self.analyzer.bias_type_summary.items():
                if voters:
                    # Shorten labels for better display
                    short_label = bias_type.split('(')[0].strip()
                    bias_types.append(short_label)
                    counts.append(len(voters))
            
            if counts:
                bars = ax2.barh(bias_types, counts, color='skyblue', edgecolor='black', linewidth=0.5)
                
                for i, (bias, count) in enumerate(zip(bias_types, counts)):
                    ax2.text(count + 0.1, i, str(count), va='center', fontweight='bold')
                
                ax2.set_xlabel('Number of Voters', fontweight='bold')
                ax2.set_title('Bias Type Classification', fontweight='bold', fontsize=12)
                ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/11_bias_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created bias pattern distribution chart")
    
    def _create_pdf_voter_correlation(self, pdf):
        """Create voter correlation heatmap page for PDF"""
        fig, ax = plt.subplots(figsize=(11, 10))
        
        voter_labels = []
        for voter in self.analyzer.voters:
            voter_num = self.analyzer.voter_analysis[voter]['voter_number']
            voter_labels.append(f'V{voter_num}')
        
        correlation_matrix = self.analyzer.voter_correlations
        
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        ax.set_xticks(np.arange(len(voter_labels)))
        ax.set_yticks(np.arange(len(voter_labels)))
        ax.set_xticklabels(voter_labels, rotation=45, ha='right')
        ax.set_yticklabels(voter_labels)
        
        for i in range(len(voter_labels)):
            for j in range(len(voter_labels)):
                val = correlation_matrix.iloc[i, j]
                if not np.isnan(val):
                    text_color = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                           color=text_color, fontsize=7)
        
        plt.colorbar(im, ax=ax, label='Correlation Coefficient')
        
        ax.set_title('Voter Rating Correlation Matrix\n(High correlation may indicate similar perspectives)', 
                    fontweight='bold', fontsize=13, pad=15)
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_pdf_bias_patterns(self, pdf):
        """Create bias pattern distribution page for PDF"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        patterns = self.analyzer.voter_patterns
        labels = []
        sizes = []
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        for pattern_name, voters in patterns.items():
            if voters:
                labels.append(f'{pattern_name}\n({len(voters)})')
                sizes.append(len(voters))
        
        if sizes:
            ax1.pie(sizes, labels=labels, colors=colors[:len(sizes)], autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 9})
            ax1.set_title('Voter Pattern Distribution', fontweight='bold', fontsize=12)
        
        if hasattr(self.analyzer, 'bias_type_summary'):
            bias_types = []
            counts = []
            
            for bias_type, voters in self.analyzer.bias_type_summary.items():
                if voters:
                    short_label = bias_type.split('(')[0].strip()
                    bias_types.append(short_label)
                    counts.append(len(voters))
            
            if counts:
                bars = ax2.barh(bias_types, counts, color='skyblue', edgecolor='black', linewidth=0.5)
                
                for i, (bias, count) in enumerate(zip(bias_types, counts)):
                    ax2.text(count + 0.1, i, str(count), va='center', fontweight='bold')
                
                ax2.set_xlabel('Number of Voters', fontweight='bold')
                ax2.set_title('Bias Type Classification', fontweight='bold', fontsize=12)
                ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_pdf_top8_comparison(self, pdf):
        """Create top 8 features comparison page for PDF"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        players = []
        overall_ratings = []
        top8_ratings = []
        
        # Sort by top 8 rating
        sorted_players = sorted(self.analyzer.top8_player_ratings.items(),
                               key=lambda x: x[1]['top8_rating'],
                               reverse=True)
        
        for player, data in sorted_players:
            overall_rating = self.analyzer.player_ratings[player]['overall_rating']
            players.append(player)
            overall_ratings.append(overall_rating)
            top8_ratings.append(data['top8_rating'])
        
        x = np.arange(len(players))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, overall_ratings, width, label='Overall Rating (6-Axis)', 
                      alpha=0.8, color='skyblue', edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, top8_ratings, width, label='Top 8 Features Average', 
                      alpha=0.8, color='coral', edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Players', fontweight='bold', fontsize=12)
        ax.set_ylabel('Rating (0-10)', fontweight='bold', fontsize=12)
        ax.set_title('Player Ratings Comparison: Overall (6-Axis) vs Top 8 Features Average\n' +
                    'Top 8 Features = Average of Player\'s 8 Highest-Rated Skills', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(players, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 10)
        
        # Add value labels on bars
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add explanation text
        explanation = (
            "This chart compares the standard overall rating (weighted average of 6 axes) "
            "with a new rating based on each player's top 8 performing skills. "
            "The Top 8 rating highlights players who excel in specific areas."
        )
        fig.text(0.5, 0.02, explanation, ha='center', fontsize=9, 
                style='italic', wrap=True, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_pdf_voter_player_bias(self, pdf):
        """Create voter-player bias heatmap page for PDF"""
        # Create matrix of deviations
        voters = []
        players = self.analyzer.players
        deviations = []
        
        for voter, bias_data in sorted(self.analyzer.voter_player_bias.items(), 
                                      key=lambda x: x[1]['voter_number']):
            voters.append(f"Voter {bias_data['voter_number']}")
            voter_devs = []
            for player in players:
                if player in bias_data['player_deviations']:
                    voter_devs.append(bias_data['player_deviations'][player]['deviation'])
                else:
                    voter_devs.append(0)
            deviations.append(voter_devs)
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        im = ax.imshow(deviations, cmap='RdYlGn', aspect='auto', vmin=-2, vmax=2)
        
        # Set ticks
        ax.set_xticks(np.arange(len(players)))
        ax.set_yticks(np.arange(len(voters)))
        ax.set_xticklabels(players, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(voters, fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Rating Deviation from Overall Mean', rotation=270, labelpad=20, fontweight='bold')
        
        # Add values to cells
        for i in range(len(voters)):
            for j in range(len(players)):
                color = "black" if abs(deviations[i][j]) < 1.0 else "white"
                text = ax.text(j, i, f'{deviations[i][j]:.1f}',
                             ha="center", va="center", color=color, fontsize=8, fontweight='bold')
        
        ax.set_title('Deep Voter Bias Analysis: Individual Voter Preferences Towards Specific Players\n' +
                    '(Green = More Favorable than Average, Red = More Critical than Average)', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xlabel('Players', fontweight='bold', fontsize=12)
        ax.set_ylabel('Voters', fontweight='bold', fontsize=12)
        
        # Add explanation
        explanation = (
            "This heatmap shows how each voter's ratings for specific players deviate from the overall average. "
            "Positive values (green) indicate the voter rated that player more favorably than average. "
            "Negative values (red) indicate more critical ratings. Values close to 0 (yellow) indicate consistency with other voters."
        )
        fig.text(0.5, 0.02, explanation, ha='center', fontsize=9, 
                style='italic', wrap=True, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_pdf_100reliable_comparison(self, pdf):
        """Create 100% reliable voters comparison page for PDF"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        players = []
        all_voters = []
        reliable_100 = []
        
        for player in self.analyzer.players:
            all_rating = self.analyzer.player_ratings[player]['overall_rating']
            reliable_rating = self.analyzer.reliable_100_player_ratings[player]['overall_rating']
            
            players.append(player)
            all_voters.append(all_rating)
            reliable_100.append(reliable_rating)
        
        # Sort by 100% reliable rating
        sorted_indices = np.argsort(reliable_100)[::-1]
        players = [players[i] for i in sorted_indices]
        all_voters = [all_voters[i] for i in sorted_indices]
        reliable_100 = [reliable_100[i] for i in sorted_indices]
        
        x = np.arange(len(players))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, all_voters, width, label='All Voters', 
                      alpha=0.8, color='skyblue', edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, reliable_100, width, label='100% Reliable Voters Only', 
                      alpha=0.8, color='gold', edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Players', fontweight='bold', fontsize=12)
        ax.set_ylabel('Rating (0-10)', fontweight='bold', fontsize=12)
        ax.set_title('100% RELIABLE DATA ANALYSIS\n' +
                    'Player Ratings: All Voters vs 100% Reliability Voters', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(players, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 10)
        
        # Add value labels on bars
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add explanation
        reliable_voters, excluded = self.analyzer.get_100_percent_reliable_voters()
        explanation = (
            f"This analysis uses only voters with 100% reliability score ({len(reliable_voters)} voters). "
            f"{len(excluded)} voters were excluded due to missing data or bias. "
            "This provides the highest quality assessment based on the most reliable data sources."
        )
        fig.text(0.5, 0.02, explanation, ha='center', fontsize=9, 
                style='italic', wrap=True, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_pdf_100reliable_radar(self, pdf):
        """Create 100% reliable voters radar charts page for PDF"""
        # Create grid for top 9 players
        sorted_players = sorted(self.analyzer.reliable_100_player_ratings.items(),
                               key=lambda x: x[1]['overall_rating'],
                               reverse=True)[:9]
        
        fig = plt.figure(figsize=(15, 12))
        
        # Axis labels
        axis_labels = [
            'Technical\nBall Control',
            'Shooting\nFinishing',
            'Offensive\nPlay',
            'Defensive\nPlay',
            'Tactical\nPsychological',
            'Physical\nCondition'
        ]
        
        axis_keys = [
            'technical_ball_control',
            'shooting_finishing',
            'offensive_play',
            'defensive_play',
            'tactical_psychological',
            'physical_condition'
        ]
        
        for idx, (player, data) in enumerate(sorted_players, 1):
            ax = fig.add_subplot(3, 3, idx, projection='polar')
            
            # Get values for this player
            values = []
            for key in axis_keys:
                if key in data['axis_ratings']:
                    values.append(data['axis_ratings'][key]['score'])
                else:
                    values.append(0)
            
            # Complete the circle
            values += values[:1]
            
            # Angles
            angles = np.linspace(0, 2 * np.pi, len(axis_labels), endpoint=False).tolist()
            angles += angles[:1]
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2.5, label=player, color='gold', markersize=6)
            ax.fill(angles, values, alpha=0.25, color='gold')
            ax.set_ylim(0, 10)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(axis_labels, size=9)
            ax.set_yticks([2, 4, 6, 8, 10])
            ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=8)
            ax.grid(True, linewidth=0.5, alpha=0.7)
            
            # Title
            overall = data['overall_rating']
            ax.set_title(f"{player}\nRating: {overall:.2f}", 
                        fontweight='bold', size=11, pad=20)
        
        plt.suptitle('100% RELIABLE DATA - Top 9 Players 6-Axis Radar Analysis\n' +
                    '(Based on 100% Reliability Voters Only)', 
                    fontweight='bold', fontsize=16, y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close()
