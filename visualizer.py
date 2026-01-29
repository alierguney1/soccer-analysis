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
        self.plot_6axis_radar()
        self.plot_voter_reliability()
        self.plot_self_bias()
        self.plot_player_heatmap()
        self.plot_voter_distributions()
        self.plot_skill_comparison()
        
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
        colors = plt.cm.RdYlGn(np.array(ratings) / 10)
        
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
        """Create radar charts for top players"""
        # Get top 6 players
        summary_data = []
        for player, data in self.analyzer.player_ratings.items():
            summary_data.append({
                'Player': player,
                'Overall': data['overall_rating']
            })
        summary_df = pd.DataFrame(summary_data).sort_values('Overall', ascending=False)
        top_players = summary_df.head(6)['Player'].tolist()
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(projection='polar'))
        axes = axes.flatten()
        
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
        
        for idx, player in enumerate(top_players):
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
            ax.set_xticklabels(categories, size=9)
            ax.set_ylim(0, 10)
            ax.set_yticks([2, 4, 6, 8, 10])
            ax.set_yticklabels(['2', '4', '6', '8', '10'], size=8)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{player}\nOverall: {data["overall_rating"]:.2f}', 
                        fontweight='bold', size=11, pad=15)
        
        plt.suptitle('Top 6 Players - 6-Axis Radar Analysis', 
                    fontweight='bold', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/02_6axis_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created 6-axis radar charts")
    
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
    
    def plot_self_bias(self):
        """Create self-bias analysis chart"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Collect self-bias data
        bias_data = []
        for voter, data in self.analyzer.voter_analysis.items():
            if 'self_vs_others_diff' in data['bias_indicators']:
                bias_data.append({
                    'Voter': f"Voter {data['voter_number']}",
                    'Player': data['bias_indicators'].get('player_name', 'Unknown'),
                    'Difference': data['bias_indicators']['self_vs_others_diff'],
                    'Self_Rating': data['bias_indicators']['self_rating_avg'],
                    'Others_Rating': data['bias_indicators']['others_rating_avg']
                })
        
        if bias_data:
            df = pd.DataFrame(bias_data)
            df = df.sort_values('Difference', ascending=False)
            
            # Create grouped bar chart
            x = np.arange(len(df))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, df['Self_Rating'], width, 
                          label='Self Rating', color='#e57373', edgecolor='black')
            bars2 = ax.bar(x + width/2, df['Others_Rating'], width, 
                          label='Others Rating', color='#64b5f6', edgecolor='black')
            
            # Add difference annotations
            for i, row in df.iterrows():
                diff = row['Difference']
                if diff > 1.0:
                    ax.annotate(f'+{diff:.2f}', 
                              xy=(i, max(row['Self_Rating'], row['Others_Rating']) + 0.3),
                              ha='center', fontweight='bold', color='red', fontsize=11)
            
            ax.set_xlabel('Voter (Player Name)', fontweight='bold', fontsize=12)
            ax.set_ylabel('Average Rating', fontweight='bold', fontsize=12)
            ax.set_title('Self-Bias Analysis: Self-Rating vs Rating Others', 
                        fontweight='bold', fontsize=14, pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{row['Voter']}\n({row['Player']})" 
                               for _, row in df.iterrows()], rotation=0)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 10)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/04_self_bias.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Created self-bias analysis chart")
        else:
            print("⚠ No self-bias data to visualize")
    
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
        """Create skill comparison across top players"""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Get top 8 players
        summary_data = []
        for player, data in self.analyzer.player_ratings.items():
            summary_data.append({
                'Player': player,
                'Overall': data['overall_rating']
            })
        summary_df = pd.DataFrame(summary_data).sort_values('Overall', ascending=False)
        top_players = summary_df.head(8)['Player'].tolist()
        
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
        width = 0.1
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_players)))
        
        for i, player in enumerate(top_players):
            data = self.analyzer.player_ratings[player]
            values = [0] * len(categories)
            
            for axis_name, pos in axis_map.items():
                if axis_name in data['axis_ratings']:
                    values[pos] = data['axis_ratings'][axis_name]['score']
            
            offset = width * (i - len(top_players)/2)
            ax.bar(x + offset, values, width, label=player[:15], 
                  color=colors[i], edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Skill Category', fontweight='bold', fontsize=12)
        ax.set_ylabel('Rating (0-10)', fontweight='bold', fontsize=12)
        ax.set_title('Top 8 Players - Skill Category Comparison', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
        ax.set_ylim(0, 10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/07_skill_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created skill comparison chart")
    
    def create_pdf_report(self):
        """Create comprehensive PDF report"""
        pdf_path = f'{self.output_dir}/Soccer_Analysis_Report.pdf'
        
        with PdfPages(pdf_path) as pdf:
            # Page 1: Player Rankings
            img = plt.imread(f'{self.output_dir}/01_player_rankings.png')
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(img)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 2: Radar Charts
            img = plt.imread(f'{self.output_dir}/02_6axis_radar.png')
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(img)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 3: Voter Reliability
            img = plt.imread(f'{self.output_dir}/03_voter_reliability.png')
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(img)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 4: Self Bias
            if os.path.exists(f'{self.output_dir}/04_self_bias.png'):
                img = plt.imread(f'{self.output_dir}/04_self_bias.png')
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.imshow(img)
                ax.axis('off')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # Page 5: Heatmap
            img = plt.imread(f'{self.output_dir}/05_player_heatmap.png')
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(img)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 6: Distributions
            img = plt.imread(f'{self.output_dir}/06_voter_distributions.png')
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(img)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 7: Skill Comparison
            img = plt.imread(f'{self.output_dir}/07_skill_comparison.png')
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(img)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        print(f"\n✓ PDF report created: {pdf_path}")
