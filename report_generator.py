#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a summary text report from the analysis
"""

import json
from datetime import datetime


def generate_text_report(analyzer, output_file='analysis_output/SUMMARY_REPORT.txt'):
    """Generate a comprehensive text summary report"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("HALІ SAHA OYUNCU ANALİZ RAPORU\n")
        f.write("6 Eksenli Değerlendirme Sistemi ve Oylayıcı Güvenilirlik Analizi\n")
        f.write("=" * 80 + "\n")
        f.write(f"Rapor Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n")
        f.write(f"Toplam Oyuncu Sayısı: {len(analyzer.players)}\n")
        f.write(f"Toplam Oylayıcı Sayısı: {len(analyzer.voters)}\n")
        f.write("=" * 80 + "\n\n")
        
        # Executive Summary
        f.write("YÖNETİCİ ÖZETİ\n")
        f.write("-" * 80 + "\n")
        
        # Top 3 players
        summary_data = []
        for player, data in analyzer.player_ratings.items():
            summary_data.append((player, data['overall_rating']))
        summary_data.sort(key=lambda x: x[1], reverse=True)
        
        f.write("\nEn İyi 3 Oyuncu (Tüm Oylar):\n")
        for i, (player, rating) in enumerate(summary_data[:3], 1):
            f.write(f"  {i}. {player}: {rating:.2f}/10\n")
        
        # Voter reliability summary
        reliable_voters = sum(1 for v in analyzer.voter_analysis.values() 
                            if v['reliability_score'] >= 75)
        total_voters = len(analyzer.voter_analysis)
        f.write(f"\nVeri Kalitesi: {reliable_voters}/{total_voters} oylayıcı güvenilir (≥75%)\n")
        
        # Harsh voters summary
        harsh_voters = sum(1 for v in analyzer.voter_analysis.values() 
                         if v['bias_indicators'].get('too_harsh', False))
        f.write(f"Çok Sert Oylayıcı: {harsh_voters} oylayıcı tespit edildi\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Detailed Player Rankings
        f.write("DETAYLI OYUNCU SIRALAMALARI\n")
        f.write("-" * 80 + "\n\n")
        
        for rank, (player, rating) in enumerate(summary_data, 1):
            data = analyzer.player_ratings[player]
            f.write(f"{rank}. {player}\n")
            f.write(f"   Genel Puan: {rating:.2f}/10\n")
            f.write(f"   Eksen Puanları:\n")
            
            axis_names_tr = {
                'technical_ball_control': 'Teknik Top Kontrolü',
                'shooting_finishing': 'Şut ve Bitiricilik',
                'offensive_play': 'Hücum Oyunu',
                'defensive_play': 'Savunma Oyunu',
                'tactical_psychological': 'Taktik/Psikolojik',
                'physical_condition': 'Fiziksel/Kondisyon'
            }
            
            for axis_name, axis_data in data['axis_ratings'].items():
                tr_name = axis_names_tr.get(axis_name, axis_name)
                f.write(f"     - {tr_name}: {axis_data['score']:.2f}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n\n")
        
        # Voter Analysis
        f.write("OYLAYICI GÜVENİLİRLİK ANALİZİ\n")
        f.write("-" * 80 + "\n\n")
        
        # Sort voters by reliability
        voter_list = [(v, d) for v, d in analyzer.voter_analysis.items()]
        voter_list.sort(key=lambda x: x[1]['reliability_score'], reverse=True)
        
        f.write("Güvenilirlik Sıralaması:\n\n")
        for voter, data in voter_list:
            if 'statistics' not in data or not data['statistics']:
                continue
            
            f.write(f"Oylayıcı {data['voter_number']}: {data['reliability_score']}/100\n")
            f.write(f"  - Ortalama Puan: {data['statistics']['mean']:.2f}\n")
            f.write(f"  - Standart Sapma: {data['statistics']['std']:.2f}\n")
            
            # Flags (self-bias removed since voter order is random)
            if data['bias_indicators'].get('too_generous'):
                f.write(f"  ⚠️  ÇOK CÖMERT (ortalama üstü puanlama)\n")
            
            if data['bias_indicators'].get('too_harsh'):
                f.write(f"  ⚠️  ÇOK SERT (ortalama altı puanlama)\n")
            
            if data['bias_indicators'].get('high_variance'):
                f.write(f"  ⚠️  TUTARSIZ (yüksek varyans)\n")
            
            if data['bias_indicators'].get('low_variance'):
                f.write(f"  ⚠️  AYRIM YAPAMIYOR (düşük varyans)\n")
            
            f.write("\n")
        
        f.write("=" * 80 + "\n\n")
        
        # Key Findings
        f.write("TEMEL BULGULAR VE ÖNERİLER\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("1. VERİ KALİTESİ:\n")
        quality_pct = (reliable_voters / total_voters) * 100
        f.write(f"   - Güvenilirlik oranı: %{quality_pct:.0f}\n")
        if quality_pct >= 80:
            f.write("   ✓ Veri kalitesi YÜKSEK - güvenilir analiz mümkün\n")
        elif quality_pct >= 60:
            f.write("   ⚠️  Veri kalitesi ORTA - dikkatli yorumlanmalı\n")
        else:
            f.write("   ❌ Veri kalitesi DÜŞÜK - sonuçlar şüpheli\n")
        
        f.write("\n2. BIAS VE ÖNYARGı:\n")
        # Note: Self-bias detection removed since voter order is random
        f.write("   NOT: Kendine bias tespiti yapılamamaktadır (oylayıcı sırası rastgele)\n")
        
        generous = sum(1 for v in analyzer.voter_analysis.values() 
                      if v['bias_indicators'].get('too_generous', False))
        harsh = sum(1 for v in analyzer.voter_analysis.values() 
                   if v['bias_indicators'].get('too_harsh', False))
        
        if generous > 0:
            f.write(f"   - {generous} oylayıcı çok cömert puanlama yapıyor\n")
        if harsh > 0:
            f.write(f"   - {harsh} oylayıcı çok sert puanlama yapıyor\n")
        
        f.write("\n3. EN İYİ PERFORMANS GÖSTERENLER:\n")
        f.write("   Genel değerlendirmeye göre en iyi 5 oyuncu:\n")
        for i, (player, rating) in enumerate(summary_data[:5], 1):
            f.write(f"   {i}. {player} ({rating:.2f}/10)\n")
        
        f.write("\n4. GELİŞİM GEREKTİREN ALANLAR:\n")
        # Find bottom 3
        f.write("   Gelişim potansiyeli yüksek oyuncular:\n")
        for i, (player, rating) in enumerate(summary_data[-3:], 1):
            f.write(f"   {i}. {player} ({rating:.2f}/10)\n")
        
        f.write("\n5. EKSİN BAZINDA ANALİZ:\n")
        
        # Calculate average per axis across all players
        axis_averages = {}
        axis_names_tr = {
            'technical_ball_control': 'Teknik Top Kontrolü',
            'shooting_finishing': 'Şut ve Bitiricilik',
            'offensive_play': 'Hücum Oyunu',
            'defensive_play': 'Savunma Oyunu',
            'tactical_psychological': 'Taktik/Psikolojik',
            'physical_condition': 'Fiziksel/Kondisyon'
        }
        
        for axis_name in axis_names_tr.keys():
            scores = []
            for player_data in analyzer.player_ratings.values():
                if axis_name in player_data['axis_ratings']:
                    scores.append(player_data['axis_ratings'][axis_name]['score'])
            if scores:
                axis_averages[axis_name] = sum(scores) / len(scores)
        
        # Sort by average
        sorted_axes = sorted(axis_averages.items(), key=lambda x: x[1], reverse=True)
        
        f.write("   Takım güçlü yönleri (ortalama puan):\n")
        for axis_name, avg in sorted_axes[:3]:
            tr_name = axis_names_tr.get(axis_name, axis_name)
            f.write(f"   ✓ {tr_name}: {avg:.2f}\n")
        
        f.write("\n   Takım gelişim alanları (ortalama puan):\n")
        for axis_name, avg in sorted_axes[-3:]:
            tr_name = axis_names_tr.get(axis_name, axis_name)
            f.write(f"   ⚠️  {tr_name}: {avg:.2f}\n")
        
        # FILTERED ANALYSIS SECTION
        if hasattr(analyzer, 'filtered_player_ratings') and analyzer.filtered_player_ratings:
            f.write("\n" + "=" * 80 + "\n\n")
            f.write("FİLTRELENMİŞ ANALİZ\n")
            f.write("(Çok Sert ve Güvenilirliği <%75 Olan Oylayıcılar Hariç)\n")
            f.write("-" * 80 + "\n\n")
            
            # Get filtered voters info
            filtered_voters, excluded = analyzer.get_filtered_voters()
            f.write(f"Dahil Edilen Oylayıcı Sayısı: {len(filtered_voters)}\n")
            f.write(f"Hariç Tutulan Oylayıcı Sayısı: {len(excluded)}\n\n")
            
            if excluded:
                f.write("Hariç Tutulan Oylayıcılar:\n")
                for ex in excluded:
                    f.write(f"   - Oylayıcı {ex['voter_num']}: {ex['reason']}\n")
                f.write("\n")
            
            # Filtered top 5 players
            filtered_summary = []
            for player, data in analyzer.filtered_player_ratings.items():
                filtered_summary.append((player, data['overall_rating']))
            filtered_summary.sort(key=lambda x: x[1], reverse=True)
            
            f.write("FİLTRELENMİŞ SIRALAMA - EN İYİ 5 OYUNCU:\n")
            for i, (player, rating) in enumerate(filtered_summary[:5], 1):
                # Also show difference from all-voters rating
                all_rating = analyzer.player_ratings[player]['overall_rating']
                diff = rating - all_rating
                diff_str = f"({diff:+.2f})" if diff != 0 else ""
                f.write(f"   {i}. {player}: {rating:.2f}/10 {diff_str}\n")
            
            # Comparison table
            f.write("\n\nKARŞILAŞTIRMA TABLOSU (Tüm Oylar vs Filtrelenmiş):\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Oyuncu':<25} {'Tüm Oylar':>12} {'Filtrelenmiş':>12} {'Fark':>8}\n")
            f.write("-" * 60 + "\n")
            
            for player, filt_rating in filtered_summary:
                all_rating = analyzer.player_ratings[player]['overall_rating']
                diff = filt_rating - all_rating
                f.write(f"{player:<25} {all_rating:>12.2f} {filt_rating:>12.2f} {diff:>+8.2f}\n")
            
            f.write("-" * 60 + "\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("RAPOR SONU\n")
        f.write("Detaylı grafikler için 'analysis_output' klasöründeki PNG ve PDF dosyalarına bakınız.\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n✓ Text summary report created: {output_file}")


if __name__ == '__main__':
    # This module should be imported, not run directly
    print("This module should be imported by soccer_analysis.py")
