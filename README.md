# Soccer Analysis System

Detaylı futbol oyuncu analiz sistemi - 6 eksenli oyuncu değerlendirme ve oylayıcı güvenilirlik analizi.

## Özellikler

### 6 Eksenli Oyuncu Değerlendirme Sistemi

1. **Teknik Top Kontrolü** (Technical Ball Control)
   - Topu kontrol etme
   - Dribling
   - Pas verme

2. **Şut ve Bitiricilik** (Shooting & Finishing)
   - Şut gücü ve isabeti
   - Gol bitirme yeteneği

3. **Hücum Oyunu** (Offensive Play)
   - Hücumda katkı
   - Pozisyon alma

4. **Savunma Oyunu** (Defensive Play)
   - Savunma katkısı
   - Topa müdahale

5. **Taktik/Psikolojik** (Tactical/Psychological)
   - Disiplin
   - Karar verme
   - Takım çalışması

6. **Fiziksel/Kondisyon** (Physical/Condition)
   - Fiziksel dayanıklılık
   - Kondisyon seviyesi

### Oylayıcı Analizi

- **Güvenilirlik Skoru**: Her oylayıcı için 0-100 arası güvenilirlik değerlendirmesi
- **Kendine Puan Verme Bias**: Oylayıcıların kendilerine aşırı puan verip vermediklerinin analizi
- **Tutarlılık Analizi**: Benzer yeteneklere tutarlı puan verilip verilmediğinin kontrolü
- **İstatistiksel Anormallikler**: Çok cömert veya çok sert oylayıcıların tespiti

## Kurulum

```bash
# Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt
```

## Kullanım

```bash
# Analizi çalıştırın
python3 soccer_analysis.py
```

Bu komut:
1. CSV dosyasından verileri yükler
2. 6 eksenli oyuncu değerlendirmelerini hesaplar
3. Oylayıcı güvenilirlik analizini yapar
4. Detaylı raporlar oluşturur
5. Görselleştirmeler üretir (PNG ve PDF formatında)

## Çıktılar

Analiz tamamlandığında `analysis_output/` klasöründe şu dosyalar oluşturulur:

1. **01_player_rankings.png** - Oyuncu sıralamaları
2. **02_6axis_radar.png** - En iyi 6 oyuncu için radar grafikleri
3. **03_voter_reliability.png** - Oylayıcı güvenilirlik skorları
4. **04_self_bias.png** - Kendine puan verme bias analizi
5. **05_player_heatmap.png** - Tüm oyuncular için yetenek ısı haritası
6. **06_voter_distributions.png** - Oylayıcı skor dağılımları
7. **07_skill_comparison.png** - En iyi 8 oyuncu için yetenek karşılaştırması
8. **Soccer_Analysis_Report.pdf** - Tüm grafikleri içeren PDF rapor

## Veri Formatı

CSV dosyası (`upk_halisaha.csv`) şu formatta olmalıdır:
- Sütunlar: Name, Skill, Voter 1, Voter 2, ..., Voter N
- Her satır bir oyuncunun bir yeteneği için aldığı puanları içerir
- Puanlar 1-10 arası olmalıdır

## Analiz Detayları

### Oyuncu Değerlendirme
- Her eksende ortalama skor hesaplanır
- Genel skor ağırlıklı ortalama ile bulunur
- Standart sapma ile tutarlılık değerlendirilir

### Oylayıcı Güvenilirlik
- Yüksek varyans: Tutarsız puanlama
- Düşük varyans: Ayrım yapamama
- Ortalama sapma: Çok cömert/sert olma
- Kendine bias: Kendine başkalarından yüksek puan verme

## Geliştirici

Bu analiz sistemi halı saha maçlarında oyuncuların birbirlerini değerlendirmesi için geliştirilmiştir.