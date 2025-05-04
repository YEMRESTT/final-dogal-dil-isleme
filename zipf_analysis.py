import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import logging
import re

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dosya yolları
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "processed_data")
plots_dir = os.path.join(base_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# İşlenmiş veriyi oku
input_csv = os.path.join(data_dir, "processed_turk_medeni_kanunu.csv")
df = pd.read_csv(input_csv)
logger.info(f"Veri okundu: {input_csv}, {len(df)} satır bulundu.")


# Kök (lemmatized) ve token (stemmed) verilerini çıkar
def extract_data():
    """DataFrame'den token ve kök verilerini çıkarır"""

    # token ve root verilerinin string olarak saklanma ihtimaline karşı parse işlemi
    def parse_list_string(list_str):
        if pd.isna(list_str) or not isinstance(list_str, str):
            return []
        # Liste formatındaki string'i gerçek listeye çevir
        try:
            # Strip köşeli parantezleri ve tek tırnakları temizle
            clean_str = list_str.strip('[]').replace("'", "").replace('"', '')
            # Virgülle ayrılmış değerleri listele
            return [item.strip() for item in clean_str.split(',') if item.strip()]
        except:
            return []

    # Her bölümdeki token ve kökleri çıkar
    all_tokens = []
    all_roots = []

    # CSV formatına göre uyarla
    token_column = 'tokens' if 'tokens' in df.columns else 'unique_tokens'
    root_column = 'roots' if 'roots' in df.columns else 'analyzed_tokens'

    # Her satırı kontrol et ve token/root verilerini çıkar
    for _, row in df.iterrows():
        try:
            # Tokenler için
            if token_column in row:
                tokens = parse_list_string(row[token_column])
                all_tokens.extend(tokens)

            # Kökler için
            if root_column in row:
                roots = parse_list_string(row[root_column])
                all_roots.extend(roots)

        except Exception as e:
            logger.error(f"Veri çıkarma hatası: {str(e)}")

    logger.info(f"Toplam {len(all_tokens)} token ve {len(all_roots)} kök çıkarıldı")

    # Eğer hiç veri çıkarılamadıysa, raw_text'ten token çıkarmayı deneyelim
    if not all_tokens and 'raw_text' in df.columns:
        logger.warning("Token verisi bulunamadı, raw_text'ten token çıkarma deneniyor...")
        for _, row in df.iterrows():
            if isinstance(row['raw_text'], str):
                # Basit tokenization
                tokens = re.findall(r'\b\w+\b', row['raw_text'].lower())
                all_tokens.extend(tokens)
        logger.info(f"Raw text'ten {len(all_tokens)} token çıkarıldı")

    # Temizleme işlemi: Boş string, None değerleri filtrele
    all_tokens = [token for token in all_tokens if token and isinstance(token, str) and len(token) > 1]
    all_roots = [root for root in all_roots if root and isinstance(root, str) and len(root) > 1]

    return all_tokens, all_roots


# Verileri çıkar
tokens, roots = extract_data()

# Eğer token/root yoksa uyarı ver
if not tokens:
    logger.warning("Hiç token bulunamadı!")
if not roots:
    logger.warning("Hiç kök bulunamadı!")


# CSV'leri oluştur
def create_csv_from_data(data, filename, column_name):
    """Veriyi CSV dosyasına kaydeder"""
    output_df = pd.DataFrame({column_name: data})
    output_path = os.path.join(data_dir, filename)
    output_df.to_csv(output_path, index=False)
    logger.info(f"CSV kaydedildi: {output_path}")
    return output_path


# Token ve kök verilerini CSV olarak kaydet
if tokens:
    stemmed_csv = create_csv_from_data(tokens, "stemmed_data.csv", "stemmed_text")
if roots:
    lemmatized_csv = create_csv_from_data(roots, "lemmatized_data.csv", "lemmatized_text")


# Zipf analizi fonksiyonu
def plot_zipf(data, title, filename):
    """Zipf analizi grafiği oluşturur"""
    if not data:
        logger.warning(f"{title} için veri bulunamadı, Zipf grafiği oluşturulamıyor.")
        return None

    # Kelime frekanslarını hesapla
    word_counts = Counter(data)

    # Frekans sırasına göre sırala
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    ranks = np.arange(1, len(sorted_word_counts) + 1)
    frequencies = [count for _, count in sorted_word_counts]

    # Log-log grafiği çiz
    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, frequencies, marker='.', linestyle='none')

    # Teorik Zipf eğrisi (1/rank)
    k = frequencies[0]  # en sık geçen kelimenin frekansı
    theoretical_zipf = [k / r for r in ranks]
    plt.loglog(ranks, theoretical_zipf, 'r-', alpha=0.7, label='Teorik Zipf (1/rank)')

    plt.title(title)
    plt.xlabel('Rank (log scale)')
    plt.ylabel('Frequency (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Grafiği kaydet
    output_path = os.path.join(plots_dir, filename)
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Zipf grafiği kaydedildi: {output_path}")
    return output_path


# Zipf grafiklerini oluştur
if tokens:
    stemmed_zipf_plot = plot_zipf(tokens, 'Zipf Analizi - Stemmed Veri (Tokens)', 'stemmed_zipf.png')
if roots:
    lemmatized_zipf_plot = plot_zipf(roots, 'Zipf Analizi - Lemmatized Veri (Roots)', 'lemmatized_zipf.png')

# Raw text'ten de zipf analizi yapalım
raw_texts = df['raw_text'].fillna('').astype(str).tolist()
raw_tokens = []
for text in raw_texts:
    raw_tokens.extend(re.findall(r'\b\w+\b', text.lower()))

raw_zipf_plot = plot_zipf(raw_tokens, 'Zipf Analizi - Ham Veri', 'raw_zipf.png')

# Veri setinin istatistiklerini yazdır
print("\nVeri Seti İstatistikleri:")
print(f"Döküman Sayısı: {len(df)}")
print(f"Ham Token Sayısı: {len(raw_tokens)}")
print(f"Stemmed Token Sayısı: {len(tokens)}")
print(f"Lemmatized Kök Sayısı: {len(roots)}")
print(f"Ham Veri Benzersiz Kelime Sayısı: {len(set(raw_tokens))}")
print(f"Stemmed Veri Benzersiz Kelime Sayısı: {len(set(tokens))}")
print(f"Lemmatized Veri Benzersiz Kelime Sayısı: {len(set(roots))}")

# En sık geçen kelimeler
print("\nHam Veri En Sık Geçen 20 Kelime:")
for word, count in Counter(raw_tokens).most_common(20):
    print(f"  {word}: {count}")

if tokens:
    print("\nStemmed Veri En Sık Geçen 20 Kelime:")
    for word, count in Counter(tokens).most_common(20):
        print(f"  {word}: {count}")

if roots:
    print("\nLemmatized Veri En Sık Geçen 20 Kelime:")
    for word, count in Counter(roots).most_common(20):
        print(f"  {word}: {count}")

print("\nİşlem tamamlandı!")