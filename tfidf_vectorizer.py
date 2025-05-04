import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

# Proje dizin yapısına göre dosya yolları
base_dir = os.path.abspath('.')  # Mevcut çalışma dizini
lemmatized_path = os.path.join(base_dir, "processed_data", "lemmatized_data.csv")
stemmed_path = os.path.join(base_dir, "processed_data", "stemmed_data.csv")
output_dir = os.path.join(base_dir, "processed_data")
plots_dir = os.path.join(base_dir, "plots", "tfidf_output")

# Dizinleri oluştur
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

print("Veri dosyaları yükleniyor...")
# Verileri oku
lemmatized_df = pd.read_csv(lemmatized_path)
stemmed_df = pd.read_csv(stemmed_path)

# Veri kontrolü
print(f"Lemmatized veri boyutu: {lemmatized_df.shape}")
print(f"Stemmed veri boyutu: {stemmed_df.shape}")

# Sütun isimlerini kontrol et
print(f"Lemmatized sütunları: {lemmatized_df.columns.tolist()}")
print(f"Stemmed sütunları: {stemmed_df.columns.tolist()}")

# Doğru sütun isimlerini kullan
lemmatized_text_column = 'lemmatized_text'
stemmed_text_column = 'stemmed_text'

# NaN değerleri temizle
lemmatized_df[lemmatized_text_column] = lemmatized_df[lemmatized_text_column].fillna('')
stemmed_df[stemmed_text_column] = stemmed_df[stemmed_text_column].fillna('')


# TF-IDF vektörleştirme fonksiyonu
def create_tfidf_matrix(texts, max_features=3000):
    """
    Metinler için TF-IDF vektörleştirme yapar ve DataFrame olarak döndürür
    """
    # NaN değerleri kontrol et ve temizle
    texts = ['' if pd.isna(text) else str(text) for text in texts]

    print(f"TF-IDF vektörleştirme yapılıyor (max_features={max_features})...")

    # TF-IDF vektörleştirmeyi yap
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Özellik isimleri
    feature_names = vectorizer.get_feature_names_out()

    print(f"Toplam özellik sayısı: {len(feature_names)}")

    # TF-IDF matrix'i DataFrame'e dönüştür
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    return df_tfidf, tfidf_matrix, vectorizer, feature_names


# Lemmatized veriler için TF-IDF
print("\nLemmatized veri için TF-IDF hesaplanıyor...")
lemmatized_texts = lemmatized_df[lemmatized_text_column].tolist()
lemmatized_tfidf_df, lemmatized_tfidf_matrix, lemmatized_vectorizer, lemmatized_features = create_tfidf_matrix(
    lemmatized_texts)

# Stemmed veriler için TF-IDF
print("\nStemmed veri için TF-IDF hesaplanıyor...")
stemmed_texts = stemmed_df[stemmed_text_column].tolist()
stemmed_tfidf_df, stemmed_tfidf_matrix, stemmed_vectorizer, stemmed_features = create_tfidf_matrix(
    stemmed_texts)

# CSV olarak kaydet
lemmatized_tfidf_df.to_csv(os.path.join(output_dir, "tfidf_lemmatized.csv"), index=False)
stemmed_tfidf_df.to_csv(os.path.join(output_dir, "tfidf_stemmed.csv"), index=False)

print(f"Lemmatized TF-IDF DataFrame kaydedildi. Boyut: {lemmatized_tfidf_df.shape}")
print(f"Stemmed TF-IDF DataFrame kaydedildi. Boyut: {stemmed_tfidf_df.shape}")

# İlk satırları göster
print("\nLemmatized TF-IDF matrisinin ilk 5 satırı:")
print(lemmatized_tfidf_df.head())

print("\nStemmed TF-IDF matrisinin ilk 5 satırı:")
print(stemmed_tfidf_df.head())

# İlk cümle için TF-IDF skorlarını analiz et
if not lemmatized_tfidf_df.empty:
    print("\nLemmatized veride ilk satırda en yüksek TF-IDF skoruna sahip 10 kelime:")
    first_sentence_vector = lemmatized_tfidf_df.iloc[0]
    top_10_words = first_sentence_vector.sort_values(ascending=False).head(10)
    print(top_10_words)

if not stemmed_tfidf_df.empty:
    print("\nStemmed veride ilk satırda en yüksek TF-IDF skoruna sahip 10 kelime:")
    first_sentence_vector = stemmed_tfidf_df.iloc[0]
    top_10_words = first_sentence_vector.sort_values(ascending=False).head(10)
    print(top_10_words)


# Doküman benzerliği analiz fonksiyonu
def analyze_document_similarity(tfidf_matrix, top_n=5):
    """
    Dokümanlararası benzerliği analiz eder ve en benzer doküman çiftlerini döndürür
    """
    # Tüm dokümanlar arası kosinüs benzerliğini hesapla
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Benzerlik matrisi DataFrame'e çevir
    similarity_df = pd.DataFrame(cosine_sim)

    # En benzer doküman çiftlerini bul
    sim_pairs = []
    for i in range(len(similarity_df)):
        # Kendisi hariç en benzer dokümanı bul
        most_similar = similarity_df.iloc[i].drop(i).nlargest(top_n)
        for idx, sim_score in most_similar.items():
            sim_pairs.append((i, idx, sim_score))

    # Benzerlik skoruna göre sırala
    sim_pairs.sort(key=lambda x: x[2], reverse=True)

    return sim_pairs[:top_n]


# TF-IDF vektörlerinden heatmap görselleştirme
def plot_tfidf_heatmap(tfidf_df, title, filename, sample_size=20, top_terms=30):
    """
    TF-IDF vektörlerinin heatmap görselleştirmesini oluşturur
    """
    # Örnek dökümanları ve en yüksek TF-IDF değerlerine sahip terimleri seç
    if len(tfidf_df) > sample_size:
        sample_df = tfidf_df.sample(sample_size, random_state=42)
    else:
        sample_df = tfidf_df

    # Sütunlardaki ortalama değerlere göre en önemli terimleri bul
    top_terms_idx = tfidf_df.mean().nlargest(top_terms).index
    sample_df = sample_df[top_terms_idx]

    plt.figure(figsize=(16, 10))
    sns.heatmap(sample_df, cmap='viridis', xticklabels=True, yticklabels=True)
    plt.title(f"{title} - Top {top_terms} Terimler", fontsize=14)
    plt.xticks(fontsize=8, rotation=90)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    output_path = os.path.join(plots_dir, filename)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"TF-IDF heatmap görselleştirme kaydedildi: {output_path}")


# Kelime frekansı dağılımı görselleştirme
def plot_term_frequency(tfidf_df, title, filename, top_n=30):
    """
    En yüksek ortalama TF-IDF değerine sahip terimleri görselleştirir
    """
    # Terimlerin ortalama TF-IDF değerlerini hesapla
    mean_tfidf = tfidf_df.mean().sort_values(ascending=False)
    top_terms = mean_tfidf.head(top_n)

    plt.figure(figsize=(14, 8))
    sns.barplot(x=top_terms.values, y=top_terms.index)
    plt.title(f"{title} - En Yüksek Ortalama TF-IDF Değerine Sahip {top_n} Terim", fontsize=14)
    plt.xlabel('Ortalama TF-IDF Değeri', fontsize=12)
    plt.ylabel('Terim', fontsize=12)
    plt.tight_layout()

    output_path = os.path.join(plots_dir, filename)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Terim frekansı grafiği kaydedildi: {output_path}")


# Kelime benzerliği analizi
def analyze_word_similarity(tfidf_matrix, feature_names, target_words, top_n=10):
    """
    Belirli kelimelere en benzer kelimeleri bulur
    """
    results = {}

    for target_word in target_words:
        if target_word not in feature_names:
            print(f"'{target_word}' kelimesi veri setinde bulunamadı.")
            continue

        # Hedef kelimenin indeksini bul
        target_index = feature_names.tolist().index(target_word)

        # Hedef kelimenin vektörünü al
        target_vector = tfidf_matrix[:, target_index].toarray()

        # Tüm kelimelerin vektörlerini al
        tfidf_vectors = tfidf_matrix.toarray()

        # Kosinüs benzerliğini hesapla
        similarities = cosine_similarity(target_vector.T, tfidf_vectors.T)
        similarities = similarities.flatten()

        # En benzer kelimeleri bul (hedef kelime dahil)
        top_indices = similarities.argsort()[-(top_n + 1):][::-1]

        # Sonuçları kaydet
        word_similarities = []
        for i, index in enumerate(top_indices):
            if i == 0 and feature_names[index] == target_word:
                continue  # Kelimenin kendisini atla
            word_similarities.append((feature_names[index], similarities[index]))

        results[target_word] = word_similarities

    return results


# Görselleştirmeleri oluştur
print("\nTF-IDF heatmap görselleştirmeleri oluşturuluyor...")
plot_tfidf_heatmap(lemmatized_tfidf_df, "Lemmatized TF-IDF Vektörleri", "lemmatized_tfidf_heatmap.png")
plot_tfidf_heatmap(stemmed_tfidf_df, "Stemmed TF-IDF Vektörleri", "stemmed_tfidf_heatmap.png")

print("\nTerim frekansı grafikleri oluşturuluyor...")
plot_term_frequency(lemmatized_tfidf_df, "Lemmatized", "lemmatized_top_terms.png")
plot_term_frequency(stemmed_tfidf_df, "Stemmed", "stemmed_top_terms.png")

# Doküman benzerliği analizi
print("\nDokümanlararası benzerlik analizi yapılıyor...")
lemmatized_sim_pairs = analyze_document_similarity(lemmatized_tfidf_matrix)
stemmed_sim_pairs = analyze_document_similarity(stemmed_tfidf_matrix)

print("\nLemmatized veride en benzer doküman çiftleri:")
for i, (doc1, doc2, score) in enumerate(lemmatized_sim_pairs[:5]):
    print(f"{i + 1}. Doküman {doc1} ve Doküman {doc2}: Benzerlik skoru = {score:.4f}")

print("\nStemmed veride en benzer doküman çiftleri:")
for i, (doc1, doc2, score) in enumerate(stemmed_sim_pairs[:5]):
    print(f"{i + 1}. Doküman {doc1} ve Doküman {doc2}: Benzerlik skoru = {score:.4f}")

# Türk Medeni Kanunu için önemli hukuki terimler
legal_terms = ['hak', 'kanun', 'hukuk', 'hakim', 'miras', 'taşınmaz', 'sorumluluk', 'borç', 'evlilik', 'boşanma']

# Lemmatized veri için analiz
print("\nLemmatized veri için hukuki terim analizi:")
lemmatized_legal_terms = [term for term in legal_terms if term in lemmatized_features]
if lemmatized_legal_terms:
    print(f"Bulunan hukuki terimler: {lemmatized_legal_terms}")
    lemmatized_word_similarities = analyze_word_similarity(lemmatized_tfidf_matrix, lemmatized_features,
                                                           lemmatized_legal_terms)

    for term, similarities in lemmatized_word_similarities.items():
        print(f"\n'{term}' kelimesine en benzer kelimeler:")
        for word, sim_score in similarities[:5]:
            print(f"  {word}: {sim_score:.4f}")
else:
    print("Belirtilen hukuki terimler lemmatized veri setinde bulunamadı.")

# Stemmed veri için analiz
print("\nStemmed veri için hukuki terim analizi:")
stemmed_legal_terms = [term for term in legal_terms if term in stemmed_features]
if stemmed_legal_terms:
    print(f"Bulunan hukuki terimler: {stemmed_legal_terms}")
    stemmed_word_similarities = analyze_word_similarity(stemmed_tfidf_matrix, stemmed_features, stemmed_legal_terms)

    for term, similarities in stemmed_word_similarities.items():
        print(f"\n'{term}' kelimesine en benzer kelimeler:")
        for word, sim_score in similarities[:5]:
            print(f"  {word}: {sim_score:.4f}")
else:
    print("Belirtilen hukuki terimler stemmed veri setinde bulunamadı.")

# Processed veriden özellik çıkarma ve ek analiz
processed_path = os.path.join(output_dir, "processed_turk_medeni_kanunu.csv")
if os.path.exists(processed_path):
    print("\nİşlenmiş Türk Medeni Kanunu verisi analiz ediliyor...")
    processed_df = pd.read_csv(processed_path)

    # Madde numaralarını ve başlıklarını kontrol et
    if 'madde_no' in processed_df.columns and 'madde_baslik' in processed_df.columns:
        print(f"Toplam madde sayısı: {len(processed_df)}")

        # Metinleri vektörleştir - raw_text veya normalized_text sütununu kullan
        text_column = 'normalized_text' if 'normalized_text' in processed_df.columns else 'raw_text'
        processed_df[text_column] = processed_df[text_column].fillna('')

        print(f"Madde metinleri TF-IDF vektörleştiriliyor...")
        madde_texts = processed_df[text_column].tolist()
        madde_tfidf_df, madde_tfidf_matrix, madde_vectorizer, madde_features = create_tfidf_matrix(madde_texts)

        # Madde benzerliği analizi
        madde_similarity = cosine_similarity(madde_tfidf_matrix)

        # En benzer maddeleri bul
        print("\nEn benzer madde çiftleri:")
        for i in range(min(5, len(processed_df))):
            # Kendisi hariç en benzer maddeyi bul
            most_similar_idx = madde_similarity[i].argsort()[-2]  # -1 kendisi olur
            similarity_score = madde_similarity[i][most_similar_idx]

            madde1 = processed_df.iloc[i]['madde_no']
            madde2 = processed_df.iloc[most_similar_idx]['madde_no']
            baslik1 = processed_df.iloc[i]['madde_baslik']
            baslik2 = processed_df.iloc[most_similar_idx]['madde_baslik']

            print(f"Madde {madde1} ({baslik1}) ve Madde {madde2} ({baslik2}): {similarity_score:.4f}")

        # Kategorik analiz - Madde başlıklarına göre
        if len(processed_df['madde_baslik'].unique()) < 100:  # Fazla kategori yoksa
            # Madde başlıklarına göre gruplama
            baslik_groups = processed_df.groupby('madde_baslik')[text_column].apply(' '.join).reset_index()

            if len(baslik_groups) > 1:  # En az 2 farklı başlık varsa
                # Başlık metinlerini vektörleştir
                baslik_texts = baslik_groups[text_column].tolist()
                baslik_tfidf_df, baslik_tfidf_matrix, _, _ = create_tfidf_matrix(baslik_texts)

                # Başlıklar arası benzerlik matrisi
                baslik_similarity = cosine_similarity(baslik_tfidf_matrix)

                # Başlıklar arası benzerlik görselleştirmesi
                if len(baslik_groups) <= 20:  # Görselleştirme için uygun boyut
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(baslik_similarity,
                                annot=True,
                                cmap='viridis',
                                xticklabels=baslik_groups['madde_baslik'],
                                yticklabels=baslik_groups['madde_baslik'])
                    plt.title("Madde Başlıkları Arası Benzerlik Matrisi", fontsize=14)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, "madde_baslik_similarity_heatmap.png"), dpi=300)
                    plt.close()
                    print(f"Madde başlıkları arası benzerlik matrisi kaydedildi.")

print("\nTF-IDF analizi tamamlandı!")