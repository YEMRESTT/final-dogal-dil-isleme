# Türk Medeni Kanunu NLP Pipeline

Bu depo, Türk Medeni Kanunu'nun (Türk Medeni Kanunu) Doğal Dil İşleme (NLP) teknikleriyle işlenmesi, analiz edilmesi ve modellenmesi için kapsamlı bir pipeline sunar. Proje; veri çıkarımı, ön işleme, tokenizasyon, kök bulma, lemmatizasyon, frekans analizi, Zipf analizi, TF-IDF ve Word2Vec model eğitimi için betikler içerir.

## İçindekiler
- [Proje Genel Bakış](#proje-genel-bakis)
- [Veri Setinin Amacı](#veri-setinin-amaci)
- [Kurulum & Gereksinimler](#kurulum--gereksinimler)
- [Adım Adım Model Oluşturma](#adim-adim-model-olusturma)
- [Kullanım Örnekleri](#kullanim-ornekleri)
- [Çıktılar](#ciktilar)
### Hariç bırakılan dosyalar
100 MB boyutunu geçtiği için aşağıdaki dosyalar hariç bırakılmıştır bu dosyaları elde etmek için [TF-IDF vektörleştirme](#3-tf-idf-vektörleştirme) adımına gidiniz. 
```
processed_data/tfidf_lemmatized.csv
processed_data/tfidf_stemmed.csv
```

## Proje Genel Bakış
Bu proje, Türk Medeni Kanunu PDF'ini işler, metni çıkarır ve temizler, bölümlere ayırır ve gelişmiş NLP teknikleri uygular. Pipeline şunları içerir:
- PDF metin çıkarımı
- Metin temizleme ve normalleştirme
- Bölüm ve madde segmentasyonu
- Tokenizasyon, kök bulma ve lemmatizasyon (Zemberek kullanılarak)
- Frekans ve Zipf analizi
- TF-IDF vektörleştirme ve görselleştirme
- Word2Vec model eğitimi ve değerlendirmesi

## Veri Setinin Amacı
İşlenmiş veri seti ve modeller şunlar için kullanılabilir:
- Türkçe hukuki metin analizi
- NLP araştırmaları ve deneyleri
- Türkçe için kelime gömme (embedding) modellerinin eğitimi ve değerlendirilmesi
- Anlamsal benzerlik ve belge kümeleme
- Hesaplamalı dilbilim alanında eğitim amaçlı

## Kurulum & Gereksinimler
### Gerekli Kütüphaneler
- Python 3.8+
- pandas
- numpy
- matplotlib
- scikit-learn
- gensim
- PyPDF2
- zemberek-python (Türkçe NLP için)

### Kurulum
Gerekli kütüphaneleri pip ile yükleyin:

```bash
pip install pandas numpy matplotlib scikit-learn gensim PyPDF2
```

Zemberek için, Türkçe NLP araçlarını kurmak için [zemberek-python kurulum rehberini](https://github.com/ahmetaa/zemberek-python) takip edin.

## Adım Adım Model Oluşturma
### 1. Veri Ön İşleme
- Türk Medeni Kanunu PDF dosyasını (`Turk_Medeni_Kanunu.pdf`) proje dizinine yerleştirin.
- Metni çıkarmak, temizlemek ve bölümlere ayırmak için ön işleme betiğini çalıştırın:

```bash
python turk_medeni_kanun_preprocessing.py
```
- Bu işlem, `processed_data/` dizininde işlenmiş CSV dosyaları oluşturacaktır.

### 2. Zipf ve Frekans Analizi
- Kelime frekans dağılımlarını görselleştirmek için Zipf analizi betiğini çalıştırın:

```bash
python zipf_analysis.py
```
- Grafikler `plots/` dizininde kaydedilecektir.

### 3. TF-IDF Vektörleştirme
- Terim önemi ve belge benzerliğini analiz etmek için şunu çalıştırın:

```bash
python tfidf_vectorizer.py
```
- Görselleştirmeler `plots/tfidf_output/` dizininde kaydedilecektir.

### 4. Word2Vec Model Eğitimi
- İşlenmiş veriler üzerinde Word2Vec modellerini eğitin:

```bash
python word2vec_train.py
```
- Modeller `models/` dizininde kaydedilir. t-SNE görselleştirmeleri `plots/models_output/` dizininde saklanır.

## Kullanım Örnekleri
- Eğitilmiş modelleri kelime benzerliği, analoji görevleri veya aşağı akış NLP uygulamalarında gömme olarak kullanın.
- Dilbilimsel veya hukuki araştırmalar için frekans ve TF-IDF çıktılarının analizini yapın.

## Çıktılar
- `processed_data/`: Temizlenmiş ve tokenleştirilmiş CSV dosyaları
- `models/`: Eğitilmiş Word2Vec modelleri
- `plots/`: Görselleştirmeler (Zipf, TF-IDF, t-SNE)

