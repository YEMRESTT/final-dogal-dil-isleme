import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import ast


# CSV dosyalarını okuyalım
def read_csv_files():
    print("CSV dosyaları okunuyor...")
    lemmatized_data = pd.read_csv('processed_data/lemmatized_data.csv')
    stemmed_data = pd.read_csv('processed_data/stemmed_data.csv')
    processed_kanun = pd.read_csv('processed_data/processed_turk_medeni_kanunu.csv')
    root_frequency = pd.read_csv('processed_data/root_frequency.csv')
    section_summary = pd.read_csv('processed_data/section_summary.csv')
    token_frequency = pd.read_csv('processed_data/token_frequency.csv')

    return {
        'lemmatized_data': lemmatized_data,
        'stemmed_data': stemmed_data,
        'processed_kanun': processed_kanun,
        'root_frequency': root_frequency,
        'section_summary': section_summary,
        'token_frequency': token_frequency
    }


# Veri setini inceleyip içeriğini anlayalım
def explore_data(dataframes):
    for name, df in dataframes.items():
        print(f"\n{name} veri seti:")
        print(f"Boyut: {df.shape}")
        print("İlk 5 satır:")
        print(df.head())
        print("Sütunlar:", df.columns.tolist())


# Eğitim verisi hazırlama
def prepare_training_data(dataframes):
    print("\nEğitim verileri hazırlanıyor...")
    lemmatized_data = dataframes['lemmatized_data']
    stemmed_data = dataframes['stemmed_data']

    tokenized_corpus_lemmatized = []
    tokenized_corpus_stemmed = []

    try:
        if 'tokens' in lemmatized_data.columns:
            for tokens in lemmatized_data['tokens']:
                try:
                    if isinstance(tokens, str):
                        tokenized_corpus_lemmatized.append(ast.literal_eval(tokens))
                    else:
                        tokenized_corpus_lemmatized.append(tokens)
                except (ValueError, SyntaxError):
                    tokenized_corpus_lemmatized.append(tokens.split())

        elif 'lemmatized_text' in lemmatized_data.columns:
            for text in lemmatized_data['lemmatized_text']:
                if isinstance(text, str):
                    tokenized_corpus_lemmatized.append(text.split())

        if 'tokens' in stemmed_data.columns:
            for tokens in stemmed_data['tokens']:
                try:
                    if isinstance(tokens, str):
                        tokenized_corpus_stemmed.append(ast.literal_eval(tokens))
                    else:
                        tokenized_corpus_stemmed.append(tokens)
                except (ValueError, SyntaxError):
                    tokenized_corpus_stemmed.append(tokens.split())

        elif 'stemmed_text' in stemmed_data.columns:
            for text in stemmed_data['stemmed_text']:
                if isinstance(text, str):
                    tokenized_corpus_stemmed.append(text.split())

        if len(tokenized_corpus_lemmatized) == 0:
            processed_kanun = dataframes['processed_kanun']
            if 'lemmatized_text' in processed_kanun.columns:
                for text in processed_kanun['lemmatized_text']:
                    if isinstance(text, str):
                        tokenized_corpus_lemmatized.append(text.split())

            if 'stemmed_text' in processed_kanun.columns:
                for text in processed_kanun['stemmed_text']:
                    if isinstance(text, str):
                        tokenized_corpus_stemmed.append(text.split())

    except Exception as e:
        print(f"Veri hazırlama hatası: {e}")

    print(f"Lemmatize edilmiş corpus boyutu: {len(tokenized_corpus_lemmatized)}")
    print(f"Stem edilmiş corpus boyutu: {len(tokenized_corpus_stemmed)}")

    return tokenized_corpus_lemmatized, tokenized_corpus_stemmed


# Model eğitme ve kaydetme
def train_and_save_model(corpus, params, model_name):
    print(
        f"\nModel eğitiliyor: {model_name}_{params['model_type']}_window{params['window']}_dim{params['vector_size']}")

    if corpus and len(corpus) > 0:
        try:
            model = Word2Vec(
                corpus,
                vector_size=params['vector_size'],
                window=params['window'],
                min_count=1,
                workers=4,
                sg=1 if params['model_type'] == 'skipgram' else 0
            )

            os.makedirs("models", exist_ok=True)

            model_filename = f"{model_name}_{params['model_type']}_window{params['window']}_dim{params['vector_size']}.model"
            model_path = os.path.join("models", model_filename)

            model.save(model_path)
            print(f"{model_path} kaydedildi!")

            evaluate_model(model, model_name, params)

            return model
        except Exception as e:
            print(f"Model eğitim hatası: {e}")
            return None
    else:
        print("Corpus boş! Model eğitilemedi.")
        return None


# Model değerlendirme ve görselleştirme
def evaluate_model(model, model_name, params):
    try:
        vocab_size = len(model.wv.index_to_key)
        print(f"Modelin kelime dağarcığı boyutu: {vocab_size}")

        print("En sık kullanılan 10 kelime:")
        for i, word in enumerate(model.wv.index_to_key[:10]):
            print(f"{i + 1}. {word}")

        sample_words = model.wv.index_to_key[:5]
        for word in sample_words:
            try:
                similar_words = model.wv.most_similar(word, topn=5)
                print(f"\n'{word}' için en benzer 5 kelime:")
                for similar_word, similarity in similar_words:
                    print(f"  {similar_word}: {similarity:.4f}")
            except KeyError:
                print(f"'{word}' kelimesi modelde bulunamadı.")

        if vocab_size > 100:
            words = model.wv.index_to_key[:100]
        else:
            words = model.wv.index_to_key

        X = model.wv[words]
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(words) - 1))
        X_tsne = tsne.fit_transform(X)

        os.makedirs("plots/models_output", exist_ok=True)

        plt.figure(figsize=(10, 8))
        for i, word in enumerate(words):
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1])
            plt.annotate(word, (X_tsne[i, 0], X_tsne[i, 1]))

        title = f"{model_name}_{params['model_type']}_window{params['window']}_dim{params['vector_size']} t-SNE"
        plt.title(title)
        plt.tight_layout()

        plot_filename = f"{model_name}_{params['model_type']}_window{params['window']}_dim{params['vector_size']}_tsne.png"
        plot_path = os.path.join("plots/models_output", plot_filename)
        plt.savefig(plot_path)
        plt.close()

        print(f"t-SNE görselleştirmesi kaydedildi: {plot_path}")

    except Exception as e:
        print(f"Model değerlendirme hatası: {e}")


# Ana fonksiyon
def main():
    dataframes = read_csv_files()
    explore_data(dataframes)
    tokenized_corpus_lemmatized, tokenized_corpus_stemmed = prepare_training_data(dataframes)

    parameters = [
        {'model_type': 'cbow', 'window': 2, 'vector_size': 100},
        {'model_type': 'skipgram', 'window': 2, 'vector_size': 100},
        {'model_type': 'cbow', 'window': 4, 'vector_size': 100},
        {'model_type': 'skipgram', 'window': 4, 'vector_size': 100},
        {'model_type': 'cbow', 'window': 2, 'vector_size': 300},
        {'model_type': 'skipgram', 'window': 2, 'vector_size': 300},
        {'model_type': 'cbow', 'window': 4, 'vector_size': 300},
        {'model_type': 'skipgram', 'window': 4, 'vector_size': 300}
    ]

    if tokenized_corpus_lemmatized:
        print("\nLemmatize edilmiş veri seti ile model eğitimi başlatılıyor...")
        for param in parameters:
            train_and_save_model(tokenized_corpus_lemmatized, param, "lemmatized_model")
    else:
        print("Lemmatize edilmiş veri seti bulunamadı veya boş!")

    if tokenized_corpus_stemmed:
        print("\nStemlenmiş veri seti ile model eğitimi başlatılıyor...")
        for param in parameters:
            train_and_save_model(tokenized_corpus_stemmed, param, "stemmed_model")
    else:
        print("Stemlenmiş veri seti bulunamadı veya boş!")

    print("\nTüm modellerin eğitimi tamamlandı!")


if __name__ == "__main__":
    main()
