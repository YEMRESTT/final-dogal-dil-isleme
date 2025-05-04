import os
import re
import pandas as pd
import PyPDF2
from collections import Counter
import logging
from zemberek import (
    TurkishMorphology,
    TurkishTokenizer,
    TurkishSentenceExtractor,
    TurkishSentenceNormalizer,
    TurkishSpellChecker
)

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Zemberek için gerekli nesneleri başlat
def initialize_zemberek():
    """Zemberek modüllerini başlatır"""
    logger.info("Zemberek modülleri başlatılıyor...")
    morphology = TurkishMorphology.create_with_defaults()
    tokenizer = TurkishTokenizer.DEFAULT
    sentence_extractor = TurkishSentenceExtractor()
    normalizer = TurkishSentenceNormalizer(morphology)
    spell_checker = TurkishSpellChecker(morphology)

    logger.info("Zemberek modülleri başarıyla başlatıldı")
    return {
        "morphology": morphology,
        "tokenizer": tokenizer,
        "sentence_extractor": sentence_extractor,
        "normalizer": normalizer,
        "spell_checker": spell_checker
    }


def extract_text_from_pdf(pdf_path):
    """PDF dosyasından metin çıkarma"""
    logger.info(f"PDF dosyası okunuyor: {pdf_path}")
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            logger.info(f"Toplam {num_pages} sayfa bulundu")

            for page_num in range(num_pages):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                text += page_text + "\n\n"

        logger.info(f"PDF metni başarıyla çıkarıldı, toplam {len(text)} karakter")
        # Kısa bir örnek göster
        logger.info(f"Örnek metin (ilk 200 karakter): {text[:200]}")
        return text
    except Exception as e:
        logger.error(f"PDF okuma hatası: {str(e)}")
        return ""


def clean_text(text):
    """Metni temizleme"""
    logger.info("Metin temizleniyor...")

    # Alt tireler ve satır sonları
    text = re.sub(r"_+", " ", text)

    # Birden fazla boşlukları tek boşluğa dönüştür
    text = re.sub(r"\s+", " ", text)

    # Madde başlıklarını düzgün format
    text = re.sub(r"(MADDE\s+\d+\s*-)(\w)", r"\1 \2", text)

    logger.info("Metin temizleme tamamlandı")
    return text.strip()


def get_law_sections(text):
    """Metni kanun maddelerine ayırma"""
    logger.info("Metin maddelere ayrılıyor...")

    # Madde başlıklarını bul - Farklı regex seçenekleri deneyelim
    patterns = [
        r"(MADDE\s+\d+\s*-.*?)(?=(MADDE\s+\d+\s*-)|\Z)",  # Standart MADDE formatı
        r"(Madde\s+\d+\s*-.*?)(?=(Madde\s+\d+\s*-)|\Z)",  # Küçük harf Madde formatı
        r"(MADDE\s+\d+\s*\(.*?\).*?)(?=(MADDE\s+\d+)|\Z)",  # Parantezli MADDE formatı
        r"(Madde\s+\d+\s*\(.*?\).*?)(?=(Madde\s+\d+)|\Z)",  # Parantezli Madde formatı
        r"(MADDE\s+\d+.*?)(?=(MADDE\s+\d+)|\Z)",  # Basit MADDE formatı
    ]

    sections = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            logger.info(f"'{pattern}' formatıyla {len(matches)} madde bulundu")
            for match in matches:
                if isinstance(match, tuple):
                    sections.append(match[0].strip())
                else:
                    sections.append(match.strip())
            break  # İlk başarılı desen bulunduysa diğerlerini deneme

    # Eğer madde bulunamadıysa, başka bir bölümleme yöntemi dene
    if not sections:
        logger.warning("Hiçbir madde formatı bulunamadı, bölüm başlıklarını deniyorum...")
        patterns = [
            r"([A-Z]\.\s+[A-Za-zçğıöşüÇĞİÖŞÜ\s]+\n.*?)(?=([A-Z]\.\s+[A-Za-zçğıöşüÇĞİÖŞÜ\s]+\n)|\Z)",
            r"(BİRİNCİ\s+[A-Z]+.*?)(?=(İKİNCİ\s+[A-Z]+|ÜÇÜNCÜ\s+[A-Z]+|DÖRDÜNCÜ\s+[A-Z]+)|\Z)",
            r"(Bölüm\s+\d+:.*?)(?=(Bölüm\s+\d+:)|\Z)"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                logger.info(f"Alternatif format ile {len(matches)} bölüm bulundu")
                for match in matches:
                    if isinstance(match, tuple):
                        sections.append(match[0].strip())
                    else:
                        sections.append(match.strip())
                break

    # Hiçbir şey bulunamadıysa, metni paragraf paragraf böl
    if not sections:
        logger.warning("Hiçbir bölüm formatı bulunamadı. Son çare: metni paragraf paragraf bölüyorum...")
        # İki veya daha fazla yeni satır ile bölme
        paragraphs = re.split(r'\n\s*\n', text)
        # Boş olmayan paragrafları al
        sections = [p.strip() for p in paragraphs if p.strip()]
        logger.info(f"Paragraf bölme ile {len(sections)} bölüm oluşturuldu")

    logger.info(f"Toplam {len(sections)} madde/bölüm bulundu")

    # Birkaç örnek göster
    if sections:
        for i in range(min(3, len(sections))):
            logger.info(f"Bölüm {i + 1} örneği: {sections[i][:100]}...")

    return sections


def process_with_zemberek(text, zemberek_tools):
    """Zemberek ile metin işleme"""
    logger.info("Zemberek ile metin işleniyor...")

    # Cümlelere ayır
    sentences = zemberek_tools["sentence_extractor"].from_paragraph(text)

    # Cümleleri normalize et
    normalized_sentences = [zemberek_tools["normalizer"].normalize(sentence) for sentence in sentences]

    # Tokenize et
    tokens = []
    for sentence in normalized_sentences:
        sentence_tokens = zemberek_tools["tokenizer"].tokenize(sentence)
        tokens.extend([token.content for token in sentence_tokens if token.type_.name == "Word"])

    # Morfolojik analiz yap
    analyzed_tokens = []
    try:
        for token in tokens:
            analysis = zemberek_tools["morphology"].analyze(token)
            if analysis:
                # get_stems() yerine get_stem() kullan - DÜZELTME BURADA
                roots = [result.get_stem() for result in analysis]
                if roots:
                    analyzed_tokens.append(roots[0])
    except Exception as e:
        logger.error(f"Morfolojik analiz hatası: {str(e)}")
        # En azından tokenları kullanabiliriz
        analyzed_tokens = tokens[:] if not analyzed_tokens else analyzed_tokens

    logger.info(f"Zemberek işleme tamamlandı: {len(tokens)} token, {len(analyzed_tokens)} analiz edilmiş kök")
    return {
        "sentences": sentences,
        "normalized_sentences": normalized_sentences,
        "tokens": tokens,
        "analyzed_tokens": analyzed_tokens
    }


def process_pdf(pdf_path, output_dir=".", section_limit=None):
    """PDF işleme ana fonksiyonu"""
    # Çıktı dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)

    # Zemberek'i başlat
    zemberek_tools = initialize_zemberek()

    # PDF'den metni çıkar
    raw_text = extract_text_from_pdf(pdf_path)

    # PDF metin çıkarma başarısız olduysa
    if not raw_text:
        logger.error("PDF'den metin çıkarılamadı. Lütfen dosya yolunu kontrol edin.")
        return None

    # Metni temizle
    cleaned_text = clean_text(raw_text)

    # Metin temizleme sonrası örnek göster
    logger.info(f"Temizlenmiş metin örneği: {cleaned_text[:200]}...")

    # Bölümlere ayır
    sections = get_law_sections(cleaned_text)

    # Hiç bölüm bulunamadıysa hata ver
    if not sections:
        logger.error("Metin bölümlere ayrılamadı, işlem durduruluyor.")
        return None

    # Eğer section_limit belirtilmişse, sadece ilk birkaç bölümü al
    if section_limit and section_limit > 0:
        sections = sections[:section_limit]
        logger.info(f"Section limit uygulandı, {len(sections)} bölüm işlenecek.")

    # Bölümleri işle
    processed_sections = []

    for i, section in enumerate(sections):
        logger.info(f"Bölüm {i + 1}/{len(sections)} işleniyor...")

        # Madde numarasını ve başlığını ayıkla
        match = re.search(r"(MADDE|Madde)\s+(\d+)\s*-\s*([A-Za-zçğıöşüÇĞİÖŞÜ0-9\s]+)?", section)

        if match:
            madde_no = match.group(2)
            madde_baslik = match.group(3).strip() if match.group(3) else ""
        else:
            madde_no = f"Bölüm-{i + 1}"
            madde_baslik = ""

        # Zemberek ile işleme
        zemberek_result = process_with_zemberek(section, zemberek_tools)

        # Sonuçları kaydet
        processed_sections.append({
            "madde_no": madde_no,
            "madde_baslik": madde_baslik,
            "raw_text": section,
            "normalized_text": " ".join(zemberek_result["normalized_sentences"]),
            "tokens": zemberek_result["tokens"],
            "token_count": len(zemberek_result["tokens"]),
            "unique_tokens": list(set(zemberek_result["tokens"])),
            "unique_token_count": len(set(zemberek_result["tokens"])),
            "roots": zemberek_result["analyzed_tokens"],
            "sentence_count": len(zemberek_result["sentences"])
        })

    # Eğer hiçbir bölüm işlenmediyse hata ver
    if not processed_sections:
        logger.error("Hiçbir bölüm işlenemedi, işlem durduruluyor.")
        return None

    # DataFrame oluştur
    df = pd.DataFrame(processed_sections)

    # İşlenmiş dosyayı kaydet
    output_csv = os.path.join(output_dir, "processed_turk_medeni_kanunu.csv")
    df.to_csv(output_csv, index=False)
    logger.info(f"[✓] İşleme tamamlandı. Veriler '{output_csv}' dosyasına kaydedildi.")

    # Token frekans analizi
    all_tokens = []
    for section in processed_sections:
        all_tokens.extend(section["tokens"])

    token_freq = Counter(all_tokens)

    # Kök frekans analizi
    all_roots = []
    for section in processed_sections:
        all_roots.extend(section["roots"])

    root_freq = Counter(all_roots)

    # Sık kullanılan kelimeleri kaydet
    freq_df = pd.DataFrame(token_freq.most_common(), columns=["token", "frequency"])
    freq_csv = os.path.join(output_dir, "token_frequency.csv")
    freq_df.to_csv(freq_csv, index=False)
    logger.info(f"[✓] Token frekansları '{freq_csv}' dosyasına kaydedildi.")

    # Sık kullanılan kökleri kaydet
    root_freq_df = pd.DataFrame(root_freq.most_common(), columns=["root", "frequency"])
    root_freq_csv = os.path.join(output_dir, "root_frequency.csv")
    root_freq_df.to_csv(root_freq_csv, index=False)
    logger.info(f"[✓] Kök frekansları '{root_freq_csv}' dosyasına kaydedildi.")

    # Bölümlerin özet analizi
    # df'in boş olmadığından ve gerekli sütunlara sahip olduğundan emin ol
    if not df.empty and all(col in df.columns for col in
                            ["madde_no", "madde_baslik", "token_count", "unique_token_count", "sentence_count"]):
        section_summary = df[["madde_no", "madde_baslik", "token_count", "unique_token_count", "sentence_count"]]
        summary_csv = os.path.join(output_dir, "section_summary.csv")
        section_summary.to_csv(summary_csv, index=False)
        logger.info(f"[✓] Bölüm özetleri '{summary_csv}' dosyasına kaydedildi.")
    else:
        logger.warning("DataFrame boş veya gerekli sütunlar eksik, bölüm özeti oluşturulamadı.")

    # İstatistikleri yazdır
    print("\nVeri İstatistikleri:")
    print(f"Toplam bölüm sayısı: {len(processed_sections)}")
    print(f"Toplam token sayısı: {len(all_tokens)}")
    print(f"Benzersiz token sayısı: {len(token_freq)}")
    print(f"Toplam kök sayısı: {len(all_roots)}")
    print(f"Benzersiz kök sayısı: {len(root_freq)}")

    if processed_sections:
        print(f"Bölüm başına ortalama token: {len(all_tokens) / len(processed_sections):.1f}")

    print(f"En sık kullanılan 20 token:")
    for token, freq in token_freq.most_common(20):
        print(f"  {token}: {freq}")

    print(f"\nEn sık kullanılan 20 kök:")
    for root, freq in root_freq.most_common(20):
        print(f"  {root}: {freq}")

    return {
        "dataframe": df,
        "token_frequency": freq_df,
        "root_frequency": root_freq_df,
        "section_summary": section_summary if not df.empty else None
    }


if __name__ == "__main__":
    # Kullanım örneği:
    pdf_path = "Turk_Medeni_Kanunu.pdf"
    output_dir = "processed_data"

    # PDF dosyasının varlığını kontrol et
    if not os.path.exists(pdf_path):
        logger.error(f"PDF dosyası bulunamadı: {pdf_path}")
        print(f"HATA: '{pdf_path}' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
    else:
        # PDF'yi işle
        results = process_pdf(pdf_path, output_dir, section_limit=None)

        if results:
            print("\nİşlem başarıyla tamamlandı!")
        else:
            print("\nİşlem sırasında hatalar oluştu. Lütfen log çıktılarını kontrol edin.")