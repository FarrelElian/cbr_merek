import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=False)
print("Pustaka dasar dan dependensi NLTK lokal siap!")

import os
import json
import numpy as np
import pandas as pd

# Arahkan CWD ke root proyek (satu level di atas folder notebooks/)
# agar seluruh path relatif ke data/ dapat ditemukan dengan benar.
_script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_script_dir) == "notebooks":
    os.chdir(os.path.dirname(_script_dir))

BASE_PATH = os.getcwd()
DATA_PATH = os.path.join(BASE_PATH, "data")
PROCESSED_PATH = os.path.join(DATA_PATH, "processed")
EVAL_PATH = os.path.join(DATA_PATH, "eval")

# Pastikan folder Eval sudah dibuat untuk menyimpan visualisasi HTML nanti
os.makedirs(EVAL_PATH, exist_ok=True)
print(f"Direktori kerja aktif: {BASE_PATH}")

# %% [markdown]
# ## **Tahap 2: Load Case Base & Inisialisasi Model Search**
# *Memuat database kasus terstruktur (`cases.json`) secara lokal dari harddisk Anda.*

# %%
# 1. Muat database kasus terstruktur
json_file_path = os.path.join(PROCESSED_PATH, "cases.json")
with open(json_file_path, "r", encoding="utf-8") as f:
    cases_list = json.load(f)

# 2. Siapkan TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [case["text_full"] for case in cases_list]
indonesian_stopwords = [
    'yang', 'untuk', 'dan', 'dalam', 'dari', 'pada', 'with', 'dengan', 'bahwa', 
    'tersebut', 'oleh', 'atas', 'gugatan', 'sengketa', 'merek', 'perkara', 'putusan', 'dituduh', 'adalah', 'kepada', 'atau', 'ini', 'ia', 'telah', 'sebagai'
]
tfidf_vectorizer = TfidfVectorizer(stop_words=indonesian_stopwords)
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# 3. Siapkan IndoBERT Encoder
import torch
from transformers import AutoTokenizer, AutoModel

# Deteksi ketersediaan CUDA (GPU) lokal atau gunakan CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan komputasi lokal berbasis: {device}")

model_name = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name).to(device)

def get_bert_embedding(text, max_length=512):
    """Menghasilkan representasi vektor semantik menggunakan IndoBERT"""
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()[0]

# Ekstrak ulang embedding semua kasus lama untuk proses retrieval berbasis semantik
print("Mengekstrak kembali representasi semantik IndoBERT untuk basis kasus (Case-Base)...")
bert_embeddings = [get_bert_embedding(case["text_full"]) for case in cases_list]
bert_embeddings_matrix = np.array(bert_embeddings)

print("Inisialisasi database dan representasi vektor ganda sukses!")

# %% [markdown]
# ## **Tahap 3: Pemetaan Kelas Keputusan Biner**
# *Mendefinisikan kelas putusan biner untuk seluruh kasus hukum lama (DIKABULKAN vs DITOLAK).*

# %%
# Memetakan case_id ke putusan biner (DIKABULKAN atau DITOLAK)
case_classes = {}
for case in cases_list:
    c_id = case["case_id"]
    
    # Menggunakan .get() dengan default string kosong jika 'argumen_hukum' tidak dibentuk
    argumen_text = case.get("argumen_hukum", "").lower()
    
    # Kita cek jika teksnya kosong, kita kategorikan secara default ke DITOLAK 
    # atau sesuaikan dengan status asli Kasus 2 tersebut
    if argumen_text == "":
        case_classes[c_id] = "DITOLAK"  # Mengantisipasi Kasus 2 yang tidak memiliki argumen hukum
    elif "kabul" in argumen_text:
        case_classes[c_id] = "DIKABULKAN"
    else:
        case_classes[c_id] = "DITOLAK"

print("Pemetaan keputusan hukum biner aman dari KeyError!")

# %% [markdown]
# ## **Tahap 4: Pembuatan Fungsi Pembungkus Probabilitas (LIME Wrapper)**
# *Fungsi pembungkus untuk mengubah pencocokan berbasis Cosine Similarity menjadi distribusi probabilitas.*

# %%
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_for_xai(query: str, k: int = 5, method: str = 'tfidf'):
    """Fungsi pembantu retrieve khusus untuk modul XAI"""
    cleaned_query = query.lower().strip()
    
    if method == 'tfidf':
        query_vector = tfidf_vectorizer.transform([cleaned_query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    elif method == 'bert':
        query_vector = get_bert_embedding(cleaned_query).reshape(1, -1)
        similarities = cosine_similarity(query_vector, bert_embeddings_matrix).flatten()
        
    top_indices = np.argsort(similarities)[::-1][:k]
    
    results = []
    for idx in top_indices:
        results.append({
            "case_id": cases_list[idx]["case_id"],
            "similarity_score": float(similarities[idx])
        })
    return results

def cbr_predict_proba(texts, method='tfidf', k=3):
    """
    Wrapper fungsi prediksi CBR agar kompatibel dengan LIME.
    Menerima list teks kueri, mengembalikan array numpy berdimensi (N, 2)
    yang mewakili probabilitas kelas [DITOLAK, DIKABULKAN].
    """
    probabilities = []
    
    for text in texts:
        # Cari Top-K kasus terdekat
        top_k = retrieve_for_xai(text, k=k, method=method)
        
        # Akumulasikan skor kemiripan sebagai bobot suara untuk masing-masing kelas
        weights = {"DITOLAK": 0.01, "DIKABULKAN": 0.01}
        
        for case in top_k:
            cls = case_classes[case["case_id"]]
            weights[cls] += case["similarity_score"]
            
        # Normalisasi bobot menjadi nilai probabilitas total = 1.0
        total_weight = sum(weights.values())
        prob_ditolak = weights["DITOLAK"] / total_weight
        prob_dikabulkan = weights["DIKABULKAN"] / total_weight
        
        probabilities.append([prob_ditolak, prob_dikabulkan])
        
    return np.array(probabilities)

print("CBR Predict Proba Wrapper untuk LIME berhasil dideklarasikan!")

# %% [markdown]
# ## **Tahap 5: Inisialisasi LIME Text Explainer**
# *Mempersiapkan objek penjelas LIME.*

# %%
from lime.lime_text import LimeTextExplainer

class_names = ["DITOLAK", "DIKABULKAN"]
explainer = LimeTextExplainer(class_names=class_names)

print("LIME Text Explainer sukses diinisialisasi!")

# %% [markdown]
# ## **Tahap 6: Eksekusi LIME & Ekspor ke File HTML**
# *Di VS Code, selain merender di notebook interaktif, kita akan menyimpan penjelasan visual LIME menjadi sebuah file HTML lokal.*

# %%
# Kueri kasus baru yang akan diuji
test_query = "Gugatan sengketa merek rokok WIN dituduh tidak aktif dipasarkan selama tiga tahun berturut-turut."

print("=== MEMULAI ANALISIS EXPLAINABLE AI (LIME) ===")
print(f"Kueri Uji: '{test_query}'\n")

# Menggunakan pencarian TF-IDF untuk analisis LIME
predict_fn = lambda x: cbr_predict_proba(x, method='tfidf', k=3)

# Hasilkan eksplanasi dari LIME
exp = explainer.explain_instance(
    test_query, 
    predict_fn, 
    num_features=6, 
    num_samples=500, 
    labels=(1,) 
)

# 1. Tampilkan hasil penjelasan LIME dalam format teks di terminal VS Code
print("KONTRIBUSI KATA TERHADAP PUTUSAN 'DIKABULKAN' (Hasil Analisis Terminal):")
explanation_list = exp.as_list(label=1)
for word, score in explanation_list:
    influence = "MENDUKUNG Kemenangan" if score > 0 else "MENDORONG Kekalahan (Noise)"
    print(f"  -> Kata: '{word:<12}' | Skor Kontribusi: {score:+.4f} | Pengaruh: {influence}")

# 2. Ekspor hasil visualisasi LIME menjadi File HTML lokal
# Ini sangat krusial di VS Code agar Anda bisa membuka visualisasi grafisnya secara independen!
html_output_path = os.path.join(EVAL_PATH, "lime_explanation.html")
exp.save_to_file(html_output_path)

print("\n" + "="*80)
print(f"SUKSES! File visualisasi interaktif LIME disimpan di: {html_output_path}")
print("Silakan klik kanan file tersebut di VS Code dan pilih 'Open with Live Server' atau buka langsung di Google Chrome!")
print("="*80)

# Jika Anda menjalankan file ini menggunakan ekstensi 'Interactive Window' (Jupyter) di VS Code,
# baris di bawah ini akan tetap menampilkan visualisasi interaktif langsung di panel VS Code Anda.
try:
    exp.show_in_notebook(text=True)
except Exception:
    pass

# %% [markdown]
# ## **Tahap 7: Pendekatan Alternatif - Atribusi Fitur Leksikal Eksak**
# *Atribusi kemiripan leksikal murni (Dot-Product TF-IDF) yang berjalan sangat cepat dan andal secara lokal.*

# %%
def explain_lexical_similarity(query: str, retrieved_case_id: str):
    """Menghitung kata mana saja yang berkontribusi nyata terhadap skor Cosine Similarity TF-IDF"""
    doc_idx = None
    for idx, case in enumerate(cases_list):
        if case["case_id"] == retrieved_case_id:
            doc_idx = idx
            break
            
    if doc_idx is None:
        print("Kasus rujukan tidak ditemukan!")
        return
        
    query_vector = tfidf_vectorizer.transform([query.lower().strip()]).toarray()[0]
    doc_vector = tfidf_matrix[doc_idx].toarray()[0]
    contribution_vector = query_vector * doc_vector
    
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    word_contributions = []
    for word_idx, score in enumerate(contribution_vector):
        if score > 0:
            word_contributions.append((feature_names[word_idx], score))
            
    word_contributions = sorted(word_contributions, key=lambda x: x[1], reverse=True)
    
    print(f"\n=== ANALISIS COGNITIVE ATRIBUSI LEKSIKAL (TF-IDF EXACT MATCH) ===")
    print(f"Kueri Kasus Baru   : '{query}'")
    print(f"Kasus Rujukan Terdekat : ID {retrieved_case_id} (Putusan No: {cases_list[doc_idx]['no_perkara']})")
    print(f"Daftar Kata yang Cocok & Berkontribusi terhadap Cosine Similarity:")
    
    if not word_contributions:
        print("  -> Tidak ada kata yang cocok secara leksikal (Sistem mengandalkan sinonim/semantik).")
    else:
        for word, score in word_contributions[:6]:
            print(f"  -> Kata yang Cocok: '{word:<12}' | Skor Kontribusi Vektor: {score:.4f}")

# Jalankan pengujian atribusi eksak untuk Kasus Rujukan ID case_01
explain_lexical_similarity(test_query, retrieved_case_id="case_01")