# -*- coding: utf-8 -*-
"""
Tugas Penalaran Komputer - SIKLUS CBR (Tahap 3: Case Retrieval)
Studi Kasus: Sengketa Merek & Indikasi Geografis (UU No. 20 Tahun 2016)
Fakultas Teknik - Informatika UMM
"""

import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC

# =====================================================================
# KONFIGURASI DAN PENYEDIAAN DIREKTORI
# =====================================================================
PROCESSED_JSON_PATH = "data/processed/cases.json"
EVAL_DIR = "data/eval"
QUERIES_JSON_PATH = os.path.join(EVAL_DIR, "queries.json")

os.makedirs(EVAL_DIR, exist_ok=True)

# =====================================================================
# FUNGSI RAPI PENYEDIAAN TEKS (PREPROCESSING)
# =====================================================================
def preprocess_query(text):
    """
    Membersihkan teks pertanyaan/query sebelum ditukar menjadi representasi vektor.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s\-\/\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =====================================================================
# 1. MEMBACA PANGKALAN DATA KES (LOAD CASE BASE)
# =====================================================================
if not os.path.exists(PROCESSED_JSON_PATH):
    print("[RALAT] Fail data terstruktur 'cases.json' tidak ditemui!")
    print("[INFO] Sila jalankan skrip Tahap 2 (02_case_representation.py) terlebih dahulu.")
    exit()

with open(PROCESSED_JSON_PATH, "r", encoding="utf-8") as f:
    cases_db = json.load(f)

print(f"[INFO] Berjaya memuatkan {len(cases_db)} kes dari pangkalan data.")

# =====================================================================
# 2. PEMBAHAGIAN DATA (SPLITTING DATA)
# =====================================================================
# Kita membahagikan data kepada data latihan (Train) dan data ujian (Test)
# dengan nisbah standard akademis 80:20 mengikut arahan modul tugasan.
train_cases, test_cases = train_test_split(cases_db, test_size=0.2, random_state=42)
print(f"[INFO] Pembahagian Data Selesai (Nisbah 80:20):")
print(f"       └─ Data Latihan (Train Set): {len(train_cases)} kes")
print(f"       └─ Data Ujian (Test Set)   : {len(test_cases)} kes")

# =====================================================================
# 3. REPRESENTASI VEKTOR (TF-IDF VECTORIZATION)
# =====================================================================
# Membina pembina vektor TF-IDF menggunakan teks penuh yang telah dibersihkan
train_texts = [case["text_full"] for case in train_cases]
vectorizer = TfidfVectorizer(preprocessor=preprocess_query)
tfidf_train_matrix = vectorizer.fit_transform(train_texts)

# Sediakan juga matriks TF-IDF untuk keseluruhan database bagi tujuan carian penuh
all_texts = [case["text_full"] for case in cases_db]
tfidf_all_matrix = vectorizer.transform(all_texts)

print(f"[INFO] Selesai membina vektor TF-IDF. Jumlah dimensi kosa kata: {len(vectorizer.vocabulary_)} kata.")

# =====================================================================
# 4. MODEL RETRIEVAL / KLASIFIKASI (SVM MODEL)
# =====================================================================
# Melatih model Machine Learning Support Vector Machine (SVM) pada representasi TF-IDF
# untuk mengkelaskan keputusan sengketa (solusi_hukum).
train_labels = [case["solusi_hukum"] for case in train_cases]

# Kami telah membuang parameter probability=True untuk menghilangkan FutureWarning
# pada pustaka Scikit-Learn terbaru (1.9+) karena kita hanya memerlukan fungsi prediksi kelas langsung.
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(tfidf_train_matrix, train_labels)

print("[INFO] Model Klasifikasi SVM berjaya dilatih pada representasi TF-IDF.")

# =====================================================================
# 5. FUNGSI RETRIEVAL UTAMA (RETRIEVE FUNCTION)
# =====================================================================
def retrieve(query: str, k: int = 5):
    """
    Mencari k-kes terdahulu yang paling serupa dengan kes baru (query).
    
    Langkah Kerja:
    1) Pre-process query
    2) Hitung vektor query
    3) Hitung cosine-similarity dengan semua case vectors
    4) Kembalikan top-k case_id beserta skor kemiripan
    """
    # 1) Pre-process query
    cleaned_query = preprocess_query(query)
    
    # 2) Hitung vektor query
    query_vector = vectorizer.transform([cleaned_query])
    
    # 3) Hitung cosine-similarity dengan semua case vectors dalam pangkalan data
    similarities = cosine_similarity(query_vector, tfidf_all_matrix).flatten()
    
    # 4) Dapatkan indeks top-k kes teratas mengikut kemiripan tertinggi
    top_k_indices = np.argsort(similarities)[::-1][:k]
    
    retrieved_results = []
    for idx in top_k_indices:
        case = cases_db[idx]
        retrieved_results.append({
            "case_id": case["case_id"],
            "no_perkara": case["no_perkara"],
            "pihak": case["pihak"],
            "merek_penggugat": case["merek_penggugat"],
            "merek_tergugat": case["merek_tergugat"],
            "similarity": float(similarities[idx]),
            "solusi_hukum": case["solusi_hukum"],
            "ringkasan_fakta": case["ringkasan_fakta"]
        })
        
    return retrieved_results

# =====================================================================
# 6. PENYEDIAAN KES UJIAN SECARA DINAMIK (DYNAMICAL TEST QUERY CREATOR)
# =====================================================================
# Untuk mengelakkan ralat 'hardcoded case_id' yang tidak sepadan dengan data asli anda,
# kami membina penjana query automatik yang mengambil fakta dari pangkalan data sebenar
# dan menetapkannya sebagai Ground-Truth secara dinamik!

def generate_test_queries():
    """
    Menghasilkan fail queries.json secara automatik berdasarkan kes sebenar
    yang ada di dalam processed/cases.json untuk kegunaan fasa penilaian.
    """
    print("[INFO] Menjana kes ujian pengesahan (queries.json) secara dinamik...")
    
    test_queries_list = []
    # Ambil maksimum 6 kes berbeza dari pangkalan data sebagai asas soalan pengesahan
    selected_samples = cases_db[:6] if len(cases_db) >= 6 else cases_db
    
    for idx, case in enumerate(selected_samples):
        q_id = f"Q{idx+1:03d}"
        
        # Contoh rekaan pertanyaan hukum baru yang meniru teks ringkasan fakta kes terdahulu
        query_text = (
            f"Terdapat pendaftaran merek terdaftar atas nama pihak lawan yang dinilai memiliki persamaan pada pokoknya "
            f"dengan merek terkenal milik kami yaitu '{case['merek_penggugat']}'. "
            f"Objek sengketa adalah merek '{case['merek_tergugat']}' untuk kelas barang yang sejenis. "
            f"Kami mohon pembatalan pendaftaran merek tersebut karena didasari iktikad tidak baik."
        )
        
        # Jika kes adalah tentang penyingkiran/penghapusan merek akibat tidak digunakan (non-use)
        if case["fakta_non_use"] == "YA":
            query_text = (
                f"Kami mengajukan gugatan penghapusan terhadap merek terdaftar '{case['merek_tergugat']}' "
                f"dengan nomor perkara tersebut karena terbukti tidak digunakan secara berturut-turut selama "
                f"3 tahun dalam perdagangan barang sejak tanggal pendaftaran terakhir."
            )
            
        test_queries_list.append({
            "query_id": q_id,
            "query_text": query_text,
            "ground_truth_case_id": case["case_id"],
            "ground_truth_solusi": case["solusi_hukum"]
        })
        
    with open(QUERIES_JSON_PATH, "w", encoding="utf-8") as json_out:
        json.dump(test_queries_list, json_out, indent=4)
        
    print(f"[✔] Fail '{QUERIES_JSON_PATH}' berjaya dijana dengan {len(test_queries_list)} kes ujian.")
    return test_queries_list

test_queries = generate_test_queries()

# =====================================================================
# 7. PENGUJIAN AWAL PIPELINE RETRIEVAL
# =====================================================================
def run_initial_retrieval_test():
    print("\n" + "="*80)
    # Menampilkan tajuk ujian menggunakan bahasa Indonesia/Melayu formal
    print(" DEMO PENGUJIAN AWAL RETRIEVAL (TF-IDF & COSINE SIMILARITY)")
    print("="*80)
    
    for q in test_queries[:3]:  # Papar 3 query sahaja untuk demo konsol yang bersih
        print(f"\n[Query Ujian] ID: {q['query_id']}")
        print(f"  └─ Fakta Baru : \"{q['query_text'][:120]}...\"")
        print(f"  └─ Sasaran GT : {q['ground_truth_case_id']} ({q['ground_truth_solusi']})")
        
        # Jalankan fungsi retrieve() utama
        results = retrieve(q["query_text"], k=3)
        
        # Gunakan model SVM untuk meramal kelas sengketa secara langsung
        query_vector = vectorizer.transform([preprocess_query(q["query_text"])])
        predicted_class = svm_model.predict(query_vector)[0]
        
        print(f"  └─ Hasil Carian CBR (Top-3 Kes Serupa):")
        for rank, res in enumerate(results):
            indicator = "⭐ [TEPAT]" if res["case_id"] == q["ground_truth_case_id"] else " "
            print(f"     {rank+1}. Kes ID: {res['case_id']} | Nilai Mirip: {res['similarity']:.4f} | Merek: {res['merek_penggugat']} VS {res['merek_tergugat']} {indicator}")
            
        print(f"  └─ Ramalan Keputusan Model (SVM): {predicted_class}")
        print("-" * 60)

if __name__ == "__main__":
    run_initial_retrieval_test()