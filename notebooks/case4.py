# -*- coding: utf-8 -*-
"""
Tugas Penalaran Komputer - SIKLUS CBR (Tahap 4: Case Solution Reuse)
Studi Kasus: Sengketa Merek & Indikasi Geografis (UU No. 20 Tahun 2016)
Fakultas Teknik - Informatika UMM
"""

import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================================
# KONFIGURASI JALUR DIREKTORI
# =====================================================================
PROCESSED_JSON_PATH = "data/processed/cases.json"
QUERIES_JSON_PATH = "data/eval/queries.json"
RESULTS_DIR = "data/results"
PREDICTIONS_CSV_PATH = os.path.join(RESULTS_DIR, "predictions.csv")

# Memastikan folder penyimpanan hasil prediksi sudah siap
os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================================================================
# LOAD DATABASE KASUS & QUERIES UJI
# =====================================================================
if not os.path.exists(PROCESSED_JSON_PATH):
    print("[ERROR] Database kasus terstruktur 'cases.json' tidak ditemukan!")
    print("[INFO] Silakan jalankan Tahap 2 (02_case_representation.py) terlebih dahulu.")
    exit()

if not os.path.exists(QUERIES_JSON_PATH):
    print("[ERROR] File queries uji 'queries.json' tidak ditemukan!")
    print("[INFO] Silakan jalankan Tahap 3 (03_retrieval.py) untuk menjana queries uji terlebih dahulu.")
    exit()

with open(PROCESSED_JSON_PATH, "r", encoding="utf-8") as f:
    cases_db = json.load(f)

with open(QUERIES_JSON_PATH, "r", encoding="utf-8") as f:
    test_queries = json.load(f)

print(f"[INFO] Sukses memuat {len(cases_db)} kasus dari database hukum merek.")
print(f"[INFO] Sukses memuat {len(test_queries)} skenario kasus uji dari evaluasi.")

# =====================================================================
# PERSAPAN RETRIEVAL ENGINE (TF-IDF & COSINE SIMILARITY)
# =====================================================================
def preprocess_text(text):
    """Membersihkan teks query baru sebelum diproses."""
    text = text.lower()
    text = re.sub(r'[^\w\s\-\/\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Inisialisasi Vectorizer TF-IDF menggunakan teks penuh dari pangkalan data kasus
all_texts = [case["text_full"] for case in cases_db]
vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
tfidf_matrix = vectorizer.fit_transform(all_texts)

def retrieve(query: str, k: int = 5):
    """Fungsi pencarian untuk mengambil top-k kasus terdekat."""
    cleaned_query = preprocess_text(query)
    query_vector = vectorizer.transform([cleaned_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_k_indices = np.argsort(similarities)[::-1][:k]
    
    results = []
    for idx in top_k_indices:
        results.append({
            "case_id": cases_db[idx]["case_id"],
            "similarity": float(similarities[idx]),
            "solusi_hukum": cases_db[idx]["solusi_hukum"],
            "detail_amar": cases_db[idx]["detail_amar"]
        })
    return results

# =====================================================================
# IMPLEMENTASI ALGORITMA PREDIKSI (SOLUTION REUSE)
# =====================================================================
def predict_outcome(query: str, k: int = 5, method: str = "weighted") -> dict:
    """
    Memprediksi keputusan hukum kasus baru berdasarkan top-k kasus termirip.
    
    Langkah Kerja:
    i. Ekstrak Solusi: Ambil amar putusan/klasifikasi dari kasus top-k.
    ii. Algoritma Prediksi: 
        1. Majority Vote (pilih keputusan yang paling banyak muncul)
        2. Weighted Similarity (skor kesamaan digunakan sebagai bobot suara)
    """
    # 1. Ambil top-k kasus termirip menggunakan fungsi retrieve()
    top_k_cases = retrieve(query, k=k)
    
    # Jika database kosong atau kemiripan nol
    if not top_k_cases:
        return {
            "predicted_solution": "BELUM DAPAT DIPUTUSKAN (DATA KURANG)",
            "top_k_case_ids": []
        }
    
    # Ekstrak data solusi dan skor kemiripan dari top-k kasus
    solutions_list = [c["solusi_hukum"] for c in top_k_cases]
    similarities_list = [c["similarity"] for c in top_k_cases]
    case_ids_list = [c["case_id"] for c in top_k_cases]
    
    predicted_solution = ""
    
    # Pendekatan A: Majority Vote (Suara Terbanyak)
    if method == "majority":
        # Menghitung frekuensi kemunculan setiap kategori putusan
        unique_solutions, counts = np.unique(solutions_list, return_counts=True)
        winner_index = np.argmax(counts)
        predicted_solution = unique_solutions[winner_index]
        
    # Pendekatan B: Weighted Similarity (Bobot Nilai Kemiripan - Direkomendasikan)
    else:
        # Menyimpan akumulasi nilai bobot kemiripan untuk setiap kelas solusi hukum
        weight_map = {}
        for sol, sim in zip(solutions_list, similarities_list):
            weight_map[sol] = weight_map.get(sol, 0.0) + sim
            
        # Kelas dengan total bobot kemiripan tertinggi keluar sebagai pemenang
        predicted_solution = max(weight_map, key=weight_map.get)

    return {
        "predicted_solution": predicted_solution,
        "top_k_case_ids": case_ids_list,
        "top_k_cases_detail": top_k_cases
    }

# =====================================================================
# DEMO MANUAL & EKSPOR HASIL PREDIKSI (Sesuai Tabel Instruksi Dosen)
# =====================================================================
def run_prediction_pipeline():
    print("="*80)
    print(" MEMULAI TAHAP 4: CASE SOLUTION REUSE (PREDIKSI PUTUSAN BARU)")
    print("="*80)
    
    # Kita proses semua contoh kasus baru untuk demo manual sesuai rubrik tugas
    demo_queries = test_queries
    
    predictions_table = []
    
    for q in demo_queries:
        q_id = q["query_id"]
        q_text = q["query_text"]
        gt_solusi = q["ground_truth_solusi"]
        
        # Jalankan prediksi dengan K=5 menggunakan metode Weighted Similarity
        prediction_results = predict_outcome(q_text, k=5, method="weighted")
        pred_solusi = prediction_results["predicted_solution"]
        top_ids = prediction_results["top_k_case_ids"]
        
        # Format list case_ids menjadi string yang rapi untuk kolom CSV
        top_ids_str = ", ".join(top_ids)
        
        # Tambahkan data ke tabel laporan (Sesuai kolom tabel dosen di PDF)
        predictions_table.append({
            "query_id": q_id,
            "predicted_solution": pred_solusi,
            "top_5_case_ids": top_ids_str
        })
        
        # Tampilkan hasil demo yang informatif dan interaktif di konsol
        print(f"\n[Kasus Baru] ID: {q_id}")
        print(f"  Ringkasan Gugatan: \"{q_text[:110]}...\"")
        print(f"  Top-5 Kasus Mirip : {top_ids}")
        print(f"  Skor Kemiripan   : {[round(c['similarity'], 4) for c in prediction_results['top_k_cases_detail']]}")
        print(f"  Solusi Sebenarnya : {gt_solusi}")
        print(f"  Prediksi CBR (Weighted): {pred_solusi}")
        
        # Indikator ketepatan sistem
        if pred_solusi == gt_solusi:
            print("  Hasil Keputusan: TEPAT (Sesuai dengan Putusan Asli Hakim MA RI)")
        else:
            print("  Hasil Keputusan: Terdapat Perbedaan (Butuh Analisis Hukum Tambahan)")
        print("-" * 75)
        
    # --- PROSES EKSPOR KE CSV ---
    # Menggunakan penanganan eksepsi try-except untuk mencegah error gembok Microsoft Excel
    try:
        df_predictions = pd.DataFrame(predictions_table)
        df_predictions.to_csv(PREDICTIONS_CSV_PATH, index=False)
        print(f"\n[OK] Sukses mengekspor hasil ke berkas: '{PREDICTIONS_CSV_PATH}'")
    except PermissionError:
        print("\n" + "!"*80)
        print(" GAGAL MENYIMPAN FILE HASIL PREDIKSI!")
        print("!"*80)
        print(f"Error: Permission denied pada '{PREDICTIONS_CSV_PATH}'")
        print("Solusi: Silakan TUTUP MICROSOFT EXCEL Anda terlebih dahulu, lalu jalankan kembali skrip ini.")
        print("!"*80 + "\n")

if __name__ == "__main__":
    run_prediction_pipeline()