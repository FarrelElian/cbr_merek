# -*- coding: utf-8 -*-
"""
Tugas Penalaran Komputer - SIKLUS CBR (Tahap 5: Model Evaluation)
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =====================================================================
# KONFIGURASI JALUR DIREKTORI
# =====================================================================
PROCESSED_JSON_PATH = "data/processed/cases.json"
QUERIES_JSON_PATH = "data/eval/queries.json"
PREDICTIONS_CSV_PATH = "data/results/predictions.csv"
EVAL_DIR = "data/eval"

RETRIEVAL_METRICS_PATH = os.path.join(EVAL_DIR, "retrieval_metrics.csv")
PREDICTION_METRICS_PATH = os.path.join(EVAL_DIR, "prediction_metrics.csv")

# Memastikan folder evaluasi tersedia
os.makedirs(EVAL_DIR, exist_ok=True)

# =====================================================================
# LOAD SELURUH DATASET & HASIL PREDIKSI
# =====================================================================
if not os.path.exists(PROCESSED_JSON_PATH):
    print("[ERROR] Database kasus terstruktur 'cases.json' tidak ditemukan!")
    exit()

if not os.path.exists(QUERIES_JSON_PATH):
    print("[ERROR] Berkas queries uji 'queries.json' tidak ditemukan!")
    exit()

if not os.path.exists(PREDICTIONS_CSV_PATH):
    print("[ERROR] Hasil prediksi 'predictions.csv' tidak ditemukan!")
    print("[INFO] Silakan jalankan Tahap 4 (04_predict.py) terlebih dahulu untuk membuat file prediksi.")
    exit()

with open(PROCESSED_JSON_PATH, "r", encoding="utf-8") as f:
    cases_db = json.load(f)

with open(QUERIES_JSON_PATH, "r", encoding="utf-8") as f:
    test_queries = json.load(f)

df_predictions = pd.read_csv(PREDICTIONS_CSV_PATH)

print(f"[INFO] Sukses memuat {len(cases_db)} kasus dari database.")
print(f"[INFO] Sukses memuat {len(test_queries)} data ground-truth query uji.")
print(f"[INFO] Sukses memuat {len(df_predictions)} baris hasil prediksi sistem.")

# =====================================================================
# PERSAPAN RETRIEVAL ENGINE UNTUK EVALUASI DINAMIS
# =====================================================================
def preprocess_text(text):
    """Membersihkan teks sebelum ditransformasikan ke vektor."""
    text = text.lower()
    text = re.sub(r'[^\w\s\-\/\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Mengubah seluruh isi dokumen di database kasus menjadi matriks TF-IDF
all_texts = [case["text_full"] for case in cases_db]
vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
tfidf_matrix = vectorizer.fit_transform(all_texts)

def retrieve_for_eval(query: str, k: int = 5):
    """Fungsi pembantu untuk meretrieve daftar case_id teratas."""
    cleaned_query = preprocess_text(query)
    query_vector = vectorizer.transform([cleaned_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_k_indices = np.argsort(similarities)[::-1][:k]
    return [cases_db[idx]["case_id"] for idx in top_k_indices]

# =====================================================================
# i. EVALUASI RETRIEVAL (Mendukung Hit Rate dan Mean Reciprocal Rank)
# =====================================================================
def eval_retrieval(queries, ground_truth, k):
    """
    Fungsi Evaluasi Retrieval sesuai spesifikasi tugas dosen.
    Mengukur seberapa akurat sistem pencari dokumen kemiripan kosinus.
    
    Metrik yang digunakan:
    1. Hit Rate@K (Accuracy@K) : Berapa persen query yang dokumen aslinya berhasil ditemukan di Top-K.
    2. Mean Reciprocal Rank (MRR@K) : Memperhitungkan posisi peringkat berkas yang benar.
    """
    hits = 0
    reciprocal_ranks = []
    evaluation_logs = []
    
    # Loop setiap query -> hitung metrics
    for q_data, gt_id in zip(queries, ground_truth):
        query_text = q_data["query_text"]
        query_id = q_data["query_id"]
        
        # Jalankan retrieval untuk query ini
        retrieved_ids = retrieve_for_eval(query_text, k=k)
        
        # 1. Cek Hit Rate (Apakah Ground Truth case_id ada di dalam Top-K?)
        is_hit = 1 if gt_id in retrieved_ids else 0
        hits += is_hit
        
        # 2. Cek Reciprocal Rank
        rank_val = 0
        if gt_id in retrieved_ids:
            rank_index = retrieved_ids.index(gt_id)
            rank_val = 1 / (rank_index + 1) # Peringkat 1 mendapat nilai 1.0, Peringkat 2 mendapat 0.5, dst.
        reciprocal_ranks.append(rank_val)
        
        evaluation_logs.append({
            "query_id": query_id,
            "ground_truth_case": gt_id,
            "is_hit_at_k": is_hit,
            "reciprocal_rank": rank_val,
            "retrieved_cases": ", ".join(retrieved_ids)
        })
        
    # Hitung nilai rata-rata keseluruhan query
    avg_hit_rate = hits / len(queries)
    avg_mrr = np.mean(reciprocal_ranks)
    
    return avg_hit_rate, avg_mrr, evaluation_logs

# Jalankan evaluasi retrieval dengan k=5 sesuai default sistem
ground_truths_retrieval = [q["ground_truth_case_id"] for q in test_queries]
hit_rate_5, mrr_5, raw_retrieval_logs = eval_retrieval(test_queries, ground_truths_retrieval, k=5)

# Menyimpan hasil evaluasi retrieval ke format terstruktur data/eval/retrieval_metrics.csv
df_ret_metrics = pd.DataFrame([
    {"Metric": "Hit_Rate@5", "Score": hit_rate_5, "Description": "Persentase kasus relevan berhasil masuk Top-5"},
    {"Metric": "MRR@5", "Score": mrr_5, "Description": "Mean Reciprocal Rank (Kualitas urutan hasil pencarian)"}
])

try:
    df_ret_metrics.to_csv(RETRIEVAL_METRICS_PATH, index=False)
    print(f"[✔] Sukses mengekspor metrik retrieval ke: '{RETRIEVAL_METRICS_PATH}'")
except PermissionError:
    print("[WARNING] Gagal menulis retrieval_metrics.csv, pastikan file sedang tidak dibuka di Excel.")

# =====================================================================
# ii. EVALUASI PREDIKSI (Accuracy, Precision, Recall, F1-Score)
# =====================================================================
def eval_prediction():
    """
    Mengevaluasi akurasi keputusan hukum akhir dari sistem CBR
    dengan membandingkan hasil prediksi dengan nilai ground-truth asli.
    """
    # Menghubungkan prediksi dengan ground truth berdasarkan query_id
    y_true = []
    y_pred = []
    
    # Ambil data pembanding
    for q in test_queries:
        q_id = q["query_id"]
        gt_solusi = q["ground_truth_solusi"]
        
        # Ambil hasil tebakan sistem dari predictions.csv
        prediction_row = df_predictions[df_predictions["query_id"] == q_id]
        if not prediction_row.empty:
            pred_solusi = prediction_row.iloc[0]["predicted_solution"]
            y_true.append(gt_solusi)
            y_pred.append(pred_solusi)
            
    # Menghitung metrik klasifikasi formal menggunakan sklearn.metrics
    # Menggunakan average='macro' karena sengketa memiliki multi-kelas solusi (kabul vs tolak)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Menyimpan metrik ke dalam data/eval/prediction_metrics.csv
    df_pred_metrics = pd.DataFrame([
        {"Metric": "Accuracy", "Score": acc},
        {"Metric": "Precision (Macro)", "Score": prec},
        {"Metric": "Recall (Macro)", "Score": rec},
        {"Metric": "F1-Score (Macro)", "Score": f1}
    ])
    
    try:
        df_pred_metrics.to_csv(PREDICTION_METRICS_PATH, index=False)
        print(f"[✔] Sukses mengekspor metrik prediksi ke: '{PREDICTION_METRICS_PATH}'")
    except PermissionError:
        print("[WARNING] Gagal menulis prediction_metrics.csv, pastikan file sedang tidak dibuka di Excel.")
        
    return acc, prec, rec, f1

acc_score, prec_score, rec_score, f1_score_val = eval_prediction()

# =====================================================================
# iii. VISUALISASI & LAPORAN ERROR ANALYSIS (KONSOL)
# =====================================================================
def display_evaluation_dashboard():
    print("\n" + "="*80)
    print("      🏆 HASIL EVALUASI AKHIR SISTEM CBR - HUKUM MEREK UMM (TAHAP 5) 🏆")
    print("="*80)
    
    # 1. Tampilkan Metrik Riset Pencarian Dokumen (Retrieval)
    print("\n[A. PERFORMA RETRIEVAL KASUS (COSINE SIMILARITY)]")
    print(f"  ├─ Hit Rate@5 (Akurasi Carian)  : {hit_rate_5:.2%}")
    print(f"  ├─ MRR@5 (Kualitas Urutan Rank) : {mrr_5:.4f}")
    
    # Visualisasi Bar Chart Sederhana di Konsol
    bar_hit = "█" * int(hit_rate_5 * 20) + "░" * (20 - int(hit_rate_5 * 20))
    bar_mrr = "█" * int(mrr_5 * 20) + "░" * (20 - int(mrr_5 * 20))
    print(f"  ├─ Visualisasi Hit Rate@5       : [{bar_hit}] {hit_rate_5:.1%}")
    print(f"  └─ Visualisasi MRR@5            : [{bar_mrr}] {mrr_5:.3f}")
    
    # 2. Tampilkan Metrik Keputusan Solusi Akhir (Prediction)
    print("\n[B. PERFORMA PREDIKSI PUTUSAN HUKUM (WEIGHTED SIMILARITY)]")
    print(f"  ├─ Akurasi Klasifikasi (Accuracy) : {acc_score:.2%}")
    print(f"  ├─ Presisi Model (Precision)      : {prec_score:.2%}")
    print(f"  ├─ Sensitivitas Model (Recall)    : {rec_score:.2%}")
    print(f"  └─ Skor F1 (F1-Score)             : {f1_score_val:.2%}")
    
    bar_acc = "█" * int(acc_score * 20) + "░" * (20 - int(acc_score * 20))
    print(f"  └─ Visualisasi Akurasi Akhir      : [{bar_acc}] {acc_score:.1%}")
    
    # 3. ANALISIS KEGAGALAN (ERROR ANALYSIS & DISKUSI REJECTION)
    print("\n[C. ANALISIS KEGAGALAN & DISKUSI AKADEMIS (ERROR ANALYSIS)]")
    print("  1. Isu Kegagalan Lokasi Berkas PT TUN:")
    print("     - Berkas jenis peradilan Tata Usaha Negara (PT TUN) sering kali")
    print("       tidak menggunakan diksi sengketa merek langsung, melainkan gugatan")
    print("       administratif terhadap DJKI. Hal ini menurunkan nilai kemiripan")
    print("       kosinus jika dicari menggunakan kata kunci sengketa perdata biasa.")
    print("  2. Solusi & Rekomendasi Perbaikan Sistem:")
    print("     - Rekomendasi A: Terapkan pembagian database kasus hukum merek terpisah")
    print("       antara rumpun perdata khusus (Niaga) dan rumpun administratif (PT TUN).")
    print("     - Rekomendasi B: Gunakan representasi Text Embedding (seperti IndoBERT)")
    print("       di masa depan untuk menangkap makna semantik kalimat hukum yang rumit,")
    print("       menggantikan pendekatan TF-IDF yang murni bergantung pada frekuensi kata.")
    print("="*80 + "\n")

if __name__ == "__main__":
    display_evaluation_dashboard()