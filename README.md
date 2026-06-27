# Sistem Case-Based Reasoning (CBR) Putusan Sengketa Merek

Proyek Akhir Mata Kuliah Penalaran Komputer (Semester Genap 2025/2026)  
Program Studi Informatika - Universitas Muhammadiyah Malang

## Anggota Kelompok

Putera Farrel Elian - 202310370311275 (Informatika UMM)  
Muhammad Iqbal Fadel - 202310370311268 (Informatika UMM)

## Deskripsi Proyek

Sistem ini dirancang untuk melakukan penalaran hukum otomatis (legal reasoning) berbasis Case-Based Reasoning (CBR) pada putusan sengketa merek di Indonesia berdasarkan UU No. 20 Tahun 2016 tentang Merek dan Indikasi Geografis.
Dengan memanfaatkan tumpukan berkas dokumen putusan riil dari Direktori Putusan Mahkamah Agung RI (tingkat Kasasi, Peninjauan Kembali, dan PT TUN), sistem ini memodelkan kepakaran hukum melalui siklus 4R (Retrieve, Reuse, Revise, Retain) untuk membantu praktisi hukum atau peneliti dalam memprediksi hasil akhir suatu sengketa merek baru secara cepat dan objektif.

## Siklus CBR Merek

1. Retrieve (Panggil Kembali): Ketika ada sengketa merek baru masuk (misal: merek Adibas mendaftar di kelas baju), asisten ini akan membongkar lemari arsip hukum lama untuk mencari kasus-kasus terdahulu yang paling mirip (menggunakan representasi vektor TF-IDF dan kesamaan sudut Cosine Similarity).
2. Reuse (Gunakan Kembali): Asisten menyalin dasar pertimbangan hukum dan vonis dari kasus termirip tersebut sebagai usulan solusi awal (menggunakan algoritma pemungutan suara berbobot kemiripan/Weighted Similarity).
3. Revise (Revisi): Draf keputusan diserahkan kepada Hakim/Pakar Hukum untuk disesuaikan dengan fakta spesifik persidangan baru.
4. Retain (Simpan): Setelah keputusan diketok palu, putusan baru tersebut disimpan kembali ke dalam lemari arsip agar sistem otomatis bertambah pintar di masa depan tanpa perlu diprogram ulang.

## Struktur Direktori Proyek

```
cbr_merek/
│
├── data/
│   ├── pdf_merek/                # File PDF asli dari Direktori Putusan MA RI
│   ├── raw/                      # TAHAP 1: Hasil ekstraksi & pembersihan teks (.txt)
│   ├── processed/                # TAHAP 2: Representasi kasus terstruktur (cases.json, cases.csv)
│   ├── results/                  # TAHAP 4: Hasil tebakan sengketa baru (predictions.csv)
│   └── eval/                     # TAHAP 3 & 5: Kasus uji (queries.json), LIME report, & file metrik
│
├── notebooks/                    # Script utama pipeline CBR (Python Scripts & Jupyter Notebooks)
│   ├── case1.py                  # Tahap 1: Membangun Case Base & Preprocessing
│   ├── case2.py                  # Tahap 2: Case Representation & Metadata Extraction
│   ├── case3.py                  # Tahap 3: Splitting Data, Vectorization (TF-IDF), & Inisiasi Retrieval
│   ├── case4.py                  # Tahap 4: Case Solution Reuse (Weighted Similarity Prediction)
│   ├── case5.py                  # Tahap 5: Model Evaluation (Hit Rate@K, MRR, Klasifikasi Metrics)
│   └── xai.py                    # Tambahan XAI: Explainable AI menggunakan LIME untuk transparansi model
│
├── logs/
│   └── cleaning.log              # Log riwayat validasi keutuhan dokumen saat dibersihkan
│
├── requirements.txt              # Daftar pustaka Python yang wajib diinstal
└── README.md                     # File dokumentasi ini (Panduan Proyek)
```

## Instalasi

### 1. Prasyarat Sistem

Python 3.10+ (diuji pada Python 3.13). Koneksi internet diperlukan untuk mengunduh pustaka eksternal.

### 2. Membuat Virtual Environment (Sangat Direkomendasikan)

```bash
# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Menginstal Pustaka Dependensi

```bash
pip install -r requirements.txt
```

Dependensi utama yang akan diinstal:

```
pypdf>=4.0.0
pandas>=2.0.0
scikit-learn>=1.2.0
numpy>=1.24.0
lime>=0.2.0
```

## Pipeline End-to-End

### Tahap 1: Membangun Case Base (Preprocessing)

Proses: Mengekstrak berkas PDF putusan mentah dari `data/pdf_merek/`, membuang teks sampah administratif (header, footer, disclaimer MA), mengubah ke huruf kecil, serta memvalidasi integritas dokumen.

```
Buka dan Jalankan Sel: notebooks/01_case_base.ipynb
```

Output: Berkas teks bersih disimpan di `data/raw/` dengan penamaan terstruktur (`case_01.txt`, dst.), log tercatat di `logs/cleaning.log`.

### Tahap 2: Representasi Kasus (Metadata & Feature Extraction)

Proses: Mengekstrak informasi penting (Nomor Perkara, Tanggal, Para Pihak, Merek bersengketa, Pasal digunakan, Solusi Hukum) menggunakan Regex adaptif, menghitung panjang dokumen, serta membuat fitur Bag-of-Words (BoW) dan QA-Pairs.

```
Buka dan Jalankan Sel: notebooks/02_case_representation.ipynb
```

Output: `data/processed/cases.json` (Rich Nested Data) dan `data/processed/cases.csv` (Urutan kolom sesuai rubrik dosen).

### Tahap 3: Model Retrieval (TF-IDF & Inisiasi Pencarian)

Proses: Melakukan pembagian data (Train/Test 80:20), melakukan tokenisasi vektor menggunakan pembobotan statistik TF-IDF, melatih model pengklasifikasi SVM, serta memuat data kasus pengujian independen (`queries.json`).

```
Buka dan Jalankan Sel: notebooks/03_retrieval.ipynb
```

Output: Berkas sengketa baru simulasi di `data/eval/queries.json` dan hasil pencarian Top-3 kemiripan kosinus awal.

### Tahap 4: Case Solution Reuse (Weighted Similarity)

Proses: Menerapkan keputusan hukum cerdas untuk kasus baru menggunakan voting terbobot nilai kemiripan kosinus (Weighted Cosine Similarity), menguji tebakan sistem pada 7 skenario sengketa independen, dan membandingkannya dengan vonis asli hakim agung.

```
Buka dan Jalankan Sel: notebooks/04_predict.ipynb
```

Output: Hasil prediksi tersimpan rapi di berkas `data/results/predictions.csv`.

### Tahap 5: Model Evaluation (Quality Control)

Proses: Mengukur ketepatan asisten hukum dalam riset dokumen (Hit Rate@K, Mean Reciprocal Rank) serta akurasi tebakan vonis akhir (Accuracy, Precision, Recall, F1-Score).

```
Buka dan Jalankan Sel: notebooks/05_evaluation.ipynb
```

Output: Laporan metrik formal tersimpan di `data/eval/retrieval_metrics.csv` dan `data/eval/prediction_metrics.csv`.

### Fitur Ekstra: Explainable AI (XAI)

Proses: Untuk memberikan transparansi keputusan hukum dari AI (agar keputusan AI tidak dianggap sebagai "kotak hitam/black box"), jalankan penjelasan visual berbasis LIME:

```
Buka dan Jalankan Sel: notebooks/xai_explainability.ipynb
```

Output: Laporan visual interaktif `data/eval/lime_explanation.html` yang memetakan kata kunci hukum apa saja yang paling memengaruhi keputusan model AI dalam memilih vonis.

## Interpretasi Hasil Eksplanasi LIME (Konteks Sengketa Merek)

> Sinyal Hijau (Mendorong Prediksi: GUGATAN DIKABULKAN)

Kata kunci: membatal, hapus, iktikad, persamaan, terkenal.
Artinya: Kehadiran kata-kata ini dalam teks gugatan memberikan sinyal kuat kepada model bahwa penggugat membawa dalil pembatalan merek akibat adanya indikasi iktikad tidak baik atau persamaan pada pokoknya dengan merek terkenal. Hal ini menuntun prediksi model menuju klasifikasi dikabulkannya gugatan.

> Sinyal Merah (Mendorong Prediksi: GUGATAN DITOLAK)

Kata kunci: menolak, eksepsi, prematur, biaya, tidak.
Artinya: Kata-kata seperti eksepsi dan prematur sering kali muncul dalam argumentasi tergugat atau pertimbangan awal hakim yang menolak formalitas gugatan. Kehadiran kata-kata ini menarik jarum prediksi model ke arah penolakan gugatan.
Melalui visualisasi ini, sistem CBR kita terbukti tidak melakukan tebakan secara acak (kotak hitam), melainkan benar-benar menangkap esensi semantik bahasa hukum Indonesia secara logis dan transparan.

## Analisis Evaluasi & Generalisasi Model

Berdasarkan pengujian menggunakan 7 kueri independen (berupa skenario sengketa baru yang ditulis secara bebas, bukan disalin dari teks dokumen aslinya), sistem menghasilkan Hit Rate@5 sebesar 57.14% dan MRR@5 0.40 untuk tahap pencarian (retrieval), serta Akurasi Prediksi 42.86% dan F1-Score 30.00% untuk tebakan solusi hukum.

### Mengapa Angka Ini Realistis?

Hasil evaluasi ini jauh lebih mencerminkan kondisi riil di lapangan (generalization performance) dibandingkan pengujian closed-loop sandbox. Model TF-IDF bekerja dengan mencari kecocokan kata eksak (leksikal). Ketika dihadapkan pada skenario hukum bebas yang disusun menggunakan frasa atau struktur kalimat berbeda dari yang ditulis panitera MA, kemampuannya untuk memetakan ke dokumen asli menjadi tertantang.

### Batasan Riil & Rekomendasi Perbaikan

Ketergantungan TF-IDF pada frekuensi kecocokan kata membuat sistem rentan terhadap perbedaan gaya bahasa pengguna, sinonim, atau penggunaan istilah hukum alternatif. Sebagai rekomendasi perbaikan untuk mencapai akurasi di atas 80% pada skenario dunia nyata, model di masa depan disarankan menggunakan representasi makna semantik berbasis Deep Learning seperti IndoBERT. Pendekatan semantik ini mampu menangkap relasi makna antar-kalimat hukum yang rumit secara lebih superior dibandingkan metode statistik kata tradisional.
