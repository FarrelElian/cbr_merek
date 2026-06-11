**Sistem Case-Based Reasoning (CBR) Putusan Sengketa Merek**
Proyek Akhir Mata Kuliah Penalaran Komputer (Semester Genap 2025/2026)
Program Studi Informatika - Universitas Muhammadiyah Malang

**Anggota Kelompok** 
Putera Farrel Elian - 202310370311275 (Informatika UMM)
Muhammad Iqbal Fadel - 202310370311268 (Informatika UMM)

**Deskripsi Proyek**

Sistem ini dirancang untuk melakukan penalaran hukum otomatis (legal reasoning) berbasis Case-Based Reasoning (CBR) pada putusan sengketa merek di Indonesia berdasarkan UU No. 20 Tahun 2016 tentang Merek dan Indikasi Geografis.
Dengan memanfaatkan tumpukan berkas dokumen putusan riil dari Direktori Putusan Mahkamah Agung RI (tingkat Kasasi, Peninjauan Kembali, dan PT TUN), sistem ini memodelkan kepakaran hukum melalui siklus 4R (Retrieve, Reuse, Revise, Retain) untuk membantu praktisi hukum atau peneliti dalam memprediksi hasil akhir suatu sengketa merek baru secara cepat dan objektif.

**Siklus CBR Merek**
1. Retrieve (Panggil Kembali): Ketika ada sengketa merek baru masuk (misal: merek Adibas mendaftar di kelas baju), asisten ini akan membongkar lemari arsip hukum lama untuk mencari kasus-kasus terdahulu yang paling mirip (menggunakan representasi vektor TF-IDF dan kesamaan sudut Cosine Similarity).
2. Reuse (Gunakan Kembali): Asisten menyalin dasar pertimbangan hukum dan vonis dari kasus termirip tersebut sebagai usulan solusi awal (menggunakan algoritma pemungutan suara berbobot kemiripan/Weighted Similarity).
3. Revise (Revisi): Draf keputusan diserahkan kepada Hakim/Pakar Hukum untuk disesuaikan dengan fakta spesifik persidangan baru.
4. Retain (Simpan): Setelah keputusan diketok palu, putusan baru tersebut disimpan kembali ke dalam lemari arsip agar sistem otomatis bertambah pintar di masa depan tanpa perlu diprogram ulang.

**Struktur Direktori Proyek**
cbr_merek/
│
├── data/
│   ├── pdf_merek/                # 📥 File PDF asli dari Direktori Putusan MA RI
│   ├── raw/                      # ⚙️ TAHAP 1: Hasil ekstraksi & pembersihan teks (.txt)
│   ├── processed/                # ⚙️ TAHAP 2: Representasi kasus terstruktur (cases.json, cases.csv)
│   ├── results/                  # ⚙️ TAHAP 4: Hasil tebakan sengketa baru (predictions.csv)
│   └── eval/                     # ⚙️ TAHAP 3 & 5: Kasus uji (queries.json), LIME report, & file metrik
│
├── notebooks/                    # 🐍 SCRIPT UTAMA PIPELINE CBR (Python Scripts)
│   ├── case1.py                  # Tahap 1: Membangun Case Base & Preprocessing
│   ├── case2.py                  # Tahap 2: Case Representation & Metadata Extraction
│   ├── case3.py                  # Tahap 3: Splitting Data, Vectorization (TF-IDF), & Inisiasi Retrieval
│   ├── case4.py                  # Tahap 4: Case Solution Reuse (Weighted Similarity Prediction)
│   └── case5.py                  # Tahap 5: Model Evaluation (Hit Rate@K, MRR, Klasifikasi Metrics)
│
├── logs/
│   └── cleaning.log              # Log riwayat validasi keutuhan dokumen saat dibersihkan
│
├── requirements.txt              # Daftar pustaka Python yang wajib diinstal
├── xai.py                        # Tambahan XAI: Explainable AI menggunakan LIME untuk transparansi model
├── README.md                     # File dokumentasi ini (Panduan Proyek)


**Instalasi (Requirements)**
1. Prasyarat Sistem
Python versi 3.10 hingga 3.13 (Direkomendasikan menggunakan Python 3.11 atau 3.12).
Koneksi internet (untuk mengunduh pustaka eksternal).

2. Membuat Virtual Environment (Sangat Direkomendasikan)
# Windows
python -m venv venv
venv\Scripts\activate

3. Menginstal Pustaka Dependensi
pip install -r requirements.txt

pypdf>=4.0.0
pandas>=2.0.0
scikit-learn>=1.2.0
numpy>=1.24.0
lime>=0.2.0


**Pipeline End-to-End**
Tahap 1: Membangun Case Base (Preprocessing)
Proses: Mengekstrak berkas PDF putusan mentah dari data/pdf_merek/, membuang teks sampah administratif (header, footer, disclaimer MA), mengubah ke huruf kecil, serta memvalidasi integritas dokumen.
python notebooks/case1.py
Output: Berkas teks bersih disimpan di data/raw/*.txt, log tercatat di logs/cleaning.log.

Tahap 2: Representasi Kasus (Metadata & Feature Extraction)
proses: Mengekstrak informasi penting (Nomor Perkara, Tanggal, Para Pihak, Merek bersengketa, Pasal digunakan, Solusi Hukum) menggunakan Regex adaptif, menghitung panjang dokumen, serta membuat fitur Bag-of-Words (BoW) dan QA-Pairs.
python notebooks/case2.py
Output: data/processed/cases.json (Rich Nested Data) dan data/processed/cases.csv (Urutan kolom sesuai rubrik dosen).

Tahap 3: Model Retrieval (TF-IDF & Inisiasi Pencarian)
proses: Melakukan pembagian data (Train/Test 80:20), melakukan tokenisasi vektor menggunakan pembobotan statistik TF-IDF, melatih model pengklasifikasi SVM, serta menjana berkas kasus pengujian evaluasi secara dinamis.
python notebooks/case3.py
Output: Berkas sengketa baru simulasi di data/eval/queries.json dan hasil pencarian Top-3 kemiripan kosinus awal.

Tahap 4: Case Solution Reuse (Weighted Similarity)
proses: Menerapkan keputusan hukum cerdas untuk kasus baru menggunakan voting terbobot nilai kemiripan kosinus (Weighted Cosine Similarity), menguji tebakan sistem pada 5 skenario sengketa, dan membandingkannya dengan vonis asli hakim agung.
python notebooks/case4.py
Output: Hasil prediksi tersimpan rapi di berkas data/results/predictions.csv.

Tahap 5: Model Evaluation (Quality Control)
proses: Mengukur ketepatan asisten hukum dalam riset dokumen (Hit Rate@K, Mean Reciprocal Rank) serta akurasi tebakan vonis akhir (Accuracy, Precision, Recall, F1-Score).
python notebooks/case5.py
Output: Laporan metrik formal tersimpan di data/eval/retrieval_metrics.csv dan data/eval/prediction_metrics.csv.

Fitur Ekstra: Explainable AI (XAI)
proses: Untuk memberikan transparansi keputusan hukum dari AI (agar keputusan AI tidak dianggap sebagai "kotak hitam/black box"), kamu dapat menjalankan penjelasan visual berbasis LIME:
python xai.py
Output: Laporan visual interaktif data/eval/lime_explanation.html yang memetakan kata kunci hukum apa saja yang paling memengaruhi keputusan model AI dalam memilih vonis.

**Interpretasi Hasil Eksplanasi LIME (Konteks Sengketa Merek)**
> Sinyal Hijau (Mendorong Prediksi: GUGATAN DIKABULKAN)
Kata kunci: membatal, hapus, iktikad, persamaan, terkenal.
Artinya: Kehadiran kata-kata ini dalam teks gugatan memberikan sinyal kuat kepada model bahwa penggugat membawa dalil pembatalan merek akibat adanya indikasi iktikad tidak baik atau persamaan pada pokoknya dengan merek terkenal. Hal ini menuntun prediksi model menuju klasifikasi dikabulkannya gugatan.

> Sinyal Merah (Mendorong Prediksi: GUGATAN DITOLAK)
Kata kunci: menolak, eksepsi, prematur, biaya, tidak.
Artinya: Kata-kata seperti eksepsi dan prematur sering kali muncul dalam argumentasi tergugat atau pertimbangan awal hakim yang menolak formalitas gugatan. Kehadiran kata-kata ini menarik jarum prediksi model ke arah penolakan gugatan.
Melalui visualisasi ini, sistem CBR kita terbukti tidak melakukan tebakan secara acak (kotak hitam), melainkan benar-benar menangkap esensi semantik bahasa hukum Indonesia secara logis dan transparan!

**Analisis Evaluasi & Batasan Laboratorium (Overfitting Discussion)**
> Saat menjalankan pengujian penuh pada pipeline di atas, sistem akan menghasilkan metrik evaluasi yang menyentuh angka 1.0 (100%) pada seluruh aspek klasifikasi.
Mengapa Angka Ini Bisa Terjadi?
Hasil evaluasi sempurna ini tercapai karena Skenario Pengujian Terkontrol (Closed-loop Sandbox):
  - Berkas query uji (queries.json) dibuat secara dinamis dengan mengekstrak pola kalimat dari database kasus asli (cases.json). Jika dalam analogi murid di sekolah maka seperti murid yang diberi soal ujian dari kisi-kisi kalimat buku paket yang ia hafalkan secara persis.
  - Ukuran database kasus sengketa merek saat ini masih berada pada taraf minimum rubrik akademis (30+ dokumen). Dengan jumlah data uji yang kecil (5 kasus uji manual), probabilitas matematika bagi kemiripan kosinus untuk menebak 5 dari 5 soal secara tepat sangatlah besar.

> Batasan Riil & Solusi Masa Depan:
Performa sistem diproyeksikan akan mengalami generalization gap dan turun ke rentang 75% - 85% akibat gaya bahasa pengguna yang bebas, adanya saltik (typo), serta dokumen hukum yang tidak rapi. Sebagai rekomendasi perbaikan, model di masa depan disarankan menggunakan representasi makna semantik berbasis Deep Learning seperti IndoBERT untuk menggantikan TF-IDF yang murni mengandalkan frekuensi kecocokan kata dasar.
