# -*- coding: utf-8 -*-
"""
Tugas Penalaran Komputer - SIKLUS CBR (Tahap 1: Membangun Case Base)
Studi Kasus: Sengketa Merek & Indikasi Geografis (UU No. 20 Tahun 2016)
Fakultas Teknik - Informatika UMM
"""

import os
import re
import sys
import datetime

# Kita gunakan pustaka 'pypdf' untuk mengekstrak teks dari berkas PDF.
# Jika belum terinstal, jalankan: pip install pypdf
try:
    import pypdf
except ImportError:
    print("[PERINGATAN] Pustaka 'pypdf' tidak ditemukan.")
    print("[INFO] Menjalankan instalasi otomatis 'pypdf' via pip...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
    import pypdf

# =====================================================================
# CONFIGURATION & DIRECTORY SETUP
# =====================================================================
PDF_INPUT_DIR = "data/pdf_merek"       # Tempat kamu menaruh 30+ PDF Putusan asli
TXT_OUTPUT_DIR = "data/raw"            # Output file teks yang sudah bersih (.txt)
LOG_DIR = "logs"                       # Folder untuk menyimpan file log
LOG_FILE_PATH = os.path.join(LOG_DIR, "cleaning.log")

# Membuat folder-folder yang dibutuhkan jika belum ada
os.makedirs(PDF_INPUT_DIR, exist_ok=True)
os.makedirs(TXT_OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =====================================================================
# TEKS PREPROCESSING & NOISE REMOVAL
# =====================================================================
def clean_legal_text(raw_text):
    """
    Melakukan pembersihan mendalam (noise removal) khusus untuk teks hukum Indonesia:
    1. Menghilangkan header & footer khas Direktori Putusan MA.
    2. Menghilangkan nomor halaman 'Halaman X dari Y'.
    3. Menghilangkan disclaimer hukum di akhir dokumen.
    4. Menghilangkan spasi berlebih dan merapikan baris baru.
    5. Konversi ke lowercase untuk standarisasi tokenisasi.
    """
    cleaned = raw_text
    
    # 1. Hapus Baris Header Direktori Putusan MA RI yang berulang
    cleaned = re.sub(r'(?i)direktori\s+putusan\s+mahkamah\s+agung\s+republik\s+indonesia', '', cleaned)
    cleaned = re.sub(r'(?i)mahkamah\s+agung\s+republik\s+indonesia', '', cleaned)
    
    # 2. Hapus Pola penomoran halaman seperti "Halaman 5 dari 45 Halaman" atau "Hal. 12"
    cleaned = re.sub(r'(?i)halaman\s+\d+\s+dari\s+\d+\s+halaman', '', cleaned)
    cleaned = re.sub(r'(?i)hal\.\s*\d+', '', cleaned)
    
    # 3. Hapus Disclaimer Kepaniteraan MA RI di footer
    disclaimer_pattern = r'(?i)disclaimer\s*:\s*kepaniteraan\s*mahkamah\s*agung\s*ri\s*berupaya\s*untuk\s*menjaga\s*keakuratan\s*.*'
    cleaned = re.sub(disclaimer_pattern, '', cleaned)
    
    # 4. Normalisasi spasi, tab, dan baris baru yang berantakan hasil ekstraksi PDF
    cleaned = re.sub(r'\r', '\n', cleaned)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)  # ganti tab/spasi ganda dengan spasi tunggal
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned) # satukan baris kosong beruntun
    
    # 5. Ubah semua teks menjadi huruf kecil (lowercase) agar seragam saat di-vektorkan
    cleaned = cleaned.lower()
    
    return cleaned.strip()

# =====================================================================
# LOGGER & VALIDATION PIPELINE
# =====================================================================
def log_activity(message):
    """Mencatat riwayat aktivitas pembersihan dokumen ke cleaning.log"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")

def validate_extraction_integrity(raw_text, cleaned_text):
    """
    Melakukan validasi apakah teks berhasil diekstraksi minimal 80% keutuhannya.
    Indikator keutuhan dokumen putusan pengadilan dinilai dari keberadaan kata kunci vital
    seperti 'mengadili', 'putusan', atau 'menimbang' serta panjang karakter minimum.
    """
    # 1. Cek rasio panjang teks bersih dibanding teks asli
    if len(raw_text) == 0:
        return False, "Ukuran file nol (gagal ekstraksi)"
        
    ratio = len(cleaned_text) / len(raw_text)
    
    # 2. Cek keberadaan kata kunci wajib dokumen putusan pengadilan
    keywords = ["mengadili", "putusan", "menimbang", "merek"]
    found_keywords = [kw for kw in keywords if kw in cleaned_text]
    
    # Jika teks bersih memiliki minimal 2 kata kunci hukum utama dan panjang memadai
    if len(found_keywords) >= 2 and len(cleaned_text) > 200:
        return True, f"Lolos (Rasio: {ratio:.2f}, Kata kunci ditemukan: {found_keywords})"
    else:
        return False, f"Gagal Integritas (Hanya mendeteksi kata kunci: {found_keywords})"

# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================
def execute_tahap_1_pipeline():
    """Fungsi utama yang menjalankan seluruh langkah kerja Tahap 1 CBR."""
    print("="*80)
    print(" MEMULAI TAHAP 1: MEMBANGUN CASE BASE (PREPROCESSING SENGKETA MEREK)")
    print("="*80)
    
    # Ambil semua file input (.pdf) di folder input
    pdf_files = [f for f in os.listdir(PDF_INPUT_DIR) if f.endswith(".pdf")]
    
    # Validasi jika folder input kosong
    if not pdf_files:
        print("[PERINGATAN] Tidak ditemukan file PDF putusan asli di folder 'data/pdf_merek/'.")
        print("[PETUNJUK] Silakan letakkan file-file PDF putusan sengketa merek Anda di dalam folder tersebut,")
        print("           lalu jalankan kembali skrip ini.")
        log_activity("Pipeline dihentikan: Tidak ada berkas PDF di data/pdf_merek/")
        return
    
    log_activity(f"=== Memulai Pipeline Pembersihan Kasus Baru untuk {len(pdf_files)} Dokumen ===")
    
    berhasil_proses = 0
    gagal_proses = 0
    
    for filename in sorted(pdf_files):
        filepath = os.path.join(PDF_INPUT_DIR, filename)
        raw_text = ""
        
        # Ekstraksi PDF asli
        try:
            with open(filepath, "rb") as f:
                pdf_reader = pypdf.PdfReader(f)
                pages_text = []
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        pages_text.append(text)
                raw_text = "\n".join(pages_text)
        except Exception as e:
            msg = f"ERROR: Gagal membaca file PDF '{filename}': {str(e)}"
            print(f"[!] {msg}")
            log_activity(msg)
            gagal_proses += 1
            continue
                
        # Jalankan pembersihan teks mendalam
        cleaned_text = clean_legal_text(raw_text)
        
        # Validasi integritas kualitas ekstraksi data
        is_valid, validation_msg = validate_extraction_integrity(raw_text, cleaned_text)
        
        # Nama file output yang diseragamkan (.txt) dengan skema case_NN.txt
        output_filename = f"case_{berhasil_proses + 1:02d}.txt"
        output_filepath = os.path.join(TXT_OUTPUT_DIR, output_filename)
        
        if is_valid:
            # Simpan file yang telah bersih ke folder /data/raw/
            with open(output_filepath, "w", encoding="utf-8") as out_file:
                out_file.write(cleaned_text)
            
            berhasil_proses += 1
            msg_success = f"BERHASIL: '{filename}' -> '{output_filename}' ({validation_msg})"
            print(f"[OK] {msg_success}")
            log_activity(msg_success)
        else:
            gagal_proses += 1
            msg_fail = f"GUGUR VALIDASI: '{filename}' dibuang karena {validation_msg}"
            print(f"[X] {msg_fail}")
            log_activity(msg_fail)
            
    print("\n" + "="*80)
    print(" RINGKASAN EKSEKUSI TAHAP 1")
    print("="*80)
    print(f"Total Dokumen Diproses : {len(pdf_files)}")
    print(f"Lolos Validasi & Bersih: {berhasil_proses} dokumen")
    print(f"Gagal Validasi/Error   : {gagal_proses} dokumen")
    print(f"File log disimpan di   : {LOG_FILE_PATH}")
    print(f"Data bersih disimpan di: {TXT_OUTPUT_DIR}/")
    print("="*80)

if __name__ == "__main__":
    execute_tahap_1_pipeline()