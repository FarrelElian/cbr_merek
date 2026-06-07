# -*- coding: utf-8 -*-
"""
Tugas Penalaran Komputer - SIKLUS CBR (Tahap 2: Case Representation)
Studi Kasus: Sengketa Merek & Indikasi Geografis (UU No. 20 Tahun 2016)
Fakultas Teknik - Informatika UMM
"""

import os
import re
import json
import pandas as pd

# =====================================================================
# CONFIGURATION & DIRECTORY SETUP
# =====================================================================
TXT_INPUT_DIR = "data/raw"             # Tempat menyimpan file teks hasil Tahap 1
PROCESSED_DIR = "data/processed"       # Direktori output untuk representasi kasus
JSON_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "cases.json")
CSV_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "cases.csv")

# Memastikan folder output sudah siap
os.makedirs(PROCESSED_DIR, exist_ok=True)

# =====================================================================
# METADATA EXTRACTION ENGINE (TANGGUH & ADAPTIF)
# =====================================================================
def extract_metadata_from_text(text, filename):
    """
    Menggunakan ekspresi reguler (Regex) adaptif tingkat tinggi untuk mengekstrak 
    metadata penting secara otomatis dari berbagai tingkatan putusan sengketa merek.
    """
    
    # 1. Ekstraksi Nomor Perkara (Mendukung PN, Kasasi/K, PK, dan TUN)
    no_perkara = "Tidak Terdeteksi"
    no_perkara_patterns = [
        r"nomor\s+(\d+\s+[a-z]+(?:\s+sus)?/pdt\.\s*sus-hki/\d+)",
        r"nomor\s+(\d+\s+[a-z]+/pdt\.\s*sus[^\n]+)",
        r"nomor\s*:\s*(\d+/pdt\.sus-hki/merek/\d+/pn\.[a-z\.]+)",
        r"putusan\s+nomor\s*:\s*(\d+/pdt\.sus[^\n]+)",
        r"nomor\s+(\d+/pdt\.sus-hki[^\n]+)",
        r"register\s+no\.\s*(\d+/[^\n]+)"
    ]
    for pattern in no_perkara_patterns:
        match = re.search(pattern, text)
        if match:
            no_perkara = match.group(1).strip().upper()
            break

    # 2. Ekstraksi Tanggal Putusan (Mencari pola tanggal Indonesia)
    tanggal_putusan = "Tidak Terdeteksi"
    tanggal_match = re.search(r'(?:tanggal|tgl)\s+(\d{1,2}\s+[a-zA-Z\s]+\s+\d{4})', text)
    if tanggal_match:
        tanggal_putusan = tanggal_match.group(1).strip()
    else:
        # Cadangan pencarian tahun di nomor perkara jika tanggal sidang tidak ketemu
        tahun_match = re.search(r'/20\d{2}', no_perkara)
        if tahun_match:
            tanggal_putusan = "Sidang Tahun " + tahun_match.group(0).replace("/", "").strip()

    # 3. Ekstraksi Pihak Penggugat & Tergugat (Metode Pembagi "Lawan" Super Tangguh)
    penggugat = "TIDAK TERDETEKSI"
    tergugat = "TIDAK TERDETEKSI"
    
    # Mencari blok besar paragraf yang mendefinisikan para pihak sebelum masuk ke substansi putusan
    party_block_match = re.search(
        r'(?:dalam perkara|memutus sebagai berikut|perkara antara|perdata khusus[^\n]*dalam perkara)\s*:(.*?)(?:mahkamah agung tersebut|membaca surat-surat|menimbang|duduk perkara|pengadilan negeri tersebut|pengadilan niaga tersebut)', 
        text, 
        re.DOTALL | re.IGNORECASE
    )
    
    if party_block_match:
        party_text = party_block_match.group(1)
        # Memisahkan blok Penggugat dan Tergugat menggunakan pembatas kata "Lawan"
        parts = re.split(r'\n\s*lawan\s*\n|\s+lawan\s+', party_text, flags=re.IGNORECASE)
        if len(parts) >= 2:
            raw_p = parts[0].strip()
            raw_t = parts[1].strip()
            
            def clean_party_name(block):
                # Merapikan spasi ganda
                block = re.sub(r'\s+', ' ', block).strip()
                # Membuang penomoran di depan nama jika ada
                block = re.sub(r'^\d+[\s\.\)]+', '', block).strip()
                
                # Mengambil nama entitas utama sebelum kata penjelas hukum yang panjang muncul
                delimiters = [
                    r',\s*yang\s+diwakili', r',\s*berkedudukan', r',\s*beralamat', 
                    r',\s*dalam\s+hal\s+ini', r'\s+yang\s+diwakili', r'\s+berkedudukan', 
                    r'\s+beralamat', r'\s+dalam\s+hal\s+ini', r',\s*selaku', r';', r'\.'
                ]
                pattern = '|'.join(delimiters)
                split_parts = re.split(pattern, block, flags=re.IGNORECASE)
                if split_parts:
                    candidate = split_parts[0].strip()
                    # Membersihkan sisa simbol di ujung string
                    candidate = re.sub(r'^[,\s\-\.\:\(\)]+|[,\s\-\.\:\(\)]+$', '', candidate)
                    # Membuang prefix sebutan pihak jika tidak sengaja tertangkap
                    candidate = re.sub(r'^(?:penggugat|tergugat|pemohon|termohon|pembanding|terbanding|turut\s+tergugat)\s*:\s*', '', candidate, flags=re.IGNORECASE)
                    
                    if len(candidate) > 3:
                        return candidate.strip().upper()
                return block[:80].strip().upper()
                
            penggugat = clean_party_name(raw_p)
            tergugat = clean_party_name(raw_t)

    # Filter Pengaman tambahan agar kalimat sampah tidak lolos menjadi nama pihak
    bad_keywords = ["eksepsi", "menyatakan", "bahwa", "tersebut", "menimbang", "mengadili", "biaya perkara", "melakukan", "tidak"]
    if any(kw in penggugat.lower() for kw in bad_keywords) or len(penggugat) > 120 or penggugat == "TIDAK TERDETEKSI":
        # Fallback pencarian PT / CV terdekat di awal berkas jika blok jackpot utama gagal
        company_match = re.findall(r'\b(?:pt|cv|co\s*\.?\s*ltd|ltd)\s+[a-z0-9\s\.,\-\(&\)]+', text, re.IGNORECASE)
        if company_match:
            penggugat = re.split(r',\s*yang|,\s*berkedudukan|,\s*beralamat', company_match[0], flags=re.IGNORECASE)[0].strip().upper()
        else:
            penggugat = "TIDAK TERDETEKSI"

    if any(kw in tergugat.lower() for kw in bad_keywords) or len(tergugat) > 120 or tergugat == "TIDAK TERDETEKSI":
        company_match = re.findall(r'\b(?:pt|cv|co\s*\.?\s*ltd|ltd)\s+[a-z0-9\s\.,\-\(&\)]+', text, re.IGNORECASE)
        if company_match and len(company_match) >= 2:
            tergugat = re.split(r',\s*yang|,\s*berkedudukan|,\s*beralamat', company_match[1], flags=re.IGNORECASE)[0].strip().upper()
        else:
            tergugat = "DJKI / DIREKTORAT JENDERAL KEKAYAAN INTELEKTUAL" # Fallback umum sengketa merek

    pihak_gabungan = f"{penggugat} VS {tergugat}"

    # 4. Ekstraksi Merek (Merek Terkenal vs Merek Tiruan)
    raw_merek_matches = re.findall(r'merek\s+["\'“‘]([^"\'”’]+)["\'”’]', text)
    if not raw_merek_matches:
        # Fallback tanpa tanda kutip (merek kata tunggal/ganda)
        raw_merek_matches = re.findall(r'merek\s+([a-z0-9\-]+(?:\s+[a-z0-9\-]+){0,1})', text)

    stop_words_merek = {
        "terdaftar", "milik", "yang", "pada", "adalah", "selaku", "dalam", "nomor", 
        "no", "dan", "tergugat", "penggugat", "itu", "tersebut", "di", "bawah", "dengan",
        "atas", "telah", "mempunyai", "persamaan", "pokoknya", "sebagai", "gugatan", "dibatalkan",
        "agar", "oleh", "ia", "ini", "untuk"
    }
    
    filtered_merek = []
    for m in raw_merek_matches:
        words_in_match = m.split()
        clean_m = []
        for w in words_in_match:
            if w in stop_words_merek:
                break
            clean_m.append(w)
        if clean_m:
            filtered_merek.append(" ".join(clean_m).upper())
            
    filtered_merek = list(dict.fromkeys([m.strip() for m in filtered_merek if len(m.strip()) > 1 and len(m.strip()) < 45]))
    
    merek_penggugat = filtered_merek[0] if len(filtered_merek) > 0 else "TIDAK DIKETAHUI"
    merek_tergugat = filtered_merek[1] if len(filtered_merek) > 1 else (filtered_merek[0] if len(filtered_merek) > 0 else "TIDAK DIKETAHUI")

    # 5. Ekstraksi Pasal rujukan utama
    pasal_list = re.findall(r'(pasal\s+\d+\s+(?:ayat\s*\(\d+\)\s+)?(?:uu|undang-undang)\s+(?:no\.|nomor\s+)?\d+[\/\w\s]*)', text)
    pasal_list = list(dict.fromkeys([p.strip() for p in pasal_list if len(p.strip()) > 5]))
    pasal_terpakai = pasal_list[0].upper() if pasal_list else "UU NO. 20 TAHUN 2016 TENTANG MEREK"

    # 6. Solusi Hukum & Detail Amar
    solusi_hukum = "GUGATAN DITOLAK"
    amar_match = re.search(r'mengadili\s*[:\.]?\s*\n?(.*?)(?:\n\s*\n|\n\s*menimbang|\n\s*demikian|halaman|$)', text, re.DOTALL)
    if amar_match:
        solusi_content = amar_match.group(1).strip()
        if any(keyword in solusi_content.lower() for keyword in ["kabul", "mengabulkan", "membatalkan", "hapus", "menghapuskan"]):
            solusi_hukum = "GUGATAN DIKABULKAN (PEMBATALAN / PENGHAPUSAN MEREK TERGUGAT)"
            
    detail_solusi = amar_match.group(1).strip() if amar_match else "Amar putusan diserahkan pada pertimbangan majelis hakim."
    detail_solusi = re.sub(r'\s+', ' ', detail_solusi)[:250] + "..."

    return {
        "no_perkara": no_perkara,
        "tanggal": tanggal_putusan,
        "pihak": pihak_gabungan,
        "penggugat_clean": penggugat,
        "tergugat_clean": tergugat,
        "merek_penggugat": merek_penggugat,
        "merek_tergugat": merek_tergugat,
        "pasal_rujukan": pasal_terpakai,
        "solusi_hukum": solusi_hukum,
        "detail_solusi": detail_solusi
    }

# =====================================================================
# FEATURE ENGINEERING, BOW, & QA-PAIR CREATION
# =====================================================================
def perform_bag_of_words(text):
    """
    Menghasilkan representasi Bag-of-Words (BoW) sederhana berisi 
    10 kata hukum sengketa merek terpopuler beserta frekuensinya (di luar stop words).
    """
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    
    stop_words = {
        "yang", "dalam", "bahwa", "dengan", "untuk", "pada", "oleh", "atas", "tidak", 
        "adalah", "terhadap", "dari", "telah", "tentang", "sebagaimana", "atau", "merupakan", 
        "kepada", "karena", "kami", "saya", "ini", "itu", "halaman", "putusan", "nomor"
    }
    
    word_counts = {}
    for w in words:
        if w not in stop_words:
            word_counts[w] = word_counts.get(w, 0) + 1
            
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    return dict(sorted_words)


def perform_feature_engineering(text, metadata):
    """
    Melakukan ekstraksi fitur statistik panjang teks, analisis kualitatif sengketa merek, 
    dan menyusun pasangan Tanya-Jawab (QA-Pairs) dinamis.
    """
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    
    # 1. Deteksi Indikator Sengketa Merek Utama
    has_persamaan_pokok = "YA" if any(x in text.lower() for x in ["persamaan pada pokoknya", "kemiripan", "mirip"]) else "TIDAK"
    has_iktikad_tidak_baik = "YA" if any(x in text.lower() for x in ["iktikad tidak baik", "itikad tidak baik", "membonceng"]) else "TIDAK"
    has_non_use = "YA" if any(x in text.lower() for x in ["tidak menggunakan", "berturut-turut", "penghapusan merek"]) else "TIDAK"
    
    # 2. Rekayasa Ringkasan Fakta Dinamis (Menghubungkan Penggugat, Tergugat, dan Merek)
    ringkasan_fakta = (
        f"Kasus sengketa merek antara Penggugat ({metadata['penggugat_clean']}) selaku pemilik merek asli '{metadata['merek_penggugat']}' "
        f"melawan Tergugat ({metadata['tergugat_clean']}) terkait penggunaan merek '{metadata['merek_tergugat']}'."
    )
    if has_non_use == "YA":
        ringkasan_fakta += " Penggugat memohon penghapusan merek Tergugat dari Daftar Umum Merek karena dinilai tidak digunakan selama 3 tahun berturut-turut."
    elif has_persamaan_pokok == "YA":
        ringkasan_fakta += " Penggugat menuntut pembatalan merek Tergugat karena memiliki persamaan pada pokoknya dengan merek terkenal milik Penggugat."

    # 3. Rekayasa Pasangan Tanya-Jawab Sederhana (QA-Pairs)
    qa_pairs = [
        {"q": "Siapa pemilik merek asli yang sah?", "a": metadata["penggugat_clean"]},
        {"q": "Merek apa saja yang menjadi objek sengketa?", "a": f"'{metadata['merek_penggugat']}' VS '{metadata['merek_tergugat']}'"},
        {"q": "Apakah terdeteksi unsur persamaan pada pokoknya?", "a": has_persamaan_pokok},
        {"q": "Apakah terdapat gugatan penghapusan akibat non-use (merek pasif)?", "a": has_non_use},
        {"q": "Apa putusan akhir dari Majelis Hakim?", "a": metadata["solusi_hukum"]}
    ]
    
    return {
        "panjang_kata": word_count,
        "panjang_karakter": char_count,
        "ringkasan_fakta": ringkasan_fakta,
        "fitur_persamaan_pokok": has_persamaan_pokok,
        "fitur_iktikad_tidak_baik": has_iktikad_tidak_baik,
        "fitur_non_use": has_non_use,
        "qa_pairs": qa_pairs
    }

# =====================================================================
# MAIN PIPELINE EXECUTION (TAHAP 2)
# =====================================================================
def execute_tahap_2_pipeline():
    print("="*80)
    print(" MEMULAI TAHAP 2: CASE REPRESENTATION (STRUKTURISASI DATA MEREK)")
    print("="*80)
    
    raw_files = [f for f in os.listdir(TXT_INPUT_DIR) if f.endswith(".txt")]
    
    if not raw_files:
        print("[PERINGATAN] Folder 'data/raw/' kosong.")
        print("[PETUNJUK] Pastikan Anda telah sukses menjalankan Tahap 1 (01_membangun_case_base.py).")
        return
        
    structured_cases = []
    
    for filename in sorted(raw_files):
        filepath = os.path.join(TXT_INPUT_DIR, filename)
        case_id = os.path.splitext(filename)[0]
        
        with open(filepath, "r", encoding="utf-8") as f:
            text_content = f.read()
            
        # 1. Jalankan Ekstraksi Metadata Hukum
        metadata = extract_metadata_from_text(text_content, filename)
        
        # 2. Hitung Fitur Bag-of-Words (BoW)
        bow_features = perform_bag_of_words(text_content)
        
        # 3. Jalankan Feature Engineering & Ringkasan Fakta
        features = perform_feature_engineering(text_content, metadata)
        
        # 4. Susun representasi Kasus Lengkap
        case_representation = {
            "case_id": case_id,
            "no_perkara": metadata["no_perkara"],
            "tanggal": metadata["tanggal"],
            "pihak": metadata["pihak"],
            "pasal": metadata["pasal_rujukan"],
            "merek_penggugat": metadata["merek_penggugat"],
            "merek_tergugat": metadata["merek_tergugat"],
            "ringkasan_fakta": features["ringkasan_fakta"],
            "fakta_persamaan_pokok": features["fitur_persamaan_pokok"],
            "fakta_iktikad_tidak_baik": features["fitur_iktikad_tidak_baik"],
            "fakta_non_use": features["fitur_non_use"],
            "length_kata": features["panjang_kata"],
            "length_karakter": features["panjang_karakter"],
            "solusi_hukum": metadata["solusi_hukum"],
            "detail_amar": metadata["detail_solusi"],
            "bag_of_words": bow_features,
            "qa_pairs": features["qa_pairs"],
            "text_full": text_content
        }
        
        structured_cases.append(case_representation)
        # Menampilkan cetakan yang informatif, melampirkan Pihak sekaligus nama Merek agar mudah diverifikasi
        print(f"[✔] Terstruktur: {case_id}\n"
              f"    └─ No: {metadata['no_perkara']} | {metadata['tanggal']}\n"
              f"    └─ Pihak: {metadata['pihak']}\n"
              f"    └─ Merek: {metadata['merek_penggugat']} VS {metadata['merek_tergugat']}\n")
        
    # --- PROSES PENYIMPANAN DATA ---
    # Menggunakan try-except block untuk mencegah crash jika file sedang dikunci oleh Excel
    try:
        with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as json_file:
            json.dump(structured_cases, json_file, indent=4)
            print(f"[✔] Sukses menyimpan representasi kaya ke JSON.")
    except PermissionError:
        print("\n" + "!"*80)
        print(" GAGAL MENYIMPAN FILE JSON!")
        print("!"*80)
        print(f"Error: Permission denied pada '{JSON_OUTPUT_PATH}'")
        print("Solusi: Pastikan file JSON tersebut tidak sedang dibuka di text editor eksternal.")
        print("!"*80 + "\n")
        return
        
    try:
        tabular_data = []
        for case in structured_cases:
            case_copy = case.copy()
            case_copy.pop("qa_pairs", None)
            case_copy.pop("bag_of_words", None)
            tabular_data.append(case_copy)
            
        df = pd.DataFrame(tabular_data)
        kolom_wajib = ["case_id", "no_perkara", "tanggal", "ringkasan_fakta", "pasal", "pihak", "text_full"]
        kolom_tambahan = [c for c in df.columns if c not in kolom_wajib]
        df_final = df[kolom_wajib + kolom_tambahan]
        
        df_final.to_csv(CSV_OUTPUT_PATH, index=False)
        print(f"[✔] Sukses mengekspor tabel ke CSV.")
    except PermissionError:
        print("\n" + "!"*80)
        print(" GAGAL MENYIMPAN FILE CSV!")
        print("!"*80)
        print(f"Error: Permission denied pada '{CSV_OUTPUT_PATH}'")
        print("Penyebab Utama: Anda sedang membuka file 'cases.csv' di MICROSOFT EXCEL atau program lain.")
        print("Solusi        : SILAKAN TUTUP MICROSOFT EXCEL terlebih dahulu, lalu jalankan kembali skrip ini.")
        print("!"*80 + "\n")
        return
    
    print("\n" + "="*80)
    print(" RINGKASAN EKSEKUSI TAHAP 2 (REVISI STRUKTUR LENGKAP)")
    print("="*80)
    print(f"Total Kasus Berhasil Direpresentasikan: {len(structured_cases)} dokumen")
    print(f"File Terstruktur JSON (Rich Data)     : {JSON_OUTPUT_PATH}")
    print(f"File Terstruktur CSV (Sesuai Dosen)   : {CSV_OUTPUT_PATH}")
    print("="*80)

if __name__ == "__main__":
    execute_tahap_2_pipeline()