# Prediksi Skor Pertandingan Liga Primer Inggris (EPL)

Prototipe aplikasi web berbasis **Streamlit** dan **Machine Learning (XGBoost)**
untuk memprediksi skor pertandingan Liga Primer Inggris berdasarkan performa
tim pra-pertandingan. Dibangun mengikuti Bab 4 "Desain Sistem" (diagram use
case & diagram aktivitas) pada dokumen Metodologi Penelitian
*"Pengembangan Sistem Cerdas Berbasis Web untuk Memprediksi Skor Pertandingan
Liga Primer Inggris Menggunakan Machine Learning dan Analisis Performa Tim
Pra-Pertandingan"*.

## Kuesioner Usabilitas (System Usability Scale)

Kuesioner memakai 10 pernyataan baku **System Usability Scale (SUS)**
(Brooke, 1986) dengan polaritas bergantian (ganjil positif, genap negatif),
skala Likert 1-5, sesuai Bab 6.2 metodologi penelitian. Setiap jawaban
otomatis dikonversi menjadi skor SUS 0-100 memakai rumus baku
`((jumlah skor per-item) x 2.5)` dan diberi interpretasi (Poor/OK/Good/
Excellent).

**Penyimpanan jawaban** memakai dua backend dengan fallback otomatis:
1. **Google Sheets** (direkomendasikan untuk penyebaran publik) — dipakai
   otomatis bila kredensial sudah dikonfigurasi (lihat bagian "Integrasi
   Google Sheets" di bawah).
2. **File CSV lokal** (`usability_responses.csv`) — dipakai otomatis
   sebagai fallback jika Google Sheets belum dikonfigurasi atau gagal
   diakses. Cocok untuk pengujian lokal, tapi **akan hilang** jika
   aplikasi di-restart/redeploy di hosting gratis seperti Streamlit
   Community Cloud.

Halaman kuesioner selalu menampilkan backend mana yang sedang aktif, serta
rata-rata skor SUS dan rata-rata skor per pernyataan dari seluruh jawaban
yang sudah masuk.

### Integrasi Google Sheets

Agar jawaban responden tersimpan permanen (tidak hilang saat aplikasi
di-restart), hubungkan ke Google Sheets dengan langkah berikut:

1. **Buat Google Sheet baru** (kosong saja), lalu salin ID-nya dari URL:
   `https://docs.google.com/spreadsheets/d/ID_SPREADSHEET_ADA_DI_SINI/edit`.
2. Buka [Google Cloud Console](https://console.cloud.google.com/), buat
   project baru (atau pakai yang sudah ada), lalu aktifkan **Google
   Sheets API** dan **Google Drive API**.
3. Buat **Service Account** (menu *IAM & Admin -> Service Accounts ->
   Create Service Account*), lalu buat **key** baru bertipe **JSON** dan
   unduh filenya.
4. Buka file JSON tersebut, salin nilainya ke `.streamlit/secrets.toml`
   mengikuti format pada `.streamlit/secrets.toml.example` (field
   `client_email`, `private_key`, `project_id`, dst. persis sama dengan
   isi file JSON). Isi juga `sheet_id` dengan ID dari langkah 1.
5. **Bagikan (Share) Google Sheet** yang dibuat di langkah 1 ke alamat
   email `client_email` dari file JSON tadi (mis.
   `nama-app@project-id.iam.gserviceaccount.com`) dengan akses **Editor**
   — tanpa langkah ini, aplikasi tidak akan bisa menulis ke sheet.
6. Jalankan aplikasi secara lokal — jika berhasil, halaman kuesioner akan
   menampilkan "Jawaban akan disimpan ke: **Google Sheets**".
7. **Untuk Streamlit Community Cloud**: jangan commit `secrets.toml` asli
   ke GitHub. Sebagai gantinya, buka pengaturan aplikasi di Streamlit
   Cloud -> **Settings -> Secrets**, lalu tempel seluruh isi
   `secrets.toml` Anda di sana.

Jika kredensial salah atau Google Sheets tidak bisa diakses, aplikasi akan
menampilkan peringatan dan otomatis kembali memakai CSV lokal — jawaban
tidak pernah hilang begitu saja saat proses submit.

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi akan otomatis mengunduh data pertandingan dari **DataHub.io** saat
pertama kali dijalankan (butuh koneksi internet), lalu melatih model di
memori. Proses ini di-cache oleh Streamlit sehingga hanya berjalan sekali
selama sesi server aktif.

## Struktur Proyek

| File | Fungsi |
|---|---|
| `app.py` | Antarmuka Streamlit utama (4 menu: Prediksi, Statistik & Klasemen, Tentang Sistem, Kuesioner SUS) |
| `data_utils.py` | Pengambilan data sekunder dari DataHub.io (10 musim terakhir) |
| `features.py` | Rekayasa fitur performa tim pra-pertandingan (form 5 laga, head-to-head) tanpa data leakage |
| `models.py` | Pelatihan & evaluasi XGBoost (utama), Random Forest (benchmark), Logistic Regression (baseline), serta interpretasi SHAP |
| `standings.py` | Perhitungan klasemen liga & statistik kandang/tandang per tim per musim |
| `sheets_utils.py` | Integrasi penyimpanan jawaban kuesioner ke Google Sheets (opsional, fallback ke CSV lokal) |
| `requirements.txt` | Daftar dependensi Python |
| `.streamlit/secrets.toml.example` | Contoh format kredensial Google Sheets (salin jadi `secrets.toml`, isi nilai asli) |

## Sumber Data

- Data pertandingan: [DataHub.io - English Premier League](https://datahub.io/football/english-premier-league)
  (mendistribusikan ulang data dari [Football-Data.co.uk](https://www.football-data.co.uk/englandm.php)).
- Musim yang dipakai secara default: 10 musim terakhir dengan format 20
  tim/38 pekan (`season-1617` s.d. `season-2526`, yaitu musim 2016/17 s.d.
  2025/26), sesuai Bab 3.2 "Sampel dan Ukuran Sampel" pada metodologi
  penelitian. Cutoff sengaja diletakkan pada musim 2025/26 karena EPL saat
  ini sudah memasuki musim 2026/27, sehingga 2025/26 adalah musim terakhir
  yang datanya sudah lengkap 38 pekan. Bisa diubah lewat `DEFAULT_SEASONS`
  di `data_utils.py`.
- Jika DataHub.io tidak dapat diakses (mis. jaringan terbatas), aplikasi
  otomatis memakai **dataset sintetis darurat** agar tetap bisa didemokan,
  dan menampilkan peringatan di sidebar.

## Pemetaan ke Desain Sistem

**Diagram Use Case:**
- *End User*: menu "Prediksi Pertandingan" (pilih tim, lihat skor & probabilitas,
  lihat penjelasan SHAP, lihat statistik pra-pertandingan), menu "Statistik
  Tim & Klasemen Liga" (klasemen musim terbaru & statistik kandang/tandang
  per tim), dan menu "Kuesioner Usabilitas".
- *Peneliti*: menu "Tentang Sistem & Model" menampilkan hasil pelatihan,
  evaluasi (Accuracy/Precision/Recall/F1, RMSE/MAE), dan status model yang
  sudah "di-deploy" ke aplikasi.

**Diagram Aktivitas** (alur menu "Prediksi Pertandingan"):
Buka aplikasi → pilih tim kandang & tandang → validasi (tim tidak boleh sama)
→ ambil fitur pra-pertandingan → jalankan model XGBoost → tampilkan skor &
probabilitas → tampilkan penjelasan SHAP → ganti tim untuk mencoba lagi, atau
lanjut ke kuesioner usabilitas.

## Metode Machine Learning

- **Fitur**: rata-rata gol dicetak/kebobolan 5 laga terakhir, rata-rata poin,
  rata-rata tembakan & tembakan on-target **yang dilakukan**, rata-rata
  tembakan & tembakan on-target **yang dihadapi** (tekanan defensif), rasio
  akurasi tembakan sendiri (SOT/tembakan) dan akurasi tembakan lawan yang
  dihadapi, jumlah pertemuan head-to-head, win rate head-to-head, dan
  rata-rata selisih gol head-to-head — semua dihitung murni dari data
  **sebelum** tanggal pertandingan (time-aware, anti-leakage).
- **Model utama**: `XGBClassifier` (hasil H/D/A) + dua `XGBRegressor`
  (gol kandang & gol tandang).
- **Benchmark**: `RandomForestClassifier`/`RandomForestRegressor`.
- **Baseline**: `LogisticRegression`.
- **Pembagian data**: time-based split (bukan acak) — musim-musim awal
  sebagai data latih, laga-laga terbaru sebagai data uji.
- **Interpretabilitas**: SHAP (`TreeExplainer`) pada model klasifikasi.

## Statistik Tim & Klasemen Liga

Menu ini menghitung klasemen (peringkat, main, menang, seri, kalah, gol
dicetak/kebobolan, selisih gol, clean sheet, poin) langsung dari data
pertandingan yang sudah diunduh — bisa dipilih per musim (default: musim
terbaru yang tersedia, misalnya 2025/26), dilengkapi statistik detail per
tim yang dipecah performa kandang vs tandang.

## Kuesioner Usabilitas (System Usability Scale)

Kuesioner memakai 10 pernyataan baku **System Usability Scale (SUS)**
(Brooke, 1986) dengan polaritas bergantian (ganjil positif, genap negatif),
skala Likert 1-5, sesuai Bab 6.2 metodologi penelitian. Setiap jawaban
otomatis dikonversi menjadi skor SUS 0-100 dan diberi interpretasi
(Poor/OK/Good/Excellent). Lihat bagian "Kuesioner Usabilitas" di atas
untuk detail penyimpanan (Google Sheets / CSV lokal).

## Catatan & Keterbatasan

- Ini adalah **prototipe** untuk keperluan penelitian/skripsi, bukan produk
  produksi — akurasi model bergantung pada kualitas & jumlah data historis
  yang tersedia.
- **Possession (penguasaan bola) TIDAK dipakai** karena tidak tersedia di
  dataset Football-Data.co.uk/DataHub.io (kolom yang tersedia hanya
  gol, tembakan, tembakan on-target, pelanggaran, kartu, dan sepak
  pojok). Menambahkannya membutuhkan sumber data tambahan (mis. FBref/
  Understat) di luar cakupan DataHub.io yang dipakai sistem ini.
- Nama tim harus konsisten di semua file musim (mengikuti penamaan asli
  Football-Data.co.uk); tim yang baru promosi ke EPL akan memiliki riwayat
  form terbatas hingga cukup data terkumpul.
- Modul `models.py` dan `features.py` menggunakan `st.cache_resource`
  sehingga pelatihan ulang hanya terjadi jika data sumber berubah atau
  proses Streamlit di-restart.
