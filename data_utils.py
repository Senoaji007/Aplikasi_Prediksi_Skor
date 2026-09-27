"""
data_utils.py
=============
Modul pengumpulan data sekunder sesuai Bab 2 & 5 Metodologi Penelitian:
mengunduh data pertandingan historis Liga Primer Inggris dari DataHub.io
(media distribusi untuk data asli Football-Data.co.uk).

Setiap musim tersedia sebagai file CSV stabil di:
    https://datahub.io/football/english-premier-league/_r/-/season-XXYY.csv

Untuk mempercepat waktu muat (terutama saat aplikasi "bangun" dari mode
tidur di hosting gratis), modul ini:
  1. Mengunduh semua musim SECARA PARALEL (bukan satu per satu).
  2. Menyimpan hasil gabungan ke cache file di disk, sehingga proses
     berikutnya (selama file cache masih segar) cukup membaca file lokal
     tanpa perlu mengunduh ulang dari DataHub.io.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st

DATAHUB_BASE_URL = "https://datahub.io/football/english-premier-league/_r/-/{season}.csv"

DISK_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".cache_epl_data.parquet")
DISK_CACHE_MAX_AGE_SECONDS = 12 * 60 * 60  # selaras dengan ttl st.cache_data di bawah

# 10 musim kompetisi terakhir (format 20 tim / 38 pekan) -> sesuai Bab 3.2
# "Sampel dan Ukuran Sampel" pada dokumen metodologi penelitian.
# Cutoff pada musim 2025/2026 (season-2526) karena EPL saat ini sudah
# memasuki musim 2026/2027, sehingga 2025/2026 adalah musim terakhir yang
# sudah selesai penuh (38 pekan) dan layak dipakai sebagai data latih/uji.
DEFAULT_SEASONS = [
    "season-1617",
    "season-1718",
    "season-1819",
    "season-1920",
    "season-2021",
    "season-2122",
    "season-2223",
    "season-2324",
    "season-2425",
    "season-2526",
]

REQUIRED_COLUMNS = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR", "HS", "AS", "HST", "AST",
    "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
]

NUMERIC_COLUMNS = [
    "FTHG", "FTAG", "HTHG", "HTAG", "HS", "AS", "HST", "AST",
    "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
]


def _fetch_season_csv(season: str, timeout: int = 12) -> pd.DataFrame | None:
    """Mengunduh satu file CSV musim dari DataHub.io. Mengembalikan None bila gagal."""
    url = DATAHUB_BASE_URL.format(season=season)
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        if df.empty:
            return None
        df["Season"] = season.replace("season-", "")
        return df
    except Exception:
        return None


def _fetch_all_seasons_parallel(seasons: list[str], max_workers: int = 6):
    """
    Mengunduh semua musim SECARA PARALEL memakai thread pool, jauh lebih
    cepat dibanding mengunduh satu per satu terutama saat latensi jaringan
    ke DataHub.io cukup tinggi (mis. dari server hosting di luar negeri).
    """
    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_season = {executor.submit(_fetch_season_csv, s): s for s in seasons}
        for future in as_completed(future_to_season):
            season = future_to_season[future]
            df = future.result()
            if df is not None:
                frames.append(df)
            else:
                failed.append(season)
    return frames, failed


def _generate_fallback_dataset(n_teams: int = 20, n_seasons: int = 3, seed: int = 42) -> pd.DataFrame:
    """
    Dataset sintetis darurat, dipakai HANYA jika DataHub.io tidak dapat
    diakses (mis. jaringan terbatas), agar aplikasi tetap dapat didemokan.
    Data ini TIDAK merepresentasikan pertandingan Liga Primer Inggris nyata.
    """
    rng = np.random.default_rng(seed)
    teams = [f"Tim {chr(65 + i)}" for i in range(n_teams)]
    rows = []
    start = pd.Timestamp("2022-08-01")
    for s in range(n_seasons):
        season_label = f"Sintetis-{s + 1}"
        matchday = 0
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                matchday += 1
                date = start + pd.Timedelta(days=matchday + s * 400)
                home_strength = rng.normal(1.4, 0.4)
                away_strength = rng.normal(1.1, 0.4)
                fthg = max(0, int(rng.poisson(max(0.2, home_strength))))
                ftag = max(0, int(rng.poisson(max(0.2, away_strength))))
                hthg = min(fthg, int(rng.integers(0, fthg + 1)))
                htag = min(ftag, int(rng.integers(0, ftag + 1)))
                rows.append({
                    "Date": date, "HomeTeam": home, "AwayTeam": away,
                    "FTHG": fthg, "FTAG": ftag,
                    "FTR": "H" if fthg > ftag else ("A" if ftag > fthg else "D"),
                    "HTHG": hthg, "HTAG": htag,
                    "HTR": "H" if hthg > htag else ("A" if htag > hthg else "D"),
                    "HS": rng.integers(6, 20), "AS": rng.integers(4, 18),
                    "HST": rng.integers(2, 10), "AST": rng.integers(1, 9),
                    "HF": rng.integers(5, 16), "AF": rng.integers(5, 16),
                    "HC": rng.integers(2, 11), "AC": rng.integers(1, 10),
                    "HY": rng.integers(0, 4), "AY": rng.integers(0, 4),
                    "HR": rng.integers(0, 1), "AR": rng.integers(0, 1),
                    "Season": season_label,
                })
    return pd.DataFrame(rows)


def _load_disk_cache() -> pd.DataFrame | None:
    """Membaca cache dataset dari disk jika ada dan masih cukup segar."""
    if not os.path.exists(DISK_CACHE_PATH):
        return None
    age = time.time() - os.path.getmtime(DISK_CACHE_PATH)
    if age > DISK_CACHE_MAX_AGE_SECONDS:
        return None
    try:
        return pd.read_parquet(DISK_CACHE_PATH)
    except Exception:
        return None


def _save_disk_cache(data: pd.DataFrame) -> None:
    try:
        data.to_parquet(DISK_CACHE_PATH, index=False)
    except Exception:
        pass  # Cache disk bersifat opsional; kegagalan menulis tidak fatal


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_epl_dataset(seasons: tuple[str, ...] | None = None):
    """
    Mengunduh & menggabungkan data pertandingan EPL dari DataHub.io.

    Urutan sumber data (dari yang tercepat): cache disk yang masih segar
    -> unduhan paralel dari DataHub.io (lalu ditulis ke cache disk) ->
    dataset sintetis darurat bila keduanya gagal.

    Returns
    -------
    data : pd.DataFrame
        Dataset gabungan yang sudah dibersihkan & diurutkan berdasarkan tanggal.
    failed_seasons : list[str]
        Daftar musim yang gagal diunduh (kosong bila semua berhasil, atau
        bila data diambil dari cache disk).
    used_fallback : bool
        True bila seluruh unduhan gagal dan dataset sintetis darurat dipakai.
    """
    seasons = list(seasons) if seasons else DEFAULT_SEASONS

    cached = _load_disk_cache()
    if cached is not None:
        return cached, [], False

    frames, failed = _fetch_all_seasons_parallel(seasons)

    used_fallback = False
    if not frames:
        data = _generate_fallback_dataset()
        used_fallback = True
    else:
        data = pd.concat(frames, ignore_index=True, sort=False)

    for col in REQUIRED_COLUMNS:
        if col not in data.columns:
            data[col] = pd.NA

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce", dayfirst=True)
    data = data.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])

    for col in NUMERIC_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.sort_values("Date").reset_index(drop=True)

    if not used_fallback:
        _save_disk_cache(data)

    return data, failed, used_fallback


def get_team_list(data: pd.DataFrame) -> list[str]:
    teams = sorted(set(data["HomeTeam"].dropna().unique()) | set(data["AwayTeam"].dropna().unique()))
    return teams


def format_season_label(season_code: str) -> str:
    """Mengubah kode musim mis. '2526' menjadi label '2025/26'."""
    code = str(season_code).replace("season-", "")
    if len(code) == 4 and code.isdigit():
        return f"20{code[:2]}/{code[2:]}"
    return code


def get_available_seasons(data: pd.DataFrame) -> list[str]:
    """Daftar kode musim yang tersedia di data, terurut dari terbaru ke terlama."""
    seasons = [s for s in data["Season"].dropna().unique().tolist()]
    seasons.sort(reverse=True)
    return seasons
