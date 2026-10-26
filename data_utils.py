"""
data_utils.py
=============
Pengumpulan data sekunder pertandingan Liga Primer Inggris (EPL).

PERBAIKAN (masalah: statistik/klasemen berubah-ubah, semua tim "main 15 kali",
juara tiap musim tidak sesuai):

  * DataHub.io memperbarui datasetnya SETIAP HARI lewat GitHub Actions.
    Dulu aplikasi mengunduh dari DataHub saat dijalankan, lalu menyimpan hasilnya
    di cache 12 jam TANPA pernah memeriksa isinya. Jika satu unduhan salah/terpotong,
    data salah itu ikut tersimpan dan dipakai untuk klasemen + pelatihan model.
  * Label musim dulu diambil dari NAMA FILE yang diminta, bukan dari isi data.
    Jadi file yang isinya salah tetap diberi label musim yang "benar".

Sekarang:
  1. Data dibaca dari berkas lokal  data/season-XXYY.csv  (di-commit ke repo ->
     tetap, tidak berubah sendiri, hasil bisa direproduksi).
  2. Bila berkas lokal belum ada, diunduh dari football-data.co.uk (sumber asli),
     lalu DataHub.io sebagai cadangan, kemudian DISIMPAN ke data/.
  3. Setiap musim DIVALIDASI (380 laga, 20 tim, 38 laga/tim, tanggal sesuai musim).
     Musim yang tidak lolos dibuang dan dilaporkan, bukan dipakai diam-diam.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
DATAHUB_BASE_URL = "https://datahub.io/football/english-premier-league/_r/-/{season}.csv"

EXPECTED_TEAMS = 20
EXPECTED_MATCHES = 380
EXPECTED_GAMES_PER_TEAM = 38

# 10 musim kompetisi terakhir yang sudah selesai (20 tim / 38 pekan).
DEFAULT_SEASONS = [
    "season-1617", "season-1718", "season-1819", "season-1920", "season-2021",
    "season-2122", "season-2223", "season-2324", "season-2425", "season-2526",
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

_CORE_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


def _code(season: str) -> str:
    return season.replace("season-", "")


def _local_path(season: str) -> str:
    return os.path.join(DATA_DIR, f"{season}.csv")


# ---------------------------------------------------------------------------
# Pembersihan & validasi
# ---------------------------------------------------------------------------
def _clean_season_frame(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Membersihkan satu file musim: kolom, tanggal, angka, baris kosong, duplikat."""
    df = df.copy()
    # buang baris/kolom kosong yang sering ada di akhir CSV football-data
    df = df.dropna(how="all")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    # Tanggal football-data: dd/mm/yy atau dd/mm/yyyy; DataHub kadang yyyy-mm-dd.
    raw = df["Date"].astype(str).str.strip()
    iso = raw.str.match(r"^\d{4}-\d{2}-\d{2}")
    parsed = pd.to_datetime(raw.where(~iso), errors="coerce", dayfirst=True)
    parsed = parsed.fillna(pd.to_datetime(raw.where(iso), errors="coerce"))
    df["Date"] = parsed
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=_CORE_COLUMNS)
    df = df.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
    # Label musim diturunkan dari kode musim yang DIMINTA, lalu DIVERIFIKASI
    # terhadap isi data di validate_season().
    df["Season"] = _code(season)
    return df.reset_index(drop=True)


def validate_season(df: pd.DataFrame, season: str) -> list[str]:
    """
    Mengembalikan daftar masalah (kosong = data musim valid).
    Musim yang sudah selesai harus 380 laga, 20 tim, dan tiap tim 38 laga.
    """
    problems: list[str] = []
    code = _code(season)
    if len(df) != EXPECTED_MATCHES:
        problems.append(f"jumlah laga {len(df)} (seharusnya {EXPECTED_MATCHES})")
    teams = set(df["HomeTeam"]) | set(df["AwayTeam"])
    if len(teams) != EXPECTED_TEAMS:
        problems.append(f"jumlah tim {len(teams)} (seharusnya {EXPECTED_TEAMS})")
    played = pd.concat([df["HomeTeam"], df["AwayTeam"]]).value_counts()
    if len(played) and (played != EXPECTED_GAMES_PER_TEAM).any():
        problems.append(
            f"laga per tim {int(played.min())}-{int(played.max())} "
            f"(seharusnya {EXPECTED_GAMES_PER_TEAM})"
        )
    if code.isdigit() and len(code) == 4 and len(df):
        y1, y2 = 2000 + int(code[:2]), 2000 + int(code[2:])
        lo, hi = pd.Timestamp(y1, 7, 1), pd.Timestamp(y2, 7, 31)
        outside = int(((df["Date"] < lo) | (df["Date"] > hi)).sum())
        if outside:
            problems.append(f"{outside} laga bertanggal di luar musim {y1}/{str(y2)[2:]}")
    return problems


# ---------------------------------------------------------------------------
# Pengambilan data
# ---------------------------------------------------------------------------
def _download_csv(url: str, timeout: int = 20) -> pd.DataFrame | None:
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        # football-data memakai encoding latin-1 pada sebagian musim
        text = resp.content.decode("utf-8-sig", errors="replace")
        df = pd.read_csv(StringIO(text), on_bad_lines="skip")
        return None if df.empty else df
    except Exception:
        return None


def _load_one_season(season: str) -> tuple[str, pd.DataFrame | None, list[str]]:
    """
    Memuat & memvalidasi satu musim. Urutan: berkas lokal -> football-data.co.uk
    -> DataHub.io. Hanya data yang LOLOS validasi yang dipakai/disimpan.
    Return: (season, dataframe_valid_atau_None, daftar_catatan_masalah)
    """
    notes: list[str] = []
    candidates: list[tuple[str, callable]] = []
    path = _local_path(season)
    if os.path.exists(path):
        candidates.append(("lokal", lambda: pd.read_csv(path)))
    candidates.append(("football-data.co.uk", lambda: _download_csv(FOOTBALL_DATA_URL.format(code=_code(season)))))
    candidates.append(("DataHub.io", lambda: _download_csv(DATAHUB_BASE_URL.format(season=season))))

    for source, getter in candidates:
        raw = getter()
        if raw is None:
            notes.append(f"{source}: gagal dibaca")
            continue
        df = _clean_season_frame(raw, season)
        problems = validate_season(df, season)
        if not problems:
            if source != "lokal":
                try:
                    os.makedirs(DATA_DIR, exist_ok=True)
                    raw.to_csv(path, index=False)  # simpan agar berikutnya stabil
                except Exception:
                    pass
            return season, df, notes
        notes.append(f"{source}: " + "; ".join(problems))
    return season, None, notes


def _load_all_seasons(seasons: list[str], max_workers: int = 6):
    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    notes: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_load_one_season, s) for s in seasons]
        for fut in as_completed(futures):
            season, df, n = fut.result()
            if df is not None:
                frames.append(df)
            else:
                failed.append(season)
            if n:
                notes[season] = n
    return frames, sorted(failed), notes


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


@st.cache_data(show_spinner=False)
def load_epl_dataset(seasons: tuple[str, ...] | None = None):
    """
    Memuat & menggabungkan data pertandingan EPL yang SUDAH DIVALIDASI.

    Tidak ada lagi TTL/cache disk yang menyimpan hasil unduhan mentah: data
    berasal dari data/season-XXYY.csv sehingga hasilnya selalu sama.

    Returns
    -------
    data : pd.DataFrame
    failed_seasons : list[str]   musim yang tidak ada / tidak lolos validasi
    used_fallback : bool         True bila semua gagal dan data sintetis dipakai
    """
    seasons = list(seasons) if seasons else DEFAULT_SEASONS
    frames, failed, _notes = _load_all_seasons(seasons)

    used_fallback = False
    if not frames:
        data = _generate_fallback_dataset()
        used_fallback = True
        for col in REQUIRED_COLUMNS:
            if col not in data.columns:
                data[col] = pd.NA
    else:
        data = pd.concat(frames, ignore_index=True, sort=False)
        data = data.drop_duplicates(subset=["Season", "Date", "HomeTeam", "AwayTeam"])

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for col in NUMERIC_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=_CORE_COLUMNS)
    data = data.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)
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
