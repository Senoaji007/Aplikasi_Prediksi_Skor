"""
features.py
============
Rekayasa fitur performa tim PRA-PERTANDINGAN (Bab 5.1 poin 4 Metodologi
Penelitian): form 5 laga terakhir, rata-rata gol, rata-rata statistik
tembakan, dan rekam jejak head-to-head -- dihitung secara satu-pass
kronologis agar TIDAK ADA kebocoran data (data leakage) dari hasil
pertandingan ke fitur pertandingan itu sendiri.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd
import streamlit as st

N_FORM = 5  # jumlah pertandingan terakhir untuk fitur "form"

FEATURE_COLUMNS = [
    "home_form_gf", "home_form_ga", "home_form_points",
    "home_form_shots", "home_form_sot", "home_form_shots_faced", "home_form_sot_faced",
    "home_form_shot_accuracy", "home_form_def_pressure",
    "away_form_gf", "away_form_ga", "away_form_points",
    "away_form_shots", "away_form_sot", "away_form_shots_faced", "away_form_sot_faced",
    "away_form_shot_accuracy", "away_form_def_pressure",
    "h2h_matches", "h2h_home_win_rate", "h2h_avg_goal_diff",
]

FEATURE_LABELS = {
    "home_form_gf": "Rata-rata gol dicetak tim kandang (5 laga)",
    "home_form_ga": "Rata-rata gol kebobolan tim kandang (5 laga)",
    "home_form_points": "Rata-rata poin tim kandang (5 laga)",
    "home_form_shots": "Rata-rata tembakan dilakukan tim kandang (5 laga)",
    "home_form_sot": "Rata-rata tembakan on-target dilakukan tim kandang (5 laga)",
    "home_form_shots_faced": "Rata-rata tembakan dihadapi tim kandang (5 laga)",
    "home_form_sot_faced": "Rata-rata tembakan on-target dihadapi tim kandang (5 laga)",
    "home_form_shot_accuracy": "Akurasi tembakan tim kandang (SOT/tembakan, 5 laga)",
    "home_form_def_pressure": "Akurasi tembakan lawan yang dihadapi tim kandang (5 laga)",
    "away_form_gf": "Rata-rata gol dicetak tim tandang (5 laga)",
    "away_form_ga": "Rata-rata gol kebobolan tim tandang (5 laga)",
    "away_form_points": "Rata-rata poin tim tandang (5 laga)",
    "away_form_shots": "Rata-rata tembakan dilakukan tim tandang (5 laga)",
    "away_form_sot": "Rata-rata tembakan on-target dilakukan tim tandang (5 laga)",
    "away_form_shots_faced": "Rata-rata tembakan dihadapi tim tandang (5 laga)",
    "away_form_sot_faced": "Rata-rata tembakan on-target dihadapi tim tandang (5 laga)",
    "away_form_shot_accuracy": "Akurasi tembakan tim tandang (SOT/tembakan, 5 laga)",
    "away_form_def_pressure": "Akurasi tembakan lawan yang dihadapi tim tandang (5 laga)",
    "h2h_matches": "Jumlah pertemuan head-to-head sebelumnya",
    "h2h_home_win_rate": "Win rate tim kandang saat head-to-head",
    "h2h_avg_goal_diff": "Rata-rata selisih gol head-to-head (kandang - tandang)",
}


def _empty_team_stats():
    return {
        "goals_for": deque(maxlen=N_FORM),
        "goals_against": deque(maxlen=N_FORM),
        "points": deque(maxlen=N_FORM),
        "shots": deque(maxlen=N_FORM),
        "shots_on_target": deque(maxlen=N_FORM),
        "shots_faced": deque(maxlen=N_FORM),
        "shots_on_target_faced": deque(maxlen=N_FORM),
    }


def _avg(dq: deque) -> float:
    return float(np.mean(dq)) if len(dq) > 0 else np.nan


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator is None or np.isnan(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _team_form_features(stats: dict, prefix: str) -> dict:
    shots = _avg(stats["shots"])
    sot = _avg(stats["shots_on_target"])
    shots_faced = _avg(stats["shots_faced"])
    sot_faced = _avg(stats["shots_on_target_faced"])
    return {
        f"{prefix}_form_gf": _avg(stats["goals_for"]),
        f"{prefix}_form_ga": _avg(stats["goals_against"]),
        f"{prefix}_form_points": _avg(stats["points"]),
        f"{prefix}_form_shots": shots,
        f"{prefix}_form_sot": sot,
        f"{prefix}_form_shots_faced": shots_faced,
        f"{prefix}_form_sot_faced": sot_faced,
        # Akurasi tembakan sendiri: seberapa efisien tim mengarahkan tembakan ke gawang
        f"{prefix}_form_shot_accuracy": _safe_ratio(sot, shots),
        # "Tekanan defensif": akurasi tembakan LAWAN yang berhasil dihadapi/dibiarkan tim ini
        f"{prefix}_form_def_pressure": _safe_ratio(sot_faced, shots_faced),
    }


def _h2h_features(h2h_history: dict, key: frozenset, home: str) -> dict:
    past = h2h_history.get(key, [])
    if not past:
        return {"h2h_matches": 0, "h2h_home_win_rate": np.nan, "h2h_avg_goal_diff": np.nan}
    diffs, wins = [], 0
    for pm in past:
        if pm["home"] == home:
            diff = pm["fthg"] - pm["ftag"]
            won = pm["fthg"] > pm["ftag"]
        else:
            diff = pm["ftag"] - pm["fthg"]
            won = pm["ftag"] > pm["fthg"]
        diffs.append(diff)
        wins += int(won)
    return {
        "h2h_matches": len(past),
        "h2h_home_win_rate": wins / len(past),
        "h2h_avg_goal_diff": float(np.mean(diffs)),
    }


@st.cache_resource(show_spinner=False)
def build_feature_dataset(data: pd.DataFrame):
    """
    Melakukan satu pass kronologis atas seluruh dataset untuk membangun
    fitur pra-pertandingan setiap laga, sekaligus mengembalikan STATE
    TERKINI (form & head-to-head) setiap tim -- dipakai untuk menyusun
    fitur pertandingan baru yang dipilih pengguna di aplikasi.

    Memakai itertuples() (bukan iterrows()) karena jauh lebih cepat untuk
    dataset beribu-ribu baris -- penting agar waktu muat aplikasi tetap
    singkat, terutama saat proses baru "bangun" dari mode tidur di hosting.
    """
    team_stats = defaultdict(_empty_team_stats)
    h2h_history = defaultdict(list)
    has_shots = "HS" in data.columns
    has_sot = "HST" in data.columns

    rows = []
    for m in data.itertuples(index=False):
        home, away = m.HomeTeam, m.AwayTeam
        hs, as_ = team_stats[home], team_stats[away]
        key = frozenset((home, away))
        fthg, ftag = m.FTHG, m.FTAG

        feat = {
            "Date": m.Date, "Season": getattr(m, "Season", None),
            "HomeTeam": home, "AwayTeam": away,
            **_team_form_features(hs, "home"),
            **_team_form_features(as_, "away"),
            **_h2h_features(h2h_history, key, home),
            "FTHG": fthg, "FTAG": ftag, "FTR": m.FTR,
        }
        rows.append(feat)

        # Perbarui histori SETELAH fitur diambil -> mencegah kebocoran data
        hs["goals_for"].append(fthg)
        hs["goals_against"].append(ftag)
        as_["goals_for"].append(ftag)
        as_["goals_against"].append(fthg)
        hs["points"].append(3 if fthg > ftag else (1 if fthg == ftag else 0))
        as_["points"].append(3 if ftag > fthg else (1 if fthg == ftag else 0))
        if has_shots and pd.notna(m.HS):
            hs["shots"].append(m.HS)
            as_["shots"].append(m.AS)
            hs["shots_faced"].append(m.AS)
            as_["shots_faced"].append(m.HS)
        if has_sot and pd.notna(m.HST):
            hs["shots_on_target"].append(m.HST)
            as_["shots_on_target"].append(m.AST)
            hs["shots_on_target_faced"].append(m.AST)
            as_["shots_on_target_faced"].append(m.HST)
        h2h_history[key].append({"home": home, "away": away, "fthg": fthg, "ftag": ftag})

    feature_df = pd.DataFrame(rows)
    return feature_df, team_stats, h2h_history


def get_snapshot_features(team_stats: dict, h2h_history: dict, home_team: str, away_team: str) -> dict:
    """Menyusun fitur pra-pertandingan TERKINI untuk pasangan tim yang dipilih pengguna."""
    hs = team_stats.get(home_team, _empty_team_stats())
    as_ = team_stats.get(away_team, _empty_team_stats())
    key = frozenset((home_team, away_team))
    return {
        **_team_form_features(hs, "home"),
        **_team_form_features(as_, "away"),
        **_h2h_features(h2h_history, key, home_team),
    }
