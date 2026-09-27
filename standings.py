"""
standings.py
============
Perhitungan klasemen (peringkat) liga dan statistik tim per musim,
dipakai pada menu "Statistik Tim & Klasemen Liga". Dihitung langsung
dari data pertandingan yang sama dengan yang dipakai untuk pelatihan
model, sehingga selalu konsisten dengan musim terbaru yang tersedia
di DataHub.io.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_standings(data: pd.DataFrame, season: str) -> pd.DataFrame:
    """
    Menghitung tabel klasemen liga untuk satu musim tertentu, mengikuti
    aturan standar sepak bola: menang = 3 poin, seri = 1 poin, kalah = 0
    poin; diurutkan berdasarkan poin, lalu selisih gol, lalu gol dicetak.
    """
    df = data[data["Season"] == season]
    records: dict[str, dict] = {}

    def _get(team: str) -> dict:
        return records.setdefault(team, {
            "Main": 0, "Menang": 0, "Seri": 0, "Kalah": 0,
            "GM": 0, "GK": 0, "CleanSheet": 0,
        })

    for _, m in df.iterrows():
        home, away = m["HomeTeam"], m["AwayTeam"]
        fthg, ftag = m["FTHG"], m["FTAG"]
        h, a = _get(home), _get(away)

        h["Main"] += 1
        a["Main"] += 1
        h["GM"] += fthg
        h["GK"] += ftag
        a["GM"] += ftag
        a["GK"] += fthg
        if ftag == 0:
            h["CleanSheet"] += 1
        if fthg == 0:
            a["CleanSheet"] += 1

        if fthg > ftag:
            h["Menang"] += 1
            a["Kalah"] += 1
        elif fthg < ftag:
            a["Menang"] += 1
            h["Kalah"] += 1
        else:
            h["Seri"] += 1
            a["Seri"] += 1

    rows = []
    for team, r in records.items():
        poin = r["Menang"] * 3 + r["Seri"]
        rows.append({
            "Tim": team,
            "Main": r["Main"], "M": r["Menang"], "S": r["Seri"], "K": r["Kalah"],
            "GM": r["GM"], "GK": r["GK"], "SG": r["GM"] - r["GK"],
            "Clean Sheet": r["CleanSheet"],
            "Poin": poin,
        })

    table = pd.DataFrame(rows).sort_values(
        ["Poin", "SG", "GM"], ascending=[False, False, False]
    ).reset_index(drop=True)
    table.insert(0, "Peringkat", np.arange(1, len(table) + 1))
    return table


def get_team_season_stats(data: pd.DataFrame, season: str, team: str) -> dict:
    """Statistik ringkas satu tim pada satu musim (dipecah kandang/tandang)."""
    df = data[data["Season"] == season]
    home = df[df["HomeTeam"] == team]
    away = df[df["AwayTeam"] == team]

    def _record(matches: pd.DataFrame, is_home: bool) -> dict:
        gf = matches["FTHG"] if is_home else matches["FTAG"]
        ga = matches["FTAG"] if is_home else matches["FTHG"]
        wins = int(((gf > ga)).sum())
        draws = int((gf == ga).sum())
        losses = int((gf < ga).sum())
        return {
            "Main": len(matches), "Menang": wins, "Seri": draws, "Kalah": losses,
            "Gol Dicetak": int(gf.sum()) if len(matches) else 0,
            "Gol Kebobolan": int(ga.sum()) if len(matches) else 0,
        }

    return {
        "Kandang": _record(home, True),
        "Tandang": _record(away, False),
    }
