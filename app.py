"""
Prototipe sistem berbasis web untuk memprediksi skor pertandingan
Liga Primer Inggris (EPL):
"""

from __future__ import annotations

import datetime as dt
import os
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from data_utils import (
    DEFAULT_SEASONS,
    format_season_label,
    get_available_seasons,
    get_team_list,
    load_epl_dataset,
)
from features import FEATURE_LABELS, build_feature_dataset, get_snapshot_features
from models import predict_match, train_and_evaluate
from standings import compute_standings, get_team_season_stats
import sheets_utils

WIB = ZoneInfo("Asia/Jakarta")  # Server hosting biasanya berjalan di UTC; catat waktu dalam WIB agar konsisten

QUESTIONNAIRE_FILE = os.path.join(os.path.dirname(__file__), "usability_responses.csv")

# Instrumen System Usability Scale (SUS) baku - 10 pernyataan berpolaritas
# bergantian (ganjil = positif, genap = negatif), skala Likert 1-5.
QUESTIONNAIRE_ITEMS = [
    ("Saya pikir saya akan sering menggunakan sistem prediksi skor ini.", True),
    ("Saya merasa sistem ini terlalu rumit dan tidak perlu serumit itu.", False),
    ("Saya pikir sistem ini mudah digunakan.", True),
    ("Saya rasa saya membutuhkan bantuan orang teknis untuk bisa menggunakan sistem ini.", False),
    ("Saya merasa berbagai fitur dalam sistem ini (prediksi, statistik, klasemen) terintegrasi dengan baik.", True),
    ("Saya rasa ada terlalu banyak hal yang tidak konsisten pada sistem ini.", False),
    ("Saya membayangkan kebanyakan orang akan belajar menggunakan sistem ini dengan cepat.", True),
    ("Saya merasa sistem ini sangat membingungkan saat digunakan.", False),
    ("Saya merasa percaya diri saat menggunakan sistem ini.", True),
    ("Saya perlu mempelajari banyak hal terlebih dahulu sebelum bisa menggunakan sistem ini dengan lancar.", False),
]


def _sus_score(scores: list[int]) -> float:
    """Menghitung skor SUS (0-100) sesuai rumus baku Brooke (1996)."""
    total = 0
    for (_, is_positive), s in zip(QUESTIONNAIRE_ITEMS, scores):
        total += (s - 1) if is_positive else (5 - s)
    return total * 2.5


def _sus_grade(score: float) -> str:
    if score >= 80.3:
        return "Excellent"
    if score >= 68:
        return "Good"
    if score >= 51:
        return "OK"
    return "Poor"

st.set_page_config(
    page_title="Prediksi Skor EPL",
    page_icon="\u26bd",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Pemuatan data & pelatihan model (di-cache agar tidak diulang setiap interaksi)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_pipeline():
    data, failed_seasons, used_fallback = load_epl_dataset(tuple(DEFAULT_SEASONS))
    feature_df, team_stats, h2h_history = build_feature_dataset(data)
    artifacts = train_and_evaluate(feature_df)
    return {
        "data": data,
        "failed_seasons": failed_seasons,
        "used_fallback": used_fallback,
        "feature_df": feature_df,
        "team_stats": team_stats,
        "h2h_history": h2h_history,
        "artifacts": artifacts,
    }


def _team_snapshot_table(snapshot: dict, home_team: str, away_team: str) -> pd.DataFrame:
    rows = []
    for col, label in FEATURE_LABELS.items():
        rows.append({"Fitur Performa Pra-Pertandingan": label, "Nilai": round(snapshot.get(col, float("nan")), 2)})
    return pd.DataFrame(rows)


def _save_questionnaire_response(scores: list[int]) -> str:
    """
    Menyimpan satu jawaban kuesioner. Mencoba Google Sheets dahulu (jika
    dikonfigurasi lewat st.secrets); jika gagal atau belum dikonfigurasi,
    otomatis jatuh ke penyimpanan CSV lokal. Mengembalikan backend yang
    dipakai: 'sheets' atau 'csv'.
    """
    row = {"timestamp": dt.datetime.now(WIB).isoformat(timespec="seconds")}
    for i in range(1, len(QUESTIONNAIRE_ITEMS) + 1):
        row[f"Q{i}"] = scores[i - 1]
    row["SUS_Score"] = round(_sus_score(scores), 2)

    if sheets_utils.is_configured() and sheets_utils.append_response(row):
        return "sheets"

    df_row = pd.DataFrame([row])
    header = not os.path.exists(QUESTIONNAIRE_FILE)
    df_row.to_csv(QUESTIONNAIRE_FILE, mode="a", header=header, index=False)
    return "csv"


def _load_questionnaire_responses() -> pd.DataFrame | None:
    """Mengambil seluruh jawaban kuesioner dari backend yang aktif (Google Sheets atau CSV lokal)."""
    if sheets_utils.is_configured():
        df = sheets_utils.load_responses()
        if df is not None and not df.empty:
            return df
        if df is not None:
            return df  # Sheets terkonfigurasi tapi belum ada jawaban sama sekali
    if os.path.exists(QUESTIONNAIRE_FILE):
        return pd.read_csv(QUESTIONNAIRE_FILE)
    return None


# ---------------------------------------------------------------------------
# Sidebar navigasi
# ---------------------------------------------------------------------------
st.sidebar.title("\u26bd Prediksi Skor EPL")
st.sidebar.caption("Prototipe berbasis Machine Learning & Streamlit")
page = st.sidebar.radio(
    "Menu",
    [
        "Prediksi Pertandingan",
        "Statistik Tim & Klasemen Liga",
        "Tentang Sistem & Model",
        "Kuesioner Usabilitas",
    ],
)

with st.spinner("Memuat data dari DataHub.io dan menyiapkan model..."):
    pipeline = _load_pipeline()

data = pipeline["data"]
teams = get_team_list(data)

if pipeline["used_fallback"]:
    st.sidebar.warning(
        "DataHub.io tidak dapat diakses saat ini, sehingga aplikasi memakai "
        "data sintetis sementara sebagai demonstrasi.",
        icon="\u26a0\ufe0f",
    )
elif pipeline["failed_seasons"]:
    st.sidebar.info(
        f"{len(pipeline['failed_seasons'])} dari {len(DEFAULT_SEASONS)} musim gagal diunduh dari "
        "DataHub.io dan dilewati; data musim lain tetap dipakai."
    )
else:
    st.sidebar.success(f"Data EPL berhasil dimuat dari DataHub.io ({len(DEFAULT_SEASONS)} musim, {len(data):,} laga).")


# ---------------------------------------------------------------------------
# Halaman 1: Prediksi Pertandingan (use case End User)
# ---------------------------------------------------------------------------
if page == "Prediksi Pertandingan":
    st.title("Prediksi Skor Pertandingan Liga Primer Inggris")
    st.write(
        "Pilih tim kandang (home) dan tim tandang (away) untuk memprediksi skor "
        "pertandingan berdasarkan performa pra-pertandingan menggunakan model XGBoost."
    )

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Tim Kandang (Home)", teams, index=0, key="home_team")
    with col2:
        away_default = 1 if len(teams) > 1 else 0
        away_team = st.selectbox("Tim Tandang (Away)", teams, index=away_default, key="away_team")

    # --- Diagram Aktivitas: "Valid Team Selection?" ---
    is_valid = home_team != away_team
    if not is_valid:
        st.warning("Tim kandang dan tim tandang tidak boleh sama. Silakan pilih tim yang berbeda.")

    predict_clicked = st.button("Prediksi Skor Pertandingan", type="primary", disabled=not is_valid)

    if predict_clicked and is_valid:
        # --- "Retrieve Pre-Match Performance Features" ---
        snapshot = get_snapshot_features(pipeline["team_stats"], pipeline["h2h_history"], home_team, away_team)
        # --- "Run XGBoost Prediction Model" ---
        result = predict_match(pipeline["artifacts"], snapshot)
        st.session_state["last_prediction"] = {
            "home_team": home_team, "away_team": away_team,
            "snapshot": snapshot, "result": result,
        }

    prediction = st.session_state.get("last_prediction")
    if prediction and prediction["home_team"] == home_team and prediction["away_team"] == away_team:
        result = prediction["result"]
        home_goals, away_goals = result["predicted_score"]
        proba = result["win_probability"]

        st.divider()
        st.subheader("Hasil Prediksi")

        score_col, proba_col = st.columns([1, 2])
        with score_col:
            st.metric(
                label=f"{home_team} vs {away_team}",
                value=f"{home_goals} - {away_goals}",
            )
        with proba_col:
            st.write("**Probabilitas Hasil Pertandingan**")
            proba_df = pd.DataFrame({
                "Hasil": ["Menang Kandang (H)", "Seri (D)", "Menang Tandang (A)"],
                "Probabilitas": [proba.get("H", 0.0), proba.get("D", 0.0), proba.get("A", 0.0)],
            }).set_index("Hasil")
            st.bar_chart(proba_df)

        # --- "View SHAP Feature Explanation" ---
        st.subheader("Penjelasan Prediksi (SHAP)")
        st.caption(
            "Nilai positif mendorong prediksi ke arah hasil yang paling mungkin di atas; "
            "nilai negatif menekannya."
        )
        shap_series = result["shap_values"].rename(FEATURE_LABELS).sort_values(key=abs, ascending=True)
        st.bar_chart(shap_series)

        # --- "View Pre-Match Team Statistics" ---
        st.subheader("Statistik Performa Pra-Pertandingan")
        st.dataframe(_team_snapshot_table(prediction["snapshot"], home_team, away_team), hide_index=True, width="stretch")

        st.divider()
        st.info(
            "Ingin mencoba pertandingan lain? Ganti tim di atas lalu klik "
            "**Prediksi Skor Pertandingan** kembali, atau lanjutkan ke menu "
            "**Kuesioner Usabilitas** di sidebar setelah selesai."
        )


# ---------------------------------------------------------------------------
# Halaman 2: Statistik Tim & Klasemen Liga (use case End User)
# ---------------------------------------------------------------------------
elif page == "Statistik Tim & Klasemen Liga":
    st.title("Statistik Tim & Klasemen Liga Primer Inggris")

    available_seasons = get_available_seasons(data)
    if not available_seasons:
        st.warning("Belum ada data musim yang tersedia.")
    else:
        season_labels = {s: format_season_label(s) for s in available_seasons}
        selected_season = st.selectbox(
            "Pilih Musim",
            available_seasons,
            index=0,  # musim terbaru (data terurut menurun)
            format_func=lambda s: season_labels[s],
        )
        is_latest = selected_season == available_seasons[0]
        if is_latest:
            st.caption(f"Menampilkan klasemen musim terbaru: **{season_labels[selected_season]}**.")

        table = compute_standings(data, selected_season)
        if table.empty:
            st.info("Belum ada data pertandingan untuk musim ini.")
        else:
            st.subheader(f"Klasemen Liga - Musim {season_labels[selected_season]}")
            st.dataframe(table, hide_index=True, width="stretch")

            st.divider()
            st.subheader("Statistik Detail per Tim")
            season_teams = sorted(table["Tim"].unique())
            selected_team = st.selectbox("Pilih Tim", season_teams, key="stats_team")

            stats = get_team_season_stats(data, selected_season, selected_team)
            team_row = table[table["Tim"] == selected_team].iloc[0]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Peringkat", int(team_row["Peringkat"]))
            m2.metric("Poin", int(team_row["Poin"]))
            m3.metric("Selisih Gol", int(team_row["SG"]))
            m4.metric("Clean Sheet", int(team_row["Clean Sheet"]))

            home_col, away_col = st.columns(2)
            for col, label, key in ((home_col, "Kandang", "Kandang"), (away_col, "Tandang", "Tandang")):
                rec = stats[key]
                with col:
                    st.write(f"**Performa {label}**")
                    st.dataframe(
                        pd.DataFrame([rec]).T.rename(columns={0: "Nilai"}),
                        width="stretch",
                    )


# ---------------------------------------------------------------------------
# Halaman 3: Tentang Sistem & Model (use case Peneliti)
# ---------------------------------------------------------------------------
elif page == "Tentang Sistem & Model":
    st.title("Tentang Sistem & Evaluasi Model")
    artifacts = pipeline["artifacts"]

    st.subheader("Sumber Data")
    st.write(
        "Data sekunder pertandingan historis EPL diunduh dari "
        "[DataHub.io](https://datahub.io/football/english-premier-league) "
        "(distribusi Football-Data.co.uk), mencakup 10 musim kompetisi terakhir "
        f"({format_season_label(DEFAULT_SEASONS[0])} - {format_season_label(DEFAULT_SEASONS[-1])})."
    )
    st.write(f"Total laga yang dipakai untuk pelatihan & pengujian model: **{len(pipeline['feature_df']):,}**")
    st.write(f"Data latih: **{artifacts['n_train']:,}** laga &nbsp;|&nbsp; Data uji: **{artifacts['n_test']:,}** laga (time-based split).")

    st.subheader("Evaluasi Klasifikasi Hasil Pertandingan (H/D/A)")
    metrics_df = pd.DataFrame(artifacts["metrics"]).T
    st.dataframe(metrics_df.style.format("{:.3f}"), width="stretch")

    st.subheader("Evaluasi Regresi Skor Pertandingan")
    score_metrics_df = pd.DataFrame(artifacts["score_metrics"]).T
    st.dataframe(score_metrics_df.style.format("{:.3f}"), width="stretch")

    st.caption(
        "XGBoost dipakai sebagai model utama; Random Forest sebagai benchmark dan "
        "Logistic Regression sebagai baseline klasifikasi."
    )


# ---------------------------------------------------------------------------
# Halaman 4: Kuesioner Usabilitas - System Usability Scale (use case End User)
# ---------------------------------------------------------------------------
else:
    st.title("Kuesioner Usabilitas Prototipe (System Usability Scale)")
    st.write(
        "Instrumen ini memakai 10 pernyataan baku **System Usability Scale (SUS)** "
        "(Brooke, 1996). "
        "Skala: 1 = Sangat Tidak Setuju, 5 = Sangat Setuju."
    )

    storage_label = "Google Sheets" if sheets_utils.is_configured() else "file CSV lokal"
    st.caption(f"Jawaban akan disimpan ke: **{storage_label}**.")

    with st.form("usability_form"):
        scores = []
        for i, (pernyataan, _) in enumerate(QUESTIONNAIRE_ITEMS, start=1):
            st.write(f"**{i}.** {pernyataan}")
            scores.append(st.radio("Skor", [1, 2, 3, 4, 5], index=2, horizontal=True, key=f"q{i}", label_visibility="collapsed"))
            st.write("")
        submitted = st.form_submit_button("Kirim Jawaban")

    if submitted:
        backend = _save_questionnaire_response(scores)
        skor = _sus_score(scores)
        tersimpan_di = "Google Sheets" if backend == "sheets" else "file lokal"
        st.success(
            f"Terima kasih! Jawaban Anda tersimpan di {tersimpan_di}. "
            f"Skor SUS Anda: **{skor:.1f}** ({_sus_grade(skor)})."
        )

    responses = _load_questionnaire_responses()
    if responses is not None and not responses.empty:
        st.divider()
        st.subheader("Ringkasan Hasil Kuesioner Saat Ini")
        q_cols = [c for c in responses.columns if c.startswith("Q")]
        for c in q_cols + ["SUS_Score"]:
            responses[c] = pd.to_numeric(responses[c], errors="coerce")

        avg_sus = responses["SUS_Score"].mean()
        r1, r2 = st.columns(2)
        r1.metric("Jumlah Responden", len(responses))
        r2.metric("Rata-rata Skor SUS", f"{avg_sus:.1f}", help=_sus_grade(avg_sus))
        st.caption(
            f"Interpretasi rata-rata skor SUS saat ini: **{_sus_grade(avg_sus)}** "
            "(skala 0-100; >= 68 umumnya dianggap di atas rata-rata)."
        )

        st.write("**Rata-rata skor per pernyataan (skala 1-5, sebelum konversi SUS):**")
        summary = pd.DataFrame({
            "No.": list(range(1, len(QUESTIONNAIRE_ITEMS) + 1)),
            "Pernyataan": [p for p, _ in QUESTIONNAIRE_ITEMS],
            "Polaritas": ["Positif" if pos else "Negatif" for _, pos in QUESTIONNAIRE_ITEMS],
            "Rata-rata": [responses[c].mean() for c in q_cols],
            "Std. Deviasi": [responses[c].std() for c in q_cols],
        })
        st.dataframe(
            summary.style.format({"Rata-rata": "{:.2f}", "Std. Deviasi": "{:.2f}"}),
            hide_index=True, width="stretch",
        )
