"""
sheets_utils.py
================
Penyimpanan jawaban kuesioner usabilitas (SUS) ke Google Sheets, agar
data responden TIDAK HILANG saat aplikasi di-restart atau di-redeploy
di hosting gratis seperti Streamlit Community Cloud (yang memakai
penyimpanan sementara/ephemeral).

Konfigurasi bersifat OPSIONAL: jika kredensial belum diisi lewat
st.secrets, is_configured() akan mengembalikan False dan app.py akan
otomatis memakai penyimpanan CSV lokal sebagai fallback -- lihat
Bagian "Integrasi Google Sheets" pada README.md untuk cara setup.

Format st.secrets yang dibutuhkan (mis. di .streamlit/secrets.toml):

    [gsheets]
    sheet_id = "ID_SPREADSHEET_ANDA"

    [gcp_service_account]
    type = "service_account"
    project_id = "..."
    private_key_id = "..."
    private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
    client_email = "...@....iam.gserviceaccount.com"
    client_id = "..."
    token_uri = "https://oauth2.googleapis.com/token"
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

WORKSHEET_NAME = "responses"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def is_configured() -> bool:
    """Mengecek apakah kredensial Google Sheets sudah diisi di st.secrets."""
    try:
        has_creds = "gcp_service_account" in st.secrets
        has_sheet_id = bool(st.secrets.get("gsheets", {}).get("sheet_id"))
        return has_creds and has_sheet_id
    except Exception:
        # st.secrets melempar error jika belum ada secrets.toml sama sekali
        # -> anggap saja fitur Google Sheets belum dikonfigurasi.
        return False


@st.cache_resource(show_spinner=False)
def _get_worksheet():
    import gspread
    from google.oauth2.service_account import Credentials

    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]), scopes=SCOPES
    )
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(st.secrets["gsheets"]["sheet_id"])
    try:
        worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=WORKSHEET_NAME, rows=1000, cols=20)
    return worksheet


def append_response(row: dict) -> bool:
    """
    Menambahkan satu baris jawaban ke Google Sheets (menulis header
    otomatis jika sheet masih kosong). Mengembalikan True bila berhasil,
    False bila gagal (mis. kredensial salah / tidak ada akses internet)
    -- app.py akan otomatis jatuh ke penyimpanan CSV lokal jika False.
    """
    try:
        worksheet = _get_worksheet()
        if not worksheet.get_all_values():
            worksheet.append_row(list(row.keys()))
        worksheet.append_row(list(row.values()))
        return True
    except Exception as e:
        st.warning(f"Gagal menyimpan ke Google Sheets ({e}); jawaban disimpan secara lokal.")
        return False


def load_responses() -> pd.DataFrame | None:
    """Mengambil seluruh jawaban dari Google Sheets sebagai DataFrame, atau None bila gagal."""
    try:
        worksheet = _get_worksheet()
        records = worksheet.get_all_records()
        return pd.DataFrame(records) if records else pd.DataFrame()
    except Exception:
        return None
