'''
da lanciare in modalità streamlit


c:\Users\u023779\Documents\python\Opzioni_GEX\.venv\Scripts\streamlit run c:\Users\u023779\Documents\python\Opzioni_GEX\dashboard_live.py 



v 1.0
dashboard live per visualizzare i dati GEX scaricati da BYBIT

v 1.1
mostra l'ultima giornata

'''


import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
from datetime import datetime
import os

# ==============================
# CONFIG
# ==============================
st.set_page_config(layout="wide")

CSV_PATH = r".\data_bybit_60s\2026-06-13\GEX_0DTE_ALL_BTC_2026-06-13.csv"

REFRESH_SECONDS = 60

# ==============================
# AUTO REFRESH
# ==============================
st.title("📊 BTC GEX Live Dashboard")

placeholder = st.empty()

while True:
    with placeholder.container():

        if not os.path.exists(CSV_PATH):
            st.warning("CSV non trovato")
            time.sleep(REFRESH_SECONDS)
            continue

        df = pd.read_csv(CSV_PATH)

        #if df.empty or "timestamp" not in df.columns:
        #    st.warning("CSV vuoto")
        #    time.sleep(REFRESH_SECONDS)
        #    continue

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp")

        # ✅ SOLO OGGI
        today_date = pd.to_datetime(datetime.now().date())

        df = df[df["timestamp"].dt.date == today_date.date()]
        
        if df.empty:
            st.warning("Nessun dato oggi")
            st.stop()


        # seleziona tipo
        tipo = st.selectbox("Tipo dati", ["0DTE", "ALL"])

        df = df[df["tipo"] == tipo]

        if len(df) < 5:
            st.warning("Pochi dati")
            time.sleep(REFRESH_SECONDS)
            continue

        # ==============================
        # ZOOM ULTIME 2 ORE
        # ==============================
        xmax = df["timestamp"].iloc[-1]
        xmin = xmax - pd.Timedelta(hours=2)

        df = df[df["timestamp"] >= xmin]

        # ==============================
        # FIGURA
        # ==============================
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.45, 0.3, 0.25]
        )

        # SPOT + WALL
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["spot"],
            mode="lines",
            name="Spot",
            line=dict(color="green")
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["call_wall"],
            mode="lines",
            name="Call Wall",
            line=dict(color="blue")
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["put_wall"],
            mode="lines",
            name="Put Wall",
            line=dict(color="red")
        ), row=1, col=1)

        # GEX
        colors = ["green" if v > 0 else "red" for v in df["gex_totale_mercato_M$"]]

        fig.add_trace(go.Bar(
            x=df["timestamp"],
            y=df["gex_totale_mercato_M$"],
            name="GEX",
            marker_color=colors
        ), row=2, col=1)

        # GAMMA FLIP
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["gamma_flip"],
            mode="lines",
            name="Gamma Flip",
            line=dict(color="purple")
        ), row=3, col=1)

        # ==============================
        # LAYOUT
        # ==============================
        fig.update_layout(
            height=900,
            showlegend=True,
            title=f"BTC {tipo} GEX Live"
        )

        # linea NOW
        fig.add_vline(
            x=xmax,
            line_dash="dash",
            line_color="black"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.success(f"Aggiornato alle: {xmax}")

    time.sleep(REFRESH_SECONDS)