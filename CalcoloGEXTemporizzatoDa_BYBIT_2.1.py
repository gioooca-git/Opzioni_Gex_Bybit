'''
v.10
partiamo da CalcoloGEXTemporizzatoDaYahooFinance.py e anzichè raccogliere i dati ogni 5 minuti da yahoo finance li raccoglie per bybit:


v 1.1
versione più robusta del loop temporizzato Bybit BTC con logging e retry con exponential backoff, mantenendo la logica di calcolo 0DTE + ALL 
e il salvataggio CSV cumulativo


v 1.3
ggiunge un grafico dashboard che legga l’Excel/CSV storico
e si aggiorni a ogni nuovo ciclo con quattro serie nel tempo: 
gex_totale, call_wall, put_wall, gamma_flip

v 1.4 
grafici divisi per 0DTE e ALL:
GEX_dashboard_0DTE_BTC.png.
GEX_dashboard_ALL_BTC.png.

v 1.5
aggiungi nel grafico sottostante il prezzo dello spot


v 1.6

aggiornamento ogni 60 secondi


v 1.7
rotazione dei file giornalmente

v 1.8
disegna singoli grafici

v 1.9
ridimensionamento pannelli dei grafici e aggiustamento

v 2.0
sposto le legende dal pannello siperiore a quello giusto

v 2.1
voglio la possibilità di zoomare sugli ultimi dati ricevuti in ordine temporale sul grafico


'''


import os
import json
import time
import random
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ==============================================================================
# 1. PARAMETRI
# ==============================================================================


BASE_URL = "https://api.bybit.com/v5/market/tickers"
BASE_COIN = "BTC"
CATEGORY = "option"
RISK_FREE_RATE = 0.045

OUTPUT_DIR = r".\data_bybit_60s"
os.makedirs(OUTPUT_DIR, exist_ok=True)

AGGIORNAMENTO_SECONDI = 60
MAX_RETRY = 5
BACKOFF_BASE = 1.8
TIMEOUT_SEC = 30

CURRENT_DAY = None
LOG_DIR = None
CSV_PATH = None
RAW_SNAPSHOT_DIR = None
DASHBOARD_0DTE_PNG = None
DASHBOARD_ALL_PNG = None
logger = None
session = requests.Session()


# ==============================================================================
# 2. FUNZIONI PATH GIORNALIERI
# ==============================================================================


def init_daily_paths():
    global CURRENT_DAY, LOG_DIR, CSV_PATH, RAW_SNAPSHOT_DIR, DASHBOARD_0DTE_PNG, DASHBOARD_ALL_PNG, logger

    today = datetime.now().strftime("%Y-%m-%d")
    if CURRENT_DAY == today and LOG_DIR and CSV_PATH and RAW_SNAPSHOT_DIR and DASHBOARD_0DTE_PNG and DASHBOARD_ALL_PNG:
        return

    CURRENT_DAY = today
    daily_dir = os.path.join(OUTPUT_DIR, CURRENT_DAY)
    os.makedirs(daily_dir, exist_ok=True)

    LOG_DIR = os.path.join(daily_dir, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)

    RAW_SNAPSHOT_DIR = os.path.join(daily_dir, "snapshots")
    os.makedirs(RAW_SNAPSHOT_DIR, exist_ok=True)

    CSV_PATH = os.path.join(daily_dir, f"GEX_0DTE_ALL_BTC_{CURRENT_DAY}.csv")
    DASHBOARD_0DTE_PNG = os.path.join(daily_dir, f"GEX_dashboard_0DTE_{BASE_COIN}_{CURRENT_DAY}.png")
    DASHBOARD_ALL_PNG = os.path.join(daily_dir, f"GEX_dashboard_ALL_{BASE_COIN}_{CURRENT_DAY}.png")

    if logger is not None:
        for h in list(logger.handlers):
            if isinstance(h, logging.FileHandler):
                logger.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass

        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        log_file = os.path.join(LOG_DIR, f"bybit_gex_{CURRENT_DAY}.log")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)


# ==============================================================================
# 3. LOGGING
# ==============================================================================


logger = logging.getLogger("bybit_gex")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

sh = logging.StreamHandler()
sh.setFormatter(fmt)
logger.addHandler(sh)

init_daily_paths()


# ==============================================================================
# 4. BLACK-SCHOLES & IV
# ==============================================================================


def bsm_price(S, K, T, r, sigma, option_type="C"):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == "C" else max(0.0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "C":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def calcola_iv_newton(target_price, S, K, T, r, option_type="C", max_iter=100, tolerance=1e-5):
    if T <= 0 or target_price <= 0:
        return 0.0
    intrinsic = max(0.0, S - K) if option_type == "C" else max(0.0, K - S)
    if target_price <= intrinsic:
        return 0.0
    sigma = 0.30
    for _ in range(max_iter):
        price = bsm_price(S, K, T, r, sigma, option_type)
        diff = price - target_price
        if abs(diff) < tolerance:
            return sigma
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1)
        if vega < 1e-8:
            break
        sigma = sigma - diff / vega
        if sigma <= 0 or sigma > 5.0:
            break
    return max(0.0, sigma)


def calcola_greche_esatte(S, K, T, r, sigma, option_type="C"):
    if T <= 0 or sigma <= 0:
        return (1.0 if option_type == "C" and S > K else 0.0), 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    pdf_d1 = norm.pdf(d1)
    delta = norm.cdf(d1) if option_type == "C" else norm.cdf(d1) - 1
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    return round(delta, 4), round(gamma, 8)


# ==============================================================================
# 5. HTTP RETRY
# ==============================================================================


def http_get(url, params=None, timeout=TIMEOUT_SEC, max_retry=MAX_RETRY):
    last_exc = None
    for attempt in range(1, max_retry + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                raise requests.HTTPError(f"HTTP 429 Too Many Requests: {resp.text}")
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            sleep_s = min(60, (BACKOFF_BASE ** (attempt - 1))) + random.uniform(0, 0.5)
            logger.warning(f"HTTP retry {attempt}/{max_retry} fallito: {e}. Attendo {sleep_s:.2f}s")
            time.sleep(sleep_s)
    raise last_exc


# ==============================================================================
# 6. BYBIT DOWNLOAD / PARSING
# ==============================================================================


def get_btc_options_bybit():
    params = {"category": CATEGORY, "baseCoin": BASE_COIN}
    r = http_get(BASE_URL, params=params)
    data = r.json()
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit error retCode={data.get('retCode')} retMsg={data.get('retMsg')}")
    return data["result"]["list"]


def parse_bybit_option_symbol(symbol):
    parts = symbol.split("-")
    if len(parts) < 5:
        return None
    base = parts[0]
    expiry_raw = parts[1]
    strike = float(parts[2])
    opt_type = parts[3]
    try:
        expiry_dt = datetime.strptime(expiry_raw, "%d%b%y").date().isoformat()
    except Exception:
        return None
    return {"baseCoin": base, "expiry": expiry_dt, "strike": strike, "type": opt_type}


def load_options_df():
    raw = get_btc_options_bybit()
    rows = []
    for row in raw:
        symbol = row.get("symbol", "")
        parsed = parse_bybit_option_symbol(symbol)
        if not parsed or parsed["baseCoin"] != BASE_COIN:
            continue
        bid = float(row.get("bid1Price", 0) or 0)
        ask = float(row.get("ask1Price", 0) or 0)
        last = float(row.get("lastPrice", 0) or 0)
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else last
        rows.append({
            "symbol": symbol,
            "expiry": parsed["expiry"],
            "strike": parsed["strike"],
            "type": parsed["type"],
            "bid": bid,
            "ask": ask,
            "mid_price": mid,
            "lastPrice": last,
            "markPrice": float(row.get("markPrice", 0) or 0),
            "openInterest": float(row.get("openInterest", 0) or 0),
            "underlyingPrice": float(row.get("underlyingPrice", 0) or row.get("indexPrice", 0) or 0),
            "markIv": float(row.get("markIv", 0) or 0),
            "bid1Iv": float(row.get("bid1Iv", 0) or 0),
            "ask1Iv": float(row.get("ask1Iv", 0) or 0),
            "delta_api": float(row.get("delta", 0) or 0),
            "gamma_api": float(row.get("gamma", 0) or 0),
            "vega_api": float(row.get("vega", 0) or 0),
            "theta_api": float(row.get("theta", 0) or 0),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df, np.nan
    df["underlyingPrice"] = df["underlyingPrice"].replace(0, np.nan)
    spot = float(df["underlyingPrice"].dropna().iloc[0]) if df["underlyingPrice"].notna().any() else np.nan
    return df, spot


# ==============================================================================
# 7. TIME TO EXPIRY
# ==============================================================================


def calcola_tempo_rimanente_bybit(expiry_iso):
    try:
        expiry_dt = datetime.strptime(expiry_iso, "%Y-%m-%d").replace(
            hour=8, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
        )
    except Exception:
        return 1e-6
    seconds = (expiry_dt - datetime.now(timezone.utc)).total_seconds()
    if seconds <= 0:
        return 1e-6
    return seconds / (365.0 * 24 * 60 * 60)


# ==============================================================================
# 8. GEX CALCULATION
# ==============================================================================


def calcola_gex_per_scadenza(df_all, scadenza_iso, T, r, prezzo_spot):
    df = df_all[df_all["expiry"] == scadenza_iso].copy()
    if df.empty:
        return None, None, None

    def gamma_from_row(row):
        iv = calcola_iv_newton(row["mid_price"], prezzo_spot, row["strike"], T, r, row["type"])
        if iv > 0:
            _, gamma = calcola_greche_esatte(prezzo_spot, row["strike"], T, r, iv, row["type"])
            return gamma
        gamma_api = row["gamma_api"]
        return float(gamma_api) if pd.notna(gamma_api) else 0.0

    df["gamma_calc"] = df.apply(gamma_from_row, axis=1)
    df["call_gex"] = np.where(df["type"] == "C", df["gamma_calc"] * df["openInterest"] * 100 * prezzo_spot, 0.0)
    df["put_gex"] = np.where(df["type"] == "P", df["gamma_calc"] * df["openInterest"] * 100 * prezzo_spot, 0.0)
    df["net_gex"] = df["call_gex"] - df["put_gex"]
    df["total_gex"] = df["call_gex"] + df["put_gex"]

    gex_by_strike = df.groupby("strike", as_index=False).agg(
        call_gex=("call_gex", "sum"),
        put_gex=("put_gex", "sum"),
        net_gex=("net_gex", "sum"),
        total_gex=("total_gex", "sum")
    ).sort_values("strike").reset_index(drop=True)

    gex_by_strike["call_gex_mn"] = gex_by_strike["call_gex"] / 1_000_000
    gex_by_strike["put_gex_mn"] = gex_by_strike["put_gex"] / 1_000_000
    gex_by_strike["net_gex_mn"] = gex_by_strike["net_gex"] / 1_000_000
    gex_by_strike["total_gex_mn"] = gex_by_strike["total_gex"] / 1_000_000

    if not gex_by_strike.empty:
        call_wall = gex_by_strike.loc[gex_by_strike["call_gex"].idxmax(), "strike"]
        put_wall = gex_by_strike.loc[gex_by_strike["put_gex"].idxmax(), "strike"]
    else:
        call_wall = np.nan
        put_wall = np.nan

    gex_by_strike["cum_net_gex"] = gex_by_strike["net_gex_mn"].cumsum()
    cum = gex_by_strike["cum_net_gex"].values
    strikes_vals = gex_by_strike["strike"].values

    gamma_flip = np.nan
    for i in range(1, len(cum)):
        if cum[i - 1] * cum[i] < 0:
            strike1, strike2 = strikes_vals[i - 1], strikes_vals[i]
            cum1, cum2 = cum[i - 1], cum[i]
            gamma_flip = strike1 - cum1 * (strike2 - strike1) / (cum2 - cum1)
            break

    if np.isnan(gamma_flip) and len(gex_by_strike) > 1:
        try:
            f = interp1d(strikes_vals, cum, kind="linear", fill_value="extrapolate")
            smin = strikes_vals.min() * 0.8
            smax = strikes_vals.max() * 1.2
            if f(smin) * f(smax) < 0:
                gamma_flip = brentq(f, smin, smax)
        except Exception:
            gamma_flip = np.nan

    meta = {
        "expiry": scadenza_iso,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "gamma_flip": gamma_flip,
        "max_call_gex": float(gex_by_strike["call_gex"].max()) if not gex_by_strike.empty else np.nan,
        "max_put_gex": float(gex_by_strike["put_gex"].max()) if not gex_by_strike.empty else np.nan
    }
    return df, gex_by_strike, meta


# ==============================================================================
# 9. CSV / JSON SAVE
# ==============================================================================


def append_rows_csv(rows, csv_path):
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        try:
            df_old = pd.read_csv(csv_path, encoding="utf-8")
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df_all = df_new
    else:
        df_all = df_new
    df_all.to_csv(csv_path, index=False, encoding="utf-8")


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ==============================================================================
# 10. DASHBOARD
# ==============================================================================

def aggiorna_dashboard_tipo_da_csv(csv_path, tipo, png_path):
    try:
        logger.info(f"[{tipo}] Avvio aggiornamento dashboard…")

        if not os.path.exists(csv_path):
            logger.warning("CSV non trovato")
            return

        df = pd.read_csv(csv_path, encoding="utf-8")

        if df.empty or "timestamp" not in df.columns:
            logger.warning("CSV vuoto o invalido")
            return

        # ==============================
        # CLEAN ESTREMO TIMESTAMP
        # ==============================
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # elimina TUTTO ciò che non è datetime valido
        df = df[df["timestamp"].notna()]

        # filtra tipo
        df = df[df["tipo"] == tipo]

        # se resta vuoto esci
        if df.empty:
            logger.warning(f"Nessun dato valido per {tipo}")
            return

        # rimuove timezone
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        # ordina
        df = df.sort_values("timestamp")

        # ✅ conversione DEFINITIVA (no Timestamp mai più)
        timestamps = df["timestamp"].astype("datetime64[ns]").to_numpy()

        # ==============================
        # FIGURA
        # ==============================
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.45, 0.30, 0.25]
        )

        # ===== SPOT =====
        fig.add_trace(go.Scatter(
            x=timestamps, y=df["spot"].values,
            mode="lines",
            line=dict(color="green")
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=timestamps, y=df["call_wall"].values,
            mode="lines",
            line=dict(color="blue")
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=timestamps, y=df["put_wall"].values,
            mode="lines",
            line=dict(color="red")
        ), row=1, col=1)

        # ===== GEX =====
        gex_vals = df["gex_totale_mercato_M$"].values
        colors = ["green" if v > 0 else "red" for v in gex_vals]

        fig.add_trace(go.Bar(
            x=timestamps,
            y=gex_vals,
            marker_color=colors
        ), row=2, col=1)

        # ===== GAMMA FLIP =====
        fig.add_trace(go.Scatter(
            x=timestamps, y=df["gamma_flip"].values,
            mode="lines",
            line=dict(color="purple")
        ), row=3, col=1)

        # ==============================
        # ZOOM SICURO
        # ==============================
        WINDOW_MINUTES = 120

        xmax = timestamps[-1]               # numpy datetime (SAFE)
        xmin = xmax - np.timedelta64(WINDOW_MINUTES, "m")

        fig.update_xaxes(range=[xmin, xmax])

        # ==============================
        # CONTROLLI
        # ==============================
        fig.update_layout(
            showlegend=False,
            xaxis=dict(
                type="date",
                rangeslider=dict(visible=True),
                rangeselector=dict(
                    buttons=[
                        dict(count=30, label="30m", step="minute", stepmode="backward"),
                        dict(count=60, label="1h", step="minute", stepmode="backward"),
                        dict(count=180, label="3h", step="minute", stepmode="backward"),
                        dict(step="all", label="ALL")
                    ]
                )
            )
        )

        # ==============================
        # LEGENDE INTERNE
        # ==============================
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.96,
            text="Spot<br>Call Wall<br>Put Wall",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)"
        )

        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.58,
            text="GEX positivo / negativo",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)"
        )

        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.22,
            text="Gamma Flip",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)"
        )

        # ==============================
        # LINEA NOW
        # ==============================
        fig.add_vline(x=xmax, line_dash="dash", line_color="black")

        # ==============================
        # EXPORT
        # ==============================
        logger.info("Salvataggio PNG...")

        os.makedirs(os.path.dirname(png_path), exist_ok=True)

        img_bytes = fig.to_image(format="png")

        with open(png_path, "wb") as f:
            f.write(img_bytes)

        logger.info(f"Dashboard salvata: {png_path}")

    except Exception as e:
        logger.error(f"Errore dashboard {tipo}: {e}", exc_info=True)
# ==============================================================================
# 11. CICLO
# ==============================================================================


def ciclo_calcolo_gex():
    init_daily_paths()
    now_local = datetime.now()
    timestamp_iso = now_local.isoformat()

    try:
        df_options, prezzo_spot = load_options_df()
    except Exception as e:
        logger.exception(f"Errore download Bybit: {e}")
        return

    if df_options.empty or np.isnan(prezzo_spot):
        logger.warning("Nessun dato opzioni disponibile o spot non valido.")
        return

    logger.info(f"Opzioni scaricate: {len(df_options)} | Spot: {prezzo_spot:.2f}")

    expiries = sorted(df_options["expiry"].dropna().unique().tolist())
    if not expiries:
        logger.warning("Nessuna expiry trovata.")
        return

    expiry_0dte = expiries[0]
    rows_csv = []

    try:
        T_0dte = calcola_tempo_rimanente_bybit(expiry_0dte)
        _, gex_0dte, meta_0dte = calcola_gex_per_scadenza(df_options, expiry_0dte, T_0dte, RISK_FREE_RATE, prezzo_spot)
        if gex_0dte is not None and not gex_0dte.empty:
            net = float(gex_0dte["net_gex_mn"].sum())
            call = float(gex_0dte["call_gex_mn"].sum())
            put = float(gex_0dte["put_gex_mn"].sum())
            total = float(gex_0dte["total_gex_mn"].sum())
            rows_csv.append({
                "timestamp": timestamp_iso,
                "tipo": "0DTE",
                "expiry": meta_0dte["expiry"],
                "spot": float(prezzo_spot),
                "gex_totale_mercato_M$": net,
                "call_gex_totale_M$": call,
                "put_gex_totale_M$": put,
                "total_gex_totale_M$": total,
                "call_wall": float(meta_0dte["call_wall"]) if pd.notna(meta_0dte["call_wall"]) else np.nan,
                "put_wall": float(meta_0dte["put_wall"]) if pd.notna(meta_0dte["put_wall"]) else np.nan,
                "gamma_flip": float(meta_0dte["gamma_flip"]) if pd.notna(meta_0dte["gamma_flip"]) else np.nan,
                "regime_mercato": "LONG GAMMA (Smorza Volatilità)" if net >= 0 else "SHORT GAMMA (Accelera Volatilità)"
            })
            logger.info(f"0DTE {expiry_0dte}: net={net:+.2f}M call_wall={meta_0dte['call_wall']} put_wall={meta_0dte['put_wall']} gamma_flip={meta_0dte['gamma_flip']}")
    except Exception as e:
        logger.exception(f"Errore calcolo 0DTE: {e}")

    try:
        dfs_all = []
        for exp in expiries:
            T = calcola_tempo_rimanente_bybit(exp)
            df_exp, gex_exp, meta_exp = calcola_gex_per_scadenza(df_options, exp, T, RISK_FREE_RATE, prezzo_spot)
            if df_exp is not None and gex_exp is not None:
                dfs_all.append(df_exp)

        if dfs_all:
            df_all = pd.concat(dfs_all, ignore_index=True)
            df_all["call_gex"] = np.where(df_all["type"] == "C", df_all["gamma_calc"] * df_all["openInterest"] * 100 * prezzo_spot, 0.0)
            df_all["put_gex"] = np.where(df_all["type"] == "P", df_all["gamma_calc"] * df_all["openInterest"] * 100 * prezzo_spot, 0.0)
            df_all["net_gex"] = df_all["call_gex"] - df_all["put_gex"]
            df_all["total_gex"] = df_all["call_gex"] + df_all["put_gex"]

            gex_all = df_all.groupby("strike", as_index=False).agg(
                call_gex=("call_gex", "sum"),
                put_gex=("put_gex", "sum"),
                net_gex=("net_gex", "sum"),
                total_gex=("total_gex", "sum")
            ).sort_values("strike").reset_index(drop=True)

            gex_all["call_gex_mn"] = gex_all["call_gex"] / 1_000_000
            gex_all["put_gex_mn"] = gex_all["put_gex"] / 1_000_000
            gex_all["net_gex_mn"] = gex_all["net_gex"] / 1_000_000
            gex_all["total_gex_mn"] = gex_all["total_gex"] / 1_000_000

            net = float(gex_all["net_gex_mn"].sum())
            call = float(gex_all["call_gex_mn"].sum())
            put = float(gex_all["put_gex_mn"].sum())
            total = float(gex_all["total_gex_mn"].sum())

            call_wall = float(gex_all.loc[gex_all["call_gex"].idxmax(), "strike"])
            put_wall = float(gex_all.loc[gex_all["put_gex"].idxmax(), "strike"])

            gex_all["cum_net_gex"] = gex_all["net_gex_mn"].cumsum()
            cum = gex_all["cum_net_gex"].values
            strikes = gex_all["strike"].values
            gamma_flip = np.nan
            for i in range(1, len(cum)):
                if cum[i - 1] * cum[i] < 0:
                    gamma_flip = strikes[i - 1] - cum[i - 1] * (strikes[i] - strikes[i - 1]) / (cum[i] - cum[i - 1])
                    break
            if np.isnan(gamma_flip) and len(gex_all) > 1:
                try:
                    f = interp1d(strikes, cum, kind="linear", fill_value="extrapolate")
                    smin = strikes.min() * 0.8
                    smax = strikes.max() * 1.2
                    if f(smin) * f(smax) < 0:
                        gamma_flip = brentq(f, smin, smax)
                except Exception:
                    gamma_flip = np.nan

            rows_csv.append({
                "timestamp": timestamp_iso,
                "tipo": "ALL",
                "expiry": None,
                "spot": float(prezzo_spot),
                "gex_totale_mercato_M$": net,
                "call_gex_totale_M$": call,
                "put_gex_totale_M$": put,
                "total_gex_totale_M$": total,
                "call_wall": call_wall,
                "put_wall": put_wall,
                "gamma_flip": float(gamma_flip) if pd.notna(gamma_flip) else np.nan,
                "regime_mercato": "LONG GAMMA (Smorza Volatilità)" if net >= 0 else "SHORT GAMMA (Accelera Volatilità)"
            })
            logger.info(f"ALL: net={net:+.2f}M call_wall={call_wall} put_wall={put_wall} gamma_flip={gamma_flip}")
    except Exception as e:
        logger.exception(f"Errore calcolo ALL: {e}")

    try:
        append_rows_csv(rows_csv, CSV_PATH)
        logger.info(f"CSV aggiornato: {CSV_PATH}")
    except Exception as e:
        logger.exception(f"Errore salvataggio CSV: {e}")

    try:
        save_json(
            os.path.join(RAW_SNAPSHOT_DIR, f"snapshot_{BASE_COIN}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"),
            {
                "timestamp": timestamp_iso,
                "ticker": BASE_COIN,
                "spot": float(prezzo_spot),
                "rows": len(df_options),
                "expiries": expiries
            }
        )
    except Exception as e:
        logger.exception(f"Errore salvataggio JSON snapshot: {e}")

    try:
        aggiorna_dashboard_tipo_da_csv(CSV_PATH, "0DTE", DASHBOARD_0DTE_PNG)
        aggiorna_dashboard_tipo_da_csv(CSV_PATH, "ALL", DASHBOARD_ALL_PNG)
    except Exception as e:
        logger.exception(f"Errore aggiornamento dashboard: {e}")


# ==============================================================================
# 12. LOOP 60 SECONDI
# ==============================================================================


def loop_60_secondi():
    logger.info("Start loop Bybit BTC options ogni 60 secondi")
    logger.info(f"CSV: {CSV_PATH}")
    logger.info(f"Dashboard 0DTE: {DASHBOARD_0DTE_PNG}")
    logger.info(f"Dashboard ALL: {DASHBOARD_ALL_PNG}")
    logger.info(f"Calcolo GEX + dashboard: ogni {AGGIORNAMENTO_SECONDI} secondi")

    while True:
        init_daily_paths()
        logger.info("Scarico, calcolo GEX e aggiornamento dashboard")
        ciclo_calcolo_gex()
        time.sleep(AGGIORNAMENTO_SECONDI)


# ==============================================================================
# MAIN
# ==============================================================================


if __name__ == "__main__":
    logger.info("Start script: CalcoloGEXTemporizzatoBybitBTC_60s.py")
    loop_60_secondi()

