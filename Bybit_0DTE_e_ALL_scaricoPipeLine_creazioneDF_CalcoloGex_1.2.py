'''
vers 1.1
partendo da YahooFinance_0DTE_e_ALL_scaricoPipeLine_creazioneDF_CalcoloGex_1.9.py mi creo un analogo per Bybit anzichè da yahoo finance.
In bybit le greche sono già calcolate e non devo calcolarmele io come facevo in yahho finance
senza i grafici


mi apre dentro il browser i grafici per 0DTE e ALL di Gex,call wall, put wall, gamma flip
----------------
vers 1.2
questa versione NON mi apre dentro il browser i grafici ma li salva su disco in .\data_bybit_btc_options

per leggere i dati salvati su disco usare dashboard_live1.1.py
'''

import os
import json
import re
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ==============================================================================
# 1. PARAMETRI
# ==============================================================================

BASE_URL = "https://api.bybit.com/v5/market/tickers"
BASE_COIN = "BTC"
CATEGORY = "option"
RISK_FREE_RATE = 0.045
OUTPUT_DIR = r".\data_bybit_btc_options"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
now_utc = datetime.now(timezone.utc)

# ==============================================================================
# 2. BLACK-SCHOLES & IV INVERSIONE
# ==============================================================================

def bsm_price(S, K, T, r, sigma, option_type="C"):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == "C" else max(0.0, K - S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "C":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
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

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
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

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    pdf_d1 = norm.pdf(d1)

    if option_type == "C":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1.0

    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    return round(delta, 4), round(gamma, 8)

# ==============================================================================
# 3. BYBIT DOWNLOAD
# ==============================================================================

def get_btc_options_bybit():
    params = {
        "category": CATEGORY,
        "baseCoin": BASE_COIN
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit error: {data}")
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

    return {
        "baseCoin": base,
        "expiry": expiry_dt,
        "strike": strike,
        "type": opt_type
    }

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
            "symbol_raw": symbol
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, np.nan
    df["underlyingPrice"] = df["underlyingPrice"].replace(0, np.nan)
    spot = float(df["underlyingPrice"].dropna().iloc[0]) if df["underlyingPrice"].notna().any() else np.nan
    return df, spot

# ==============================================================================
# 4. TIME TO EXPIRY
# ==============================================================================

def calcola_tempo_rimanente_bybit(expiry_iso):
    try:
        expiry_dt = datetime.strptime(expiry_iso, "%Y-%m-%d").replace(
            hour=8, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
        )
    except Exception:
        return 1e-6

    seconds = (expiry_dt - now_utc).total_seconds()
    if seconds <= 0:
        return 1e-6

    return seconds / (365.0 * 24 * 60 * 60)

# ==============================================================================
# 5. GEX CALCULATION
# ==============================================================================

def calcola_gex_per_scadenza(df_all, scadenza_iso, T, r, prezzo_spot):
    df = df_all[df_all["expiry"] == scadenza_iso].copy()
    if df.empty:
        return None, None, None

    def row_gamma(row):
        iv = calcola_iv_newton(
            row["mid_price"], prezzo_spot, row["strike"], T, r, row["type"]
        )
        if iv > 0:
            _, gamma = calcola_greche_esatte(prezzo_spot, row["strike"], T, r, iv, row["type"])
            return gamma
        return row["gamma_api"] if pd.notna(row["gamma_api"]) else 0.0

    df["gamma_calc"] = df.apply(row_gamma, axis=1)

    df["call_gex"] = np.where(
        df["type"] == "C",
        df["gamma_calc"] * df["openInterest"] * 100 * prezzo_spot,
        0.0
    )
    df["put_gex"] = np.where(
        df["type"] == "P",
        df["gamma_calc"] * df["openInterest"] * 100 * prezzo_spot,
        0.0
    )
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
        "max_call_gex": float(gex_by_strike["call_gex"].max()) if not gex_by_strike.empty else None,
        "max_put_gex": float(gex_by_strike["put_gex"].max()) if not gex_by_strike.empty else None
    }

    return df, gex_by_strike, meta

def calc_totals(gex_df):
    return {
        "net": float(gex_df["net_gex_mn"].sum()) if not gex_df.empty else 0.0,
        "call": float(gex_df["call_gex_mn"].sum()) if not gex_df.empty else 0.0,
        "put": float(gex_df["put_gex_mn"].sum()) if not gex_df.empty else 0.0,
        "total": float(gex_df["total_gex_mn"].sum()) if not gex_df.empty else 0.0
    }

# ==============================================================================
# 6. GRAFICI
# ==============================================================================

def genera_grafici_per_dataset(gex_df, prefix, titolo_dataset, prezzo_spot, output_dir, timestamp, expiry_dataset=None):
    if gex_df.empty:
        print(f"Nessun dato per {prefix}")
        return

    call_wall = gex_df.loc[gex_df["call_gex"].idxmax(), "strike"] if not gex_df.empty else np.nan
    put_wall = gex_df.loc[gex_df["put_gex"].idxmax(), "strike"] if not gex_df.empty else np.nan

    gex_copy = gex_df.copy()
    gex_copy["cum_net_gex"] = gex_copy["net_gex_mn"].cumsum()
    cum = gex_copy["cum_net_gex"].values
    strikes = gex_copy["strike"].values

    gamma_flip = np.nan
    for i in range(1, len(cum)):
        if cum[i - 1] * cum[i] < 0:
            gamma_flip = strikes[i - 1] - cum[i - 1] * (strikes[i] - strikes[i - 1]) / (cum[i] - cum[i - 1])
            break

    if np.isnan(gamma_flip) and len(gex_copy) > 1:
        try:
            f = interp1d(strikes, cum, kind="linear", fill_value="extrapolate")
            if f(strikes.min() * 0.8) * f(strikes.max() * 1.2) < 0:
                gamma_flip = brentq(f, strikes.min() * 0.8, strikes.max() * 1.2)
        except Exception:
            gamma_flip = np.nan

    totals = calc_totals(gex_df)
    regime = "LONG GAMMA (Smorza Volatilità)" if totals["net"] >= 0 else "SHORT GAMMA (Accelera Volatilità)"

    testo_box = (
        f"Net GEX: {totals['net']:+.2f} M$\n"
        f"Call GEX: {totals['call']:+.2f} M$\n"
        f"Put GEX: {totals['put']:+.2f} M$\n"
        f"Total GEX: {totals['total']:+.2f} M$\n"
    )
    if not np.isnan(gamma_flip):
        testo_box += f"Gamma Flip: {gamma_flip:.2f}\n"
    if not np.isnan(call_wall):
        testo_box += f"Call Wall: {call_wall:.2f}\n"
    if not np.isnan(put_wall):
        testo_box += f"Put Wall: {put_wall:.2f}\n"
    testo_box += f"Regime: {regime}"

    percentuale_filtro = 0.15
    strike_min = prezzo_spot * (1 - percentuale_filtro)
    strike_max = prezzo_spot * (1 + percentuale_filtro)
    grafico_df = gex_df[(gex_df["strike"] >= strike_min) & (gex_df["strike"] <= strike_max)]

    fig, ax1 = plt.subplots(figsize=(14, 6))
    colori1 = ["#2ca02c" if x >= 0 else "#d62728" for x in grafico_df["net_gex_mn"]]
    ax1.bar(grafico_df["strike"], grafico_df["net_gex_mn"], color=colori1, width=4.0, edgecolor="black", alpha=0.75)
    ax1.axvline(x=prezzo_spot, color="blue", linestyle="--", linewidth=2, label=f"Spot ({prezzo_spot:.2f})")
    if not np.isnan(call_wall):
        ax1.axvline(x=call_wall, color="green", linestyle=":", linewidth=2, label=f"Call Wall ({call_wall:.2f})")
    if not np.isnan(put_wall):
        ax1.axvline(x=put_wall, color="red", linestyle=":", linewidth=2, label=f"Put Wall ({put_wall:.2f})")
    if not np.isnan(gamma_flip):
        ax1.axvline(x=gamma_flip, color="orange", linestyle="-.", linewidth=2.5, label=f"Gamma Flip ({gamma_flip:.2f})")

    ax1.text(
        0.98, 0.93, testo_box, transform=ax1.transAxes, fontsize=10, fontweight="bold",
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="navy", alpha=0.9)
    )

    title = f"{titolo_dataset} - GEX Standard"
    if expiry_dataset:
        title += f" [{expiry_dataset}]"
    plt.title(f"{title} - {timestamp}", fontsize=12, fontweight="bold")
    plt.xlabel("Strike")
    plt.ylabel("Net GEX ($ Milioni)")
    plt.axhline(0, color="black", linewidth=1)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}Grafico_GEX_Standard_BTC_{timestamp}.png"), dpi=300)
    plt.close()

    grafico2_df = gex_df[gex_df["net_gex_mn"].abs() > 0.01]
    if not grafico2_df.empty:
        fig, ax2 = plt.subplots(figsize=(14, 6))
        colori2 = ["#2ca02c" if x >= 0 else "#d62728" for x in grafico2_df["net_gex_mn"]]
        ax2.bar(grafico2_df["strike"], grafico2_df["net_gex_mn"], color=colori2, width=5.0, edgecolor="black", alpha=0.85)
        ax2.axvline(x=prezzo_spot, color="blue", linestyle="--", linewidth=2, label=f"Spot ({prezzo_spot:.2f})")
        if not np.isnan(call_wall):
            ax2.axvline(x=call_wall, color="green", linestyle=":", linewidth=2, label=f"Call Wall ({call_wall:.2f})")
        if not np.isnan(put_wall):
            ax2.axvline(x=put_wall, color="red", linestyle=":", linewidth=2, label=f"Put Wall ({put_wall:.2f})")
        if not np.isnan(gamma_flip):
            ax2.axvline(x=gamma_flip, color="orange", linestyle="-.", linewidth=2.5, label=f"Gamma Flip ({gamma_flip:.2f})")

        ax2.text(
            0.98, 0.93, testo_box, transform=ax2.transAxes, fontsize=10, fontweight="bold",
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="navy", alpha=0.9)
        )

        plt.xlim(grafico2_df["strike"].min() - 20, grafico2_df["strike"].max() + 20)
        plt.title(f"{titolo_dataset} - GEX Ottimizzato - {timestamp}", fontsize=12, fontweight="bold")
        plt.xlabel("Strike")
        plt.ylabel("Net GEX ($ Milioni)")
        plt.axhline(0, color="black", linewidth=1)
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}Grafico_GEX_Ottimizzato_BTC_{timestamp}.png"), dpi=300)
        plt.close()

    fig, axs = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    axs = axs.flatten()
    plots = [
        ("Call GEX", "call_gex_mn", "#2ca02c"),
        ("Put GEX", "put_gex_mn", "#d62728"),
        ("Net GEX", "net_gex_mn", "#1f77b4"),
        ("Total GEX", "total_gex_mn", "#9467bd")
    ]
    subset = gex_df[(gex_df["strike"] >= strike_min) & (gex_df["strike"] <= strike_max)]

    for ax, (title, col, color) in zip(axs, plots):
        ax.bar(subset["strike"], subset[col], color=color, edgecolor="black", alpha=0.8)
        ax.axvline(x=prezzo_spot, color="blue", linestyle="--", linewidth=1.8, label=f"Spot ({prezzo_spot:.2f})")
        if not np.isnan(call_wall):
            ax.axvline(x=call_wall, color="green", linestyle=":", linewidth=1.8, label=f"Call Wall ({call_wall:.2f})")
        if not np.isnan(put_wall):
            ax.axvline(x=put_wall, color="red", linestyle=":", linewidth=1.8, label=f"Put Wall ({put_wall:.2f})")
        if not np.isnan(gamma_flip):
            ax.axvline(x=gamma_flip, color="orange", linestyle="-.", linewidth=2.5, label=f"Gamma Flip ({gamma_flip:.2f})")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="upper left")

    axs[0].set_ylabel("GEX ($ Milioni)")
    axs[2].set_ylabel("GEX ($ Milioni)")
    axs[2].set_xlabel("Strike")
    axs[3].set_xlabel("Strike")
    fig.suptitle(f"{titolo_dataset} - Analisi GEX Completa - {timestamp}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}Grafico_GEX_Completo_BTC_{timestamp}.png"), dpi=300)
    plt.close()

# ==============================================================================
# 7. MAIN
# ==============================================================================

print("Scarico opzioni BTC da Bybit...")
df_options, prezzo_spot = load_options_df()

if df_options.empty:
    raise RuntimeError("Nessun dato opzioni BTC disponibile da Bybit.")

print(f"Opzioni trovate: {len(df_options)}")
print(f"Spot BTC: {prezzo_spot:.2f}")

# Salvataggio snapshot grezzo
raw_json_path = os.path.join(OUTPUT_DIR, f"btc_options_snapshot_{timestamp}.json")
with open(raw_json_path, "w", encoding="utf-8") as f:
    json.dump(df_options.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

csv_raw_path = os.path.join(OUTPUT_DIR, f"btc_options_grezzi_{timestamp}.csv")
df_options.to_csv(csv_raw_path, index=False, encoding="utf-8")

expiries = sorted(df_options["expiry"].dropna().unique().tolist())
print(f"Scadenze trovate: {len(expiries)}")

# 0DTE = expiry più vicina
expiry_0dte = expiries[0] if expiries else None

df_0dte = pd.DataFrame()
gex_0dte = pd.DataFrame()
meta_0dte = {}

if expiry_0dte:
    T_0dte = calcola_tempo_rimanente_bybit(expiry_0dte)
    df_0dte, gex_0dte, meta_0dte = calcola_gex_per_scadenza(df_options, expiry_0dte, T_0dte, RISK_FREE_RATE, prezzo_spot)

    if df_0dte is not None and not df_0dte.empty:
        df_0dte.to_csv(os.path.join(OUTPUT_DIR, f"0DTE_dati_opzioni_grezzi_BTC_{timestamp}.csv"), index=False, encoding="utf-8")
        gex_0dte.to_csv(os.path.join(OUTPUT_DIR, f"0DTE_dati_gex_calcolati_BTC_{timestamp}.csv"), index=False, encoding="utf-8")

        metadati_0dte = {
            "ticker": "BTC",
            "tipo": "0DTE",
            "expiry": expiry_0dte,
            "timestamp_esecuzione": now_utc.isoformat(),
            "prezzo_spot": float(prezzo_spot),
            "T": float(T_0dte),
            "r": float(RISK_FREE_RATE),
        }
        metadati_0dte.update(meta_0dte)

        if not gex_0dte.empty:
            totals = calc_totals(gex_0dte)
            metadati_0dte["gex_totale_mercato_M$"] = totals["net"]
            metadati_0dte["call_gex_totale_M$"] = totals["call"]
            metadati_0dte["put_gex_totale_M$"] = totals["put"]
            metadati_0dte["total_gex_totale_M$"] = totals["total"]
            metadati_0dte["regime_mercato"] = "LONG GAMMA (Smorza Volatilità)" if totals["net"] >= 0 else "SHORT GAMMA (Accelera Volatilità)"

        with open(os.path.join(OUTPUT_DIR, f"0DTE_metadati_gex_BTC_{timestamp}.json"), "w", encoding="utf-8") as f:
            json.dump(metadati_0dte, f, indent=2, ensure_ascii=False)

# ALL EXPIRIES
dfs_all = []
gex_list_all = []
meta_all = []

for exp in expiries:
    T = calcola_tempo_rimanente_bybit(exp)
    df_exp, gex_exp, meta_exp = calcola_gex_per_scadenza(df_options, exp, T, RISK_FREE_RATE, prezzo_spot)
    if df_exp is not None and gex_exp is not None:
        dfs_all.append(df_exp)
        gex_list_all.append(gex_exp)
        meta_all.append(meta_exp)

if dfs_all:
    df_all = pd.concat(dfs_all, ignore_index=True)
    df_all.to_csv(os.path.join(OUTPUT_DIR, f"ALL_dati_opzioni_grezzi_BTC_{timestamp}.csv"), index=False, encoding="utf-8")

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

    gex_all.to_csv(os.path.join(OUTPUT_DIR, f"ALL_dati_gex_calcolati_BTC_{timestamp}.csv"), index=False, encoding="utf-8")

    metadati_all = {
        "ticker": "BTC",
        "tipo": "ALL_EXPIRIES",
        "timestamp_esecuzione": now_utc.isoformat(),
        "prezzo_spot": float(prezzo_spot),
        "r": float(RISK_FREE_RATE),
        "num_expiries": len(meta_all),
        "expiries": [m["expiry"] for m in meta_all if m.get("expiry")]
    }

    totals_all = calc_totals(gex_all)
    metadati_all["gex_totale_mercato_M$"] = totals_all["net"]
    metadati_all["call_gex_totale_M$"] = totals_all["call"]
    metadati_all["put_gex_totale_M$"] = totals_all["put"]
    metadati_all["total_gex_totale_M$"] = totals_all["total"]
    metadati_all["regime_mercato"] = "LONG GAMMA (Smorza Volatilità)" if totals_all["net"] >= 0 else "SHORT GAMMA (Accelera Volatilità)"

    if not gex_all.empty:
        metadati_all["call_wall"] = float(gex_all.loc[gex_all["call_gex"].idxmax(), "strike"])
        metadati_all["put_wall"] = float(gex_all.loc[gex_all["put_gex"].idxmax(), "strike"])

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
                if f(strikes.min() * 0.8) * f(strikes.max() * 1.2) < 0:
                    gamma_flip = brentq(f, strikes.min() * 0.8, strikes.max() * 1.2)
            except Exception:
                gamma_flip = np.nan

        if not np.isnan(gamma_flip):
            metadati_all["gamma_flip"] = float(gamma_flip)

    with open(os.path.join(OUTPUT_DIR, f"ALL_metadati_gex_BTC_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(metadati_all, f, indent=2, ensure_ascii=False)

    genera_grafici_per_dataset(
        gex_all,
        prefix="ALL_",
        titolo_dataset="BTC Options - ALL EXPIRIES",
        prezzo_spot=prezzo_spot,
        output_dir=OUTPUT_DIR,
        timestamp=timestamp
    )

    if not gex_0dte.empty:
        genera_grafici_per_dataset(
            gex_0dte,
            prefix="0DTE_",
            titolo_dataset="BTC Options - 0DTE",
            prezzo_spot=prezzo_spot,
            output_dir=OUTPUT_DIR,
            timestamp=timestamp,
            expiry_dataset=expiry_0dte
        )

print("Completato.")