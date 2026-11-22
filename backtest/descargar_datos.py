"""Descarga 25 años de precios ajustados (S&P 500 actual + SPY + T-Bill ^IRX) y los guarda en caché.

Uso:  python backtest/descargar_datos.py
Salida: backtest/data/precios.pkl  (dict con 'close', 'volume', 'irx')
"""
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
INICIO = "2001-01-01"

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
tabla = pd.read_html(StringIO(requests.get(url, headers=header, timeout=30).text))[0]
tickers = tabla["Symbol"].str.replace(".", "-", regex=False).tolist()
print(f"{len(tickers)} tickers del S&P 500 actual")

bloque = yf.download(tickers + ["SPY"], start=INICIO, progress=False, auto_adjust=True, threads=True)
close = bloque["Close"].sort_index()
volume = bloque["Volume"].sort_index()
close.columns = [c.lower() for c in close.columns]
volume.columns = [c.lower() for c in volume.columns]

irx = yf.download("^IRX", start=INICIO, progress=False, auto_adjust=True)["Close"]
irx = irx.squeeze().rename("irx") / 100.0  # tasa anual de T-Bill 13 semanas

pd.to_pickle({"close": close, "volume": volume, "irx": irx}, DATA / "precios.pkl")
print(f"Guardado: {close.shape[1]} columnas, {close.index.min().date()} a {close.index.max().date()}")
print("Tickers sin datos:", [c for c in close.columns if close[c].notna().sum() == 0])
