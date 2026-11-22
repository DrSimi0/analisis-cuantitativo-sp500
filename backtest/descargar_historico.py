"""Universo histórico (point-in-time) del S&P 500 para reducir el sesgo de supervivencia.

Fuente de composición: github.com/fja05680/sp500 (lista diaria de miembros desde 1996),
guardada en backtest/data/sp500_hist/componentes.csv.
Descarga de yfinance los precios de las acciones que fueron miembro desde 2001 y ya no lo son.
Muchas empresas deslistadas (quiebras, fusiones) NO existen en Yahoo: se reporta la cobertura.

Uso:  python backtest/descargar_historico.py
Salida: backtest/data/precios_hist.pkl  (close, volume de miembros históricos + miembros por fecha)
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA = Path(__file__).parent / "data"
comp = pd.read_csv(DATA / "sp500_hist" / "componentes.csv", parse_dates=["date"])
comp = comp[comp["date"] >= "2001-01-01"]
miembros = {d: set(t.replace(".", "-").lower() for t in s.split(",")) for d, s in zip(comp["date"], comp["tickers"])}

actuales = pd.read_pickle(DATA / "precios.pkl")
ya = set(actuales["close"].columns)
todos = set().union(*miembros.values())
faltan = sorted(todos - ya)
print(f"Miembros históricos desde 2001: {len(todos)} | ya descargados: {len(todos & ya)} | por descargar: {len(faltan)}")

bloque = yf.download([t.upper() for t in faltan], start="2001-01-01", progress=False, auto_adjust=True, threads=True)
close = bloque["Close"]
volume = bloque["Volume"]
close.columns = [c.lower() for c in close.columns]
volume.columns = [c.lower() for c in volume.columns]
con_datos = [c for c in close.columns if close[c].notna().sum() > 0]
print(f"Con datos en Yahoo: {len(con_datos)} de {len(faltan)}")

close_h = pd.concat([actuales["close"], close[con_datos]], axis=1).sort_index()
volume_h = pd.concat([actuales["volume"], volume[con_datos]], axis=1).sort_index()
pd.to_pickle({"close": close_h, "volume": volume_h, "irx": actuales["irx"], "miembros": miembros},
             DATA / "precios_hist.pkl")
print(f"Guardado: {close_h.shape[1]} columnas")
