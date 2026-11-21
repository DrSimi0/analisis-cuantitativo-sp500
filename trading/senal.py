"""Señal en vivo: réplica exacta de senal() de backtest/backtest_v4.py para la última sesión cerrada.

Score General = 35 % Sharpe + 25 % Rendimiento + 25 % Control DD + 15 % Baja Vol (min-max) -> Top 5
-> pesos de Máximo Sharpe (SLSQP) con la tasa del T-Bill de 13 semanas (^IRX).
"""
import time
from io import StringIO

import numpy as np
import pandas as pd
import requests
import scipy.optimize as sco
import yfinance as yf

import config as cfg


def tickers_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    tabla = pd.read_html(StringIO(requests.get(url, headers=header, timeout=30).text))[0]
    return tabla["Symbol"].str.replace(".", "-", regex=False).tolist()


def _descargar(tickers, intentos=3, **kw):
    ultimo_error = None
    for k in range(intentos):
        try:
            b = yf.download(tickers, progress=False, auto_adjust=True, threads=True, **kw)
            if b is not None and not b.empty:
                return b
            ultimo_error = "descarga vacía"
        except Exception as e:  # yfinance/Yahoo falla de forma intermitente
            ultimo_error = e
        time.sleep(30 * (k + 1))
    raise RuntimeError(f"No se pudieron descargar datos de Yahoo: {ultimo_error}")


def descargar_datos(hasta):
    """Precios ajustados hasta la sesión ANTERIOR a `hasta` (end es exclusivo: se descarta la barra parcial de hoy)."""
    tickers = tickers_sp500()
    hasta = pd.Timestamp(hasta).normalize()
    inicio = hasta - pd.DateOffset(years=6)         # 1260 sesiones (~5 años) + margen
    kw = {"start": inicio.strftime("%Y-%m-%d"), "end": hasta.strftime("%Y-%m-%d")}
    b = _descargar(tickers + ["SPY"], **kw)
    close = b["Close"].sort_index()
    volume = b["Volume"].sort_index()
    close.columns = [c.lower() for c in close.columns]
    volume.columns = [c.lower() for c in volume.columns]
    close = close[close.index < hasta]
    volume = volume.reindex(close.index)
    spy = close.pop("spy").dropna()
    volume = volume.drop(columns=["spy"])
    irx = _descargar("^IRX", **kw)["Close"].squeeze().dropna() / 100.0
    sin_datos = [c for c in close.columns if close[c].notna().sum() == 0]
    return {"close": close, "volume": volume, "spy": spy, "irx": irx,
            "n_tickers": len(tickers), "sin_datos": sin_datos}


def regimen(spy):
    """Filtro SMA 200 a la última sesión disponible."""
    sma = spy.rolling(cfg.SMA).mean()
    return {"fecha": str(spy.index[-1].date()), "spy": float(spy.iloc[-1]), "sma200": float(sma.iloc[-1]),
            "alcista": bool(spy.iloc[-1] > sma.iloc[-1]), "distancia": float(spy.iloc[-1] / sma.iloc[-1] - 1)}


def max_sharpe(mu, cov, rf):
    n = len(mu)

    def neg_sharpe(w):
        return -((w @ mu - rf) / np.sqrt(w @ cov @ w))

    res = sco.minimize(neg_sharpe, np.full(n, 1 / n), method="SLSQP", bounds=[(0, cfg.TOPE)] * n,
                       constraints=({"type": "eq", "fun": lambda x: x.sum() - 1},))
    w = np.clip(res.x, 0, None)
    return w / w.sum(), bool(res.success), str(res.message)


def calcular_senal(close, volume, rf):
    """Top 5 y pesos Max Sharpe con los datos hasta la última fila de `close` (misma lógica que el backtest)."""
    if len(close) < cfg.VENTANA:
        raise RuntimeError(f"Historial insuficiente: {len(close)} sesiones < {cfg.VENTANA}")
    vent_p = close.iloc[-cfg.VENTANA:]
    vent_v = volume.iloc[-cfg.VENTANA:]
    cob = vent_p.notna().mean()
    filtro = (cob >= cfg.COBERTURA) & (vent_v.mean() > cfg.VOL_MIN) & vent_p.iloc[-1].notna()
    elegibles = cob.index[filtro]
    p = vent_p[elegibles].dropna()                      # unión 'inner' como en el notebook
    r = p.pct_change().dropna()
    mu = r.mean() * 252
    vol = r.std() * np.sqrt(252)
    sharpe = (mu - rf) / vol
    mdd = (p / p.cummax() - 1).min()
    df = pd.DataFrame({"Rendimiento": mu, "Sharpe": sharpe, "Control_DD": 1 + mdd, "Baja_Vol": 1 / vol})
    nrm = (df - df.min()) / (df.max() - df.min())
    df["Score"] = 0.35 * nrm["Sharpe"] + 0.25 * nrm["Rendimiento"] + 0.25 * nrm["Control_DD"] + 0.15 * nrm["Baja_Vol"]
    top = df["Score"].nlargest(cfg.TOP_N).index.tolist()
    mu5, cov5 = mu[top].values, r[top].cov().values * 252
    w, ok, msg = max_sharpe(mu5, cov5, rf)
    exp_ret, exp_vol = float(w @ mu5), float(np.sqrt(w @ cov5 @ w))
    return {"fecha": str(close.index[-1].date()), "top": top, "pesos": [float(x) for x in w],
            "rf": float(rf), "exp_ret": exp_ret, "exp_vol": exp_vol, "exp_sharpe": (exp_ret - rf) / exp_vol,
            "n_universo": int(len(elegibles)), "n_sesiones_inner": int(len(p)), "convergio": ok, "mensaje_opt": msg,
            "scores": {t: float(df.loc[t, "Score"]) for t in top}}
