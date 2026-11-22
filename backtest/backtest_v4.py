"""Backtest walk-forward de 20 años de la estrategia de Stocks_Auto_Analisis_V4.

En cada fecha de rebalanceo, usando SOLO los datos disponibles hasta ese día:
  1. Ventana de 5 años (1260 sesiones), filtro de volumen promedio > 1M y cobertura >= 95 %.
  2. Score General = 35 % Sharpe + 25 % Rendimiento + 25 % Control DD + 15 % Baja Vol (min-max).
  3. Top 5 -> pesos de Máximo Sharpe (SLSQP, 0-100 %, rf = 4 %), igual que la celda 26.
  4. Se ejecuta al cierre de la sesión SIGUIENTE (sin mirar al futuro), con acciones enteras,
     costos de 10 pb por operación y el efectivo sobrante rinde la tasa del T-Bill (^IRX).

Variantes (variables de entorno, combinables):
  BT_TOPE=0.35           tope de peso por acción en Max Sharpe (por defecto 1.0, como el notebook)
  BT_RF=dinamica         tasa libre de riesgo del T-Bill vigente en t (por defecto 4 % fija)
  BT_FILTRO=sma200       a efectivo cuando SPY < SMA 200 (revisión a fin de cada mes)
  BT_UNIVERSO=historico  solo miembros del S&P 500 en la fecha t (requiere descargar_historico.py)

Uso:  python backtest/backtest_v4.py      (requiere haber corrido descargar_datos.py)
Salida: backtest/resultados[_variante]/  (gráficos PNG, CSV y resultados.json)
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as sco
from scipy.stats import spearmanr

BASE = Path(__file__).parent
# Tope de peso por acción en Max Sharpe: 1.0 = notebook (celda 26). Variante: BT_TOPE=0.35
TOPE = float(os.environ.get("BT_TOPE", "1.0"))
RF_DIN = os.environ.get("BT_RF", "") == "dinamica"
FILTRO = os.environ.get("BT_FILTRO", "") == "sma200"
UNIV_HIST = os.environ.get("BT_UNIVERSO", "") == "historico"
# Banda de tolerancia: una acción que SIGUE en el Top 5 solo se ajusta si su peso actual se desvía del
# objetivo más de BANDA (puntos de peso absolutos, p. ej. 0.05 = 5 pp). Entradas y salidas se ejecutan siempre.
BANDA = float(os.environ.get("BT_BANDA", "0"))
# Termómetro de volatilidad (revisión mensual): "vix" = 100 % si VIX<20, 50 % si 20-30, efectivo si >30;
# "voltarget" = exposición = 15 % / volatilidad realizada (63 días) de la cartera, máximo 100 %.
TERMO = os.environ.get("BT_TERMO", "")
# Complemento por categoría (cuadrantes de la Fase 3.2 del notebook): "alcista" = solo RSI>50;
# "sin_alta_bajista" = excluye Alta Vol - Bajista; "regimen" = si VIX>20 solo acciones de Baja Volatilidad.
CATEG = os.environ.get("BT_CATEG", "")
# Rebalanceo por objetivo: BT_REBAL=objetivo rebalancea toda la cartera cuando cualquier acción llega a
# precio_compra × (1 + rendimiento esperado anual). BT_MAXDIAS=252 añade un máximo de sesiones sin rebalancear.
REBAL = os.environ.get("BT_REBAL", "")
MAXDIAS = int(os.environ.get("BT_MAXDIAS", "0"))
VIX_BAJO, VIX_ALTO = 20.0, 30.0
VOL_OBJETIVO = float(os.environ.get("BT_VOLOBJ", "0.15"))   # volatilidad anual objetivo del termómetro "voltarget"
_sufijo = "".join([f"_tope{round(TOPE * 100)}" if TOPE < 1 else "", "_univhist" if UNIV_HIST else "",
                   "_sma200" if FILTRO else "", "_rfdin" if RF_DIN else "",
                   f"_banda{BANDA * 100:g}".replace(".", "p") if BANDA > 0 else "",
                   f"_rebal-{REBAL}" if REBAL else "", f"{MAXDIAS}d" if REBAL and MAXDIAS else "",
                   f"_termo-{TERMO}" if TERMO else "", f"{VOL_OBJETIVO * 100:g}" if TERMO == "voltarget" and VOL_OBJETIVO != 0.15 else "", f"_cat-{CATEG}" if CATEG else ""])
OUT = BASE / f"resultados{_sufijo}"
OUT.mkdir(exist_ok=True)

# ---------------- Parámetros (los mismos del notebook V4) ----------------
CAPITAL = 100_000
VENTANA = 1260            # 5 años de sesiones
RF_MODELO = 0.04          # tasa libre de riesgo fija usada por el notebook
VOL_MIN = 1_000_000
COBERTURA = 0.95
TOP_N = 5
COSTO = 0.0010            # 10 pb por cada dólar operado (comisión + deslizamiento)
ANIOS = 20
FRECUENCIAS = {"Mensual": "ME", "Trimestral": "QE", "Semestral": "6ME", "Anual": "YE"}
FREC_PRINCIPAL = "Objetivo" if os.environ.get("BT_REBAL", "") == "objetivo" else "Trimestral"
N_MC_BOOT = 5000
N_MC_RANDOM = 1000
SEED = 42

datos = pd.read_pickle(BASE / "data" / ("precios_hist.pkl" if UNIV_HIST else "precios.pkl"))
close_all = datos["close"]
volume_all = datos["volume"]
spy = close_all["spy"]
close = close_all.drop(columns=["spy"])
volume = volume_all.drop(columns=["spy"])
fechas = close.index
irx = datos["irx"].reindex(fechas).ffill().bfill()
rf_diario = (1 + irx) ** (1 / 252) - 1           # rendimiento diario del efectivo
ret = close.pct_change(fill_method=None)          # NaN donde no hay precio
spy_ret = spy.pct_change()

spy_sma200 = spy.rolling(200).mean()
_vix_pkl = BASE / "data" / "vix.pkl"
if not _vix_pkl.exists():
    import yfinance as yf
    yf.download("^VIX", start="2001-01-01", progress=False, auto_adjust=True)["Close"].squeeze().to_pickle(_vix_pkl)
vix = pd.read_pickle(_vix_pkl).reindex(fechas).ffill().bfill()
if CATEG:
    # RSI de Wilder (14) de cada acción, causal: solo usa precios hasta cada fecha
    _delta = close.diff()
    _gan = _delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    _per = (-_delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi_all = 100 - 100 / (1 + _gan / _per)
if UNIV_HIST:
    _fechas_comp = sorted(datos["miembros"])
    def miembros(t):
        """Tickers que pertenecían al S&P 500 en la fecha t (última composición publicada <= t)."""
        k = np.searchsorted(np.array(_fechas_comp, dtype="datetime64[ns]"), np.datetime64(t), side="right") - 1
        return datos["miembros"][_fechas_comp[k]]

FIN = fechas[-1]
INICIO_BT = FIN - pd.DateOffset(years=ANIOS)


# ============================ SEÑAL (réplica del notebook) ============================
EXPO_STATS = False       # backtest_v5.py lo activa para reportar la exposición promedio
FALLOS_OPT = []          # optimizaciones SLSQP que no convergieron (se reportan al final)


def max_sharpe(mu, cov, rf=RF_MODELO):
    n = len(mu)
    def neg_sharpe(w):
        vol = np.sqrt(w @ cov @ w)
        return -((w @ mu - rf) / vol)
    res = sco.minimize(neg_sharpe, np.full(n, 1 / n), method="SLSQP",
                       bounds=[(0, TOPE)] * n, constraints=({"type": "eq", "fun": lambda x: x.sum() - 1},))
    if not res.success:
        FALLOS_OPT.append(res.message)
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def senal(t):
    """Top 5, pesos Max Sharpe y métricas usando datos hasta la fecha t (inclusive)."""
    i = fechas.get_loc(t)
    if i + 1 < VENTANA:
        return None
    vent_p = close.iloc[i + 1 - VENTANA:i + 1]
    vent_v = volume.iloc[i + 1 - VENTANA:i + 1]
    rf_t = float(irx.iloc[i]) if RF_DIN else RF_MODELO
    cob = vent_p.notna().mean()
    filtro = (cob >= COBERTURA) & (vent_v.mean() > VOL_MIN) & vent_p.iloc[-1].notna()
    n_miembros = n_miembros_con_datos = None
    if UNIV_HIST:
        m = miembros(t)
        en_indice = cob.index.isin(list(m))
        n_miembros = len(m)
        n_miembros_con_datos = int((en_indice & vent_p.iloc[-1].notna().values).sum())
        filtro &= en_indice
    elegibles = cob.index[filtro]
    p = vent_p[elegibles].dropna()                      # unión 'inner' como en el notebook
    r = p.pct_change().dropna()
    mu = r.mean() * 252
    vol = r.std() * np.sqrt(252)
    sharpe = (mu - rf_t) / vol
    mdd = (p / p.cummax() - 1).min()
    df = pd.DataFrame({"Rendimiento": mu, "Sharpe": sharpe, "Control_DD": 1 + mdd, "Baja_Vol": 1 / vol})
    nrm = (df - df.min()) / (df.max() - df.min())
    df["Score"] = 0.35 * nrm["Sharpe"] + 0.25 * nrm["Rendimiento"] + 0.25 * nrm["Control_DD"] + 0.15 * nrm["Baja_Vol"]
    candidatos = df.index
    if CATEG:
        alta_vol = vol > vol.median()                       # mismo umbral que la Fase 3.2 del notebook
        alcista = rsi_all.iloc[i].reindex(df.index) > 50
        if CATEG == "alcista":
            candidatos = df.index[alcista.values]
        elif CATEG == "sin_alta_bajista":
            candidatos = df.index[~(alta_vol & ~alcista).values]
        elif CATEG == "regimen" and vix.iloc[i] > VIX_BAJO:
            candidatos = df.index[~alta_vol.values]
        if len(candidatos) < TOP_N:
            candidatos = df.index
    top = df.loc[candidatos, "Score"].nlargest(TOP_N).index.tolist()
    mu5, cov5 = mu[top].values, r[top].cov().values * 252
    w = max_sharpe(mu5, cov5, rf_t)
    exp_ret = w @ mu5
    exp_vol = np.sqrt(w @ cov5 @ w)
    return {"fecha": t, "top": top, "pesos": w, "exp_ret": exp_ret, "exp_vol": exp_vol,
            "exp_sharpe": (exp_ret - rf_t) / exp_vol, "score": df["Score"], "n_univ": len(elegibles), "mu": mu5,
            "n_miembros": n_miembros, "n_miembros_con_datos": n_miembros_con_datos}


def exposicion(d, s):
    """Fracción invertida al cierre de d: filtro SMA200 (0/1) por termómetro de volatilidad."""
    e = 1.0
    if FILTRO and not spy.loc[d] > spy_sma200.loc[d]:
        return 0.0
    if TERMO == "vix":
        v = vix.loc[d]
        e = 1.0 if v < VIX_BAJO else 0.5 if v < VIX_ALTO else 0.0
    elif TERMO == "voltarget":
        i = fechas.get_loc(d)
        rp = ret.iloc[i - 62:i + 1][s["top"]].fillna(0).values @ s["pesos"]
        vol_p = rp.std() * np.sqrt(252)
        e = min(1.0, np.floor(VOL_OBJETIVO / vol_p * 10) / 10) if vol_p > 0 else 1.0
    return e


def construir_eventos_termo(senales):
    """Revisión mensual de la exposición; se opera al cambiar >= 20 pp o en cada señal trimestral."""
    fechas_senal = {s["fecha"]: s for s in senales}
    s_fechas = pd.Series(fechas, index=fechas)
    fin_mes = set(pd.Timestamp(x) for x in s_fechas.resample("ME").last().dropna().values)
    calendario = sorted(d for d in fin_mes | set(fechas_senal) if senales[0]["fecha"] <= d < FIN)
    eventos, actual, e_prev = [], None, None
    for d in calendario:
        if d in fechas_senal:
            actual = fechas_senal[d]
        e = exposicion(d, actual)
        if d in fechas_senal or e_prev is None or abs(e - e_prev) >= 0.2 - 1e-9:
            if e <= 0:
                eventos.append({"fecha": d, "top": [], "pesos": np.array([]), "efectivo": True})
            else:
                eventos.append({**actual, "fecha": d, "pesos": actual["pesos"] * e, "exposicion": e})
            e_prev = e
    return eventos


def construir_eventos(senales):
    """Calendario de operaciones. Sin filtro = las señales. Con filtro SMA 200: a fin de cada mes,
    si SPY < SMA200 se pasa a efectivo; al volver arriba se recompra la última señal disponible."""
    if not FILTRO and not TERMO:
        return list(senales)
    if TERMO:
        return construir_eventos_termo(senales)
    fechas_senal = {s["fecha"]: s for s in senales}
    s_fechas = pd.Series(fechas, index=fechas)
    fin_mes = set(pd.Timestamp(x) for x in s_fechas.resample("ME").last().dropna().values)
    calendario = sorted(d for d in fin_mes | set(fechas_senal) if senales[0]["fecha"] <= d < FIN)
    eventos, actual, estado = [], None, None
    for d in calendario:
        if d in fechas_senal:
            actual = fechas_senal[d]
        alcista = bool(spy.loc[d] > spy_sma200.loc[d])
        if d in fechas_senal or alcista != estado:
            if alcista:
                eventos.append(actual if d in fechas_senal else {**actual, "fecha": d, "reentrada": True})
            else:
                eventos.append({"fecha": d, "top": [], "pesos": np.array([]), "efectivo": True})
            estado = alcista
    return eventos


def eventos_objetivo():
    """Calendario de rebalanceo por objetivo de rendimiento (en vez de por tiempo).
    Al ejecutar una señal, cada acción comprada recibe objetivo = cierre de ejecución × (1 + su rendimiento
    esperado anual). Cada día, si algún cierre alcanza su objetivo (o pasan MAXDIAS sesiones), se rehace el
    análisis completo ese día y se ejecuta al cierre siguiente. Con filtro SMA200 la revisión mensual sigue igual."""
    px = close.ffill().values
    col = {c: j for j, c in enumerate(close.columns)}
    s_f = pd.Series(fechas, index=fechas)
    fin_mes = set(pd.Timestamp(x) for x in s_f.resample("ME").last().dropna().values)
    t0 = pd.Timestamp(fechas_rebalanceo("QE")[0])
    eventos, senales, motivos = [], [], []
    estado = {"invertido": False, "exec": None, "obj": {}}

    def evento_senal(i, motivo):
        sg = senal(fechas[i])
        ev = {**sg, "motivo": motivo}
        eventos.append(ev); senales.append(ev); motivos.append(motivo)
        ex = i + 1
        estado.update(invertido=True, exec=ex,
                      obj={col[t]: px[ex, col[t]] * (1 + m) for t, w, m in zip(sg["top"], sg["pesos"], sg["mu"])
                           if w > 1e-3 and m > 0})

    def evento_efectivo(i):
        eventos.append({"fecha": fechas[i], "top": [], "pesos": np.array([]), "efectivo": True, "motivo": "filtro"})
        motivos.append("filtro")
        estado.update(invertido=False, exec=None, obj={})

    i0 = fechas.get_loc(t0)
    if FILTRO and not spy.iloc[i0] > spy_sma200.iloc[i0]:
        evento_efectivo(i0)
    else:
        evento_senal(i0, "inicio")
    for i in range(i0 + 1, len(fechas) - 1):
        d = fechas[i]
        if FILTRO and d in fin_mes:
            alcista = bool(spy.iloc[i] > spy_sma200.iloc[i])
            if estado["invertido"] and not alcista:
                evento_efectivo(i); continue
            if not estado["invertido"] and alcista:
                evento_senal(i, "reentrada"); continue
        if estado["invertido"] and i > estado["exec"]:
            if any(px[i, j] >= o for j, o in estado["obj"].items()):
                evento_senal(i, "objetivo")
            elif MAXDIAS and i - estado["exec"] >= MAXDIAS:
                evento_senal(i, "tiempo")
    return eventos, senales, motivos


def fechas_rebalanceo(regla):
    """Último día hábil de cada periodo dentro del horizonte del backtest."""
    s = pd.Series(fechas, index=fechas)
    f = s.resample(regla).last().dropna()
    f = f[(f >= INICIO_BT - pd.Timedelta(days=1)) & (f < FIN)]
    return list(f.values.astype("datetime64[ns]"))


# ============================ SIMULACIÓN CON ACCIONES ENTERAS ============================
def simular(senales, capital=CAPITAL, costo=COSTO):
    """Ejecuta cada señal al cierre de la sesión siguiente. Devuelve NAV diaria y bitácora."""
    px = close.ffill()
    exec_idx = {fechas.get_loc(s["fecha"]) + 1: s for s in senales}
    inicio = min(exec_idx)
    efectivo, acciones = float(capital), pd.Series(dtype=float)
    nav, costos, operado, log = {}, 0.0, 0.0, []
    n_ops = 0
    for i in range(inicio, len(fechas)):
        d = fechas[i]
        if i > inicio:
            efectivo *= 1 + rf_diario.iloc[i]
        if i in exec_idx:
            s = exec_idx[i]
            precios = px.iloc[i]
            valor = efectivo + (acciones * precios[acciones.index]).sum()
            objetivo = pd.Series(np.floor(valor * s["pesos"] * (1 - 2 * costo) / precios[s["top"]].values),
                                 index=s["top"])
            if BANDA > 0 and valor > 0:
                # posiciones que siguen en el Top 5 y están dentro de la banda: no se tocan
                for t in objetivo.index.intersection(acciones.index):
                    peso_actual = acciones[t] * precios[t] / valor
                    if abs(peso_actual - s["pesos"][s["top"].index(t)]) <= BANDA:
                        objetivo[t] = acciones[t]
                # si las posiciones conservadas dejan poco efectivo, se recortan solo las compras
                todos = acciones.index.union(objetivo.index)
                delta = objetivo.reindex(todos, fill_value=0) - acciones.reindex(todos, fill_value=0)
                flujo = (delta * precios[todos]).sum() + (delta.abs() * precios[todos]).sum() * costo
                if flujo > efectivo:
                    compras = delta[delta > 0]
                    costo_compras = (compras * precios[compras.index]).sum()
                    factor = max(0.0, 1 - (flujo - efectivo) * (1 + costo) / costo_compras) if costo_compras > 0 else 1.0
                    for t in compras.index:
                        objetivo[t] = acciones.get(t, 0) + np.floor(compras[t] * factor)
            todos = acciones.index.union(objetivo.index)
            delta = objetivo.reindex(todos, fill_value=0) - acciones.reindex(todos, fill_value=0)
            n_ops += int((delta != 0).sum())
            monto = (delta.abs() * precios[todos]).sum()
            c = monto * costo
            efectivo -= (delta * precios[todos]).sum() + c
            costos += c
            operado += monto
            acciones = objetivo[objetivo > 0]
            log.append({"senal": s["fecha"], "ejecucion": d, "valor": valor, "costo": c,
                        "rotacion": monto / valor, "acciones": acciones.to_dict()})
        nav[d] = efectivo + (acciones * px.iloc[i][acciones.index]).sum()
    simular.n_ops = n_ops
    return pd.Series(nav), log, costos, operado


# ============================ MÉTRICAS ============================
def metricas(nav, bench=None, rf=rf_diario):
    r = nav.pct_change().dropna()
    rfx = rf.reindex(r.index).fillna(0)
    ex = r - rfx
    anios = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / anios) - 1
    vol = r.std() * np.sqrt(252)
    neg = ex[ex < 0]
    dd = nav / nav.cummax() - 1
    # duración máxima bajo el agua (días calendario)
    bajo = dd < 0
    grupos = (~bajo).cumsum()
    dur = dd[bajo].groupby(grupos[bajo]).apply(lambda x: (x.index[-1] - x.index[0]).days).max() if bajo.any() else 0
    mensual = nav.resample("ME").last().pct_change().dropna()
    anual = nav.resample("YE").last().pct_change().dropna()
    var95 = np.percentile(r, 5)
    m = {
        "Valor final": nav.iloc[-1],
        "Retorno total": nav.iloc[-1] / nav.iloc[0] - 1,
        "CAGR": cagr,
        "Volatilidad anual": vol,
        "Sharpe": ex.mean() / ex.std() * np.sqrt(252),
        "Sortino": ex.mean() * 252 / (np.sqrt((neg ** 2).sum() / len(ex)) * np.sqrt(252)),
        "Max Drawdown": dd.min(),
        "Calmar": cagr / abs(dd.min()),
        "Duración máx. DD (días)": int(dur),
        "VaR 95% diario": var95,
        "CVaR 95% diario": r[r <= var95].mean(),
        "Asimetría": r.skew(),
        "Curtosis": r.kurt(),
        "% meses positivos": (mensual > 0).mean(),
        "Mejor año": anual.max(),
        "Peor año": anual.min(),
    }
    if bench is not None:
        b = bench.reindex(nav.index).pct_change().dropna().reindex(r.index)
        bx = b - rfx
        beta = np.cov(ex, bx)[0, 1] / bx.var()
        alpha = (ex.mean() - beta * bx.mean()) * 252
        te = (r - b).std() * np.sqrt(252)
        bm = bench.reindex(nav.index).resample("ME").last().pct_change().dropna().reindex(mensual.index)
        m.update({
            "Beta vs SPY": beta,
            "Alfa anual (CAPM)": alpha,
            "Correlación vs SPY": r.corr(b),
            "Tracking error": te,
            "Information ratio": (r - b).mean() * 252 / te if te > 1e-8 else np.nan,
            "Captura alcista": mensual[bm > 0].mean() / bm[bm > 0].mean(),
            "Captura bajista": mensual[bm < 0].mean() / bm[bm < 0].mean(),
            "% meses que vence a SPY": (mensual > bm).mean(),
        })
    return m


# ============================ MOTOR VECTORIZADO (para Monte Carlo y comparaciones) ============================
RF = rf_diario.values
R = np.hstack([ret.fillna(0).values, RF[:, None]])    # última columna = efectivo (rinde el T-Bill)
CASH = R.shape[1] - 1
def backtest_pesos(senales_idx, pesos, columnas, costo=COSTO):
    """Compra y mantiene entre rebalanceos (los pesos derivan). Devuelve NAV diaria normalizada."""
    n = len(fechas)
    inicio = senales_idx[0] + 1                 # primera ejecución: cierre de la sesión siguiente
    nav = np.empty(n - inicio)
    nav[0] = 1.0
    v = 1.0
    previo = {}                                 # pesos derivados antes de rebalancear {columna: peso}
    for k, i0 in enumerate(senales_idx):
        a = i0 + 1                              # se ejecuta al cierre de a
        b = senales_idx[k + 1] + 1 if k + 1 < len(senales_idx) else n - 1   # se mantiene hasta el cierre de b
        cols, w = list(columnas[k]), np.asarray(pesos[k], dtype=float)
        if cols and w.sum() < 1 - 1e-9:                    # exposición parcial: el resto queda en efectivo
            cols, w = cols + [CASH], np.append(w, 1 - w.sum())
        nuevo = dict(zip(cols, w))
        operado = sum(abs(nuevo.get(c, 0.0) - previo.get(c, 0.0)) for c in set(nuevo) | set(previo) if c != CASH)
        v *= 1 - costo * operado
        nav[a - inicio] = v                     # valor tras pagar el costo del rebalanceo
        if b > a and not cols:                  # periodo en efectivo: rinde el T-Bill
            nav[a + 1 - inicio:b + 1 - inicio] = v * np.cumprod(1 + RF[a + 1:b + 1])
            v = nav[b - inicio]
            previo = {}
        elif b > a:
            crec = np.cumprod(1 + R[a + 1:b + 1][:, cols], axis=0)   # retornos de los días a+1..b
            nav[a + 1 - inicio:b + 1 - inicio] = v * (crec @ w)
            w_fin = w * crec[-1]
            v = nav[b - inicio]
            previo = dict(zip(cols, w_fin / w_fin.sum()))
        else:
            previo = nuevo
    return pd.Series(nav, index=fechas[inicio:])


# ============================ EJECUCIÓN ============================
def main():
    """Corre el backtest completo y guarda resultados en OUT (lo reutiliza backtest_v5.py)."""
    rng = np.random.default_rng(SEED)
    col_idx = {c: j for j, c in enumerate(close.columns)}
    res = {"parametros": {"capital": CAPITAL, "ventana_dias": VENTANA, "rf_modelo": RF_MODELO, "costo_pb": COSTO * 1e4,
                          "inicio": str(INICIO_BT.date()), "fin": str(FIN.date()), "frecuencia_principal": FREC_PRINCIPAL, "tope_peso": TOPE, "banda": BANDA}}

    # ---- 1. Señales para todas las frecuencias ----
    cache = {}
    senales_por_frec = {}
    for nombre, regla in ({} if REBAL == "objetivo" else FRECUENCIAS).items():
        lst = []
        for t in fechas_rebalanceo(regla):
            t = pd.Timestamp(t)
            if t not in cache:
                cache[t] = senal(t)
            if cache[t] is not None:
                lst.append(cache[t])
        senales_por_frec[nombre] = lst
        print(f"{nombre}: {len(lst)} rebalanceos")

    # ---- 2. Simulación con $100K para cada frecuencia ----
    navs, logs, resumen_frec = {}, {}, {}
    eventos_por_frec = {k: construir_eventos(v) for k, v in senales_por_frec.items()}
    if REBAL == "objetivo":
        _ev, _sg, motivos_obj = eventos_objetivo()
        senales_por_frec["Objetivo"], eventos_por_frec["Objetivo"] = _sg, _ev
        print(f"Objetivo: {len(_ev)} eventos -> " + str(pd.Series(motivos_obj).value_counts().to_dict()))
    for nombre, lst in eventos_por_frec.items():
        nav, log, costos, operado = simular(lst)
        navs[nombre], logs[nombre] = nav, log
        m = metricas(nav, spy)
        m["Costos totales"] = costos
        m["Operaciones"] = simular.n_ops
        m["Rotación anual promedio"] = operado / nav.mean() / ((nav.index[-1] - nav.index[0]).days / 365.25) / 2
        resumen_frec[nombre] = m

    nav_bt = navs[FREC_PRINCIPAL]
    senales = senales_por_frec[FREC_PRINCIPAL]
    eventos = eventos_por_frec[FREC_PRINCIPAL]
    idx_e = [fechas.get_loc(e["fecha"]) for e in eventos]
    univ_e = [[col_idx[c] for c in e["score"].index] if e["top"] else [] for e in eventos]
    ini = nav_bt.index[0]

    # ---- 3. Benchmarks ----
    spy_nav = CAPITAL * spy.loc[ini:] / spy.loc[ini]
    idx_s = [fechas.get_loc(s["fecha"]) for s in senales]
    ew5 = CAPITAL * backtest_pesos(idx_s, [np.full(TOP_N, 1 / TOP_N)] * len(senales),
                                   [[col_idx[c] for c in s["top"]] for s in senales])
    univ_cols = [[col_idx[c] for c in s["score"].index] for s in senales]
    ew_univ = CAPITAL * backtest_pesos(idx_s, [np.full(len(c), 1 / len(c)) for c in univ_cols], univ_cols)
    bench = {"Estrategia (Top 5 Max Sharpe)": nav_bt, "Top 5 Equiponderado": ew5,
             "Universo Equiponderado": ew_univ, "SPY (S&P 500)": spy_nav}
    tabla = pd.DataFrame({k: metricas(v, spy) for k, v in bench.items()})
    tabla.loc["Beta vs SPY", "SPY (S&P 500)"] = 1.0
    res["metricas"] = tabla.to_dict()
    res["frecuencias"] = pd.DataFrame(resumen_frec).to_dict()
    tabla.to_csv(OUT / "metricas.csv")
    pd.DataFrame(resumen_frec).to_csv(OUT / "metricas_por_frecuencia.csv")

    # ---- 4. Precisión en el tiempo: predicho vs real, IC del Score ----
    filas = []
    for k, s in enumerate(senales):
        a = fechas.get_loc(s["fecha"]) + 1
        b = fechas.get_loc(senales[k + 1]["fecha"]) + 1 if k + 1 < len(senales) else len(fechas) - 1
        if b - a < 10:
            continue
        tramo = nav_bt.loc[fechas[a]:fechas[b]]
        dias = len(tramo) - 1
        r_real = tramo.iloc[-1] / tramo.iloc[0] - 1
        r_spy = spy.iloc[b] / spy.iloc[a] - 1
        vol_real = tramo.pct_change().std() * np.sqrt(252)
        fwd = close.iloc[b] / close.iloc[a] - 1
        sc = s["score"]
        f = fwd.reindex(sc.index).dropna()
        ic = spearmanr(sc[f.index], f).statistic
        dec = pd.qcut(sc[f.index], 10, labels=False)
        filas.append({
            "fecha": s["fecha"].date().isoformat(), "top": ",".join(t.upper() for t in s["top"]),
            "pesos": ",".join(f"{w:.0%}" for w in s["pesos"]), "universo": s["n_univ"],
            "ret_esperado_anual": s["exp_ret"], "ret_real_anual": (1 + r_real) ** (252 / dias) - 1,
            "vol_esperada": s["exp_vol"], "vol_real": vol_real,
            "ret_periodo": r_real, "ret_spy_periodo": r_spy, "vence_spy": r_real > r_spy,
            "positivo": r_real > 0, "ic_score": ic,
            "top_decil": f[dec == 9].mean(), "fondo_decil": f[dec == 0].mean(), "mediana_univ": f.median(),
        })
    prec = pd.DataFrame(filas)
    prec.to_csv(OUT / "precision_por_periodo.csv", index=False)
    err_ret = prec["ret_real_anual"] - prec["ret_esperado_anual"]
    res["precision"] = {
        "periodos": len(prec),
        "ret_esperado_promedio": prec["ret_esperado_anual"].mean(),
        "ret_real_promedio": prec["ret_real_anual"].mean(),
        "ret_real_mediana": prec["ret_real_anual"].median(),
        "sesgo_ret (real - esperado)": err_ret.mean(),
        "corr_esperado_real": prec[["ret_esperado_anual", "ret_real_anual"]].corr().iloc[0, 1],
        "vol_esperada_promedio": prec["vol_esperada"].mean(),
        "vol_real_promedio": prec["vol_real"].mean(),
        "corr_vol_esperada_real": prec[["vol_esperada", "vol_real"]].corr().iloc[0, 1],
        "%_periodos_positivos": prec["positivo"].mean(),
        "%_periodos_vence_spy": prec["vence_spy"].mean(),
        "IC_promedio_score": prec["ic_score"].mean(),
        "IC_t_stat": prec["ic_score"].mean() / prec["ic_score"].std() * np.sqrt(len(prec)),
        "%_periodos_IC_positivo": (prec["ic_score"] > 0).mean(),
        "spread_decil_promedio (top-fondo, por periodo)": (prec["top_decil"] - prec["fondo_decil"]).mean(),
        "%_periodos_ret_real_dentro_de_esperado±1vol": (err_ret.abs() <= prec["vol_esperada"]).mean(),
    }

    # Retorno rolling 1 año y 3 años vs SPY
    roll = pd.DataFrame({"Estrategia": nav_bt, "SPY": spy_nav})
    roll1 = roll.pct_change(252).dropna()
    roll3 = (roll / roll.shift(756)) ** (1 / 3) - 1
    res["rolling"] = {"%_ventanas_1a_vence_spy": (roll1["Estrategia"] > roll1["SPY"]).mean(),
                      "%_ventanas_3a_vence_spy": (roll3.dropna()["Estrategia"] > roll3.dropna()["SPY"]).mean(),
                      "%_ventanas_1a_negativas": (roll1["Estrategia"] < 0).mean()}

    # ---- 5. Monte Carlo A: bootstrap por bloques del retorno diario (futuros 20 años) ----
    def mc_bootstrap(serie, n_paths, horizonte=252 * ANIOS, bloque=21):
        r = serie.pct_change().dropna().values
        n = len(r)
        rng_mc = np.random.default_rng(SEED + 1)   # misma semilla => mismos bloques para cada serie (bootstrap pareado)
        finales, mdds, cagrs = np.empty(n_paths), np.empty(n_paths), np.empty(n_paths)
        percentiles_camino = []
        for p in range(n_paths):
            ini_b = rng_mc.integers(0, n - bloque, size=horizonte // bloque + 1)
            camino = np.concatenate([r[j:j + bloque] for j in ini_b])[:horizonte]
            v = CAPITAL * np.cumprod(1 + camino)
            finales[p] = v[-1]
            mdds[p] = (v / np.maximum.accumulate(v) - 1).min()
            cagrs[p] = (v[-1] / CAPITAL) ** (1 / ANIOS) - 1
            if p < 2000:
                percentiles_camino.append(v[::21])
        bandas = np.percentile(np.array(percentiles_camino), [5, 25, 50, 75, 95], axis=0)
        return finales, mdds, cagrs, bandas

    mc = {}
    bandas = {}
    for nombre, serie in {"Estrategia": nav_bt, "SPY": spy_nav}.items():
        fin_, mdd_, cagr_, band = mc_bootstrap(serie, N_MC_BOOT)
        bandas[nombre] = band
        mc[nombre] = {
            "valor_final_p5": np.percentile(fin_, 5), "valor_final_p25": np.percentile(fin_, 25),
            "valor_final_mediana": np.median(fin_), "valor_final_p75": np.percentile(fin_, 75),
            "valor_final_p95": np.percentile(fin_, 95),
            "cagr_p5": np.percentile(cagr_, 5), "cagr_mediana": np.median(cagr_), "cagr_p95": np.percentile(cagr_, 95),
            "mdd_mediana": np.median(mdd_), "mdd_p5 (peor 5%)": np.percentile(mdd_, 5),
            "prob_perder_capital": (fin_ < CAPITAL).mean(),
            "prob_duplicar": (fin_ > 2 * CAPITAL).mean(),
            "prob_mdd_peor_50%": (mdd_ < -0.5).mean(),
        }
        if nombre == "Estrategia":
            mc_fin_estr, mc_cagr_estr, mc_mdd_estr = fin_, cagr_, mdd_
        else:
            mc_fin_spy = fin_
    mc["prob_estrategia_supera_spy (bootstrap pareado)"] = (mc_fin_estr > mc_fin_spy).mean()
    res["optimizaciones_sin_converger"] = len(FALLOS_OPT)
    res["variante"] = {"tope": TOPE, "rf_dinamica": RF_DIN, "filtro_sma200": FILTRO, "universo_historico": UNIV_HIST,
                       "eventos": len(eventos), "pasos_a_efectivo": sum(1 for e in eventos if not e["top"])}
    res["variante"].update({"termometro": TERMO, "categoria": CATEG, "rebalanceo": REBAL or "tiempo", "max_dias": MAXDIAS})
    if REBAL == "objetivo":
        res["variante"]["motivos"] = pd.Series(motivos_obj).value_counts().to_dict()
        dias = np.diff([fechas.get_loc(e["fecha"]) for e in eventos])
        res["variante"]["sesiones_entre_eventos_mediana"] = float(np.median(dias))
        res["variante"]["sesiones_entre_eventos_promedio"] = float(np.mean(dias))
    if FILTRO or TERMO or REBAL or EXPO_STATS:
        expo = pd.Series(1.0, index=nav_bt.index)
        for k, e in enumerate(eventos):
            a_, b_ = fechas[idx_e[k] + 1], (fechas[idx_e[k + 1] + 1] if k + 1 < len(eventos) else fechas[-1])
            expo.loc[a_:b_] = float(e["pesos"].sum()) if e["top"] else 0.0
        res["variante"]["%_tiempo_en_efectivo"] = float((expo == 0).mean())
        res["variante"]["exposicion_promedio"] = float(expo.mean())
    if UNIV_HIST:
        cob = [s["n_miembros_con_datos"] / s["n_miembros"] for s in senales]
        res["variante"]["cobertura_universo_hist_promedio"] = float(np.mean(cob))
        res["variante"]["cobertura_universo_hist_min"] = float(np.min(cob))
        res["variante"]["cobertura_por_anio"] = {str(s["fecha"].year): s["n_miembros_con_datos"] / s["n_miembros"]
                                                 for s in senales if s["fecha"].month == 12}
    res["montecarlo_bootstrap"] = mc

    # ---- 6. Monte Carlo B: ¿habilidad o suerte? Portafolios aleatorios con las mismas reglas ----
    estr_vec = backtest_pesos(idx_e, [e["pesos"] for e in eventos], [[col_idx[c] for c in e["top"]] for e in eventos])
    cagr_estr_vec = (estr_vec.iloc[-1]) ** (365.25 / (estr_vec.index[-1] - estr_vec.index[0]).days) - 1
    cagr_rand, sharpe_rand = np.empty(N_MC_RANDOM), np.empty(N_MC_RANDOM)
    for k in range(N_MC_RANDOM):
        cols_k, w_k = [], []
        for c in univ_e:                        # mismas fechas, mismo universo y mismo filtro de mercado
            if c:
                cols_k.append(list(rng.choice(c, TOP_N, replace=False)))
                w_k.append(rng.dirichlet(np.ones(TOP_N)) * eventos[len(w_k)]["pesos"].sum())
            else:
                cols_k.append([])
                w_k.append(np.array([]))
        nv = backtest_pesos(idx_e, w_k, cols_k)
        rr = nv.pct_change().dropna()
        cagr_rand[k] = nv.iloc[-1] ** (365.25 / (nv.index[-1] - nv.index[0]).days) - 1
        sharpe_rand[k] = (rr - rf_diario.reindex(rr.index)).mean() / rr.std() * np.sqrt(252)
    rr = estr_vec.pct_change().dropna()
    sharpe_estr_vec = (rr - rf_diario.reindex(rr.index)).mean() / rr.std() * np.sqrt(252)
    res["montecarlo_aleatorio"] = {
        "simulaciones": N_MC_RANDOM,
        "cagr_estrategia": cagr_estr_vec, "sharpe_estrategia": sharpe_estr_vec,
        "cagr_aleatorio_mediana": np.median(cagr_rand), "cagr_aleatorio_p5": np.percentile(cagr_rand, 5),
        "cagr_aleatorio_p95": np.percentile(cagr_rand, 95),
        "sharpe_aleatorio_mediana": np.median(sharpe_rand),
        "percentil_cagr_estrategia": (cagr_rand < cagr_estr_vec).mean(),
        "percentil_sharpe_estrategia": (sharpe_rand < sharpe_estr_vec).mean(),
        "p_valor_cagr": (cagr_rand >= cagr_estr_vec).mean(),
        "p_valor_sharpe": (sharpe_rand >= sharpe_estr_vec).mean(),
    }

    # ---- 7. Portafolio ACTUAL del notebook aplicado hacia atrás (con sesgo de anticipación) ----
    actual = senal(fechas[-1])
    disp = close[actual["top"]].dropna()
    ini_act = max(disp.index[0], INICIO_BT)
    i_act = fechas.get_loc(disp.loc[ini_act:].index[0]) - 1
    act_nav = CAPITAL * backtest_pesos([i_act], [actual["pesos"]], [[col_idx[c] for c in actual["top"]]], costo=COSTO)
    res["portafolio_actual"] = {
        "top": [t.upper() for t in actual["top"]], "pesos": [float(w) for w in actual["pesos"]],
        "ret_esperado": actual["exp_ret"], "vol_esperada": actual["exp_vol"], "sharpe_esperado": actual["exp_sharpe"],
        "inicio_historia": str(act_nav.index[0].date()),
        "metricas_compra_y_mantener": metricas(act_nav, spy),
    }

    # Composición: frecuencia con que aparece cada acción
    conteo = pd.Series([t.upper() for s in senales for t in s["top"]]).value_counts()
    res["acciones_mas_frecuentes"] = conteo.head(15).to_dict()
    def ret_anual(v):
        """Retorno de cada año calendario; el primero se mide desde el capital inicial."""
        fa = v.resample("YE").last()
        base = pd.Series([v.iloc[0]], index=[v.index[0] - pd.Timedelta(days=1)])
        return pd.concat([base, fa]).pct_change().iloc[1:]
    anuales = pd.DataFrame({k: ret_anual(v) for k, v in {"Estrategia": nav_bt, "SPY": spy_nav}.items()})
    anuales.index = anuales.index.year
    res["anuales"] = anuales.to_dict()
    res["log_operaciones"] = [{**{k: (str(v.date()) if hasattr(v, "date") else v) for k, v in l.items() if k != "acciones"},
                               "acciones": {k.upper(): int(v) for k, v in l["acciones"].items()}} for l in logs[FREC_PRINCIPAL]]

    # ---- 8. Series para gráficos/reporte ----
    pd.DataFrame(bench).resample("W").last().to_csv(OUT / "curvas_semanales.csv")
    pd.DataFrame(bench).round(2).to_csv(OUT / "curvas_diarias.csv")
    dd = pd.DataFrame({k: v / v.cummax() - 1 for k, v in {"Estrategia": nav_bt, "SPY": spy_nav}.items()})
    dd.resample("W").min().to_csv(OUT / "drawdowns_semanales.csv")
    roll1.resample("W").last().to_csv(OUT / "rolling_1a.csv")
    np.save(OUT / "mc_bandas_estrategia.npy", bandas["Estrategia"])
    np.save(OUT / "mc_bandas_spy.npy", bandas["SPY"])
    np.save(OUT / "mc_cagr_aleatorio.npy", cagr_rand)
    np.save(OUT / "mc_finales_estrategia.npy", mc_fin_estr)
    np.save(OUT / "mc_finales_spy.npy", mc_fin_spy)

    # ---- 9. Gráficos PNG ----
    fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for k, v in bench.items():
        ax[0].plot(v.index, v, label=k, lw=1.6 if "Estrategia" in k else 1)
    ax[0].set_yscale("log"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title(f"$100K – Backtest walk-forward {FREC_PRINCIPAL.lower()} ({INICIO_BT.year}–{FIN.year})")
    ax[1].fill_between(dd.index, dd["Estrategia"], 0, alpha=.5, label="Estrategia")
    ax[1].plot(dd.index, dd["SPY"], color="k", lw=.8, label="SPY"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[1].set_ylabel("Drawdown")
    plt.tight_layout(); plt.savefig(OUT / "curva_capital.png", dpi=120); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].scatter(prec["ret_esperado_anual"], prec["ret_real_anual"], alpha=.6)
    lim = [min(prec["ret_esperado_anual"].min(), prec["ret_real_anual"].min()), prec[["ret_esperado_anual", "ret_real_anual"]].max().max()]
    ax[0].plot(lim, lim, "r--", label="Predicción perfecta"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_xlabel("Retorno esperado (modelo)"); ax[0].set_ylabel("Retorno real siguiente periodo"); ax[0].set_title("Precisión del retorno esperado")
    ax[1].bar(pd.to_datetime(prec["fecha"]), prec["ic_score"], width=60)
    ax[1].axhline(prec["ic_score"].mean(), color="r", ls="--", label=f"IC medio {prec['ic_score'].mean():.3f}")
    ax[1].set_title("IC del Score General (Spearman vs retorno futuro)"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "precision.png", dpi=120); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    meses = np.arange(bandas["Estrategia"].shape[1]) / 12
    for nombre, col in [("Estrategia", "C0"), ("SPY", "C3")]:
        b = bandas[nombre]
        ax[0].fill_between(meses, b[0], b[4], color=col, alpha=.12)
        ax[0].fill_between(meses, b[1], b[3], color=col, alpha=.25)
        ax[0].plot(meses, b[2], color=col, label=f"{nombre} (mediana, 50% y 90%)")
    ax[0].set_yscale("log"); ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_xlabel("Años")
    ax[0].set_title(f"Monte Carlo bootstrap – {N_MC_BOOT} caminos de {ANIOS} años")
    ax[1].hist(cagr_rand, bins=40, alpha=.7, label="Portafolios aleatorios")
    ax[1].axvline(cagr_estr_vec, color="r", lw=2, label=f"Estrategia ({cagr_estr_vec:.1%})")
    ax[1].legend(); ax[1].grid(alpha=.3); ax[1].set_title(f"Monte Carlo: estrategia vs {N_MC_RANDOM} selecciones aleatorias")
    plt.tight_layout(); plt.savefig(OUT / "montecarlo.png", dpi=120); plt.close()

    def limpiar(o):
        if isinstance(o, dict):
            return {str(k): limpiar(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [limpiar(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o
    (OUT / "resultados.json").write_text(json.dumps(limpiar(res), indent=2, ensure_ascii=False), encoding="utf-8")

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(tabla.to_string(float_format=lambda x: f"{x:,.4f}"))
    print("\nPor frecuencia:\n", pd.DataFrame(resumen_frec).loc[["CAGR", "Sharpe", "Max Drawdown", "Alfa anual (CAPM)", "Costos totales", "Rotación anual promedio"]].to_string(float_format=lambda x: f"{x:,.4f}"))
    for k in ["precision", "rolling", "montecarlo_bootstrap", "montecarlo_aleatorio", "portafolio_actual"]:
        print(f"\n== {k} ==")
        print(json.dumps(limpiar(res[k]), indent=1, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
