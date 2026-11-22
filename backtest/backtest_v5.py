"""Backtest V5: Score sin redundancias + indicadores técnicos complementarios.

Reutiliza el motor de backtest_v4.py (datos, simulación con acciones enteras, métricas, Monte Carlo) y
cambia solo la selección y el calendario de operaciones:

  1. Score sin redundancias: 0.35 Sharpe + 0.25 Control de drawdown. Se eliminan Rendimiento
     (correlación 0.91 con Sharpe) y Baja volatilidad (0.80 con Control DD). Los pesos NO se redistribuyen.
  2. Filtro de sobrecompra (oscilador): no se compra una acción con RSI(14) de Wilder > 70 en la señal.
  3. Gatillo de momentum (MACD 12/26/9): una acción elegida solo se compra si su MACD está sobre la señal;
     si no, su peso espera en efectivo y se compra el día que el MACD cruza hacia arriba.
  4. Salida por momentum acabado (tendencia): si la acción cierra por debajo de su SMA 50 se vende y su
     peso pasa a efectivo hasta que el gatillo se active otra vez o llegue el siguiente rebalanceo.

Todos los indicadores son causales (solo usan precios hasta cada fecha). Las decisiones del día t se
ejecutan al cierre de t+1, como en la v4. Los pesos de Máximo Sharpe siguen entre 0 y 100 %.

Uso (se combinan con las variables de la v4, p. ej. las 3 mejoras):
  python backtest/backtest_v5.py                                    # Original + V5
  BT_UNIVERSO=historico BT_FILTRO=sma200 BT_RF=dinamica python backtest/backtest_v5.py   # Las 3 mejoras + V5
  BT5_PIEZAS=score,sobrecompra,gatillo,salida (por defecto todas) para probar las piezas por separado.
Salida: backtest/resultados_v5[_piezas][_variante]/
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("backtest_v4", BASE / "backtest_v4.py")
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)

TODAS = ["score", "sobrecompra", "gatillo", "salida"]
PIEZAS = [p for p in os.environ.get("BT5_PIEZAS", ",".join(TODAS)).split(",") if p]
assert all(p in TODAS for p in PIEZAS), f"Piezas válidas: {TODAS}"
RSI_SOBRECOMPRA = 70

# ---------------- Indicadores técnicos (causales) sobre todas las acciones ----------------
close = bt.close
_delta = close.diff()
_gan = _delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
_per = (-_delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
RSI = (100 - 100 / (1 + _gan / _per)).values                         # oscilador: sobrecompra
_macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
MACD_HIST = (_macd - _macd.ewm(span=9, adjust=False).mean()).values    # > 0: MACD sobre su señal
SMA50 = close.rolling(50).mean().values                                 # tendencia de la acción
PX = close.values
COL = {c: j for j, c in enumerate(close.columns)}


def senal_v5(t):
    """Igual que la señal v4 (universo, filtros, Max Sharpe) con el Score sin redundancias y el filtro RSI."""
    fechas = bt.fechas
    i = fechas.get_loc(t)
    if i + 1 < bt.VENTANA:
        return None
    vent_p = close.iloc[i + 1 - bt.VENTANA:i + 1]
    vent_v = bt.volume.iloc[i + 1 - bt.VENTANA:i + 1]
    rf_t = float(bt.irx.iloc[i]) if bt.RF_DIN else bt.RF_MODELO
    cob = vent_p.notna().mean()
    filtro = (cob >= bt.COBERTURA) & (vent_v.mean() > bt.VOL_MIN) & vent_p.iloc[-1].notna()
    n_miembros = n_miembros_con_datos = None
    if bt.UNIV_HIST:
        m = bt.miembros(t)
        en_indice = cob.index.isin(list(m))
        n_miembros = len(m)
        n_miembros_con_datos = int((en_indice & vent_p.iloc[-1].notna().values).sum())
        filtro &= en_indice
    elegibles = cob.index[filtro]
    p = vent_p[elegibles].dropna()
    r = p.pct_change().dropna()
    mu = r.mean() * 252
    vol = r.std() * np.sqrt(252)
    sharpe = (mu - rf_t) / vol
    mdd = (p / p.cummax() - 1).min()
    df = pd.DataFrame({"Rendimiento": mu, "Sharpe": sharpe, "Control_DD": 1 + mdd, "Baja_Vol": 1 / vol})
    nrm = (df - df.min()) / (df.max() - df.min())
    if "score" in PIEZAS:
        df["Score"] = 0.35 * nrm["Sharpe"] + 0.25 * nrm["Control_DD"]          # sin redistribuir pesos
    else:
        df["Score"] = (0.35 * nrm["Sharpe"] + 0.25 * nrm["Rendimiento"] + 0.25 * nrm["Control_DD"]
                       + 0.15 * nrm["Baja_Vol"])
    candidatos = df.index
    if "sobrecompra" in PIEZAS:
        rsi_t = pd.Series(RSI[i, [COL[c] for c in df.index]], index=df.index)
        candidatos = df.index[(rsi_t <= RSI_SOBRECOMPRA).values]
    top = df.loc[candidatos, "Score"].nlargest(bt.TOP_N).index.tolist()
    mu5, cov5 = mu[top].values, r[top].cov().values * 252
    w = bt.max_sharpe(mu5, cov5, rf_t)
    exp_ret, exp_vol = w @ mu5, np.sqrt(w @ cov5 @ w)
    return {"fecha": t, "top": top, "pesos": w, "exp_ret": exp_ret, "exp_vol": exp_vol,
            "exp_sharpe": (exp_ret - rf_t) / exp_vol, "score": df["Score"], "n_univ": len(elegibles),
            "mu": mu5, "n_miembros": n_miembros, "n_miembros_con_datos": n_miembros_con_datos}


def construir_eventos_v5(senales):
    """Calendario diario: señal periódica + filtro SMA200 mensual (si aplica) + gatillo MACD y salida SMA50."""
    if "gatillo" not in PIEZAS and "salida" not in PIEZAS:
        return _eventos_v4(senales)
    fechas = bt.fechas
    por_fecha = {s["fecha"]: s for s in senales}
    s_f = pd.Series(fechas, index=fechas)
    fin_mes = set(pd.Timestamp(x) for x in s_f.resample("ME").last().dropna().values)
    i0, i_fin = fechas.get_loc(senales[0]["fecha"]), fechas.get_loc(bt.FIN)
    eventos, actual, activos, invertido = [], None, {}, False

    def alcista(i):
        return (not bt.FILTRO) or bool(bt.spy.iloc[i] > bt.spy_sma200.iloc[i])

    def emitir(i):
        if not invertido or not any(activos.values()):
            eventos.append({"fecha": fechas[i], "top": [], "pesos": np.array([]), "efectivo": True})
            return
        mascara = np.array([1.0 if activos.get(t) else 0.0 for t in actual["top"]])
        eventos.append({**actual, "fecha": fechas[i], "pesos": actual["pesos"] * mascara})

    def entrada_ok(t, i):
        return ("gatillo" not in PIEZAS) or MACD_HIST[i, COL[t]] > 0

    for i in range(i0, i_fin):
        d = fechas[i]
        cambio = False
        if d in por_fecha:                                   # rebalanceo periódico: análisis nuevo
            actual = por_fecha[d]
            invertido = alcista(i)
            activos = {t: (w > 1e-3 and entrada_ok(t, i)) for t, w in zip(actual["top"], actual["pesos"])}
            emitir(i)
            continue
        if bt.FILTRO and d in fin_mes and alcista(i) != invertido:   # filtro SMA200 de Las 3 mejoras
            invertido = alcista(i)
            if invertido:
                activos = {t: (w > 1e-3 and entrada_ok(t, i)) for t, w in zip(actual["top"], actual["pesos"])}
            emitir(i)
            continue
        if not invertido:
            continue
        for t, w in zip(actual["top"], actual["pesos"]):
            if w <= 1e-3:
                continue
            j = COL[t]
            if activos[t] and "salida" in PIEZAS and PX[i, j] < SMA50[i, j]:
                activos[t] = False; cambio = True                    # momentum acabado: vender
            elif (not activos[t] and "gatillo" in PIEZAS and MACD_HIST[i - 1, j] <= 0 < MACD_HIST[i, j]
                  and not ("salida" in PIEZAS and PX[i, j] < SMA50[i, j])):
                activos[t] = True; cambio = True                     # gatillo: cruce alcista del MACD
        if cambio:
            emitir(i)
    return eventos


# ---------------- Conectar con el motor v4 y correr ----------------
_eventos_v4 = bt.construir_eventos          # calendario original (señales + filtro SMA200 mensual)
bt.senal = senal_v5
bt.construir_eventos = construir_eventos_v5
bt.EXPO_STATS = True
sufijo_piezas = "" if PIEZAS == TODAS else "_" + "-".join(PIEZAS)
bt.OUT = BASE / f"resultados_v5{sufijo_piezas}{bt._sufijo}"
bt.OUT.mkdir(exist_ok=True)

if __name__ == "__main__":
    print(f"V5 piezas: {PIEZAS} -> {bt.OUT.name}")
    bt.main()
