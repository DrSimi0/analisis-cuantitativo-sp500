"""Exporta a CSV, para el reporte web, las dos estrategias vigentes: Original (V4) y Las 3 mejoras.

Uso:  python backtest/exportar_reporte.py   (después de correr las variantes de backtest_v4.py)
Salida: backtest/reporte/*.csv  (incluye operaciones.csv: todas las compras y ventas de los 20 años)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
OUT = BASE / "reporte"
OUT.mkdir(exist_ok=True)
VARIANTES = {
    "Original": "resultados",
    "Las 3 mejoras": "resultados_univhist_sma200_rfdin",
}
R = {v: json.loads((BASE / d / "resultados.json").read_text(encoding="utf-8")) for v, d in VARIANTES.items()}
datos = pd.read_pickle(BASE / "data" / "precios_hist.pkl")
close = datos["close"]
spy = close["spy"]
irx = datos["irx"].reindex(close.index).ffill().bfill()
rf_d = (1 + irx) ** (1 / 252) - 1

# ---------- SPY con el mismo filtro SMA200 (control) ----------
sma = spy.rolling(200).mean()
f = close.index
inicio = pd.Timestamp(R["Original"]["log_operaciones"][0]["senal"])   # misma primera fecha que la estrategia
fin = f[-1]
me = [pd.Timestamp(x) for x in pd.Series(f, index=f).resample("ME").last().dropna().values]
me = [d for d in me if inicio <= d < fin]
pos = pd.Series(np.nan, index=f)
for d in me:
    pos.iloc[f.get_loc(d) + 1] = float(spy.loc[d] > sma.loc[d])
pos = pos.ffill()
s0 = f.get_loc(me[0]) + 1
r = np.where(pos.shift(1).iloc[s0:] == 1, spy.pct_change().iloc[s0:], rf_d.iloc[s0:])
r = r - pos.iloc[s0:].diff().abs().fillna(0).values * 0.001
spy_filtro = pd.Series(100_000 * np.cumprod(1 + r), index=f[s0:])
import importlib.util
_spec = importlib.util.spec_from_file_location("bt", BASE / "backtest_v4.py")
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)
m_spyf = bt.metricas(spy_filtro, spy)

# ---------- 1. Métricas ----------
CAMPOS = {"Valor final": "valor_final", "CAGR": "cagr", "Volatilidad anual": "volatilidad", "Sharpe": "sharpe",
          "Sortino": "sortino", "Max Drawdown": "max_dd", "Calmar": "calmar", "Peor año": "peor_anio",
          "Mejor año": "mejor_anio", "Beta vs SPY": "beta", "Alfa anual (CAPM)": "alfa",
          "Information ratio": "info_ratio", "Duración máx. DD (días)": "dias_dd", "% meses positivos": "meses_pos"}
filas = []
for v, res in R.items():
    for serie, m in res["metricas"].items():
        if serie == "Top 5 Equiponderado" or (serie == "SPY (S&P 500)" and v != "Original"):
            continue
        nombre = {"Estrategia (Top 5 Max Sharpe)": "Estrategia"}.get(serie, serie)
        uni = "" if nombre != "Universo Equiponderado" else (" (histórico)" if v == "Las 3 mejoras" else " (sobrevivientes)")
        fila = {"id": f"{v}|{nombre}", "variante": v if nombre == "Estrategia" else "Referencia",
                "serie": nombre + uni}
        fila.update({c: m.get(k) for k, c in CAMPOS.items()})
        fila["sin_ventaja_pvalor"] = res["montecarlo_aleatorio"]["p_valor_sharpe"] if nombre == "Estrategia" else None
        filas.append(fila)
fila = {"id": "Referencia|SPY con filtro SMA200", "variante": "Referencia", "serie": "SPY con filtro SMA200"}
fila.update({c: m_spyf.get(k) for k, c in CAMPOS.items()})
filas.append(fila)
metricas = pd.DataFrame(filas)
metricas.loc[metricas["serie"] == "SPY (S&P 500)", ["beta", "alfa"]] = [1.0, 0.0]
metricas.to_csv(OUT / "metricas.csv", index=False, float_format="%.6g")

# ---------- 2. Curvas y drawdowns semanales ----------
curvas = {}
for v, d in VARIANTES.items():
    c = pd.read_csv(BASE / d / "curvas_semanales.csv", index_col=0, parse_dates=True)
    curvas[v] = c["Estrategia (Top 5 Max Sharpe)"]
c0 = pd.read_csv(BASE / "resultados" / "curvas_semanales.csv", index_col=0, parse_dates=True)
ch = pd.read_csv(BASE / "resultados_univhist_sma200_rfdin" / "curvas_semanales.csv", index_col=0, parse_dates=True)
curvas["SPY"] = c0["SPY (S&P 500)"]
curvas["SPY con filtro SMA200"] = spy_filtro.resample("W").last()
curvas["Universo equiponderado (sobrevivientes)"] = c0["Universo Equiponderado"]
curvas["Universo equiponderado (histórico)"] = ch["Universo Equiponderado"]
cur = pd.DataFrame(curvas).dropna(how="all")
cur.index.name = "fecha"
cur.round(0).to_csv(OUT / "curvas.csv", date_format="%Y-%m-%d")
dd = (cur / cur.cummax() - 1)
dd.round(4).to_csv(OUT / "drawdowns.csv", date_format="%Y-%m-%d")

# ---------- 3. Retornos anuales ----------
diarias = {v: pd.read_csv(BASE / d / "curvas_diarias.csv", index_col=0, parse_dates=True)["Estrategia (Top 5 Max Sharpe)"]
           for v, d in VARIANTES.items()}
diarias["SPY"] = pd.read_csv(BASE / "resultados" / "curvas_diarias.csv", index_col=0, parse_dates=True)["SPY (S&P 500)"]
diarias["SPY con filtro SMA200"] = spy_filtro


def ret_anual(v):
    """Retorno de cada año calendario; el primero se mide desde el capital inicial ($100K)."""
    v = v.dropna()
    base = pd.Series([v.iloc[0]], index=[v.index[0] - pd.Timedelta(days=1)])
    return pd.concat([base, v.resample("YE").last()]).pct_change().iloc[1:]


an = pd.DataFrame({k: ret_anual(v) for k, v in diarias.items()})
an.index = an.index.year
an.index.name = "anio"
an.round(4).to_csv(OUT / "anuales.csv")

# ---------- 4. Precisión por periodo ----------
pp = []
for v, d in VARIANTES.items():
    p = pd.read_csv(BASE / d / "precision_por_periodo.csv")
    p.insert(0, "variante", v)
    pp.append(p)
prec = pd.concat(pp)
prec["id"] = prec["variante"] + "|" + prec["fecha"]
prec.to_csv(OUT / "precision.csv", index=False, float_format="%.5g")

resumen_prec = pd.DataFrame([{"variante": v, **{k.replace("%_", "pct_").replace(" ", "_"): x for k, x in res["precision"].items()},
                              **{"rolling_" + k.replace("%_", "pct_"): x for k, x in res["rolling"].items()}}
                             for v, res in R.items()])
resumen_prec.to_csv(OUT / "precision_resumen.csv", index=False, float_format="%.5g")

# ---------- 5. Monte Carlo ----------
mc = []
for v, res in R.items():
    b, a = res["montecarlo_bootstrap"], res["montecarlo_aleatorio"]
    e = b["Estrategia"]
    mc.append({"variante": v, "final_p5": e["valor_final_p5"], "final_p25": e["valor_final_p25"],
               "final_mediana": e["valor_final_mediana"], "final_p75": e["valor_final_p75"], "final_p95": e["valor_final_p95"],
               "cagr_p5": e["cagr_p5"], "cagr_mediana": e["cagr_mediana"], "cagr_p95": e["cagr_p95"],
               "mdd_mediana": e["mdd_mediana"], "prob_mdd_50": e["prob_mdd_peor_50%"], "prob_perder": e["prob_perder_capital"],
               "prob_supera_spy": b["prob_estrategia_supera_spy (bootstrap pareado)"],
               "cagr_estrategia": a["cagr_estrategia"], "cagr_azar_mediana": a["cagr_aleatorio_mediana"],
               "cagr_azar_p5": a["cagr_aleatorio_p5"], "cagr_azar_p95": a["cagr_aleatorio_p95"],
               "percentil_cagr": a["percentil_cagr_estrategia"], "percentil_sharpe": a["percentil_sharpe_estrategia"],
               "pvalor_sharpe": a["p_valor_sharpe"]})
e = R["Original"]["montecarlo_bootstrap"]["SPY"]
mc.append({"variante": "SPY", "final_p5": e["valor_final_p5"], "final_p25": e["valor_final_p25"],
           "final_mediana": e["valor_final_mediana"], "final_p75": e["valor_final_p75"], "final_p95": e["valor_final_p95"],
           "cagr_p5": e["cagr_p5"], "cagr_mediana": e["cagr_mediana"], "cagr_p95": e["cagr_p95"],
           "mdd_mediana": e["mdd_mediana"], "prob_mdd_50": e["prob_mdd_peor_50%"], "prob_perder": e["prob_perder_capital"]})
pd.DataFrame(mc).to_csv(OUT / "montecarlo.csv", index=False, float_format="%.5g")

bandas = []
for v, d in VARIANTES.items():
    b = np.load(BASE / d / "mc_bandas_estrategia.npy")
    for k in range(b.shape[1]):
        bandas.append({"variante": v, "anio": round(k / 12, 3), "p5": b[0, k], "p25": b[1, k], "p50": b[2, k], "p75": b[3, k], "p95": b[4, k]})
b = np.load(BASE / "resultados" / "mc_bandas_spy.npy")
for k in range(b.shape[1]):
    bandas.append({"variante": "SPY", "anio": round(k / 12, 3), "p5": b[0, k], "p25": b[1, k], "p50": b[2, k], "p75": b[3, k], "p95": b[4, k]})
pd.DataFrame(bandas).round(0).to_csv(OUT / "mc_bandas.csv", index=False)

azar = []
for v, d in VARIANTES.items():
    x = np.load(BASE / d / "mc_cagr_aleatorio.npy")
    cnt, edges = np.histogram(x, bins=np.arange(-0.06, 0.32, 0.01))
    for c_, lo in zip(cnt, edges[:-1]):
        azar.append({"variante": v, "cagr_desde": round(lo, 3), "simulaciones": int(c_)})
pd.DataFrame(azar).to_csv(OUT / "mc_azar.csv", index=False)

# ---------- 6. Operaciones: todas las compras y ventas ----------
ops = []
for v, res in R.items():
    previo = {}
    for l in res["log_operaciones"]:
        d = pd.Timestamp(l["ejecucion"])
        nuevo = l["acciones"]
        for t in sorted(set(previo) | set(nuevo)):
            a0, a1 = previo.get(t, 0), nuevo.get(t, 0)
            precio = float(close.loc[:d, t.lower()].ffill().iloc[-1])
            delta = a1 - a0
            tipo = "Compra" if a0 == 0 and a1 > 0 else "Venta total" if a1 == 0 else "Aumenta" if delta > 0 else "Reduce" if delta < 0 else "Mantiene"
            ops.append({"id": f"{v}|{l['senal']}|{t}", "variante": v, "fecha_senal": l["senal"], "fecha_ejecucion": l["ejecucion"],
                        "ticker": t, "operacion": tipo, "acciones_antes": a0, "acciones_despues": a1, "delta": delta,
                        "precio": round(precio, 2), "monto_operado": round(abs(delta) * precio, 2),
                        "valor_posicion": round(a1 * precio, 2), "peso": round(a1 * precio / l["valor"], 4),
                        "valor_portafolio": round(l["valor"], 2)})
        if not nuevo:
            ops.append({"id": f"{v}|{l['senal']}|EFECTIVO", "variante": v, "fecha_senal": l["senal"],
                        "fecha_ejecucion": l["ejecucion"], "ticker": "EFECTIVO", "operacion": "A efectivo (SPY < SMA200)",
                        "acciones_antes": 0, "acciones_despues": 0, "delta": 0, "precio": None, "monto_operado": 0,
                        "valor_posicion": round(l["valor"], 2), "peso": 1.0, "valor_portafolio": round(l["valor"], 2)})
        previo = nuevo
opsdf = pd.DataFrame(ops)
opsdf.to_csv(OUT / "operaciones.csv", index=False)

# Resumen por acción
res_acc = []
for (v, t), g in opsdf[opsdf["ticker"] != "EFECTIVO"].groupby(["variante", "ticker"]):
    tenencia = g[g["acciones_despues"] > 0]
    res_acc.append({"id": f"{v}|{t}", "variante": v, "ticker": t,
                    "rebalanceos_en_cartera": len(tenencia), "veces_comprada": int((g["operacion"] == "Compra").sum()),
                    "primera_compra": tenencia["fecha_ejecucion"].min() if len(tenencia) else None,
                    "ultima_tenencia": tenencia["fecha_ejecucion"].max() if len(tenencia) else None,
                    "peso_promedio": round(tenencia["peso"].mean(), 4) if len(tenencia) else 0,
                    "monto_comprado_total": round(g.loc[g["delta"] > 0, "monto_operado"].sum(), 2),
                    "en_cartera_hoy": bool(t in R[v]["log_operaciones"][-1]["acciones"])})
pd.DataFrame(res_acc).sort_values(["variante", "rebalanceos_en_cartera"], ascending=[True, False]).to_csv(
    OUT / "acciones_resumen.csv", index=False)

# Cobertura del universo histórico
cob = R["Las 3 mejoras"]["variante"]["cobertura_por_anio"]
pd.DataFrame({"anio": list(cob), "cobertura": list(cob.values())}).to_csv(OUT / "cobertura.csv", index=False, float_format="%.4f")

for p in sorted(OUT.glob("*.csv")):
    print(f"{p.name:24s} {p.stat().st_size / 1024:8.1f} KB  {sum(1 for _ in open(p, encoding='utf-8')) - 1} filas")
print("Acciones distintas compradas por variante:")
print(opsdf[opsdf.operacion == "Compra"].groupby("variante")["ticker"].nunique().to_string())
