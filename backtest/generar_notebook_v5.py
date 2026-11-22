"""Genera Stocks_Auto_Analisis_V5_Las3Mejoras.ipynb (Colab) a partir del notebook V4 Final.

Mantiene la estructura del V4 (Pasos 1-7 + resumen) e integra las 3 mejoras validadas en el backtest:
  1. Tasa libre de riesgo dinámica (T-Bill 13 semanas, ^IRX) en lugar de 4 % fija.
  2. Filtro de régimen de mercado: SPY vs SMA 200 (si está debajo, la recomendación es efectivo/T-Bills).
  3. Universo histórico (point-in-time) del S&P 500 en el backtest walk-forward del Paso 8.

Uso:  python backtest/generar_notebook_v5.py
"""
import json
from pathlib import Path

RAIZ = Path(__file__).parent.parent
V4 = RAIZ / "Stocks_Auto_Analisis_V4 Final.ipynb"
SALIDA = RAIZ / "Stocks_Auto_Analisis_V5_Las3Mejoras.ipynb"

v4 = json.loads(V4.read_text(encoding="utf-8"))
fuente = ["".join(c["source"]) for c in v4["cells"]]


def md(texto):
    return {"cell_type": "markdown", "metadata": {}, "source": texto.strip("\n").splitlines(keepends=True)}


def code(texto):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": texto.strip("\n").splitlines(keepends=True)}


def v4code(i, reemplazos=()):
    s = fuente[i]
    for a, b in reemplazos:
        assert a in s, f"No se encontró en la celda {i}: {a[:60]}"
        s = s.replace(a, b)
    return code(s)


celdas = []

# ------------------------------------------------------------------ Portada
celdas.append(md("""
# 📈 Análisis Cuantitativo Automatizado del S&P 500 — V5 "Las 3 mejoras"

Misma estructura que **V4** (Pasos 1–7 + resumen del Top 5) con las tres mejoras que el backtest de 20 años validó:

| # | Mejora | Dónde se aplica | Qué cambia |
|---|---|---|---|
| 1 | **Tasa libre de riesgo dinámica** | Fase 1.3 y todas las fases que usan Sharpe/CAPM | Se usa la tasa actual del T-Bill de 13 semanas (`^IRX`) en lugar de 4 % fija. |
| 2 | **Filtro de mercado SMA 200** | Fase 3.3 y Recomendación final | Si SPY cierra por debajo de su media de 200 días, la recomendación es **efectivo / T-Bills**. |
| 3 | **Universo histórico (point-in-time)** | Paso 8 (backtest) | En cada fecha del backtest solo se eligen acciones que **en ese momento** eran miembros del S&P 500. Hoy el universo histórico es la lista actual. |

**Paso 8** (opcional, ~10–15 min en Colab) corre el backtest walk-forward de 20 años con las tres mejoras, Monte Carlo y la lista de todas las operaciones.

> ⚠️ Proyecto educativo. Los resultados se basan en datos históricos y **no constituyen recomendación de inversión**.

**Colab:** `Entorno de ejecución → Ejecutar todas`. La primera celda instala las dependencias.
"""))
celdas.append(code("""
# Dependencias (Colab ya trae pandas, numpy, scipy, matplotlib, seaborn y statsmodels)
!pip install -q yfinance lxml
"""))

# ------------------------------------------------------------------ Paso 1
celdas.append(md(fuente[1]))
celdas.append(md(fuente[0]))
celdas.append(v4code(2))
celdas.append(md(fuente[3]))
celdas.append(v4code(4))
celdas.append(md("""
### Fase 1.3: Tasa Libre de Riesgo Dinámica (Mejora 1)
En V4 la tasa libre de riesgo era un 4 % fijo. Aquí se descarga el rendimiento del **T-Bill de 13 semanas** (`^IRX`, en % anual) y se usa su último valor en el CAPM, el Sharpe y la optimización. En el backtest (Paso 8) se usa la tasa vigente en **cada** fecha.
"""))
celdas.append(code("""
# MEJORA 1: tasa libre de riesgo del T-Bill de 13 semanas (^IRX viene en porcentaje anual)
datos_irx = yf.download('^IRX', period='5y', progress=False, auto_adjust=True)['Close'].squeeze().dropna()
tasa_libre_riesgo_actual = float(datos_irx.iloc[-1]) / 100

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 3))
plt.plot(datos_irx.index, datos_irx / 100, color='teal')
plt.axhline(0.04, color='gray', ls='--', label='4 % fijo usado en V4')
plt.title('Tasa del T-Bill de 13 semanas (^IRX)')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1%}'))
plt.legend(); plt.grid(True); plt.show()

print(f"Tasa libre de riesgo actual (T-Bill 13 semanas): {tasa_libre_riesgo_actual:.2%}  (V4 usaba 4.00 % fijo)")
"""))

# ------------------------------------------------------------------ Paso 2
celdas.append(md(fuente[5]))
celdas.append(md(fuente[6]))
celdas.append(v4code(7, [("tasa_libre_riesgo_anual = 0.04",
                          "tasa_libre_riesgo_anual = tasa_libre_riesgo_actual  # MEJORA 1: T-Bill actual (V4: 0.04 fijo)")]))

# ------------------------------------------------------------------ Paso 3
celdas.append(md(fuente[8]))
celdas.append(md(fuente[9]))
celdas.append(v4code(10))
celdas.append(md(fuente[11]))
celdas.append(v4code(12))
celdas.append(md("""
### Fase 3.3: Filtro de Régimen de Mercado — SPY vs SMA 200 (Mejora 2)
Regla del backtest: **a fin de cada mes**, si SPY cierra por encima de su media móvil de 200 días se invierte en el portafolio del Top 5; si cierra por debajo, se pasa a **efectivo (T-Bills)** hasta que vuelva arriba.

En el backtest de 20 años este filtro evitó la caída de 2008 (−0.5 % en lugar de −61 %) y redujo la caída máxima de −68 % a −35 %. No protege en caídas rápidas (2022).
"""))
celdas.append(code("""
# MEJORA 2: filtro de régimen de mercado con SPY vs su media móvil de 200 días
spy_close = datos_spy.set_index(pd.to_datetime(datos_spy['Date']))['Close']
spy_sma200 = spy_close.rolling(200).mean()
ultimo_spy, ultima_sma200 = spy_close.iloc[-1], spy_sma200.iloc[-1]
mercado_alcista = bool(ultimo_spy > ultima_sma200)
distancia_sma200 = ultimo_spy / ultima_sma200 - 1

# Último cierre de mes (la regla se revisa a fin de mes)
fin_mes = spy_close.resample('ME').last().index[-2]
fin_mes = spy_close.loc[:fin_mes].index[-1]
alcista_fin_mes = bool(spy_close.loc[fin_mes] > spy_sma200.loc[fin_mes])

plt.figure(figsize=(12, 4))
plt.plot(spy_close.index[-756:], spy_close.tail(756), color='black', label='SPY')
plt.plot(spy_sma200.index[-756:], spy_sma200.tail(756), color='orange', label='SMA 200')
plt.fill_between(spy_close.index[-756:], spy_close.tail(756).min(), spy_close.tail(756).max(),
                 where=(spy_close.tail(756) < spy_sma200.tail(756)), color='red', alpha=0.08, label='SPY < SMA 200')
plt.title('Filtro de mercado: SPY vs SMA 200 (últimos 3 años)')
plt.legend(); plt.grid(True); plt.show()

print(f"SPY: {ultimo_spy:,.2f} | SMA 200: {ultima_sma200:,.2f} | distancia: {distancia_sma200:+.2%}")
print(f"Régimen hoy: {'ALCISTA → invertir en el Top 5' if mercado_alcista else 'BAJISTA → efectivo / T-Bills'}")
print(f"Régimen al último cierre de mes ({fin_mes.date()}): {'alcista' if alcista_fin_mes else 'bajista'} "
      f"(la regla del backtest decide con este valor)")
"""))

# ------------------------------------------------------------------ Pasos 4 a 7 (idénticos, ya usan la tasa dinámica)
for i in range(13, 27):
    celdas.append(md(fuente[i]) if v4["cells"][i]["cell_type"] == "markdown" else v4code(i))
celdas.append(v4code(27))

# ------------------------------------------------------------------ Recomendación final
celdas.append(md("""
### Recomendación Final con Filtro de Mercado y $100,000
Une el portafolio de **Máximo Sharpe del Top 5** con el **filtro SMA 200** (decisión al último cierre de mes, como en el backtest) y calcula cuántas acciones enteras comprar con el capital.
"""))
celdas.append(code("""
CAPITAL = 100_000
COSTO_OPERACION = 0.001   # 10 pb por dólar operado (mismo supuesto del backtest)

print("=" * 70)
print("        RECOMENDACIÓN FINAL — V5 (LAS 3 MEJORAS)")
print("=" * 70)
print(f"Tasa libre de riesgo (T-Bill): {tasa_libre_riesgo_anual:.2%}")
print(f"Filtro de mercado al cierre de mes {fin_mes.date()}: {'ALCISTA' if alcista_fin_mes else 'BAJISTA'} "
      f"| hoy: {'alcista' if mercado_alcista else 'bajista'} (SPY {distancia_sma200:+.2%} vs SMA 200)")

if alcista_fin_mes:
    precios_hoy = df_precios_unidos[top_tickers].iloc[-1]
    acciones_comprar = np.floor(CAPITAL * pesos_top5_ms.values * (1 - 2 * COSTO_OPERACION) / precios_hoy.values)
    inversion = acciones_comprar * precios_hoy.values
    plan = pd.DataFrame({
        'Acción': [t.upper() for t in top_tickers],
        'Peso Max Sharpe': pesos_top5_ms.values,
        'Precio': precios_hoy.values,
        'Acciones a comprar': acciones_comprar.astype(int),
        'Inversión': inversion,
    })
    plan = plan[plan['Acciones a comprar'] > 0]
    efectivo = CAPITAL - inversion.sum() - inversion.sum() * COSTO_OPERACION
    display(plan.style.format({'Peso Max Sharpe': '{:.2%}', 'Precio': '${:,.2f}', 'Inversión': '${:,.2f}'}).hide(axis='index'))
    print(f"Invertido: ${inversion.sum():,.2f} | Costos estimados: ${inversion.sum() * COSTO_OPERACION:,.2f} | "
          f"Efectivo sobrante: ${efectivo:,.2f}")
    print("\\nReglas: rebalancear cada trimestre (último día hábil) y revisar el filtro a fin de cada mes.")
else:
    print("\\nSPY está por debajo de su SMA 200: la estrategia queda 100 % en EFECTIVO / T-Bills")
    print(f"(p. ej. un ETF de T-Bills como SGOV o BIL) rindiendo ~{tasa_libre_riesgo_anual:.2%} anual.")
    print("Volver a comprar el Top 5 cuando SPY cierre un mes por encima de su SMA 200.")
"""))

# ------------------------------------------------------------------ Paso 8: Backtest
celdas.append(md("""
# Paso 8: Backtest Walk-Forward de 20 Años con las 3 Mejoras
Simula haber invertido **$100,000** hace 20 años aplicando **cada trimestre** este mismo análisis, usando **solo los datos disponibles en cada fecha**:

1. **Universo histórico (Mejora 3):** en cada fecha solo se consideran las acciones que eran miembros del S&P 500 en ese momento (composición diaria de [github.com/fja05680/sp500](https://github.com/fja05680/sp500)).
2. Ventana de 5 años, filtro de volumen > 1M y cobertura ≥ 95 %, **Score General** y Top 5 con pesos de **Máximo Sharpe** (igual que los Pasos 1–7).
3. **Tasa libre de riesgo dinámica (Mejora 1):** el T-Bill vigente en cada fecha.
4. **Filtro SMA 200 (Mejora 2):** a fin de cada mes, si SPY < SMA 200 se pasa a efectivo.
5. Ejecución al cierre del día siguiente, acciones enteras, costos de 10 pb y el efectivo rinde el T-Bill.

> ⏱️ Opcional: tarda ~10–15 minutos (descarga ~1,000 tickers con 26 años de historia). Limitación: Yahoo Finance no tiene precios de muchas empresas que quebraron o fueron adquiridas, así que el sesgo de supervivencia se reduce pero no desaparece.
"""))
celdas.append(md("### Fase 8.1: Composición Histórica del S&P 500 y Descarga de Precios"))
celdas.append(code("""
from urllib.parse import quote

ANIOS_BT = 20
VENTANA_BT = 1260          # 5 años de sesiones, igual que el análisis
TOP_N_BT = 5
VOL_MIN_BT = 1_000_000
COBERTURA_BT = 0.95
COSTO_BT = 0.001           # 10 pb por dólar operado
N_MC_BOOT = 2000           # caminos del Monte Carlo bootstrap
N_MC_AZAR = 500            # carteras aleatorias para la prueba de habilidad
SEED_BT = 42

# 1. Composición diaria histórica del S&P 500 (point-in-time)
repo = requests.get('https://api.github.com/repos/fja05680/sp500/contents/', headers=header, timeout=30).json()
nombres = [f['name'] for f in repo if isinstance(f, dict) and 'Historical Components' in f.get('name', '')]
nombre_csv = next((n for n in nombres if 'Updated' in n), nombres[0])
url_csv = 'https://raw.githubusercontent.com/fja05680/sp500/master/' + quote(nombre_csv)
composicion = pd.read_csv(StringIO(requests.get(url_csv, timeout=120).text), parse_dates=['date'])

hoy = pd.Timestamp.today().normalize()
inicio_datos = hoy - pd.DateOffset(years=ANIOS_BT + 6)          # 20 años + 5 de ventana + margen
composicion = composicion[composicion['date'] >= inicio_datos - pd.DateOffset(years=1)]
miembros_hist = {d: set(t.replace('.', '-') for t in s.split(',')) for d, s in zip(composicion['date'], composicion['tickers'])}
todos_hist = sorted(set().union(*miembros_hist.values()) | set(tickers_muestra))
print(f"Composición histórica: {len(miembros_hist)} cambios desde {composicion['date'].min().date()} | "
      f"{len(todos_hist)} tickers que alguna vez fueron miembros")

# 2. Precios ajustados de todos los miembros históricos + SPY, y la tasa del T-Bill
print("Descargando precios históricos (puede tardar varios minutos)...")
bloque_bt = yf.download(todos_hist + ['SPY'], start=inicio_datos.strftime('%Y-%m-%d'), progress=False,
                        auto_adjust=True, threads=True)
close_bt_all = bloque_bt['Close'].sort_index()
volume_bt_all = bloque_bt['Volume'].sort_index()
con_datos = close_bt_all.columns[close_bt_all.notna().any()]
close_bt_all, volume_bt_all = close_bt_all[con_datos], volume_bt_all[con_datos]
irx_bt = yf.download('^IRX', start=inicio_datos.strftime('%Y-%m-%d'), progress=False,
                     auto_adjust=True)['Close'].squeeze() / 100

spy_bt = close_bt_all['SPY']
close_bt = close_bt_all.drop(columns=['SPY'])
volume_bt = volume_bt_all.drop(columns=['SPY'])
fechas_bt = close_bt.index
irx_bt = irx_bt.reindex(fechas_bt).ffill().bfill()
rf_diario_bt = (1 + irx_bt) ** (1 / 252) - 1
sma200_bt = spy_bt.rolling(200).mean()
FIN_BT = fechas_bt[-1]
INICIO_BT = FIN_BT - pd.DateOffset(years=ANIOS_BT)

_fechas_comp = np.array(sorted(miembros_hist), dtype='datetime64[ns]')
_lista_comp = [miembros_hist[k] for k in sorted(miembros_hist)]
def miembros_en(t):
    \"\"\"Tickers que pertenecían al S&P 500 en la fecha t.\"\"\"
    return _lista_comp[np.searchsorted(_fechas_comp, np.datetime64(t), side='right') - 1]

sin_precio = len(todos_hist) - close_bt.shape[1]
print(f"Con precios en Yahoo: {close_bt.shape[1]} de {len(todos_hist)} ({sin_precio} sin datos: quiebras/adquisiciones)")
print(f"Backtest: {INICIO_BT.date()} a {FIN_BT.date()}")
"""))
celdas.append(md("### Fase 8.2: Motor del Backtest (Señal, Filtro y Simulación con Acciones Enteras)"))
celdas.append(code("""
def max_sharpe_bt(mu, cov, rf):
    n = len(mu)
    def neg_sharpe(w):
        return -((w @ mu - rf) / np.sqrt(w @ cov @ w))
    res = sco.minimize(neg_sharpe, np.full(n, 1 / n), method='SLSQP', bounds=[(0, 1)] * n,
                       constraints=({'type': 'eq', 'fun': lambda x: x.sum() - 1},))
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def senal_bt(t):
    \"\"\"Pasos 1-7 con datos hasta t: universo histórico, Score General, Top 5 y pesos Max Sharpe.\"\"\"
    i = fechas_bt.get_loc(t)
    if i + 1 < VENTANA_BT:
        return None
    vent_p = close_bt.iloc[i + 1 - VENTANA_BT:i + 1]
    vent_v = volume_bt.iloc[i + 1 - VENTANA_BT:i + 1]
    rf_t = float(irx_bt.iloc[i])                                        # MEJORA 1
    cob = vent_p.notna().mean()
    en_indice = cob.index.isin(list(miembros_en(t)))                    # MEJORA 3
    filtro = (cob >= COBERTURA_BT) & (vent_v.mean() > VOL_MIN_BT) & vent_p.iloc[-1].notna() & en_indice
    elegibles = cob.index[filtro]
    p = vent_p[elegibles].dropna()
    r = p.pct_change().dropna()
    mu, vol = r.mean() * 252, r.std() * np.sqrt(252)
    df = pd.DataFrame({'Rendimiento': mu, 'Sharpe': (mu - rf_t) / vol,
                       'Control_DD': 1 + (p / p.cummax() - 1).min(), 'Baja_Vol': 1 / vol})
    nrm = (df - df.min()) / (df.max() - df.min())
    df['Score'] = 0.35 * nrm['Sharpe'] + 0.25 * nrm['Rendimiento'] + 0.25 * nrm['Control_DD'] + 0.15 * nrm['Baja_Vol']
    top = df['Score'].nlargest(TOP_N_BT).index.tolist()
    mu5, cov5 = mu[top].values, r[top].cov().values * 252
    w = max_sharpe_bt(mu5, cov5, rf_t)
    return {'fecha': t, 'top': top, 'pesos': w, 'exp_ret': w @ mu5, 'exp_vol': np.sqrt(w @ cov5 @ w),
            'score': df['Score'], 'n_univ': len(elegibles)}


def construir_eventos(senales):
    \"\"\"MEJORA 2: a fin de mes, si SPY < SMA200 se pasa a efectivo; al volver arriba se recompra la última señal.\"\"\"
    por_fecha = {s['fecha']: s for s in senales}
    s_f = pd.Series(fechas_bt, index=fechas_bt)
    fin_mes = set(pd.Timestamp(x) for x in s_f.resample('ME').last().dropna().values)
    calendario = sorted(d for d in fin_mes | set(por_fecha) if senales[0]['fecha'] <= d < FIN_BT)
    eventos, actual, estado = [], None, None
    for d in calendario:
        if d in por_fecha:
            actual = por_fecha[d]
        alcista = bool(spy_bt.loc[d] > sma200_bt.loc[d])
        if d in por_fecha or alcista != estado:
            if alcista:
                eventos.append(actual if d in por_fecha else {**actual, 'fecha': d})
            else:
                eventos.append({'fecha': d, 'top': [], 'pesos': np.array([])})
            estado = alcista
    return eventos


def simular_bt(eventos, capital=CAPITAL, costo=COSTO_BT):
    \"\"\"Ejecuta cada evento al cierre de la sesión siguiente, con acciones enteras.\"\"\"
    px = close_bt.ffill()
    exec_idx = {fechas_bt.get_loc(e['fecha']) + 1: e for e in eventos}
    inicio = min(exec_idx)
    efectivo, acciones = float(capital), pd.Series(dtype=float)
    nav, log = {}, []
    for i in range(inicio, len(fechas_bt)):
        if i > inicio:
            efectivo *= 1 + rf_diario_bt.iloc[i]
        if i in exec_idx:
            e, precios = exec_idx[i], px.iloc[i]
            valor = efectivo + (acciones * precios[acciones.index]).sum()
            objetivo = pd.Series(np.floor(valor * e['pesos'] * (1 - 2 * costo) / precios[e['top']].values), index=e['top'])
            todos = acciones.index.union(objetivo.index)
            delta = objetivo.reindex(todos, fill_value=0) - acciones.reindex(todos, fill_value=0)
            monto = (delta.abs() * precios[todos]).sum()
            efectivo -= (delta * precios[todos]).sum() + monto * costo
            antes = acciones.to_dict()
            acciones = objetivo[objetivo > 0]
            log.append({'senal': e['fecha'], 'ejecucion': fechas_bt[i], 'valor': valor, 'antes': antes,
                        'despues': acciones.to_dict(), 'precios': precios[todos].to_dict()})
        nav[fechas_bt[i]] = efectivo + (acciones * px.iloc[i][acciones.index]).sum()
    return pd.Series(nav), log


def metricas_bt(nav, bench=None):
    r = nav.pct_change().dropna()
    ex = r - rf_diario_bt.reindex(r.index).fillna(0)
    anios = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / anios) - 1
    dd = nav / nav.cummax() - 1
    neg = ex[ex < 0]
    anual = pd.concat([pd.Series([nav.iloc[0]], index=[nav.index[0] - pd.Timedelta(days=1)]),
                       nav.resample('YE').last()]).pct_change().dropna()
    m = {'Valor final': nav.iloc[-1], 'CAGR': cagr, 'Volatilidad': r.std() * np.sqrt(252),
         'Sharpe': ex.mean() / ex.std() * np.sqrt(252),
         'Sortino': ex.mean() * 252 / (np.sqrt((neg ** 2).sum() / len(ex)) * np.sqrt(252)),
         'Max Drawdown': dd.min(), 'Calmar': cagr / abs(dd.min()),
         'VaR 95% diario': np.percentile(r, 5), 'Mejor año': anual.max(), 'Peor año': anual.min()}
    if bench is not None:
        b = bench.reindex(nav.index).pct_change().dropna().reindex(r.index)
        bx = b - rf_diario_bt.reindex(r.index).fillna(0)
        beta = np.cov(ex, bx)[0, 1] / bx.var()
        m.update({'Beta vs SPY': beta, 'Alfa anual': (ex.mean() - beta * bx.mean()) * 252,
                  'Correlación vs SPY': r.corr(b)})
    return m

print("Motor del backtest listo.")
"""))
celdas.append(md("### Fase 8.3: Ejecución y Resultados con $100,000"))
celdas.append(code("""
s_f = pd.Series(fechas_bt, index=fechas_bt)
fechas_reb = [pd.Timestamp(x) for x in s_f.resample('QE').last().dropna().values]
fechas_reb = [d for d in fechas_reb if INICIO_BT - pd.Timedelta(days=1) <= d < FIN_BT]

print(f"Calculando {len(fechas_reb)} señales trimestrales...")
senales_bt = [s for s in (senal_bt(d) for d in fechas_reb) if s is not None]
eventos_bt = construir_eventos(senales_bt)
nav_bt, log_bt = simular_bt(eventos_bt)
inicio_nav = nav_bt.index[0]

# Referencias: SPY y SPY con el mismo filtro SMA 200
spy_nav = CAPITAL * spy_bt.loc[inicio_nav:] / spy_bt.loc[inicio_nav]
posicion = pd.Series(np.nan, index=fechas_bt)
for d in [pd.Timestamp(x) for x in s_f.resample('ME').last().dropna().values]:
    if senales_bt[0]['fecha'] <= d < FIN_BT:
        posicion.iloc[fechas_bt.get_loc(d) + 1] = float(spy_bt.loc[d] > sma200_bt.loc[d])
posicion = posicion.ffill().loc[inicio_nav:]
r_filtro = np.where(posicion.shift(1) == 1, spy_bt.pct_change().loc[inicio_nav:], rf_diario_bt.loc[inicio_nav:])
r_filtro = r_filtro - posicion.diff().abs().fillna(0).values * COSTO_BT
r_filtro[0] = 0
spy_filtro_nav = pd.Series(CAPITAL * np.cumprod(1 + r_filtro), index=posicion.index)

tabla_bt = pd.DataFrame({
    'Estrategia V5 (las 3 mejoras)': metricas_bt(nav_bt, spy_bt),
    'SPY': metricas_bt(spy_nav, spy_bt),
    'SPY con filtro SMA 200': metricas_bt(spy_filtro_nav, spy_bt),
})
formatos = {'Valor final': '${:,.0f}', 'Sharpe': '{:.2f}', 'Sortino': '{:.2f}', 'Calmar': '{:.2f}',
            'Beta vs SPY': '{:.2f}', 'Correlación vs SPY': '{:.2f}'}
display(tabla_bt.apply(lambda col: [formatos.get(k, '{:.2%}').format(v) for k, v in col.items()]))

en_efectivo = sum(1 for e in eventos_bt if not e['top'])
print(f"Rebalanceos trimestrales: {len(senales_bt)} | pasos a efectivo por el filtro: {en_efectivo}")

fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
for nombre, serie, estilo in [('Estrategia V5', nav_bt, '-'), ('SPY', spy_nav, '-'), ('SPY con filtro SMA 200', spy_filtro_nav, '--')]:
    ax[0].plot(serie.index, serie, estilo, label=nombre, lw=2 if 'V5' in nombre else 1)
ax[0].set_yscale('log'); ax[0].legend(); ax[0].grid(True, alpha=.3)
ax[0].set_title(f'$100,000 — Backtest walk-forward V5 ({INICIO_BT.year}–{FIN_BT.year})')
for nombre, serie in [('Estrategia V5', nav_bt), ('SPY', spy_nav)]:
    ax[1].plot(serie.index, serie / serie.cummax() - 1, label=nombre)
ax[1].set_ylabel('Drawdown'); ax[1].legend(); ax[1].grid(True, alpha=.3)
plt.tight_layout(); plt.show()

anuales_bt = pd.DataFrame({k: pd.concat([pd.Series([v.iloc[0]], index=[v.index[0] - pd.Timedelta(days=1)]),
                                          v.resample('YE').last()]).pct_change().dropna()
                           for k, v in {'Estrategia V5': nav_bt, 'SPY': spy_nav, 'SPY con filtro': spy_filtro_nav}.items()})
anuales_bt.index = anuales_bt.index.year
print("Retorno por año:")
display(anuales_bt.style.format('{:.2%}'))
"""))
celdas.append(md("""
### Fase 8.4: ¿Qué Tan Preciso Fue el Modelo con el Tiempo?
- **Retorno prometido vs real:** el retorno esperado del optimizador contra lo que realmente pasó el trimestre siguiente.
- **IC del Score:** correlación de rangos (Spearman) entre el Score General y el retorno futuro de todo el universo. Si es ≈ 0, el Score no predice qué acciones subirán.
"""))
celdas.append(code("""
from scipy.stats import spearmanr

filas_prec = []
for k, s in enumerate(senales_bt):
    a = fechas_bt.get_loc(s['fecha']) + 1
    b = fechas_bt.get_loc(senales_bt[k + 1]['fecha']) + 1 if k + 1 < len(senales_bt) else len(fechas_bt) - 1
    if b - a < 10:
        continue
    tramo = nav_bt.loc[fechas_bt[a]:fechas_bt[b]]
    r_real = tramo.iloc[-1] / tramo.iloc[0] - 1
    fwd = (close_bt.iloc[b] / close_bt.iloc[a] - 1).reindex(s['score'].index).dropna()
    filas_prec.append({'fecha': s['fecha'].date(), 'top': ', '.join(s['top']), 'ret_esperado': s['exp_ret'],
                       'ret_real_anual': (1 + r_real) ** (252 / (len(tramo) - 1)) - 1,
                       'vence_spy': r_real > spy_bt.iloc[b] / spy_bt.iloc[a] - 1,
                       'ic': spearmanr(s['score'][fwd.index], fwd).statistic})
prec_bt = pd.DataFrame(filas_prec)

print(f"Retorno anual prometido (promedio): {prec_bt['ret_esperado'].mean():.2%}")
print(f"Retorno anual real (mediana):       {prec_bt['ret_real_anual'].median():.2%}")
print(f"Correlación prometido vs real:      {prec_bt[['ret_esperado', 'ret_real_anual']].corr().iloc[0, 1]:.3f}")
print(f"IC promedio del Score:              {prec_bt['ic'].mean():.3f} "
      f"(t = {prec_bt['ic'].mean() / prec_bt['ic'].std() * np.sqrt(len(prec_bt)):.2f})")
print(f"Trimestres que le ganó a SPY:       {prec_bt['vence_spy'].mean():.0%}")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].scatter(prec_bt['ret_esperado'], prec_bt['ret_real_anual'].clip(-1, 2.5), alpha=.6)
lim = [0, prec_bt['ret_esperado'].max()]
ax[0].plot(lim, lim, 'r--', label='Predicción perfecta'); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_xlabel('Retorno prometido (anual)'); ax[0].set_ylabel('Retorno real (anualizado)')
ax[0].set_title('Precisión del retorno esperado')
ax[1].bar(pd.to_datetime(prec_bt['fecha']), prec_bt['ic'], width=60,
          color=np.where(prec_bt['ic'] >= 0, 'seagreen', 'indianred'))
ax[1].axhline(prec_bt['ic'].mean(), color='k', ls='--', label=f"IC medio {prec_bt['ic'].mean():.3f}")
ax[1].set_title('IC del Score General por trimestre'); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
"""))
celdas.append(md("""
### Fase 8.5: Monte Carlo
1. **Bootstrap pareado:** remuestrea bloques de 21 días de los retornos diarios reales (los mismos bloques para la estrategia y SPY) para simular futuros de 20 años con $100,000.
2. **¿Habilidad o suerte?:** compara la estrategia contra carteras de 5 acciones **elegidas al azar** del mismo universo histórico, en las mismas fechas y con el mismo filtro de mercado.
"""))
celdas.append(code("""
rng_bt = np.random.default_rng(SEED_BT)

# 1. Bootstrap pareado por bloques
def mc_bootstrap(serie, n_paths, horizonte=252 * ANIOS_BT, bloque=21, seed=SEED_BT + 1):
    r = serie.pct_change().dropna().values
    rng_local = np.random.default_rng(seed)        # misma semilla => mismos bloques en cada serie
    finales, mdds, muestras = np.empty(n_paths), np.empty(n_paths), []
    for p in range(n_paths):
        idx = rng_local.integers(0, len(r) - bloque, size=horizonte // bloque + 1)
        v = CAPITAL * np.cumprod(1 + np.concatenate([r[j:j + bloque] for j in idx])[:horizonte])
        finales[p], mdds[p] = v[-1], (v / np.maximum.accumulate(v) - 1).min()
        muestras.append(v[::21])
    return finales, mdds, np.percentile(np.array(muestras), [5, 25, 50, 75, 95], axis=0)

fin_e, mdd_e, band_e = mc_bootstrap(nav_bt, N_MC_BOOT)
fin_s, mdd_s, band_s = mc_bootstrap(spy_nav, N_MC_BOOT)
resumen_mc = pd.DataFrame({
    'Estrategia V5': [np.percentile(fin_e, 5), np.median(fin_e), np.percentile(fin_e, 95), np.median(mdd_e),
                      (mdd_e < -0.5).mean(), (fin_e < CAPITAL).mean()],
    'SPY': [np.percentile(fin_s, 5), np.median(fin_s), np.percentile(fin_s, 95), np.median(mdd_s),
            (mdd_s < -0.5).mean(), (fin_s < CAPITAL).mean()],
}, index=['Valor final p5', 'Valor final mediana', 'Valor final p95', 'Caída máx. mediana',
          'Prob. caída > 50 %', 'Prob. perder capital'])
display(resumen_mc.apply(lambda c: [f'${v:,.0f}' if i < 3 else f'{v:.1%}' for i, v in enumerate(c)]))
print(f"Probabilidad de que la estrategia termine arriba de SPY en 20 años: {(fin_e > fin_s).mean():.0%}")

# 2. Carteras aleatorias con las mismas reglas (motor vectorizado, pesos que derivan entre rebalanceos)
R_bt, RF_bt = close_bt.pct_change(fill_method=None).fillna(0).values, rf_diario_bt.values
col_idx = {c: j for j, c in enumerate(close_bt.columns)}
idx_e = [fechas_bt.get_loc(e['fecha']) for e in eventos_bt]
univ_e = [[col_idx[c] for c in e['score'].index] if e['top'] else [] for e in eventos_bt]

def backtest_pesos(pesos, columnas, costo=COSTO_BT):
    n, inicio = len(fechas_bt), idx_e[0] + 1
    nav = np.empty(n - inicio); nav[0] = v = 1.0; previo = {}
    for k, i0 in enumerate(idx_e):
        a = i0 + 1
        b = idx_e[k + 1] + 1 if k + 1 < len(idx_e) else n - 1
        cols, w = list(columnas[k]), np.asarray(pesos[k], dtype=float)
        nuevo = dict(zip(cols, w))
        v *= 1 - costo * sum(abs(nuevo.get(c, 0) - previo.get(c, 0)) for c in set(nuevo) | set(previo))
        nav[a - inicio] = v
        if b > a and not cols:
            nav[a + 1 - inicio:b + 1 - inicio] = v * np.cumprod(1 + RF_bt[a + 1:b + 1]); v = nav[b - inicio]; previo = {}
        elif b > a:
            crec = np.cumprod(1 + R_bt[a + 1:b + 1][:, cols], axis=0)
            nav[a + 1 - inicio:b + 1 - inicio] = v * (crec @ w); v = nav[b - inicio]
            wf = w * crec[-1]; previo = dict(zip(cols, wf / wf.sum()))
        else:
            previo = nuevo
    return pd.Series(nav, index=fechas_bt[inicio:])

def cagr_de(nv):
    return nv.iloc[-1] ** (365.25 / (nv.index[-1] - nv.index[0]).days) - 1

cagr_estr = cagr_de(backtest_pesos([e['pesos'] for e in eventos_bt], [[col_idx[c] for c in e['top']] for e in eventos_bt]))
cagr_azar = []
for _ in range(N_MC_AZAR):
    cols_k = [list(rng_bt.choice(c, TOP_N_BT, replace=False)) if c else [] for c in univ_e]
    pesos_k = [rng_bt.dirichlet(np.ones(TOP_N_BT)) if c else np.array([]) for c in univ_e]
    cagr_azar.append(cagr_de(backtest_pesos(pesos_k, cols_k)))
cagr_azar = np.array(cagr_azar)
percentil = (cagr_azar < cagr_estr).mean()
print(f"CAGR estrategia: {cagr_estr:.2%} | azar: mediana {np.median(cagr_azar):.2%} "
      f"(p5 {np.percentile(cagr_azar, 5):.2%}, p95 {np.percentile(cagr_azar, 95):.2%})")
print(f"Percentil de la estrategia: {percentil:.0%} | p-valor: {1 - percentil:.3f} "
      f"({'significativo' if 1 - percentil < 0.05 else 'no se distingue del azar'} al 5 %)")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
meses = np.arange(band_e.shape[1]) / 12
for nombre, band, col in [('Estrategia V5', band_e, 'C0'), ('SPY', band_s, 'C3')]:
    ax[0].fill_between(meses, band[0], band[4], color=col, alpha=.12)
    ax[0].fill_between(meses, band[1], band[3], color=col, alpha=.25)
    ax[0].plot(meses, band[2], color=col, label=f'{nombre} (mediana, 50 % y 90 %)')
ax[0].set_yscale('log'); ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_xlabel('Años')
ax[0].set_title(f'Monte Carlo bootstrap – {N_MC_BOOT} caminos de {ANIOS_BT} años')
ax[1].hist(cagr_azar, bins=30, alpha=.7, label='Carteras aleatorias')
ax[1].axvline(cagr_estr, color='r', lw=2, label=f'Estrategia ({cagr_estr:.1%})')
ax[1].legend(); ax[1].grid(alpha=.3); ax[1].set_title('Estrategia vs selección al azar')
plt.tight_layout(); plt.show()
"""))
celdas.append(md("### Fase 8.6: Todas las Operaciones y Acciones Compradas en los 20 Años"))
celdas.append(code("""
filas_ops = []
for l in log_bt:
    for t in sorted(set(l['antes']) | set(l['despues'])):
        a0, a1 = l['antes'].get(t, 0), l['despues'].get(t, 0)
        delta, precio = a1 - a0, l['precios'][t]
        tipo = 'Compra' if a0 == 0 else 'Venta total' if a1 == 0 else 'Aumenta' if delta > 0 else 'Reduce' if delta < 0 else 'Mantiene'
        filas_ops.append({'Ejecución': l['ejecucion'].date(), 'Acción': t, 'Operación': tipo, 'Acciones antes': int(a0),
                          'Acciones después': int(a1), 'Precio': precio, 'Monto operado': abs(delta) * precio,
                          'Peso': a1 * precio / l['valor'], 'Valor portafolio': l['valor']})
    if not l['despues']:
        filas_ops.append({'Ejecución': l['ejecucion'].date(), 'Acción': 'EFECTIVO', 'Operación': 'A efectivo (SPY < SMA200)',
                          'Acciones antes': 0, 'Acciones después': 0, 'Precio': np.nan, 'Monto operado': 0.0,
                          'Peso': 1.0, 'Valor portafolio': l['valor']})
operaciones_bt = pd.DataFrame(filas_ops)

tenencias = operaciones_bt[(operaciones_bt['Acciones después'] > 0)]
acciones_bt = (tenencias.groupby('Acción')
               .agg(Trimestres_en_cartera=('Ejecución', 'count'), Primera_compra=('Ejecución', 'min'),
                    Ultima_vez=('Ejecución', 'max'), Peso_promedio=('Peso', 'mean'))
               .join(operaciones_bt[operaciones_bt['Operación'] == 'Compra'].groupby('Acción').size().rename('Veces_comprada'))
               .join(operaciones_bt[operaciones_bt['Monto operado'] > 0]
                     .loc[lambda d: d['Acciones después'] > d['Acciones antes']].groupby('Acción')['Monto operado'].sum()
                     .rename('Monto_comprado_total'))
               .sort_values('Trimestres_en_cartera', ascending=False))

print(f"{len(operaciones_bt)} operaciones | {len(acciones_bt)} acciones distintas compradas en {ANIOS_BT} años")
print(', '.join(f'{t} ({n})' for t, n in acciones_bt['Trimestres_en_cartera'].items()))
display(acciones_bt.style.format({'Peso_promedio': '{:.1%}', 'Monto_comprado_total': '${:,.0f}'}))
print("\\nÚltimas 25 operaciones:")
display(operaciones_bt.tail(25).style.format({'Precio': '${:,.2f}', 'Monto operado': '${:,.0f}', 'Peso': '{:.1%}',
                                               'Valor portafolio': '${:,.0f}'}).hide(axis='index'))

operaciones_bt.to_csv('operaciones_v5_backtest.csv', index=False)
acciones_bt.to_csv('acciones_v5_backtest.csv')
print("Guardado: operaciones_v5_backtest.csv y acciones_v5_backtest.csv")
try:
    from google.colab import files   # en Colab, descarga los CSV
    files.download('operaciones_v5_backtest.csv')
    files.download('acciones_v5_backtest.csv')
except ImportError:
    pass
"""))

nb = {"cells": celdas, "metadata": {
    "colab": {"provenance": [], "name": SALIDA.name},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
SALIDA.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Generado: {SALIDA.name} ({len(celdas)} celdas)")
