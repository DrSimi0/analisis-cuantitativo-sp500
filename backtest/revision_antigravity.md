# Auditoría Independiente: `backtest/backtest_v4.py`

**Alcance:** Revisión técnica y metodológica de `backtest/backtest_v4.py`, su fidelidad frente a `Stocks_Auto_Analisis_V4 Final.ipynb` (celdas 2, 4, 7, 25 y 26), la descarga de datos en `backtest/descargar_datos.py` y las cifras reportadas en `backtest/resultados/resultados.json`.

---

## 1. Sesgo de anticipación (Look-Ahead Bias)

* **¿La función `senal(t)` usa algún dato posterior a `t`?**
  * **En el cálculo sobre la serie de precios:** **No.** En `senal(t)` (líneas 73–77), `i = fechas.get_loc(t)` y el recorte `close.iloc[i + 1 - VENTANA:i + 1]` (y `volume.iloc[...]`) excluye el índice `i + 1`, por lo que la última observación incluida corresponde exactamente al cierre de `t`. Todos los filtros (`elegibles`), normalizaciones Min-Max, el *Score General* y la optimización `max_sharpe()` operan exclusivamente dentro de esa ventana `[t - 1259, t]`.
  * **Excepciones menores / externas:**
    1. En la línea 50, `irx = datos["irx"].reindex(fechas).ffill().bfill()` utiliza `.bfill()`, que propaga hacia atrás un valor futuro si existieran `NaN` al principio de la serie de 2001 (aunque en la ventana del backtest 2006–2026 ya hay datos previos y la señal usa `RF_MODELO = 0.04` fijo, no `rf_diario`).
    2. La sección 7 (`portafolio_actual`, líneas 395–406) aplica explícitamente la señal de `fechas[-1]` hacia atrás; está documentado como ejercicio ilustrativo con sesgo de anticipación y no contamina el backtest principal.
    3. El **sesgo estructural de supervivencia/inclusión** en el universo de entrada (ver Sección 4) sí introduce información futura en la lista de candidatos elegibles.

* **¿La ejecución al cierre de `t+1` en `simular()` y `backtest_pesos()` es correcta?**
  * **Sí, la sincronización temporal es correcta en ambas funciones.**
  * En `simular()` (líneas 110–134), `exec_idx` mapea cada señal al índice `fechas.get_loc(s["fecha"]) + 1` (el cierre de `t+1`), protegida contra desbordamiento porque `fechas_rebalanceo()` filtra `f < FIN` (línea 102). El efectivo devenga `rf_diario.iloc[i]` recién a partir de `i > inicio`.
  * En `backtest_pesos()` (líneas 204–212), `a = i0 + 1` es el cierre de `t+1` y los retornos acumulados se toman sobre `R[a + 1:b + 1]` (días `a+1` hasta `b` inclusive), que corresponde exactamente al retorno desde el cierre de `a` hasta el cierre de `b`.

---

## 2. Errores de cálculo y detalles de implementación

### En `metricas()` (líneas 139–190)
* **Sharpe (línea 160):** Calcula `ex.mean() / r.std() * np.sqrt(252)`. Usa el exceso de retorno `ex` en el numerador, pero la desviación estándar del retorno total `r.std()` en el denominador en lugar de `ex.std()`. Con tasas diarias suaves la diferencia es mínima (4to decimal), pero estrictamente debe ser `ex.std()`.
* **Sortino (línea 161):** `ex.mean() * 252 / (np.sqrt((neg ** 2).sum() / len(ex)) * np.sqrt(252))`. Correcto: usa `len(ex)` en el denominador de la semivarianza respecto a `rf_diario` como MAR.
* **Calmar (línea 163):** `cagr / abs(dd.min())`. Correcto (carece de guarda si `dd.min() == 0`, improbable en 20 años).
* **Duración máxima de drawdown (líneas 149–151):** `bajo = dd < 0` mide desde el primer día con `dd < 0` hasta el último día con `dd < 0`, omitiendo el día del pico anterior y el día de recuperación total; subestima ligeramente la duración real calendario.
* **Beta y Alfa (líneas 174–177):** Fórmulas consistentes (`ddof=1` tanto en `np.cov` como en `bx.var()`). En `resultados.json`, `SPY` contra `SPY` da `Beta = 1.0` y `Alfa = 1.78e-16`.
* **Tracking Error / Information Ratio en auto-comparación (líneas 178 y 185):** Al evaluar `SPY` contra `SPY` en `tabla` (línea 265), el error numérico de punto flotante deja `TE = 1.54e-15` e imprime `Information ratio = 0.1091` (línea 111 de `resultados.json`) en lugar de `0` / `NaN`.
* **Captura alcista / bajista (líneas 186–187):** Usa el promedio aritmético de los retornos mensuales de la estrategia sobre el del benchmark en los meses donde `bm > 0` y `bm < 0`. Es una convención estándar válida (`SPY` vs `SPY` = `1.0`).

### En `backtest_pesos()` (líneas 195–218)
* **Deriva de pesos (`w_fin`, líneas 213–215):** Correcta. Al final de cada tramo `a..b`, los pesos derivados se actualizan como `(w * crec[-1]) / (w * crec[-1]).sum()`, reflejando fielmente la evolución *buy-and-hold* entre rebalanceos.
* **Costos de transacción (líneas 208–209):** `operado` suma `|nuevo - previo|` (two-way turnover) y multiplica `v *= 1 - costo * operado` (`10 pb` por dólar comprado o vendido). Es coherente.
* **Desfase de 1 día en el costo inicial (líneas 200 y 212):** `nav[0] = 1.0` se fija antes de aplicar `v *= 1 - costo * operado`, por lo que el costo de la compra inicial de `a0` recién impacta en `nav[1]` (`a0 + 1`). Impacto despreciable (~0.1% en el día 1).
* **Diferencia entre `simular()` y `backtest_pesos()`:** `simular()` opera con acciones enteras, reserva `(1 - 2 * costo)` en la línea 122 y remunera el efectivo sobrante con `^IRX`; `backtest_pesos()` es fraccionario y 100% invertido. Por ello el CAGR de la estrategia difiere ligeramente entre `simular()` (`14.297%`) y `backtest_pesos()` en Monte Carlo B (`14.300%`).

### En Monte Carlo A y B (líneas 329–393)
* **Monte Carlo A — Bootstrap por bloques (líneas 329–366):**
  * El muestreo por bloques contiguos de 21 días preserva adecuadamente la autocorrelación de corto plazo y el *clustering* de volatilidad.
  * **Error metodológico en la comparación con SPY (línea 365):** `prob_estrategia_supera_spy` (`0.6757`) compara `mc_fin_estr[:, None] > mc_fin_spy[None, :1000]`, es decir, caminos generados con **bloques independientes** para la estrategia y para el SPY. Al destruir la correlación contemporánea (~`0.71`) entre ambos activos, se distorsiona la distribución de retornos relativos. Lo correcto es un **bootstrap pareado** (reutilizar los mismos índices `ini_b` en cada trayectoria para la estrategia y el SPY).
* **Monte Carlo B — Portafolios aleatorios (líneas 369–393):**
  * Bien implementado usando exactamente la misma ventana de universo elegible (`univ_cols`) y `backtest_pesos()`.
  * Usa `rng.dirichlet(np.ones(TOP_N))`, que genera portafolios bastante diversificados (~20% por activo en promedio), mientras que `max_sharpe` puede concentrar hasta 100% en 1–2 activos. Aun así, el resultado es contundente: `p_valor_cagr = 0.256` y `p_valor_sharpe = 0.207` (la estrategia no supera estadísticamente al azar al 5%).

---

## 3. Fidelidad al notebook (`Stocks_Auto_Analisis_V4 Final.ipynb`)

`senal(t)` replica con **alta fidelidad** las celdas 2, 4, 25 y 26 del notebook:

| Componente | Notebook V4 Final | `backtest_v4.py` | Evaluación |
| :--- | :--- | :--- | :--- |
| **Ventana y liquidez** | `5y` (~1260 d), `Volume.mean() > 1_000_000`, cobertura $\ge 95\%$ | `VENTANA = 1260`, `VOL_MIN = 1_000_000`, `COBERTURA = 0.95` (líneas 30–33, 76–79) | Fiel (agrega `iloc[-1].notna()` para operabilidad) |
| **Alineación `dropna()`** | `join='outer'` + `drop` <95% + `.dropna()` global tras unir `SPY` | `p = vent_p[elegibles].dropna()` (línea 80) | Fiel (ambos recortan a la fecha más reciente del conjunto elegible) |
| **Métricas Score (Celda 25)** | `mean*252`, `std*sqrt(252)`, `rf=0.04`, `1+MDD`, `1/vol`, Min-Max, pesos `35/25/25/15` | Líneas 81–89 idénticas en fórmula, pesos y `RF_MODELO = 0.04` | **Réplica exacta** |
| **Max Sharpe (Celda 26)** | `SLSQP`, `bounds=(0, 1.0)` sobre Top 5, `rf=0.04` | `max_sharpe()` en líneas 60–68 con `bounds=[(0, 1)] * n` y `rf=0.04` | **Réplica exacta** |

*Nota:* En el notebook, la celda 21 limita los pesos a `35%` sobre todo el universo, pero la celda 26 (la implementada aquí) usa `bounds=(0, 1.0)` sobre el Top 5. El backtest replica correctamente la celda 26.

---

## 4. Sesgos de diseño que inflan los resultados

1. **Sesgo de Supervivencia (*Survivorship Bias*):**
   * `descargar_datos.py` (líneas 17–23) descarga los integrantes **actuales** del S&P 500 (noviembre de 2026) con su historia desde 2001. Se excluyen todas las compañías que quebraron, fueron adquiridas con pérdidas o salieron del índice entre 2006 y 2026 (ej. Lehman Brothers, Bear Stearns, Washington Mutual, SVB, First Republic, Sears, Kodak, etc.).
   * **Impacto demostrado en `resultados.json`:** El **`Universo Equiponderado`** logra un CAGR de **14.45%** (Sharpe **0.687**) frente al **10.28%** del `SPY` (Sharpe **0.521**) y al **14.30%** de la `Estrategia Top 5 Max Sharpe` (Sharpe **0.609**). Además, el IC promedio del Score General es **-0.0058** (`t-stat = -0.24`), lo que confirma que el retorno superior frente al SPY proviene casi íntegramente del sesgo del universo sobreviviente y no del Score.
   * **Mitigación:** Utilizar datos históricos *point-in-time* (PIT) con la lista real de constituyentes del S&P 500 vigente en cada fecha `t` y series de acciones deslistadas (*delisted*).

2. **Sesgo de Inclusión Retroactiva (*Look-ahead Index Inclusion*):**
   * Empresas que crecieron siendo *small/mid cap* y entraron al S&P 500 años después (ej. en 2015 o 2020) ya figuran como candidatas desde que acumulan 5 años de historia, permitiendo comprarlas antes de pertenecer al índice.
   * **Mitigación:** Filtrar `elegibles` en `t` contra una tabla histórica de membresía diaria del S&P 500.

3. **Sobreconcentración por error de estimación en Max Sharpe (`0–100%`):**
   * Optimizar Max Sharpe sin límite por activo (`bounds=(0, 1)`) usando retornos medios de 5 años tiende a concentrar el 70%–100% en 1 o 2 ganadores recientes, generando una correlación esperado-real de **-0.0026** y un Max Drawdown de **-68.47%** (peor que SPY `-55.19%` y Universo Equiponderado `-51.29%`).
   * **Mitigación:** Imponer el tope del `35%` por activo (como la celda 21 del notebook) o aplicar *shrinkage* sobre retornos/covarianza.

4. **Tasa libre de riesgo fija (`RF_MODELO = 0.04`) en la señal:**
   * Usar 4% constante durante el periodo de tasas cero (2009–2021) distorsiona el ranking de Sharpe y los pesos de Max Sharpe respecto al entorno real de tasas en `t`.
   * **Mitigación:** Evaluar en `senal(t)` la tasa vigente `irx.loc[t]` (si se desea mantener el modelo consistente con el costo de oportunidad histórico).

---

## 5. Lista de hallazgos por severidad

| Severidad | Hallazgo | Ubicación (Archivo y Líneas) |
| :--- | :--- | :--- |
| **ALTA** | **Sesgo de supervivencia e inclusión retroactiva** al usar constituyentes actuales del S&P 500 desde 2001 (infla el universo al 14.45% CAGR vs 10.28% SPY y explica el aparente alfa). | `backtest/descargar_datos.py` L15–23; `backtest/backtest_v4.py` L43–49, L76–79 |
| **MEDIA** | **Bootstrap de Monte Carlo A no pareado** al comparar `Estrategia` vs `SPY` con trayectorias independientes, ignorando su correlación (~0.71). | `backtest/backtest_v4.py` L348–365 |
| **MEDIA** | **Recorte global `.dropna()` en la ventana** (`vent_p[elegibles].dropna()`): una sola acción con cobertura del 95–99% recorta la ventana histórica de todo el universo en `senal(t)`. | `backtest/backtest_v4.py` L80 |
| **MEDIA** | **No se verifica `res.success`** en la optimización `SLSQP` dentro de `max_sharpe()`, usando `res.x` aunque el optimizador no converja. | `backtest/backtest_v4.py` L65–68 |
| **BAJA** | **Denominador del Sharpe Ratio:** usa `r.std()` (retorno total) en lugar de `ex.std()` (exceso de retorno). | `backtest/backtest_v4.py` L160 |
| **BAJA** | **`close.ffill()` en `simular()`** congela precios indefinidamente ante posibles suspensiones o huecos sin liquidar la posición. | `backtest/backtest_v4.py` L109 |
| **BAJA** | **Desfase de 1 día en el costo inicial** en `backtest_pesos()` (`nav[0] = 1.0` antes de aplicar el primer costo) y ruido `0/0` en `Information ratio` al comparar `SPY` vs `SPY`. | `backtest/backtest_v4.py` L178, L185, L200, L209 |
| **BAJA** | **Duración de drawdown (`dd < 0`)** omite la fecha del pico previo y la de recuperación; `.bfill()` en `irx` propaga valores futuros si hubiera `NaN` iniciales en 2001. | `backtest/backtest_v4.py` L50, L149–151 |

FIN DE LA REVISION
