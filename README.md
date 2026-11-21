# 📈 Análisis Cuantitativo Automatizado del S&P 500

Pipeline en Jupyter que descarga el universo completo del **S&P 500**, filtra por liquidez y aplica análisis **fundamental, técnico, econométrico y de portafolios** para obtener un **Top 5 de acciones** con su portafolio óptimo.

> ⚠️ Proyecto educativo. Los resultados se basan en datos históricos y **no constituyen recomendación de inversión**.

---

## 🗂️ Estructura

```
Stocks/
├── notebooks/
│   └── Stocks_Auto_Analisis_V4.ipynb   ← versión actual (con resultados)
├── archivo/                            ← versiones anteriores (referencia)
│   ├── Cuantitative_Auto_Stocks_V1.ipynb
│   ├── Stocks_Auto_Analisis_V2.ipynb
│   ├── Stocks_Auto_Analisis_V3_original.ipynb
│   └── Stocks_Auto_Analisis_V3.ipynb
├── requirements.txt
└── README.md
```

## 🔄 Pipeline (V4)

| Paso | Fase | Qué hace |
|---|---|---|
| **1** | Filtro inicial | Tickers del S&P 500 (Wikipedia) → 5 años de precios ajustados (yfinance) → filtro de volumen promedio > 1M. Excluye acciones con < 95 % de historial. |
| **2** | Fundamental | Beta histórica, **CAPM**, D/E real de Yahoo y **beta de Hamada** (desapalancada y re-apalancada a la estructura de capital mediana). |
| **3** | Técnico | **RSI de Wilder**, **MACD**, **Estocástico**, SMA 20/50. Escáner de acciones saliendo de sobreventa y clasificación por cuadrante volatilidad × dirección. |
| **4** | Econométrico | Retorno, varianza, volatilidad, **Sharpe**, matrices de covarianza/correlación y test de cointegración. |
| **5** | Portafolio | Portafolio equiponderado del universo. |
| **6** | Optimización | **Máximo Sharpe**, **Mínima Varianza Global (GMV)** y **frontera eficiente** (SciPy SLSQP, peso máximo 35 %). |
| **7** | Monte Carlo y ranking | Simulación de portafolios (Dirichlet) y **Score General** para elegir el Top 5. |
| **Final** | Resumen Top 5 | Tabla con todas las métricas, pesos por portafolio, comparativo de portafolios y gráficos técnicos de cada acción. |

### Score General

```
Score = 35 % Sharpe + 25 % Rendimiento + 25 % Control de Drawdown + 15 % Baja Volatilidad
```

Cada métrica se normaliza (min-max) antes de ponderar.

## 🚀 Cómo ejecutarlo

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

Abre `notebooks/Stocks_Auto_Analisis_V4.ipynb` y ejecuta **Restart Kernel → Run All**.
La corrida completa (~500 tickers) tarda unos **10 minutos**; lo más lento es la consulta de D/E (Paso 2) y la frontera eficiente (Paso 6).

> Ejecuta siempre el notebook completo en orden: la celda final usa variables de todas las fases anteriores.

## 📌 Supuestos y limitaciones

- Tasa libre de riesgo fija de **4 %** y tasa de impuestos de **21 %**.
- **Sesgo de supervivencia**: se usa la composición *actual* del S&P 500.
- La optimización es **in-sample**: los retornos del portafolio óptimo son históricos, no esperados.
- El D/E no es significativo para bancos/aseguradoras; con capital contable negativo se deja vacío.

## 📝 Historial de versiones

| Versión | Cambios principales |
|---|---|
| **V4** | Corrige historial truncado (join `inner` → `outer` + filtro de cobertura), precios ajustados por dividendos, SPY fuera del universo, Hamada re-apalancado, RSI de Wilder unificado, frontera con retornos alcanzables, Monte Carlo con Dirichlet y límite de 35 %, compatibilidad con pandas 3, tabla resumen integral del Top 5. |
| V3 | Pipeline completo con CAPM/Hamada, optimización y Score General. |
| V1–V2 | Primeras versiones del análisis. |
