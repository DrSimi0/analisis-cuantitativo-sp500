# Pedido de auditoría independiente (para el agente de Antigravity)

Revisa `backtest/backtest_v4.py` (backtest walk-forward de 20 años de la estrategia del notebook
`Stocks_Auto_Analisis_V4 Final.ipynb`: Top 5 por Score General + pesos de Máximo Sharpe, $100K).
Resultados en `backtest/resultados/resultados.json`.

NO modifiques ningún archivo existente. Escribe SOLO el archivo `backtest/revision_antigravity.md` con:

1. **Sesgo de anticipación (look-ahead):** ¿la función `senal(t)` usa algún dato posterior a `t`?
   ¿La ejecución al cierre de t+1 en `simular()` y `backtest_pesos()` es correcta?
2. **Errores de cálculo:** revisa `metricas()` (Sharpe, Sortino, Calmar, Beta, Alfa, captura alcista/bajista),
   `backtest_pesos()` (derivación de pesos y costos) y el Monte Carlo (bootstrap por bloques y portafolios aleatorios).
3. **Fidelidad al notebook:** ¿`senal()` replica el Score General y el Max Sharpe de las celdas 25 y 26?
4. **Sesgos de diseño** que inflen los resultados (supervivencia, etc.) y cómo mitigarlos.
5. Lista de hallazgos con severidad (alta / media / baja) y línea de código.

Al terminar, la última línea del archivo debe ser exactamente: `FIN DE LA REVISION`
