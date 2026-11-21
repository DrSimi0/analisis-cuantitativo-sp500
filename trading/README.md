# Trading automático (Alpaca + GitHub Actions)

Ejecuta la variante validada del backtest (`univhist_sma200_rfdin_banda5`) en una cuenta de Alpaca.

| Cuándo (≈10:00 NY) | Qué hace |
|---|---|
| 1er día hábil de **ene/abr/jul/oct** | Señal nueva (Top 5 por Score General + Máximo Sharpe con tasa ^IRX) y rebalanceo con banda de 5 pp |
| 1er día hábil de los **demás meses** | Solo filtro SMA 200: SPY < SMA 200 al cierre del mes → todo a efectivo; vuelve arriba → recompra la última señal |

Primero vende, espera a que se llenen las ventas y luego compra con el efectivo real (acciones enteras, órdenes a mercado).
**Gestiona toda la cuenta:** vende cualquier posición que no esté en el objetivo. Usa una cuenta dedicada.

## Archivos
- `config.py` — parámetros de la estrategia, horario y seguros
- `senal.py` — descarga de datos y señal (réplica exacta de `senal()` del backtest; ver `verificar_senal.py`)
- `broker.py` — Alpaca: cuenta, posiciones, calendario, precios, órdenes
- `ejecutar.py` — orquestador
- `historial/estado.json` — última señal, si está invertido y último periodo ejecutado (lo actualiza el bot)
- `historial/ejecuciones/`, `historial/simulaciones/` — un JSON por corrida

## Seguros
- `TRADING_ACTIVO` distinto de `true` → solo simula. `ALPACA_PAPER` distinto de `false` → cuenta paper.
- No repite un mes ya ejecutado; no opera con el mercado cerrado ni con datos desactualizados.
- Aborta si Max Sharpe no converge, si falta algún precio o si las ventas no se completan (no compra).
- Cualquier error hace fallar la corrida → GitHub te avisa por correo.

## Uso local
```bash
pip install -r trading/requirements.txt
copy .env.example .env          # y pon tus claves
python trading/ejecutar.py --modo forzar --dry-run
```

## GitHub
1. Repo **privado**.
2. *Settings → Secrets and variables → Actions*
   - **Secrets:** `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
   - **Variables:** `ALPACA_PAPER=true`, `TRADING_ACTIVO=true` (pon `false` para pausar el bot)
3. *Actions → Rebalanceo automatico → Run workflow* para probar manualmente (`dry_run` marcado = simulación).

## Pasar a dinero real
Crea claves de la cuenta real, reemplaza los dos Secrets y cambia la variable `ALPACA_PAPER` a `false`.
