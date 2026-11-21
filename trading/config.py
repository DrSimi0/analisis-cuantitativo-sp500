"""Parámetros del trading automático (variante validada en el backtest: univhist + sma200 + rfdin + banda 5 %)."""
import os
from pathlib import Path

RAIZ = Path(__file__).parent
HISTORIAL = RAIZ / "historial"
ESTADO = HISTORIAL / "estado.json"


def _cargar_env():
    """Carga ../.env (solo en tu PC; en GitHub Actions las claves vienen de Secrets)."""
    archivo = RAIZ.parent / ".env"
    if not archivo.exists():
        return
    for linea in archivo.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#") and "=" in linea:
            k, v = linea.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_cargar_env()

# ---------------- Estrategia (idéntica a backtest/backtest_v4.py) ----------------
VENTANA = 1260            # 5 años de sesiones
VOL_MIN = 1_000_000       # volumen promedio mínimo
COBERTURA = 0.95          # % mínimo de historial en la ventana
TOP_N = 5
TOPE = 1.0                # peso máximo por acción en Max Sharpe (1.0 = sin tope, variante validada)
BANDA = 0.05              # no se ajusta una acción que sigue en el Top 5 si su peso está a <= 5 pp del objetivo
SMA = 200                 # filtro de régimen: SPY vs SMA 200 al cierre de cada mes
COSTO = 0.0010            # colchón de efectivo por operación (igual que el backtest)
MESES_TRIMESTRE = (1, 4, 7, 10)   # primer día hábil de estos meses = rebalanceo completo

# ---------------- Horario (hora de Nueva York) ----------------
HORA_MIN = "09:45"        # no operar en la apertura
HORA_MAX = "12:30"        # tolerancia a retrasos del cron de GitHub

# ---------------- Seguridad ----------------
EQUITY_MIN = 1_000        # no operar si la cuenta vale menos que esto
ESPERA_ORDEN_SEG = 300    # tiempo máximo esperando que se llenen las órdenes
COLCHON_COMPRAS = 0.002   # deja 0.2 % de efectivo extra al comprar (deslizamiento)

# ---------------- Alpaca ----------------
API_KEY = os.environ.get("ALPACA_API_KEY", "")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
PAPER = os.environ.get("ALPACA_PAPER", "true").lower() != "false"          # dinero real solo con ALPACA_PAPER=false
TRADING_ACTIVO = os.environ.get("TRADING_ACTIVO", "false").lower() == "true"  # interruptor: si no es "true", solo simula
