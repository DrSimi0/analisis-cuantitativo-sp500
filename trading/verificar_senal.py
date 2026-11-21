"""Comprueba que senal.calcular_senal() da el mismo Top 5 y los mismos pesos que senal() del backtest validado.

Uso:  python trading/verificar_senal.py     (requiere backtest/data/precios_hist.pkl)
"""
import os
import sys
from pathlib import Path

import numpy as np

os.environ.update({"BT_RF": "dinamica", "BT_FILTRO": "sma200", "BT_UNIVERSO": "historico", "BT_BANDA": "0.05"})
sys.path.insert(0, str(Path(__file__).parent.parent / "backtest"))
sys.stdout.reconfigure(encoding="utf-8")
import backtest_v4 as bt  # noqa: E402
from senal import calcular_senal  # noqa: E402

fechas = bt.fechas_rebalanceo("QE")[-12:]          # últimos 12 trimestres
fallos = 0
for f in fechas:
    t = bt.fechas[bt.fechas.searchsorted(f)]
    ref = bt.senal(t)
    cols = [c for c in bt.close.columns if c in bt.miembros(t)]
    i = bt.fechas.get_loc(t)
    nuevo = calcular_senal(bt.close[cols].iloc[:i + 1], bt.volume[cols].iloc[:i + 1], float(bt.irx.iloc[i]))
    w_ref = dict(zip(ref["top"], ref["pesos"]))
    w_new = dict(zip(nuevo["top"], nuevo["pesos"]))
    igual = set(w_ref) == set(w_new) and all(abs(w_ref[k] - w_new[k]) < 1e-6 for k in w_ref)
    fallos += not igual
    print(f"{t.date()}  {'OK   ' if igual else 'DIFERENTE'}  " + ", ".join(f"{k} {v:.1%}" for k, v in w_new.items()))
print("\nTodo coincide." if fallos == 0 else f"\n{fallos} fechas no coinciden.")
sys.exit(1 if fallos else 0)
