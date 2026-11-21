"""Orquestador: decide si toca operar, calcula las órdenes y las envía a Alpaca.

Calendario (igual que el backtest validado):
  * Primer día hábil de ene/abr/jul/oct -> señal nueva (Top 5 + Max Sharpe) y rebalanceo con banda de 5 pp.
  * Primer día hábil de los demás meses -> solo filtro SMA 200: si SPY cerró el mes debajo de su SMA 200
    se vende todo (efectivo); si vuelve arriba estando en efectivo, se recompra la última señal.
La decisión usa el cierre de la última sesión (fin de mes) y se ejecuta ese día por la mañana.

Uso:
  python trading/ejecutar.py                    # modo auto (lo que corre el cron)
  python trading/ejecutar.py --dry-run          # simula: calcula todo pero no envía órdenes
  python trading/ejecutar.py --modo forzar      # señal nueva y rebalanceo hoy, sin mirar el calendario
"""
import argparse
import json
import math
import os
import sys
import traceback
from datetime import datetime

import config as cfg
from broker import Broker, a_alpaca
from senal import calcular_senal, descargar_datos, regimen

sys.stdout.reconfigure(encoding="utf-8")


class Salida(Exception):
    """Fin normal sin operar (no es un error)."""


def cargar_estado():
    if cfg.ESTADO.exists():
        return json.loads(cfg.ESTADO.read_text(encoding="utf-8"))
    return {"ultimo_periodo": None, "ultima_senal": None, "invertido": None}


def guardar_json(ruta, datos):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def calcular_objetivo(posiciones, pesos, precios, equity):
    """Acciones objetivo con la banda del backtest (simular()): las que siguen en el Top y están a <= BANDA no se tocan."""
    objetivo = {t: math.floor(equity * w * (1 - 2 * cfg.COSTO) / precios[t]) for t, w in pesos.items()}
    conservadas = []
    for t, w in pesos.items():
        if t in posiciones and abs(posiciones[t] * precios[t] / equity - w) <= cfg.BANDA:
            objetivo[t] = posiciones[t]
            conservadas.append(t)
    return {t: q for t, q in objetivo.items() if q > 0}, conservadas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["auto", "forzar"], default="auto")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    simulacion = args.dry_run or not cfg.TRADING_ACTIVO

    reg = {"inicio": datetime.utcnow().isoformat() + "Z", "modo": args.modo, "simulacion": simulacion,
           "paper": cfg.PAPER, "ordenes": [], "eventos": []}
    log = lambda m: (print(m), reg["eventos"].append(m))
    estado = cargar_estado()
    b = None
    try:
        b = Broker()
        ahora, abierto = b.reloj()
        hoy = ahora.date()
        reg["hora_ny"] = ahora.isoformat()
        log(f"Cuenta {'PAPER' if cfg.PAPER else 'REAL'} | {'SIMULACIÓN' if simulacion else 'OPERANDO'} | "
            f"NY {ahora:%Y-%m-%d %H:%M} | mercado {'abierto' if abierto else 'cerrado'}")

        # 1) ¿Toca hoy?
        periodo = hoy.strftime("%Y-%m")
        if args.modo == "auto":
            if hoy != b.primer_dia_habil_del_mes(hoy):
                raise Salida(f"Hoy {hoy} no es el primer día hábil del mes.")
            if not (cfg.HORA_MIN <= f"{ahora:%H:%M}" <= cfg.HORA_MAX):
                raise Salida(f"Fuera del horario {cfg.HORA_MIN}-{cfg.HORA_MAX} NY.")
            if estado.get("ultimo_periodo") == periodo:
                raise Salida(f"El periodo {periodo} ya se ejecutó.")
        if not simulacion and not abierto:
            raise RuntimeError("El mercado está cerrado: no se envían órdenes.")

        cuenta = b.cuenta()
        reg["cuenta_inicial"] = cuenta
        if cuenta["bloqueada"] or cuenta["estado"] != "active":
            raise RuntimeError(f"Cuenta no operable: {cuenta}")
        if cuenta["equity"] < cfg.EQUITY_MIN:
            raise RuntimeError(f"Equity {cuenta['equity']:.2f} menor al mínimo {cfg.EQUITY_MIN}")

        # 2) Datos y régimen (cierre de la última sesión)
        datos = descargar_datos(hasta=hoy)
        t = datos["close"].index[-1].date()
        esperada = b.sesion_anterior(hoy)
        if t != esperada or datos["spy"].index[-1].date() != esperada:
            raise RuntimeError(f"Datos desactualizados: último cierre {t}, se esperaba {esperada}")
        r = regimen(datos["spy"])
        reg["regimen"] = r
        log(f"Cierre {t}: SPY {r['spy']:.2f} vs SMA200 {r['sma200']:.2f} ({r['distancia']:+.2%}) -> "
            f"{'ALCISTA' if r['alcista'] else 'BAJISTA'}")

        # 3) ¿Señal nueva?
        es_trimestre = hoy.month in cfg.MESES_TRIMESTRE
        senal_nueva = args.modo == "forzar" or es_trimestre or not estado.get("ultima_senal")
        if senal_nueva:
            rf = float(datos["irx"].loc[:str(t)].iloc[-1])
            s = calcular_senal(datos["close"], datos["volume"], rf)
            if not s["convergio"]:
                raise RuntimeError(f"Max Sharpe no convergió: {s['mensaje_opt']}")
            log(f"Señal nueva ({s['n_universo']} acciones elegibles, rf {rf:.2%}): " +
                ", ".join(f"{a_alpaca(x)} {w:.1%}" for x, w in zip(s["top"], s["pesos"])))
        else:
            s = estado["ultima_senal"]
            log(f"Se mantiene la señal del {s['fecha']}: " + ", ".join(a_alpaca(x) for x in s["top"]))
        reg["senal"] = s

        # 4) Decisión
        if not r["alcista"]:
            pesos, accion = {}, "A EFECTIVO (SPY < SMA 200)"
        elif senal_nueva:
            pesos, accion = {a_alpaca(x): w for x, w in zip(s["top"], s["pesos"])}, "REBALANCEO"
        elif estado.get("invertido") is False:
            pesos, accion = {a_alpaca(x): w for x, w in zip(s["top"], s["pesos"])}, "REENTRADA (SPY > SMA 200)"
        else:
            pesos, accion = None, "SIN CAMBIOS"
        reg["accion"] = accion
        log(f"Acción: {accion}")

        if pesos is not None:
            ejecutar_ordenes(b, pesos, cuenta["equity"], simulacion, reg, log)

        if not simulacion:
            estado["ultimo_periodo"] = periodo
            if senal_nueva:
                estado["ultima_senal"] = s
            if pesos is not None:
                estado["invertido"] = bool(pesos)
            estado["actualizado"] = reg["inicio"]
        reg["resultado"] = "ok"
    except Salida as e:
        reg["resultado"] = "sin_operar"
        log(str(e))
    except Exception as e:
        reg["resultado"] = "error"
        reg["error"] = f"{e}\n{traceback.format_exc()}"
        log(f"ERROR: {e}")
    finally:
        if b is not None and reg["resultado"] != "sin_operar":
            try:
                reg["cuenta_final"] = b.cuenta()
                reg["posiciones_finales"] = b.posiciones()
            except Exception:
                pass
        if reg["resultado"] != "sin_operar":
            sello = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
            carpeta = cfg.HISTORIAL / ("simulaciones" if simulacion else "ejecuciones")
            guardar_json(carpeta / f"{sello}.json", reg)
            if not simulacion:
                guardar_json(cfg.ESTADO, estado)
        escribir_resumen(reg)
    if reg["resultado"] == "error":
        sys.exit(1)


def ejecutar_ordenes(b, pesos, equity, simulacion, reg, log):
    posiciones = b.posiciones()
    reg["posiciones_iniciales"] = posiciones
    for t in pesos:
        if not b.operable(t):
            raise RuntimeError(f"{t} no es operable en Alpaca")
    precios = b.precios(set(pesos) | set(posiciones))
    faltan = [t for t in set(pesos) | set(posiciones) if not precios.get(t)]
    if faltan:
        raise RuntimeError(f"Sin precio para: {faltan}")
    objetivo, conservadas = calcular_objetivo(posiciones, pesos, precios, equity)
    if len(objetivo) > cfg.TOP_N:
        raise RuntimeError(f"Objetivo con {len(objetivo)} posiciones (> {cfg.TOP_N})")
    reg["objetivo"] = objetivo
    reg["conservadas_por_banda"] = conservadas
    todos = set(posiciones) | set(objetivo)
    delta = {t: objetivo.get(t, 0) - posiciones.get(t, 0) for t in todos}
    ventas = {t: -d for t, d in delta.items() if d < 0}
    compras = {t: d for t, d in delta.items() if d > 0}
    rotacion = sum(abs(d) * precios[t] for t, d in delta.items()) / equity
    reg["rotacion"] = rotacion
    log(f"Equity ${equity:,.2f} | rotación {rotacion:.1%} | conservadas por banda: {conservadas or 'ninguna'}")
    for t in sorted(todos):
        log(f"  {t:6s} actual {posiciones.get(t, 0):>8g}  objetivo {objetivo.get(t, 0):>8g}  "
            f"precio {precios[t]:>9.2f}  peso obj {pesos.get(t, 0):.1%}")
    if not ventas and not compras:
        log("Nada que operar.")
        return

    # Ventas primero
    if ventas:
        if simulacion:
            for t, q in ventas.items():
                reg["ordenes"].append({"simbolo": t, "lado": "venta", "qty": q, "simulada": True})
        else:
            ids = [b.enviar(t, q, "venta") for t, q in ventas.items()]
            res = b.esperar(ids)
            reg["ordenes"] += list(res.values())
            no_llenas = [r for r in res.values() if r["estado"] != "filled"]
            if no_llenas:
                raise RuntimeError(f"Ventas sin completar, no se compra: {no_llenas}")
            log(f"Ventas completadas: {len(res)}")

    # Compras con el efectivo real disponible
    if compras:
        if simulacion:
            efectivo = b.cuenta()["cash"] + sum(q * precios[t] for t, q in ventas.items())
        else:
            efectivo = b.cuenta()["cash"]
            precios.update(b.precios(set(compras)))
        costo = sum(q * precios[t] for t, q in compras.items()) * (1 + cfg.COLCHON_COMPRAS)
        if costo > efectivo:
            factor = max(0.0, efectivo / costo)
            compras = {t: math.floor(q * factor) for t, q in compras.items()}
            log(f"Efectivo insuficiente: compras escaladas al {factor:.1%}")
        compras = {t: q for t, q in compras.items() if q > 0}
        if simulacion:
            for t, q in compras.items():
                reg["ordenes"].append({"simbolo": t, "lado": "compra", "qty": q, "simulada": True})
        elif compras:
            ids = [b.enviar(t, q, "compra") for t, q in compras.items()]
            res = b.esperar(ids)
            reg["ordenes"] += list(res.values())
            no_llenas = [r for r in res.values() if r["estado"] != "filled"]
            if no_llenas:
                raise RuntimeError(f"Compras sin completar: {no_llenas}")
            log(f"Compras completadas: {len(res)}")
    for o in reg["ordenes"]:
        log(f"  {'SIM ' if o.get('simulada') else ''}{o['lado'].upper():6s} {o['qty']:>8g} {o['simbolo']}"
            + (f" @ {o['precio_promedio']:.2f}" if o.get("precio_promedio") else ""))


def escribir_resumen(reg):
    """Resumen en la página de la corrida de GitHub Actions."""
    ruta = os.environ.get("GITHUB_STEP_SUMMARY")
    if not ruta:
        return
    lineas = [f"## Rebalanceo — {reg['resultado'].upper()}",
              f"Cuenta {'paper' if reg['paper'] else 'REAL'} · {'simulación' if reg['simulacion'] else 'operando'}"
              f" · modo {reg['modo']}", "", "```", *reg["eventos"], "```"]
    with open(ruta, "a", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")


if __name__ == "__main__":
    main()
