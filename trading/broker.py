"""Conexión a Alpaca: cuenta, posiciones, calendario, precios y órdenes."""
import time
from datetime import timedelta

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import GetCalendarRequest, MarketOrderRequest

import config as cfg

ESTADOS_FINALES = {"filled", "canceled", "expired", "rejected", "done_for_day", "stopped", "suspended"}


def a_alpaca(ticker):
    """'brk-b' (Yahoo) -> 'BRK.B' (Alpaca)."""
    return ticker.upper().replace("-", ".")


def _valor(x):
    return str(getattr(x, "value", x)).lower()


class Broker:
    def __init__(self):
        if not cfg.API_KEY or not cfg.SECRET_KEY:
            raise RuntimeError("Faltan ALPACA_API_KEY / ALPACA_SECRET_KEY (archivo .env local o GitHub Secrets)")
        self.tc = TradingClient(cfg.API_KEY, cfg.SECRET_KEY, paper=cfg.PAPER)
        self.dc = StockHistoricalDataClient(cfg.API_KEY, cfg.SECRET_KEY)

    # ---------- cuenta ----------
    def cuenta(self):
        a = self.tc.get_account()
        return {"equity": float(a.equity), "cash": float(a.cash), "estado": _valor(a.status),
                "bloqueada": bool(a.trading_blocked or a.account_blocked)}

    def posiciones(self):
        return {p.symbol: float(p.qty) for p in self.tc.get_all_positions() if float(p.qty) != 0}

    # ---------- calendario ----------
    def reloj(self):
        c = self.tc.get_clock()
        return c.timestamp, bool(c.is_open)

    def dias_habiles(self, inicio, fin):
        return [d.date for d in self.tc.get_calendar(GetCalendarRequest(start=inicio, end=fin))]

    def primer_dia_habil_del_mes(self, fecha):
        return self.dias_habiles(fecha.replace(day=1), fecha.replace(day=10))[0]

    def sesion_anterior(self, fecha):
        return self.dias_habiles(fecha - timedelta(days=10), fecha - timedelta(days=1))[-1]

    # ---------- mercado ----------
    def precios(self, simbolos):
        if not simbolos:
            return {}
        r = self.dc.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=list(simbolos), feed=DataFeed.IEX))
        return {s: float(t.price) for s, t in r.items()}

    def operable(self, simbolo):
        try:
            return bool(self.tc.get_asset(simbolo).tradable)
        except Exception:
            return False

    # ---------- órdenes ----------
    def enviar(self, simbolo, qty, lado):
        o = self.tc.submit_order(MarketOrderRequest(
            symbol=simbolo, qty=qty, side=OrderSide.BUY if lado == "compra" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY))
        return str(o.id)

    def esperar(self, ids, timeout=cfg.ESPERA_ORDEN_SEG):
        """Espera a que las órdenes terminen. Devuelve {id: {estado, qty_llenada, precio_promedio, simbolo}}."""
        limite = time.time() + timeout
        res = {}
        while True:
            for i in ids:
                o = self.tc.get_order_by_id(i)
                res[i] = {"simbolo": o.symbol, "lado": _valor(o.side), "estado": _valor(o.status),
                          "qty": float(o.qty or 0), "qty_llenada": float(o.filled_qty or 0),
                          "precio_promedio": float(o.filled_avg_price) if o.filled_avg_price else None}
            if all(r["estado"] in ESTADOS_FINALES for r in res.values()) or time.time() > limite:
                return res
            time.sleep(5)
