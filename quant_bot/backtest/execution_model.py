"""
execution_model.py — Modelo de ejecución realista.

Principios:
  • Señal en barra[i] → ejecución en open de barra[i+1]  (sin look-ahead)
  • Spread modelado como coste fijo aplicado al fill price
  • Slippage aleatorio uniforme dentro de un rango configurable
  • Position sizing por fracción fija de riesgo (% equity)

Este módulo modela FILLS, no un order book.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ExecConfig:
    """Parámetros de ejecución."""
    spread_pips: float = 1.0         # Spread típico ECN para EURUSD
    slippage_pips: float = 0.5       # Slippage máximo por fill
    pip: float = 0.0001              # 1 pip para EURUSD
    commission_per_lot: float = 0.0  # Comisión por lote (USD)
    risk_pct: float = 0.01           # 1% riesgo por trade
    max_lots: float = 1.0
    min_lots: float = 0.01


@dataclass
class Trade:
    """Registro completo de un trade cerrado."""
    entry_time: object              # pd.Timestamp
    exit_time: object
    direction: int                  # +1 long, -1 short
    entry_price: float
    exit_price: float
    lots: float
    pnl_gross: float = 0.0         # P&L bruto (USD)
    pnl_net: float = 0.0           # Neto (después de costes)
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    commission: float = 0.0
    bars_held: int = 0
    mae: float = 0.0               # Max Adverse Excursion (pips)
    mfe: float = 0.0               # Max Favorable Excursion (pips)


def fill_price(
    raw_price: float,
    direction: int,
    cfg: ExecConfig,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """
    Calcula precio de fill realista.

    LONG  → fill = raw + half_spread + slippage  (compras en ask)
    SHORT → fill = raw - half_spread - slippage  (vendes en bid)

    Returns: (fill_price, spread_cost_per_unit, slippage_per_unit)
    """
    half_spread = (cfg.spread_pips * cfg.pip) / 2.0
    slip = rng.uniform(0, cfg.slippage_pips * cfg.pip)

    if direction == 1:  # BUY
        fp = raw_price + half_spread + slip
    else:               # SELL
        fp = raw_price - half_spread - slip

    return fp, cfg.spread_pips * cfg.pip, slip


def position_size(
    equity: float,
    entry: float,
    stop: float,
    cfg: ExecConfig,
) -> float:
    """
    Position sizing por fracción fija de riesgo.

    lots = (equity * risk_pct) / (stop_distance_pips * pip_value_per_lot)

    Para EURUSD standard lot: 1 pip ≈ $10.
    """
    risk_usd = equity * cfg.risk_pct
    stop_dist = abs(entry - stop)

    if stop_dist < cfg.pip:
        return cfg.min_lots

    stop_pips = stop_dist / cfg.pip
    pip_value = 10.0  # $10/pip/lot para EURUSD (aprox)

    lots = risk_usd / (stop_pips * pip_value)
    return round(max(cfg.min_lots, min(cfg.max_lots, lots)), 2)
