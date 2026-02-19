"""
engine.py — Motor de backtest bar-a-bar.

REGLAS CRÍTICAS (anti-sesgo):
  1. Señal en barra[i] se ejecuta en open de barra[i+1]  → NO look-ahead
  2. Costes reales en cada fill (spread + slippage + comisión)
  3. SL/TP chequeados intra-bar (usa high/low de la barra)
  4. Cada trade registra MAE/MFE para análisis posterior
  5. RNG con seed → 100% reproducible

El motor es SIMPLE a propósito: un solo instrumento, una sola posición.
Complejidad extra = bugs ocultos = sesgos invisibles.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.execution_model import (
    ExecConfig,
    Trade,
    fill_price,
    position_size,
)

logger = logging.getLogger(__name__)

# Constante: unidades por lote estándar FX
UNITS_PER_LOT = 100_000


@dataclass
class BacktestResult:
    """Contenedor de resultados del backtest."""
    equity_curve: pd.Series     # Equity bar-a-bar
    trades: list[Trade]         # Lista de trades cerrados
    returns: pd.Series          # Retornos bar-a-bar
    config: ExecConfig
    initial_equity: float
    final_equity: float


def run(
    data: pd.DataFrame,
    signals: pd.Series,
    initial_equity: float = 10_000.0,
    cfg: ExecConfig | None = None,
    sl_pips: float | None = None,
    tp_pips: float | None = None,
    seed: int = 42,
) -> BacktestResult:
    """
    Ejecuta backtest bar-a-bar.

    Parameters
    ----------
    data : DataFrame con columnas open/high/low/close, DatetimeIndex.
    signals : Series alineada con data.
              +1 = abrir/mantener long, -1 = short, 0 = flat/cerrar.
              signals[i] se ejecuta en data[i+1].open.
    initial_equity : Capital inicial en USD.
    cfg : Configuración de ejecución (spread, slippage, etc.)
    sl_pips : Stop-loss en pips desde entry (None = sin SL).
    tp_pips : Take-profit en pips desde entry (None = sin TP).
    seed : Semilla RNG para reproducibilidad del slippage.

    Returns
    -------
    BacktestResult
    """
    if cfg is None:
        cfg = ExecConfig()

    rng = np.random.default_rng(seed)
    signals = signals.reindex(data.index).fillna(0).astype(int)

    n = len(data)
    equity = np.full(n, initial_equity)
    cur_equity = initial_equity

    # Estado de posición
    pos = 0            # Dirección actual: +1, -1, 0
    entry_px = 0.0
    entry_time = None
    entry_bar = 0
    lots = 0.0
    mae_pips = 0.0     # Max Adverse Excursion (pips)
    mfe_pips = 0.0     # Max Favorable Excursion (pips)
    trades: list[Trade] = []

    opens = data['open'].values
    highs = data['high'].values
    lows  = data['low'].values
    times = data.index

    def _close_position(bar_idx: int, exit_raw: float) -> None:
        """Cierra posición actual y registra trade."""
        nonlocal pos, entry_px, lots, cur_equity, mae_pips, mfe_pips

        fp, sp_cost, sl_cost = fill_price(exit_raw, -pos, cfg, rng)
        gross = (fp - entry_px) * pos * lots * UNITS_PER_LOT
        comm = cfg.commission_per_lot * lots
        net = gross - comm

        trades.append(Trade(
            entry_time=entry_time,
            exit_time=times[bar_idx],
            direction=pos,
            entry_price=entry_px,
            exit_price=fp,
            lots=lots,
            pnl_gross=gross,
            pnl_net=net,
            spread_cost=sp_cost * lots * UNITS_PER_LOT,
            slippage_cost=sl_cost * lots * UNITS_PER_LOT,
            commission=comm,
            bars_held=bar_idx - entry_bar,
            mae=mae_pips,
            mfe=mfe_pips,
        ))

        cur_equity += net
        pos = 0
        entry_px = 0.0
        lots = 0.0
        mae_pips = 0.0
        mfe_pips = 0.0

    def _open_position(bar_idx: int, direction: int) -> None:
        """Abre nueva posición."""
        nonlocal pos, entry_px, entry_time, entry_bar, lots, mae_pips, mfe_pips

        raw = opens[bar_idx]

        if sl_pips is not None:
            stop_px = raw - direction * sl_pips * cfg.pip
            lots_calc = position_size(cur_equity, raw, stop_px, cfg)
        else:
            lots_calc = cfg.min_lots

        fp, _, _ = fill_price(raw, direction, cfg, rng)

        pos = direction
        entry_px = fp
        entry_time = times[bar_idx]
        entry_bar = bar_idx
        lots = lots_calc
        mae_pips = 0.0
        mfe_pips = 0.0

    # ── Loop principal ──
    for i in range(1, n):
        sig = signals.iloc[i - 1]  # Señal del bar ANTERIOR

        # ── Chequear SL/TP si hay posición abierta ──
        if pos != 0:
            # Calcular excursión en pips
            if pos == 1:
                fav = (highs[i] - entry_px) / cfg.pip
                adv = (entry_px - lows[i]) / cfg.pip
            else:
                fav = (entry_px - lows[i]) / cfg.pip
                adv = (highs[i] - entry_px) / cfg.pip

            mfe_pips = max(mfe_pips, fav)
            mae_pips = max(mae_pips, adv)

            # Stop-loss
            if sl_pips is not None and adv >= sl_pips:
                if pos == 1:
                    sl_px = entry_px - sl_pips * cfg.pip
                else:
                    sl_px = entry_px + sl_pips * cfg.pip
                _close_position(i, sl_px)

            # Take-profit (solo si SL no cerró)
            if pos != 0 and tp_pips is not None and fav >= tp_pips:
                if pos == 1:
                    tp_px = entry_px + tp_pips * cfg.pip
                else:
                    tp_px = entry_px - tp_pips * cfg.pip
                _close_position(i, tp_px)

        # ── Procesar señal ──
        if pos == 0 and sig != 0:
            # Abrir nueva posición
            _open_position(i, sig)

        elif pos != 0 and sig == 0:
            # Cerrar posición (señal flat)
            _close_position(i, opens[i])

        elif pos != 0 and sig != 0 and sig != pos:
            # Reversa: cerrar y abrir en dirección opuesta
            _close_position(i, opens[i])
            _open_position(i, sig)

        equity[i] = cur_equity

    # ── Construir resultado ──
    eq_series = pd.Series(equity, index=data.index, name='equity')
    ret_series = eq_series.pct_change().fillna(0)

    result = BacktestResult(
        equity_curve=eq_series,
        trades=trades,
        returns=ret_series,
        config=cfg,
        initial_equity=initial_equity,
        final_equity=cur_equity,
    )

    logger.info(f"Backtest: {len(trades)} trades | "
                f"Equity ${initial_equity:,.0f} → ${cur_equity:,.0f}")
    return result
