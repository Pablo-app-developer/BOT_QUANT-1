"""
metrics.py — Métricas cuantitativas profesionales.

Cada función está documentada con:
  • Qué mide
  • Cómo interpretarla
  • Valores de referencia

Organización:
  1. Métricas de retorno (Sharpe, Sortino, Calmar)
  2. Drawdown
  3. Métricas de trade (Expectancy, PF, WR, Payoff)
  4. Distribución (Skew, Kurtosis)
  5. Riesgo (Risk of Ruin)
  6. Reporte completo
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
# 1. MÉTRICAS DE RETORNO
# ═══════════════════════════════════════════════

def sharpe(ret: pd.Series, rf: float = 0.0, ann: int = 252) -> float:
    """
    Sharpe Ratio = (mean - rf) / std * √periods

    Qué mide: retorno ajustado por riesgo TOTAL (up+down vol).
    Referencia: >1 aceptable, >2 bueno, >3 excelente (sospechoso).
    """
    excess = ret - rf / ann
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(ann))


def sortino(ret: pd.Series, rf: float = 0.0, ann: int = 252) -> float:
    """
    Sortino Ratio = (mean - rf) / downside_std * √periods

    Qué mide: retorno ajustado solo por DOWNSIDE volatility.
    Mejor que Sharpe para estrategias con sesgo positivo.
    Referencia: >1.5 bueno, >2.5 excelente.
    """
    excess = ret - rf / ann
    down = excess[excess < 0]
    if len(down) == 0 or down.std() == 0:
        return 0.0
    return float(excess.mean() / down.std() * np.sqrt(ann))


def calmar(ret: pd.Series, eq: pd.Series, ann: int = 252) -> float:
    """
    Calmar Ratio = Retorno Anualizado / |Max Drawdown|

    Qué mide: cuánto retorno obtienes por cada unidad de drawdown.
    Referencia: >1 bueno, >3 excelente.
    Crítico para FTMO: si DD > 10%, estás fuera.
    """
    ar = ann_return(ret, ann)
    mdd = max_dd(eq)
    if mdd == 0:
        return 0.0
    return float(ar / abs(mdd))


def ann_return(ret: pd.Series, ann: int = 252) -> float:
    """Retorno compuesto anualizado."""
    total = (1 + ret).prod() - 1
    n = len(ret)
    if n == 0:
        return 0.0
    return float((1 + total) ** (ann / n) - 1)


def ann_vol(ret: pd.Series, ann: int = 252) -> float:
    """Volatilidad anualizada (std)."""
    return float(ret.std() * np.sqrt(ann))


# ═══════════════════════════════════════════════
# 2. DRAWDOWN
# ═══════════════════════════════════════════════

def max_dd(eq: pd.Series) -> float:
    """
    Max Drawdown = peor caída pico-a-valle (fracción negativa).

    Qué mide: la PEOR pérdida desde un máximo histórico.
    Referencia FTMO: límite absoluto -10% (diario -5%).
    Ejemplo: -0.08 = caíste 8% desde el pico.
    """
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return float(dd.min())


def dd_series(eq: pd.Series) -> pd.Series:
    """Serie completa de drawdown (fraccional, negativa)."""
    peak = eq.cummax()
    return (eq - peak) / peak


def max_dd_duration(eq: pd.Series) -> int:
    """
    Duración máxima de drawdown (en barras).

    Qué mide: cuánto tiempo estuviste onder water.
    Períodos largos = estrés psicológico + posible régimen desfavorable.
    """
    peak = eq.cummax()
    underwater = eq < peak
    if not underwater.any():
        return 0
    groups = (~underwater).cumsum()
    uw = groups[underwater]
    if len(uw) == 0:
        return 0
    return int(uw.value_counts().max())


# ═══════════════════════════════════════════════
# 3. MÉTRICAS DE TRADE
# ═══════════════════════════════════════════════

def expectancy(pnls: np.ndarray) -> float:
    """
    Expectancy = E[PnL por trade]

    Qué mide: cuánto esperas ganar/perder EN PROMEDIO por trade.
    DEBE ser positivo después de costes para que exista edge.
    Si expectancy < 0 → no hay edge, punto.
    """
    return float(np.mean(pnls)) if len(pnls) > 0 else 0.0


def profit_factor(pnls: np.ndarray) -> float:
    """
    Profit Factor = Σ(ganancias) / |Σ(pérdidas)|

    Qué mide: por cada dólar perdido, cuántos ganaste.
    Referencia: >1.0 rentable, >1.5 bueno, >2.0 fuerte.
    PF < 1.0 = pérdida neta.
    """
    wins = pnls[pnls > 0].sum()
    losses = abs(pnls[pnls < 0].sum())
    if losses == 0:
        return float('inf') if wins > 0 else 0.0
    return float(wins / losses)


def win_rate(pnls: np.ndarray) -> float:
    """
    Win Rate = trades ganadores / total trades

    Qué mide: frecuencia de trades ganadores.
    CUIDADO: winrate alto NO garantiza rentabilidad.
    Importa la COMBINACIÓN winrate × payoff ratio.
    Fórmula del edge: WR × avg_win - (1-WR) × avg_loss > 0
    """
    if len(pnls) == 0:
        return 0.0
    return float(np.sum(pnls > 0) / len(pnls))


def payoff_ratio(pnls: np.ndarray) -> float:
    """
    Payoff Ratio = |avg_win| / |avg_loss|

    Qué mide: ratio entre ganancia media y pérdida media.
    Junto con winrate, define si hay edge aritmético.
    """
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    return float(abs(np.mean(wins) / np.mean(losses)))


# ═══════════════════════════════════════════════
# 4. DISTRIBUCIÓN
# ═══════════════════════════════════════════════

def skewness(ret: pd.Series) -> float:
    """
    Skewness de la distribución de retornos.

    Positivo = cola derecha gorda (wins grandes ocasionales) → deseable.
    Negativo = cola izquierda gorda (losses grandes) → peligroso.
    """
    return float(ret.skew())


def kurtosis(ret: pd.Series) -> float:
    """
    Exceso de kurtosis.

    >0 = colas más gordas que normal → más eventos extremos de lo esperado.
    Alta kurtosis = el riesgo real es MAYOR que lo que el std sugiere.
    """
    return float(ret.kurtosis())


# ═══════════════════════════════════════════════
# 5. RIESGO
# ═══════════════════════════════════════════════

def risk_of_ruin(
    wr: float,
    avg_win: float,
    avg_loss: float,
    risk_per_trade: float = 0.01,
    ruin_pct: float = 0.50,
) -> float:
    """
    Risk of Ruin (aproximación binomial simplificada).

    Qué mide: probabilidad de perder `ruin_pct` del capital.
    Referencia: <1% = muy seguro, <5% = aceptable, >10% = peligroso.

    Lógica: si edge < 0, ruina es 100%.
    Si edge > 0, calcula P(n pérdidas consecutivas hasta ruina).
    """
    if avg_loss == 0 or wr >= 1.0:
        return 0.0
    if wr <= 0.0:
        return 1.0

    edge = wr * avg_win - (1 - wr) * avg_loss
    if edge <= 0:
        return 1.0  # Sin edge → ruina eventual

    n_losses = int(ruin_pct / risk_per_trade)
    return float((1 - wr) ** n_losses)


# ═══════════════════════════════════════════════
# 6. REPORTE COMPLETO
# ═══════════════════════════════════════════════

def full_report(
    eq: pd.Series,
    ret: pd.Series,
    pnls: np.ndarray,
    risk_per_trade: float = 0.01,
) -> dict:
    """Genera dict con todas las métricas."""
    wr = win_rate(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    aw = float(np.mean(wins)) if len(wins) > 0 else 0.0
    al = float(np.mean(np.abs(losses))) if len(losses) > 0 else 0.0

    return {
        # Retorno
        'sharpe': sharpe(ret),
        'sortino': sortino(ret),
        'calmar': calmar(ret, eq),
        'ann_return': ann_return(ret),
        'ann_vol': ann_vol(ret),
        # Drawdown
        'max_dd': max_dd(eq),
        'max_dd_duration': max_dd_duration(eq),
        # Trades
        'n_trades': len(pnls),
        'win_rate': wr,
        'expectancy': expectancy(pnls),
        'profit_factor': profit_factor(pnls),
        'payoff_ratio': payoff_ratio(pnls),
        'avg_win': aw,
        'avg_loss': al,
        # Distribución
        'skewness': skewness(ret),
        'kurtosis': kurtosis(ret),
        # Riesgo
        'risk_of_ruin': risk_of_ruin(wr, aw, al, risk_per_trade),
        # Resumen
        'initial_equity': float(eq.iloc[0]),
        'final_equity': float(eq.iloc[-1]),
        'total_return': float(eq.iloc[-1] / eq.iloc[0] - 1),
    }


def print_report(r: dict) -> None:
    """Imprime reporte formateado."""
    print("\n" + "=" * 55)
    print("    REPORTE DE PERFORMANCE CUANTITATIVO")
    print("=" * 55)

    sections = [
        ("RETORNO", [
            ("Sharpe Ratio",       'sharpe',      '.2f'),
            ("Sortino Ratio",      'sortino',     '.2f'),
            ("Calmar Ratio",       'calmar',      '.2f'),
            ("Retorno Anualizado", 'ann_return',  '.2%'),
            ("Volatilidad Anual",  'ann_vol',     '.2%'),
        ]),
        ("DRAWDOWN", [
            ("Max Drawdown",       'max_dd',          '.2%'),
            ("DD Duration (bars)", 'max_dd_duration',  'd'),
        ]),
        ("TRADES", [
            ("Nº Trades",         'n_trades',       'd'),
            ("Win Rate",          'win_rate',        '.2%'),
            ("Expectancy ($/op)", 'expectancy',      '.2f'),
            ("Profit Factor",     'profit_factor',   '.2f'),
            ("Payoff Ratio",      'payoff_ratio',    '.2f'),
            ("Avg Win ($)",       'avg_win',         '.2f'),
            ("Avg Loss ($)",      'avg_loss',        '.2f'),
        ]),
        ("DISTRIBUCIÓN", [
            ("Skewness",  'skewness',  '.3f'),
            ("Kurtosis",  'kurtosis',  '.3f'),
        ]),
        ("RIESGO", [
            ("Risk of Ruin", 'risk_of_ruin', '.6f'),
        ]),
        ("RESUMEN", [
            ("Equity Inicial",  'initial_equity', ',.2f'),
            ("Equity Final",    'final_equity',   ',.2f'),
            ("Retorno Total",   'total_return',   '.2%'),
        ]),
    ]

    for section, items in sections:
        print(f"\n  ── {section} ──")
        for label, key, fmt in items:
            v = r.get(key, 'N/A')
            if isinstance(v, (int, float)):
                print(f"  {label:<24s} {v:{fmt}}")
            else:
                print(f"  {label:<24s} {v}")

    print("\n" + "=" * 55)
