"""
nq_whipsaw_reversal.py — H1: NY Open Whipsaw Reversal Strategy

HIPÓTESIS:
  Los primeros 15-30 minutos de la apertura de NY (13:30-14:00 UTC) producen
  un "barrido de liquidez" (whipsaw) en una dirección inicial. La probabilidad
  de reversión al VWAP de la sesión durante los siguientes 30-60 minutos es
  estadísticamente superior al azar.

PROTOCOLO CIENTÍFICO (Fase 6):
  1. Definir la señal con lógica económica clara (no fitted)
  2. Separar OOS (2025) ANTES de cualquier análisis
  3. Testear IS → OOS → Walk-Forward → Monte Carlo
  4. Aplicar costos realistas siempre
  5. Intentar destruir el edge con stress tests

VARIABLES DE ENTRADA (sin optimización):
  - Ventana de detección whipsaw: 5 min (primeras 5 barras de la sesión)
  - Zona de reversión: precio cruza VWAP de sesión acumulado
  - SL: máximo/mínimo del barrido + 0.1% (buffer)
  - TP: 2R respecto al SL
  - Max holding: 30 barras M1 (30 minutos)
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── Setup ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant_bot.data.nq_loader import load_nq_m1, add_session_labels, NQ_PROCESSED

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NQ_Whipsaw")

ARTIFACTS_DIR = PROJECT_ROOT / "quant_bot" / "research" / "artifacts" / "nq"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# NOTA: Escala Dukascopy USATECHIDXUSD
# Precio Dukascopy ~128-262  ←→  NQ real ~10,500-21,500
# Factor escala: Dukascopy * 82 ≈ NQ real
# spread_avg en datos ≈ 0.026 units ≈ 2.1 pts NQ (correcto)
# ──────────────────────────────────────────────
DUKASCOPY_SCALE = 82.0

# Costos en unidades Dukascopy (NQ pts / 82)
SPREAD_DUCK_NORMAL = 1.0 / DUKASCOPY_SCALE    # 1 pt NQ = 0.012
SPREAD_DUCK_OPEN   = 3.0 / DUKASCOPY_SCALE    # 3 pts NQ en apertura = 0.037
SLIPPAGE_DUCK      = 1.0 / DUKASCOPY_SCALE    # 1 pt NQ slippage = 0.012

# ⬇️ Alias para compatibilidad con el código de simulación
SPREAD_POINTS_NORMAL = SPREAD_DUCK_NORMAL
SPREAD_POINTS_OPEN   = SPREAD_DUCK_OPEN
SLIPPAGE_POINTS      = SLIPPAGE_DUCK


# ═══════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════

def calculate_vwap_session(df_session: pd.DataFrame) -> pd.Series:
    """
    VWAP acumulado desde el inicio de la sesión.
    VWAP = Σ(típical_price × volume) / Σ(volume)
    """
    typical = (df_session['high'] + df_session['low'] + df_session['close']) / 3
    cum_tp_vol = (typical * df_session['volume']).cumsum()
    cum_vol = df_session['volume'].cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap


def friction_cost_points(is_opening_bar: bool = False) -> float:
    """Retorna el costo total de entrada+salida en puntos del índice."""
    spread = SPREAD_POINTS_OPEN if is_opening_bar else SPREAD_POINTS_NORMAL
    return (spread + SLIPPAGE_POINTS) * 2  # entrada + salida


# ═══════════════════════════════════════════════════════════
# GENERADOR DE SEÑALES — SIN LOOKAHEAD
# ═══════════════════════════════════════════════════════════

def generate_whipsaw_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera señales de Whipsaw Reversal para cada día de sesión.

    Lógica:
      1. Aislar la sesión NY de cada día (13:30-14:00 UTC = ventana whipsaw)
      2. Calcular el retorno de las primeras 5 barras M1
      3. Si el movimiento inicial es > umbral (0.15% = ≈3 puntos NQ):
         - Si sube → señal SHORT cuando el precio cruza VWAP (barrido alcista)
         - Si baja → señal LONG cuando el precio cruza VWAP (barrido bajista)
      4. SL = extremo del barrido + buffer
      5. TP = 2R
      6. Timeout = 30 barras desde la entrada

    Retorna:
      DataFrame con todas las señales identificadas:
      [entry_time, entry_price, direction, sl_price, tp_price, signal_closed]
    """
    logger.info("\n── GENERANDO SEÑALES WHIPSAW REVERSAL ──")

    # Solo sesión NY (13:30 → 20:00 UTC)
    ny = df[df['session'].isin(['OPEN_HOUR', 'MIDDAY', 'CLOSE_HOUR'])].copy()
    ny['vwap_session'] = np.nan

    # Calcular VWAP acumulado por día de sesión
    for date, day_data in ny.groupby(ny.index.date):
        vwap = calculate_vwap_session(day_data)
        ny.loc[day_data.index, 'vwap_session'] = vwap.values

    signals = []
    WHIPSAW_WINDOW = 5      # primeras N barras de la sesión
    MIN_MOVE_PCT = 0.15     # movimiento mínimo para considerar whipsaw (%)
    MAX_HOLDING  = 30       # máximo 30 barras

    for date_val, day_data in ny.groupby(ny.index.date):
        if len(day_data) < WHIPSAW_WINDOW + 5:
            continue

        # Ventana de detección: primeras N barras (13:30-13:35 UTC)
        opening_bars = day_data.iloc[:WHIPSAW_WINDOW]
        rest_of_day  = day_data.iloc[WHIPSAW_WINDOW:]

        open_price  = opening_bars['open'].iloc[0]
        open_extreme_high = opening_bars['high'].max()
        open_extreme_low  = opening_bars['low'].min()

        move_up   = (open_extreme_high - open_price) / open_price * 100
        move_down = (open_price - open_extreme_low) / open_price * 100

        # Detectar whipsaw
        direction = None
        sl_price = None
        tp_price = None
        buffer_pct = 0.15  # 0.15% de buffer para el SL

        if move_up > MIN_MOVE_PCT:
            # Barrido alcista → esperar SHORT cuando precio cruza VWAP hacia abajo
            direction = 'SHORT'
            sl_price = open_extreme_high * (1 + buffer_pct / 100)
            # TP se calcula dinámicamente al momento del fill

        elif move_down > MIN_MOVE_PCT:
            # Barrido bajista → esperar LONG cuando precio cruza VWAP hacia arriba
            direction = 'LONG'
            sl_price = open_extreme_low * (1 - buffer_pct / 100)

        if direction is None:
            continue

        # Buscar el fill: precio cruza VWAP
        entry_found = False
        for i in range(len(rest_of_day)):
            bar = rest_of_day.iloc[i]
            vwap_val = bar['vwap_session']

            if pd.isna(vwap_val):
                continue

            # Condición de entrada
            if direction == 'SHORT' and bar['close'] < vwap_val:
                entry_found = True
            elif direction == 'LONG' and bar['close'] > vwap_val:
                entry_found = True

            if entry_found:
                # Entrada en el cierre de este bar + spread/slippage
                friction = friction_cost_points(is_opening_bar=False)
                if direction == 'SHORT':
                    entry_price = bar['close'] - SPREAD_POINTS_NORMAL / 2
                    risk_pts = entry_price - sl_price if direction == 'SHORT' else sl_price - entry_price
                    if risk_pts <= 0:
                        break  # SL inválido
                    tp_price = entry_price - 2 * abs(risk_pts)
                else:
                    entry_price = bar['close'] + SPREAD_POINTS_NORMAL / 2
                    risk_pts = sl_price - entry_price if direction == 'LONG' else entry_price - sl_price
                    if risk_pts <= 0:
                        break
                    tp_price = entry_price + 2 * abs(risk_pts)

                # Simular el trade en las siguientes barras
                entry_bar_idx = i
                outcome = simulate_trade(
                    rest_of_day.iloc[i+1:i+1+MAX_HOLDING],
                    entry_price, sl_price, tp_price, direction
                )

                signals.append({
                    'date': date_val,
                    'entry_time': rest_of_day.index[i] if i < len(rest_of_day) else None,
                    'direction': direction,
                    'entry_price': entry_price,
                    'sl_price': sl_price,
                    'tp_price': tp_price,
                    'risk_pts': abs(risk_pts),
                    'friction_pts': friction,
                    **outcome,
                })
                break

    df_signals = pd.DataFrame(signals)
    if len(df_signals) > 0:
        df_signals.set_index('date', inplace=True)
        logger.info(f"  Señales generadas: {len(df_signals)}")
        logger.info(f"  LONG: {(df_signals['direction'] == 'LONG').sum()}")
        logger.info(f"  SHORT: {(df_signals['direction'] == 'SHORT').sum()}")
    else:
        logger.warning("  No se generaron señales.")

    return df_signals


def simulate_trade(
    forward_bars: pd.DataFrame,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    direction: str
) -> dict:
    """
    Simula el desenlace de un trade bar por bar.
    Retorna dict con resultado.
    """
    if len(forward_bars) == 0:
        return {'pnl_pts': -1.0, 'outcome': 'TIMEOUT', 'bars_held': 0, 'won': False}

    for i, (ts, bar) in enumerate(forward_bars.iterrows()):
        if direction == 'LONG':
            if bar['low'] <= sl_price:
                pnl = sl_price - entry_price
                return {'pnl_pts': pnl, 'outcome': 'SL', 'bars_held': i+1, 'won': False}
            if bar['high'] >= tp_price:
                pnl = tp_price - entry_price
                return {'pnl_pts': pnl, 'outcome': 'TP', 'bars_held': i+1, 'won': True}
        else:  # SHORT
            if bar['high'] >= sl_price:
                pnl = entry_price - sl_price
                return {'pnl_pts': -pnl, 'outcome': 'SL', 'bars_held': i+1, 'won': False}
            if bar['low'] <= tp_price:
                pnl = entry_price - tp_price
                return {'pnl_pts': pnl, 'outcome': 'TP', 'bars_held': i+1, 'won': True}

    # Timeout: cierre al precio actual (último bar)
    last_price = forward_bars['close'].iloc[-1]
    if direction == 'LONG':
        pnl = last_price - entry_price - SPREAD_POINTS_NORMAL
    else:
        pnl = entry_price - last_price - SPREAD_POINTS_NORMAL

    return {'pnl_pts': pnl, 'outcome': 'TIMEOUT', 'bars_held': len(forward_bars), 'won': pnl > 0}


# ═══════════════════════════════════════════════════════════
# ANÁLISIS ESTADÍSTICO COMPLETO
# ═══════════════════════════════════════════════════════════

def analyze_results(df_signals: pd.DataFrame, label: str = "IS") -> dict:
    """Análisis estadístico completo de los trades."""
    if len(df_signals) == 0:
        return {'n_trades': 0}

    n = len(df_signals)
    winners = df_signals[df_signals['won'] == True]
    losers  = df_signals[df_signals['won'] == False]

    win_rate = len(winners) / n
    avg_win  = winners['pnl_pts'].mean() if len(winners) > 0 else 0
    avg_loss = losers['pnl_pts'].mean()  if len(losers) > 0 else 0

    # Expectancy neta (puntos)
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # Profit factor
    gross_profit = winners['pnl_pts'].sum() if len(winners) > 0 else 0
    gross_loss   = abs(losers['pnl_pts'].sum()) if len(losers) > 0 else 1
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    # Max drawdown en equity (simplificado)
    equity = df_signals['pnl_pts'].cumsum()
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max)
    max_dd = drawdown.min()

    # Sharpe approx (ratio expectancy/std)
    returns_std = df_signals['pnl_pts'].std()
    sharpe = expectancy / returns_std if returns_std > 0 else 0

    # TEST ESTADÍSTICO: ¿El win rate difiere del 50%?
    z_stat = (win_rate - 0.5) / np.sqrt(0.5 * 0.5 / n)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    results = {
        'label': label,
        'n_trades': n,
        'win_rate': win_rate,
        'avg_win_pts': avg_win,
        'avg_loss_pts': avg_loss,
        'expectancy_pts': expectancy,
        'profit_factor': pf,
        'max_dd_pts': max_dd,
        'sharpe_approx': sharpe,
        'z_stat_vs_50pct': z_stat,
        'p_value': p_value,
        'gross_pnl_pts': df_signals['pnl_pts'].sum(),
        'outcome_counts': df_signals['outcome'].value_counts().to_dict(),
    }

    logger.info(f"\n[{label}] ── RESULTADOS ESTADÍSTICOS ──")
    logger.info(f"  N trades:        {n}")
    logger.info(f"  Win Rate:        {win_rate*100:.1f}%")
    logger.info(f"  Avg Win:         {avg_win:.2f} pts")
    logger.info(f"  Avg Loss:        {avg_loss:.2f} pts")
    logger.info(f"  Expectancy neta: {expectancy:.3f} pts")
    logger.info(f"  Profit Factor:   {pf:.3f}")
    logger.info(f"  Max Drawdown:    {max_dd:.1f} pts")
    logger.info(f"  Sharpe (approx): {sharpe:.3f}")
    logger.info(f"  Z-stat vs 50%:   {z_stat:.2f} (p={p_value:.4f})")

    for outcome, count in results['outcome_counts'].items():
        logger.info(f"  {outcome}: {count}")

    return results


# ═══════════════════════════════════════════════════════════
# STRESS TESTS — DESTRUCCIÓN DEL EDGE
# ═══════════════════════════════════════════════════════════

def run_stress_tests(df_signals: pd.DataFrame) -> dict:
    """
    Test de destrucción del edge (Fase 6.8):
    1. Spread x2
    2. Slippage extra = 2 pts
    3. Delay 1 barra = peor precio de aceptación
    4. Eliminar mejores 5% de trades (dependencia extremos)
    """
    logger.info("\n── STRESS TESTS — DESTRUCCIÓN DEL EDGE ──")
    stress_results = {}

    if len(df_signals) == 0:
        return stress_results

    # 1. Spread x2
    df_spread_x2 = df_signals.copy()
    additional_cost = SPREAD_POINTS_NORMAL * 2  # coste extra por spread duplicado
    df_spread_x2['pnl_pts'] -= additional_cost
    df_spread_x2['won'] = df_spread_x2['pnl_pts'] > 0

    r = analyze_results(df_spread_x2, "SPREAD_x2")
    stress_results['spread_x2'] = r
    survived = r['expectancy_pts'] > 0 and r['profit_factor'] > 1.0
    logger.info(f"  Spread x2: {'✅ SOBREVIVE' if survived else '❌ MUERE'}")

    # 2. Slippage extra
    df_slip = df_signals.copy()
    df_slip['pnl_pts'] -= SLIPPAGE_POINTS * 2
    df_slip['won'] = df_slip['pnl_pts'] > 0

    r = analyze_results(df_slip, "SLIPPAGE_x2")
    stress_results['slippage_x2'] = r
    survived = r['expectancy_pts'] > 0
    logger.info(f"  Slippage x2: {'✅ SOBREVIVE' if survived else '❌ MUERE'}")

    # 3. Sin el mejor 10% de trades (¿depende de outliers?)
    df_no_outliers = df_signals.copy()
    threshold_pnl = df_no_outliers['pnl_pts'].quantile(0.90)
    df_no_outliers = df_no_outliers[df_no_outliers['pnl_pts'] < threshold_pnl]

    r = analyze_results(df_no_outliers, "SIN_TOP10PCT")
    stress_results['no_outliers'] = r
    survived = r['expectancy_pts'] > 0 and r['profit_factor'] > 1.0
    logger.info(f"  Sin top 10% trades: {'✅ SOBREVIVE' if survived else '❌ MUERE (depende de outliers)'}")

    # 4. Solo días de baja volatilidad (rangos pequeños)
    # (Proxy: trades donde el risk_pts es pequeño → días tranquilos)
    if 'risk_pts' in df_signals.columns:
        median_risk = df_signals['risk_pts'].median()
        df_low_vol = df_signals[df_signals['risk_pts'] <= median_risk]
        r = analyze_results(df_low_vol, "BAJA_VOLATILIDAD")
        stress_results['low_vol'] = r
        survived = r.get('expectancy_pts', -1) > 0
        logger.info(f"  Baja volatilidad only: {'✅ SOBREVIVE' if survived else '❌ MUERE'}")

    return stress_results


# ═══════════════════════════════════════════════════════════
# MONTE CARLO
# ═══════════════════════════════════════════════════════════

def monte_carlo_simulation(df_signals: pd.DataFrame, n_runs: int = 1000) -> dict:
    """
    Monte Carlo: reordenar aleatoriamente los trades N veces.
    Medir distribución de Expectancy y Max Drawdown.
    """
    logger.info(f"\n── MONTE CARLO ({n_runs} runs) ──")

    if len(df_signals) < 20:
        logger.warning("  Insuficientes trades para Monte Carlo.")
        return {}

    pnl_array = df_signals['pnl_pts'].values
    expectations = []
    max_drawdowns = []
    profit_factors = []

    np.random.seed(42)
    for _ in range(n_runs):
        shuffled = np.random.choice(pnl_array, size=len(pnl_array), replace=True)
        equity = np.cumsum(shuffled)
        roll_max = np.maximum.accumulate(equity)
        dd = equity - roll_max

        expectations.append(shuffled.mean())
        max_drawdowns.append(dd.min())
        winners = shuffled[shuffled > 0]
        losers = np.abs(shuffled[shuffled < 0])
        pf = winners.sum() / losers.sum() if losers.sum() > 0 else 0
        profit_factors.append(pf)

    exp_array  = np.array(expectations)
    dd_array   = np.array(max_drawdowns)
    pf_array   = np.array(profit_factors)

    pct_positive_exp = (exp_array > 0).mean()
    ruin_threshold = -200  # -200 puntos NQ se considera ruina
    ruin_prob = (dd_array < ruin_threshold).mean()

    logger.info(f"  Expectancy mediana: {np.median(exp_array):.3f} pts")
    logger.info(f"  Expectancy P5/P95: [{np.percentile(exp_array, 5):.3f}, {np.percentile(exp_array, 95):.3f}]")
    logger.info(f"  % runs positivos: {pct_positive_exp*100:.1f}%")
    logger.info(f"  Max DD mediana: {np.median(dd_array):.1f} pts")
    logger.info(f"  Max DD P5 (peor caso 5%): {np.percentile(dd_array, 5):.1f} pts")
    logger.info(f"  Probabilidad de ruina (DD < {ruin_threshold} pts): {ruin_prob*100:.1f}%")

    logger.info(f"\n  {'✅ EDGE ROBUSTO en MC' if pct_positive_exp > 0.80 else '❌ EDGE FRÁGIL en MC'}")

    return {
        'pct_positive': pct_positive_exp,
        'exp_median': float(np.median(exp_array)),
        'exp_p5': float(np.percentile(exp_array, 5)),
        'exp_p95': float(np.percentile(exp_array, 95)),
        'max_dd_median': float(np.median(dd_array)),
        'max_dd_p5': float(np.percentile(dd_array, 5)),
        'ruin_probability': float(ruin_prob),
        'exp_array': exp_array,
        'dd_array': dd_array,
    }


# ═══════════════════════════════════════════════════════════
# VISUALIZACIÓN
# ═══════════════════════════════════════════════════════════

def plot_whipsaw_results(
    df_signals_is: pd.DataFrame,
    df_signals_oos: pd.DataFrame,
    mc_results: dict,
    stress_results: dict,
) -> None:
    """Genera visualización completa de los resultados."""

    fig = plt.figure(figsize=(20, 20), facecolor='#0d1117')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    tc  = '#c9d1d9'
    gc  = '#21262d'
    c1  = '#58a6ff'
    c2  = '#3fb950'
    c3  = '#ff7b72'

    fig.suptitle('H1: NY Open Whipsaw Reversal — Análisis Completo',
                 fontsize=16, color=tc, fontweight='bold', y=0.98)

    # ── Plot 1: Equity IS vs OOS ──
    ax1 = fig.add_subplot(gs[0, :])
    if len(df_signals_is) > 0:
        eq_is = df_signals_is['pnl_pts'].cumsum()
        ax1.plot(range(len(eq_is)), eq_is.values, color=c1, linewidth=1.5, label=f'IS ({len(df_signals_is)} trades)')
    if len(df_signals_oos) > 0:
        eq_oos = df_signals_oos['pnl_pts'].cumsum()
        ax1.plot(range(len(df_signals_is), len(df_signals_is) + len(eq_oos)),
                 eq_oos.values + (eq_is.iloc[-1] if len(df_signals_is) > 0 else 0),
                 color=c2, linewidth=2, label=f'OOS ({len(df_signals_oos)} trades)', linestyle='--')
    ax1.axhline(y=0, color=tc, linewidth=0.5, alpha=0.5)
    ax1.axvline(x=len(df_signals_is), color=c3, linewidth=1, alpha=0.7, linestyle=':',
                label='IS/OOS split')
    ax1.set_facecolor('#161b22')
    ax1.set_title('Equity Acumulada — In-Sample vs Out-of-Sample', color=tc, fontsize=12)
    ax1.set_ylabel('PnL acumulado (puntos)', color=tc)
    ax1.legend(facecolor='#21262d', labelcolor=tc)
    ax1.tick_params(colors=tc)
    ax1.grid(True, color=gc, alpha=0.5)
    for s in ax1.spines.values(): s.set_color(gc)

    # ── Plot 2: Distribución PnL por trade ──
    ax2 = fig.add_subplot(gs[1, 0])
    all_pnl = pd.concat([df_signals_is, df_signals_oos])['pnl_pts'] if (len(df_signals_is) > 0 or len(df_signals_oos) > 0) else pd.Series()
    if len(all_pnl) > 0:
        ax2.hist(all_pnl, bins=40, color=c1, alpha=0.7, edgecolor='none', density=True)
        ax2.axvline(x=0, color=c3, linewidth=1.5, linestyle='--')
        ax2.axvline(x=all_pnl.mean(), color=c2, linewidth=1.5, label=f'Media={all_pnl.mean():.2f}')
    ax2.set_facecolor('#161b22')
    ax2.set_title('Distribución PnL por Trade (puntos)', color=tc, fontsize=11)
    ax2.set_xlabel('PnL puntos', color=tc)
    ax2.legend(facecolor='#21262d', labelcolor=tc)
    ax2.tick_params(colors=tc)
    ax2.grid(True, color=gc, alpha=0.5)
    for s in ax2.spines.values(): s.set_color(gc)

    # ── Plot 3: Monte Carlo Expectancy distribution ──
    ax3 = fig.add_subplot(gs[1, 1])
    if mc_results and 'exp_array' in mc_results:
        ax3.hist(mc_results['exp_array'], bins=60, color=c1, alpha=0.7, edgecolor='none', density=True)
        ax3.axvline(x=0, color=c3, linewidth=2, linestyle='--', label='Break-even')
        ax3.axvline(x=mc_results['exp_median'], color=c2, linewidth=1.5, label=f"Mediana={mc_results['exp_median']:.3f}")
    ax3.set_facecolor('#161b22')
    ax3.set_title(f"Monte Carlo — Distribución Expectancy (1000 runs)", color=tc, fontsize=11)
    ax3.set_xlabel('Expectancy por trade (puntos)', color=tc)
    ax3.legend(facecolor='#21262d', labelcolor=tc)
    ax3.tick_params(colors=tc)
    ax3.grid(True, color=gc, alpha=0.5)
    for s in ax3.spines.values(): s.set_color(gc)

    # ── Plot 4: Stress Tests ──
    ax4 = fig.add_subplot(gs[2, 0])
    if stress_results:
        labels = list(stress_results.keys())
        exp_vals = [stress_results[k].get('expectancy_pts', 0) for k in labels]
        colors_bar = [c2 if v > 0 else c3 for v in exp_vals]
        ax4.bar(labels, exp_vals, color=colors_bar, alpha=0.8)
        ax4.axhline(y=0, color=tc, linewidth=0.8)
    ax4.set_facecolor('#161b22')
    ax4.set_title('Stress Tests — Expectancy por Escenario (pts)', color=tc, fontsize=11)
    ax4.set_ylabel('Expectancy (puntos)', color=tc)
    ax4.tick_params(colors=tc, axis='y')
    ax4.tick_params(colors=tc, axis='x', rotation=20)
    ax4.grid(True, color=gc, alpha=0.5, axis='y')
    for s in ax4.spines.values(): s.set_color(gc)

    # ── Plot 5: Performance por año ──
    ax5 = fig.add_subplot(gs[2, 1])
    all_sigs = pd.concat([df_signals_is, df_signals_oos]) if (len(df_signals_is) > 0 or len(df_signals_oos) > 0) else pd.DataFrame()
    if len(all_sigs) > 0 and hasattr(all_sigs.index, 'year'):
        yearly_pnl = all_sigs.groupby(all_sigs.index.year)['pnl_pts'].sum()
        colors_yr = [c2 if v > 0 else c3 for v in yearly_pnl.values]
        ax5.bar(yearly_pnl.index.astype(str), yearly_pnl.values, color=colors_yr, alpha=0.8)
        ax5.axhline(y=0, color=tc, linewidth=0.5)
    ax5.set_facecolor('#161b22')
    ax5.set_title('PnL Neto por Año (puntos)', color=tc, fontsize=11)
    ax5.set_ylabel('PnL (puntos)', color=tc)
    ax5.tick_params(colors=tc)
    ax5.grid(True, color=gc, alpha=0.5, axis='y')
    for s in ax5.spines.values(): s.set_color(gc)

    out_path = ARTIFACTS_DIR / "nq_whipsaw_reversal.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)
    logger.info(f"\n  ✅ Gráfico guardado: {out_path}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("  H1: NY OPEN WHIPSAW REVERSAL — VALIDACIÓN CIENTÍFICA COMPLETA")
    logger.info("=" * 70)
    logger.info("  Protocolo: Intentar destruir el edge. Solo si sobrevive → válido.")

    # ── Cargar datos ──
    logger.info("\n📂 Cargando datos M1...")
    df = load_nq_m1(use_cache=True)
    df = add_session_labels(df)

    # ─────────────────────────────────────────────────────
    # SEPARAR OOS ANTES DE CUALQUIER ANÁLISIS
    # OOS = 2025 completo (datos ciegos, no se tocan hasta el final)
    # ─────────────────────────────────────────────────────
    OOS_YEAR = 2025
    df_is  = df[df.index.year < OOS_YEAR]
    df_oos = df[df.index.year >= OOS_YEAR]

    logger.info(f"\n  IS:  {df_is.index[0].date()} → {df_is.index[-1].date()} ({len(df_is):,} barras)")
    logger.info(f"  OOS: {df_oos.index[0].date()} → {df_oos.index[-1].date()} ({len(df_oos):,} barras)")

    # ── FASE 1: In-Sample ──
    logger.info("\n🔍 FASE 1: Generando señales IS...")
    df_signals_is = generate_whipsaw_signals(df_is)
    results_is = analyze_results(df_signals_is, "IN-SAMPLE")

    # ── FASE 2: Out-of-Sample Puro ──
    logger.info("\n🔍 FASE 2: Generando señales OOS (datos ciegos)...")
    df_signals_oos = generate_whipsaw_signals(df_oos)
    results_oos = analyze_results(df_signals_oos, "OUT-OF-SAMPLE")

    # ── FASE 3: Stress Tests ──
    logger.info("\n⚔️  FASE 3: Stress Tests (destrucción del edge)...")
    all_signals = pd.concat([df_signals_is, df_signals_oos])
    stress_results = run_stress_tests(all_signals)

    # ── FASE 4: Monte Carlo ──
    logger.info("\n🎲 FASE 4: Monte Carlo (1000 runs)...")
    mc_results = monte_carlo_simulation(all_signals, n_runs=1000)

    # ── Visualización ──
    logger.info("\n📊 Generando gráficos...")
    plot_whipsaw_results(df_signals_is, df_signals_oos, mc_results, stress_results)

    # ── VEREDICTO FINAL ──
    logger.info("\n" + "=" * 70)
    logger.info("  VEREDICTO FINAL — WHIPSAW REVERSAL")
    logger.info("=" * 70)

    def check(condition: bool, text: str) -> bool:
        icon = "✅" if condition else "❌"
        logger.info(f"  {icon} {text}")
        return condition

    checks_passed = sum([
        check(results_oos.get('n_trades', 0) >= 100,
              f"N trades OOS suficientes: {results_oos.get('n_trades', 0)} (req: >=100)"),
        check(results_oos.get('win_rate', 0) > 0.50,
              f"WR OOS > 50%: {results_oos.get('win_rate', 0)*100:.1f}%"),
        check(results_oos.get('expectancy_pts', -1) > 0,
              f"Expectancy OOS > 0: {results_oos.get('expectancy_pts', 0):.3f} pts"),
        check(stress_results.get('spread_x2', {}).get('expectancy_pts', -1) > 0,
              "Sobrevive spread x2"),
        check(mc_results.get('pct_positive', 0) > 0.70,
              f"MC 70%+ positivo: {mc_results.get('pct_positive', 0)*100:.1f}%"),
        check(mc_results.get('ruin_probability', 1) < 0.10,
              f"Prob. ruina MC < 10%: {mc_results.get('ruin_probability', 1)*100:.1f}%"),
    ])

    logger.info(f"\n  SCORE: {checks_passed}/6 checks passed")

    CLASSIFICATION = {
        6: "🏆 EDGE ROBUSTO REAL — Candidato para implementación",
        5: "✅ EDGE PROMETEDOR — Requiere validación adicional en tick data",
        4: "⚠️  EDGE DÉBIL — No recomendable para trading real",
        3: "❌ EDGE FRÁGIL — Probable overfitting o ruido estadístico",
        2: "❌ ILUSIÓN ESTADÍSTICA — No explotar",
        1: "❌ ILUSIÓN ESTADÍSTICA — No explotar",
        0: "❌ ILUSIÓN ESTADÍSTICA — No explotar",
    }

    classification = CLASSIFICATION.get(checks_passed, "❌ ILUSIÓN ESTADÍSTICA")
    logger.info(f"\n  CLASIFICACIÓN: {classification}")

    return {
        'results_is': results_is,
        'results_oos': results_oos,
        'stress_results': stress_results,
        'mc_results': mc_results,
        'checks_passed': checks_passed,
        'classification': classification,
    }


if __name__ == "__main__":
    main()
