"""
nq_h3_execution.py — Fase 6.6: Validación de Ejecución Realista H3v2

PREGUNTA CRÍTICA (Fase 6.7):
  ¿El simulador asume ejecución perfecta?
  → Si sí → edge inválido (teórico)

TESTS:
  1. Entry timing: CLOSE 14:29 vs OPEN 14:30 vs OPEN 14:31 (latencias)
  2. Spread real observado en los datos a la hora de entrada
  3. Simulación de latencia 0 / 1 / 2 / 5 min
  4. Monte Carlo profundo sobre H3v2 filtrado
  5. Bootstrap IC del Sharpe OOS
  6. Poder estadístico: ¿cuántos trades necesitamos para p<0.05?
  7. Impacto de slippage variable vs fijo
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS_DIR = PROJECT_ROOT / "quant_bot" / "research" / "artifacts" / "nq"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ARTIFACTS_DIR / "h3_execution.log"),
    ]
)
logger = logging.getLogger("H3_Execution")

DUKASCOPY_SCALE = 82.0
OOS_YEAR = 2025


# ════════════════════════════════════════════════════
# CONSTRUCCIÓN CON PRECIOS GRANULARES DE ENTRADA
# ════════════════════════════════════════════════════

def build_execution_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada día construye señal H3v2 con múltiples puntos de entrada:
      - entry_close:  cierre del último bar de OPEN_HOUR (14:29 UTC)
      - entry_1m_lag: apertura del primer bar de MIDDAY  (14:30 UTC, +1 min)
      - entry_2m_lag: apertura del segundo bar de MIDDAY (14:31 UTC, +2 min)
      - entry_5m_lag: precio 5 min después de señal     (14:34 UTC, +5 min)
    Simulamos así latencias de 0/60/120/300 segundos.
    """
    ny = df[df['session'].isin(['OPEN_HOUR', 'MIDDAY', 'CLOSE_HOUR'])].copy()
    records = []

    for date, group in ny.groupby(ny.index.date):
        oh   = group[group['session'] == 'OPEN_HOUR']
        post = group[group['session'].isin(['MIDDAY', 'CLOSE_HOUR'])]
        if len(oh) < 30 or len(post) < 10:
            continue

        oh_open  = oh['open'].iloc[0]
        oh_close = oh['close'].iloc[-1]
        if oh_open <= 0:
            continue

        first_hour_ret = (oh_close - oh_open) / oh_open
        oh_atr = (oh['high'].max() - oh['low'].min()) / oh_open
        directionality = abs(first_hour_ret) / oh_atr if oh_atr > 0 else 0

        # Spread real en el momento de la señal (últimos 5 bars de OPEN_HOUR)
        real_spread_at_signal = oh['spread_avg'].tail(5).mean()

        # Puntos de entrada con distintas latencias
        entry_close = oh_close  # ideal: fill en cierre exacto

        post_bars = post.reset_index()
        entry_1m  = post_bars['open'].iloc[0]  if len(post_bars) > 0 else np.nan
        entry_2m  = post_bars['open'].iloc[1]  if len(post_bars) > 1 else np.nan
        entry_5m  = post_bars['open'].iloc[4]  if len(post_bars) > 4 else np.nan
        entry_10m = post_bars['open'].iloc[9]  if len(post_bars) > 9 else np.nan

        # Exit: cierre de NY (~20:00 UTC)
        exit_price = post['close'].iloc[-1]

        # Prior day return (necesitamos calcularlo después con shift)
        records.append({
            'date':            pd.Timestamp(date, tz='UTC'),
            'year':            date.year,
            'dow':             date.weekday(),
            'first_hour_ret':  float(first_hour_ret),
            'oh_atr':          float(oh_atr),
            'directionality':  float(directionality),
            'real_spread':     float(real_spread_at_signal),
            'entry_close':     float(entry_close),
            'entry_1m':        float(entry_1m)  if not np.isnan(entry_1m)  else np.nan,
            'entry_2m':        float(entry_2m)  if not np.isnan(entry_2m)  else np.nan,
            'entry_5m':        float(entry_5m)  if not np.isnan(entry_5m)  else np.nan,
            'entry_10m':       float(entry_10m) if not np.isnan(entry_10m) else np.nan,
            'exit_price':      float(exit_price),
            'day_open':        float(oh_open),
        })

    df_out = pd.DataFrame(records).set_index('date')
    df_out['day_return_full'] = (df_out['exit_price'] - df_out['day_open']) / df_out['day_open']
    df_out['prior_1d_ret']   = df_out['day_return_full'].shift(1)
    df_out['prior_bearish']  = df_out['prior_1d_ret'] < -0.001

    # Calcular retornos con cada punto de entrada
    for col in ['entry_close', 'entry_1m', 'entry_2m', 'entry_5m', 'entry_10m']:
        valid = df_out[col].notna() & (df_out[col] > 0)
        df_out.loc[valid, f'ret_{col}'] = (
            (df_out.loc[valid, 'exit_price'] - df_out.loc[valid, col])
            / df_out.loc[valid, col]
        )

    # Vol rolling
    df_out['vol_10d'] = df_out['oh_atr'].rolling(10).mean()
    df_out['high_vol'] = df_out['oh_atr'] > df_out['vol_10d']

    return df_out.dropna(subset=['prior_1d_ret', 'ret_entry_close', 'first_hour_ret'])


def _backtest(rets: np.ndarray, cost_pct: float = 0.0, label: str = "") -> dict:
    """Backtest rápido."""
    rn = rets - cost_pct
    if len(rn) < 5 or rn.std() == 0:
        return {'n': len(rn), 'sharpe': 0.0, 'annual': 0.0, 'pvalue': 1.0,
                'win_rate': float((rn > 0).mean()), 'max_dd': 0.0}
    sh  = (rn.mean() / rn.std()) * np.sqrt(252)
    eq  = np.cumprod(1 + rn)
    ann = eq[-1] ** (252 / len(rn)) - 1
    pk  = np.maximum.accumulate(eq)
    dd  = ((eq - pk) / pk).min()
    wr  = (rn > 0).mean()
    _, p = stats.ttest_1samp(rn, 0)
    return {'n': len(rn), 'sharpe': float(sh), 'annual': float(ann),
            'pvalue': float(p), 'win_rate': float(wr), 'max_dd': float(dd)}


# ════════════════════════════════════════════════════
# 1. TEST DE LATENCIA DE ENTRADA
# ════════════════════════════════════════════════════

def entry_latency_test(df: pd.DataFrame, cost_pts: float = 2.0) -> pd.DataFrame:
    """
    ¿Qué tan sensible es el edge al momento exacto de entrada?
    Un edge real debe ser robusto a 1-2 min de latencia.
    """
    logger.info("\n" + "═"*68)
    logger.info("  1. TEST DE LATENCIA DE ENTRADA (H3v2 filtrado)")
    logger.info("═"*68)
    logger.info("  Filtro activo: Previo bajista (>0.1%) + |1H| > 0.3%")

    # Filtro H3v2
    filt = (df['prior_bearish']) & (np.abs(df['first_hour_ret']) > 0.003)
    df_f = df[filt].copy()

    entry_cols = {
        'CLOSE 14:29 (ideal)':    'ret_entry_close',
        'OPEN  14:30 (+1 min)':   'ret_entry_1m',
        'OPEN  14:31 (+2 min)':   'ret_entry_2m',
        'OPEN  14:34 (+5 min)':   'ret_entry_5m',
        'OPEN  14:39 (+10 min)':  'ret_entry_10m',
    }

    rows = []
    for label, col in entry_cols.items():
        if col not in df_f.columns:
            continue
        sub   = df_f.dropna(subset=[col]).copy()
        sig   = np.sign(sub['first_hour_ret'].values)
        gross = sub[col].values * sig
        cost  = cost_pts / DUKASCOPY_SCALE / sub['entry_close'].mean()
        r     = _backtest(gross, cost_pct=cost, label=label)
        icon  = "✅" if r['sharpe'] > 1.0 else "⚠️" if r['sharpe'] > 0 else "❌"
        logger.info(f"  {icon} {label}: n={r['n']:4d}  Sharpe={r['sharpe']:.3f}"
                    f"  Ann={r['annual']*100:.1f}%  WR={r['win_rate']*100:.1f}%  p={r['pvalue']:.4f}")
        rows.append({'entry': label, **r})

    df_lat = pd.DataFrame(rows)

    # Degradación por minuto de latencia
    if len(df_lat) > 1:
        sh_0 = df_lat.iloc[0]['sharpe']
        sh_2 = df_lat.iloc[2]['sharpe'] if len(df_lat) > 2 else sh_0
        logger.info(f"\n  Degradación Sharpe por 2 min latencia: {sh_0:.3f} → {sh_2:.3f}"
                    f" ({(sh_2-sh_0)/sh_0*100:+.1f}%)")
        if sh_2 > 0.8 * sh_0:
            logger.info("  ✅ Edge ROBUSTO a latencia de 2 min")
        else:
            logger.info("  ❌ Edge SENSIBLE a latencia — requiere ejecución rápida")

    return df_lat


# ════════════════════════════════════════════════════
# 2. SPREAD REAL EN EL MOMENTO DE LA SEÑAL
# ════════════════════════════════════════════════════

def real_spread_analysis(df: pd.DataFrame) -> dict:
    """Analiza el spread real observado en los datos a las 14:29-14:30 UTC."""
    logger.info("\n" + "═"*68)
    logger.info("  2. SPREAD REAL EN EL MOMENTO DE LA SEÑAL (14:29-14:30 UTC)")
    logger.info("═"*68)

    filt = (df['prior_bearish']) & (np.abs(df['first_hour_ret']) > 0.003)
    df_f = df[filt].dropna(subset=['real_spread']).copy()

    spread_duck = df_f['real_spread'].values
    spread_nq   = spread_duck * DUKASCOPY_SCALE

    p25, p50, p75, p95 = np.percentile(spread_nq, [25, 50, 75, 95])

    logger.info(f"  N días con señal H3v2: {len(df_f)}")
    logger.info(f"  Spread en señal (pts NQ equivalente):")
    logger.info(f"    P25:  {p25:.2f} pts")
    logger.info(f"    Med:  {p50:.2f} pts")
    logger.info(f"    P75:  {p75:.2f} pts")
    logger.info(f"    P95:  {p95:.2f} pts")
    logger.info(f"    Max:  {spread_nq.max():.2f} pts")

    # ¿Es mejor o peor que el costo asumido (2 pts entry)?
    assumed_spread = 2.0  # pts NQ por lado
    pct_above = (spread_nq > assumed_spread * 2).mean()
    logger.info(f"\n  % días donde spread real > {assumed_spread*2:.0f} pts (nuestra asunción): "
                f"{pct_above*100:.1f}%")

    if pct_above < 0.20:
        logger.info("  ✅ Spread real es generalmente < nuestra asunción conservadora")
    else:
        logger.info("  ⚠️  Spread real supera la asunción en >20% de los días")

    return {
        'n': int(len(df_f)),
        'p25': float(p25), 'median': float(p50),
        'p75': float(p75), 'p95': float(p95),
        'pct_above_assumption': float(pct_above),
    }


# ════════════════════════════════════════════════════
# 3. IMPACTO DE SLIPPAGE VARIABLE
# ════════════════════════════════════════════════════

def slippage_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modela slippage variable basado en la volatilidad del día.
    Días de alta volatilidad → mayor slippage.
    """
    logger.info("\n" + "═"*68)
    logger.info("  3. IMPACTO DE SLIPPAGE VARIABLE (proporcional a ATR)")
    logger.info("═"*68)

    filt = (df['prior_bearish']) & (np.abs(df['first_hour_ret']) > 0.003)
    df_f = df[filt].dropna(subset=['ret_entry_close']).copy()
    sig  = np.sign(df_f['first_hour_ret'].values)
    gross = df_f['ret_entry_close'].values * sig
    entry_mean = df_f['entry_close'].mean()

    rows = []
    # Slip fijo vs proporcional a ATR
    scenarios = [
        ('Slip fijo: 0 pts',    0.0,   0.0),
        ('Slip fijo: 1 pt',     1.0,   0.0),
        ('Slip fijo: 2 pts',    2.0,   0.0),
        ('Slip fijo: 3 pts',    3.0,   0.0),
        ('Slip proporcional 0.5×ATR', 0.0, 0.5),
        ('Slip proporcional 1.0×ATR', 0.0, 1.0),
    ]

    spread_base_pts = 2.0  # spread fijo asumido en todos los escenarios

    for label, slip_pts, slip_atr_mult in scenarios:
        costs = np.zeros(len(df_f))
        for i in range(len(df_f)):
            spread_cost = (spread_base_pts / DUKASCOPY_SCALE) / entry_mean
            if slip_atr_mult > 0:
                atr_slip = float(df_f['oh_atr'].iloc[i]) * slip_atr_mult
                costs[i] = spread_cost + atr_slip
            else:
                slip_cost = (slip_pts / DUKASCOPY_SCALE) / entry_mean
                costs[i] = spread_cost + slip_cost
        rn = gross - costs
        if rn.std() == 0:
            continue
        sh   = (rn.mean() / rn.std()) * np.sqrt(252)
        eq   = np.cumprod(1 + rn)
        ann  = eq[-1] ** (252 / len(rn)) - 1
        wr   = (rn > 0).mean()
        icon = "✅" if sh > 1.0 else "⚠️" if sh > 0 else "❌"
        logger.info(f"  {icon} {label:35s}: Sharpe={sh:.3f}  Ann={ann*100:.1f}%  WR={wr*100:.1f}%")
        rows.append({'scenario': label, 'sharpe': float(sh), 'annual': float(ann),
                     'win_rate': float(wr)})

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════
# 4. MONTE CARLO PROFUNDO — H3v2
# ════════════════════════════════════════════════════

def monte_carlo_h3v2(df_is: pd.DataFrame, n_runs: int = 5000, cost_pts: float = 2.0) -> dict:
    """
    Monte Carlo profundo:
      - Aleatorizar el orden de trades (H0: el orden no importa)
      - Si el equity real es superior al percentil P95 del MC → señal real
      - Calcular IC del Sharpe
    """
    logger.info("\n" + "═"*68)
    logger.info(f"  4. MONTE CARLO PROFUNDO ({n_runs} runs) — H3v2 IS")
    logger.info("═"*68)

    filt = (df_is['prior_bearish']) & (np.abs(df_is['first_hour_ret']) > 0.003)
    df_f = df_is[filt].dropna(subset=['ret_entry_close']).copy()
    sig  = np.sign(df_f['first_hour_ret'].values)
    cost = cost_pts / DUKASCOPY_SCALE / df_f['entry_close'].mean()
    rets = df_f['ret_entry_close'].values * sig - cost

    n = len(rets)
    real_sharpe = (rets.mean() / rets.std()) * np.sqrt(252)
    real_eq     = np.cumprod(1 + rets)
    real_total  = real_eq[-1] - 1
    real_dd     = ((real_eq - np.maximum.accumulate(real_eq)) / np.maximum.accumulate(real_eq)).min()

    rng = np.random.default_rng(42)
    mc_sharpes = []
    mc_totals  = []
    mc_dds     = []

    for _ in range(n_runs):
        sh_r = rng.permutation(rets)
        eq   = np.cumprod(1 + sh_r)
        pk   = np.maximum.accumulate(eq)
        dd   = ((eq - pk) / pk).min()
        sh   = (sh_r.mean() / sh_r.std()) * np.sqrt(252)
        mc_sharpes.append(sh)
        mc_totals.append(eq[-1] - 1)
        mc_dds.append(dd)

    mc_sharpes = np.array(mc_sharpes)
    mc_totals  = np.array(mc_totals)
    mc_dds     = np.array(mc_dds)

    pct_pos        = (mc_totals > 0).mean()
    pct_beat_real  = (mc_totals > real_total).mean()  # % runs que superan el real
    p5_dd          = np.percentile(mc_dds, 5)
    ruin_30        = (mc_dds < -0.30).mean()
    p_value_mc     = (mc_sharpes >= real_sharpe).mean()  # MC p-value

    # Bootstrap IC del Sharpe (sobre los retornos reales)
    boot_sharpes = []
    for _ in range(2000):
        boot = rng.choice(rets, size=n, replace=True)
        boot_sh = (boot.mean() / boot.std()) * np.sqrt(252) if boot.std() > 0 else 0
        boot_sharpes.append(boot_sh)
    boot_sharpes = np.array(boot_sharpes)
    sh_ci_lo = np.percentile(boot_sharpes, 2.5)
    sh_ci_hi = np.percentile(boot_sharpes, 97.5)

    logger.info(f"  N trades:           {n}")
    logger.info(f"  Sharpe real:        {real_sharpe:.3f}")
    logger.info(f"  95% CI Bootstrap:   [{sh_ci_lo:.3f}, {sh_ci_hi:.3f}]")
    logger.info(f"  Sharpe MC mediana:  {np.median(mc_sharpes):.3f}")
    logger.info(f"  p-value MC:         {p_value_mc:.4f}  (H0: el orden no importa)")
    logger.info(f"  % runs positivos:   {pct_pos*100:.1f}%")
    logger.info(f"  % runs > real:      {pct_beat_real*100:.1f}%")
    logger.info(f"  Max DD mediana:     {np.median(mc_dds)*100:.1f}%")
    logger.info(f"  Max DD P5:          {p5_dd*100:.1f}%")
    logger.info(f"  P(DD < -30%):       {ruin_30*100:.1f}%")

    if p_value_mc < 0.05 and pct_pos > 0.65:
        logger.info("  ✅ MC: Edge robusto y estadísticamente significativo")
    elif p_value_mc < 0.10:
        logger.info("  ⚠️  MC: Edge marginalmente significativo")
    else:
        logger.info("  ❌ MC: El orden importa poco — puede ser ruido")

    return {
        'n': int(n), 'real_sharpe': float(real_sharpe),
        'ci_lo': float(sh_ci_lo), 'ci_hi': float(sh_ci_hi),
        'mc_pvalue': float(p_value_mc),
        'pct_positive': float(pct_pos),
        'pct_beat_real': float(pct_beat_real),
        'p5_dd': float(p5_dd), 'ruin_30': float(ruin_30),
        'mc_sharpes': mc_sharpes.tolist(),
        'mc_totals': mc_totals.tolist(),
        'mc_dds': mc_dds.tolist(),
    }


# ════════════════════════════════════════════════════
# 5. PODER ESTADÍSTICO — ¿Cuántos trades?
# ════════════════════════════════════════════════════

def statistical_power_analysis(df_is: pd.DataFrame, cost_pts: float = 2.0) -> dict:
    """
    ¿Cuántos trades OOS necesitamos para confirmar p < 0.05?
    Dado el Sharpe IS = 2.19 y la varianza observada.
    """
    logger.info("\n" + "═"*68)
    logger.info("  5. PODER ESTADÍSTICO — ¿Cuántos trades OOS para p<0.05?")
    logger.info("═"*68)

    filt = (df_is['prior_bearish']) & (np.abs(df_is['first_hour_ret']) > 0.003)
    df_f = df_is[filt].dropna(subset=['ret_entry_close']).copy()
    sig  = np.sign(df_f['first_hour_ret'].values)
    cost = cost_pts / DUKASCOPY_SCALE / df_f['entry_close'].mean()
    rets = df_f['ret_entry_close'].values * sig - cost

    mu  = rets.mean()
    std = rets.std()

    logger.info(f"  μ IS = {mu*100:.4f}%  σ = {std*100:.4f}%")
    logger.info(f"  Para α=0.05, 1-tailed t-test:")

    results = {}
    for power in [0.50, 0.70, 0.80, 0.90]:
        # n necesario: n = (z_alpha + z_power)^2 / (mu/std)^2
        z_alpha = stats.norm.ppf(0.95)   # α=0.05 1-tailed
        z_power = stats.norm.ppf(power)
        effect  = mu / std
        n_required = int(np.ceil(((z_alpha + z_power) / effect) ** 2))
        # Estimación de tiempo: ~45 días de mercado al año × freq filtro
        # El filtro da ~48 trades en ~247 días OOS → ~0.19 trades/día
        trade_freq = 48 / 247  # trades por día de mercado
        days_required = n_required / trade_freq
        months_required = days_required / 21  # 21 días de mercado por mes

        logger.info(f"  Poder {power*100:.0f}%: n={n_required:4d} trades "
                    f"≈ {months_required:.1f} meses de datos OOS")
        results[f'power_{int(power*100)}'] = {
            'n_required': n_required,
            'months': float(months_required),
        }

    # Proyección: con 2025 completo ya tenemos ~48 trades en ~12 meses
    # ¿Cuándo alcanzaremos n suficiente a esta tasa?
    accumulated = 48
    rate_per_month = 48 / 12
    n_for_80 = results.get('power_80', {}).get('n_required', 200)
    months_to_80 = max(0, (n_for_80 - accumulated) / rate_per_month)

    logger.info(f"\n  Trades OOS actuales (2025): {accumulated}")
    logger.info(f"  Tasa actual: {rate_per_month:.1f} trades/mes")
    logger.info(f"  Para 80% poder (n={n_for_80}): {months_to_80:.1f} meses adicionales de paper trading")
    logger.info(f"  Fecha estimada de confirmación: ~{months_to_80:.0f} meses desde hoy")

    results['accumulated'] = accumulated
    results['rate_per_month'] = float(rate_per_month)
    results['months_to_80pct_power'] = float(months_to_80)

    return results


# ════════════════════════════════════════════════════
# 6. CURVA DE EQUITY OOS DETALLADA
# ════════════════════════════════════════════════════

def oos_equity_detailed(df_oos: pd.DataFrame, cost_pts: float = 2.0) -> dict:
    """Equity OOS con IC de bootstrap para visualizar la incertidumbre."""
    filt = (df_oos['prior_bearish']) & (np.abs(df_oos['first_hour_ret']) > 0.003)
    df_f = df_oos[filt].dropna(subset=['ret_entry_close']).copy().sort_index()
    sig  = np.sign(df_f['first_hour_ret'].values)
    cost = cost_pts / DUKASCOPY_SCALE / df_f['entry_close'].mean()
    rets = df_f['ret_entry_close'].values * sig - cost
    eq   = np.cumprod(1 + rets)

    rng = np.random.default_rng(42)
    boot_finals = [np.cumprod(1 + rng.choice(rets, size=len(rets), replace=True))[-1]
                   for _ in range(1000)]
    ci_lo = np.percentile(boot_finals, 5)
    ci_hi = np.percentile(boot_finals, 95)

    logger.info(f"\n  OOS Equity final: {eq[-1]-1:.3f} (+{(eq[-1]-1)*100:.1f}%)")
    logger.info(f"  90% CI Bootstrap: [{(ci_lo-1)*100:.1f}%, {(ci_hi-1)*100:.1f}%]")

    return {
        'equity': eq.tolist(),
        'dates': [str(d.date()) for d in df_f.index],
        'ci_lo': float(ci_lo), 'ci_hi': float(ci_hi),
        'n': int(len(rets)),
    }


# ════════════════════════════════════════════════════
# VISUALIZACIÓN
# ════════════════════════════════════════════════════

def plot_execution(lat: pd.DataFrame, slip: pd.DataFrame, mc: dict,
                   power: dict, oos_eq: dict, spread_stats: dict) -> None:

    fig = plt.figure(figsize=(20, 18), facecolor='#0d1117')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)
    GOLD='#FFD700'; GREEN='#00FF88'; RED='#FF4444'; BLUE='#4488FF'; GRAY='#888888'; BG='#161b22'

    def ax_style(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, color=GOLD, fontsize=10, fontweight='bold', pad=8)
        ax.tick_params(colors=GRAY)
        ax.spines[:].set_color('#333333')
        for l in ax.get_xticklabels() + ax.get_yticklabels(): l.set_color(GRAY)

    # 1. Latencia
    ax1 = fig.add_subplot(gs[0, 0])
    if not lat.empty:
        colors1 = [GREEN if s > 1.0 else (GOLD if s > 0 else RED) for s in lat['sharpe']]
        ax1.bar(range(len(lat)), lat['sharpe'], color=colors1)
        ax1.set_xticks(range(len(lat)))
        ax1.set_xticklabels([e[:12] for e in lat['entry']], rotation=35, fontsize=7)
        ax1.axhline(0, color=GRAY, lw=0.8)
        ax1.axhline(1.0, color=GOLD, lw=1, ls='--', label='Sharpe=1.0')
        ax1.legend(facecolor=BG, labelcolor='white', fontsize=8)
    ax_style(ax1, "Robustez a Latencia\n(Sharpe por Timing de Entrada)")
    ax1.set_ylabel("Sharpe IS", color=GRAY)

    # 2. Slippage scenarios
    ax2 = fig.add_subplot(gs[0, 1])
    if not slip.empty:
        colors2 = [GREEN if s > 1.0 else (GOLD if s > 0 else RED) for s in slip['sharpe']]
        ax2.barh(range(len(slip)), slip['sharpe'], color=colors2)
        ax2.set_yticks(range(len(slip)))
        ax2.set_yticklabels(slip['scenario'].str[:30], fontsize=7)
        ax2.axvline(0, color=GRAY, lw=0.8)
        ax2.axvline(1.0, color=GOLD, lw=1, ls='--')
    ax_style(ax2, "Impacto de Slippage\n(Sharpe IS)")
    ax2.tick_params(axis='y', labelcolor=GRAY)

    # 3. Monte Carlo Sharpe distribution
    ax3 = fig.add_subplot(gs[0, 2])
    if mc.get('mc_sharpes'):
        mcs = np.array(mc['mc_sharpes'])
        try:
            ax3.hist(mcs, bins='auto', color=GRAY, alpha=0.5, label='MC aleatorio')
        except ValueError:
            ax3.hist(mcs, bins=20, range=(mcs.min()-0.01, mcs.max()+0.01), color=GRAY, alpha=0.5, label='MC aleatorio')
        ax3.axvline(mc['real_sharpe'], color=GOLD, lw=2.5, ls='--',
                    label=f"Real={mc['real_sharpe']:.2f}")
        ax3.axvline(mc['ci_lo'], color=BLUE, lw=1.5, ls=':',
                    label=f"CI [{mc['ci_lo']:.2f},{mc['ci_hi']:.2f}]")
        ax3.axvline(mc['ci_hi'], color=BLUE, lw=1.5, ls=':')
        ax3.legend(facecolor=BG, labelcolor='white', fontsize=7)
    ax_style(ax3, f"Monte Carlo Sharpe\np-val={mc.get('mc_pvalue',1):.4f}")
    ax3.set_xlabel("Sharpe", color=GRAY)

    # 4. MC equity distribution
    ax4 = fig.add_subplot(gs[1, 0])
    if mc.get('mc_totals'):
        mct = np.array(mc['mc_totals'])
        try:
            ax4.hist(mct*100, bins='auto', color=GRAY, alpha=0.5)
        except ValueError:
            ax4.hist(mct*100, bins=20, range=(mct.min()*100-0.1, mct.max()*100+0.1), color=GRAY, alpha=0.5)
        ax4.axvline(mc.get('real_sharpe', 0)*10, color=GOLD, lw=2, ls='--')
        pct_pos = mc.get('pct_positive', 0)
        ax4.set_title(f"MC Distribución Equity\n{pct_pos*100:.0f}% positivos",
                      color=GOLD, fontsize=10, fontweight='bold')
    ax4.set_facecolor(BG); ax4.tick_params(colors=GRAY)
    ax4.spines[:].set_color('#333333')
    for l in ax4.get_xticklabels() + ax4.get_yticklabels(): l.set_color(GRAY)
    ax4.set_xlabel("Equity final (%)", color=GRAY)

    # 5. MC DrawDown distribution
    ax5 = fig.add_subplot(gs[1, 1])
    if mc.get('mc_dds'):
        mcd = np.array(mc['mc_dds'])
        try:
            ax5.hist(mcd*100, bins='auto', color=RED, alpha=0.5)
        except ValueError:
            ax5.hist(mcd*100, bins=20, range=(mcd.min()*100-0.1, mcd.max()*100+0.1), color=RED, alpha=0.5)
        ax5.axvline(mc.get('p5_dd', 0)*100, color=GOLD, lw=2, ls='--',
                    label=f"P5={mc['p5_dd']*100:.1f}%")
        ax5.legend(facecolor=BG, labelcolor='white', fontsize=8)
    ax_style(ax5, f"MC Drawdown  P(DD<-30%)={mc.get('ruin_30',0)*100:.1f}%")
    ax5.set_xlabel("Max DD (%)", color=GRAY)

    # 6. OOS Equity con IC
    ax6 = fig.add_subplot(gs[1, 2])
    if oos_eq.get('equity'):
        eq = np.array(oos_eq['equity'])
        color6 = GREEN if eq[-1] > 1.0 else RED
        ax6.plot(eq, color=color6, lw=2.5, label=f"OOS n={oos_eq['n']}")
        ax6.axhline(1.0, color=GRAY, lw=0.8, ls='--')
        ax6.fill_between(range(len(eq)), 1.0, eq, alpha=0.2, color=color6)
        ci_lo = oos_eq.get('ci_lo', 1.0)
        ci_hi = oos_eq.get('ci_hi', 1.0)
        ax6.axhline(ci_lo, color=BLUE, lw=1, ls=':', label=f"90% CI")
        ax6.axhline(ci_hi, color=BLUE, lw=1, ls=':')
        ax6.legend(facecolor=BG, labelcolor='white', fontsize=8)
    ax_style(ax6, "OOS 2025 Equity + Bootstrap 90% CI")
    ax6.set_ylabel("Equity", color=GRAY)

    # 7. Poder estadístico
    ax7 = fig.add_subplot(gs[2, :2])
    power_vals = [(k, v) for k, v in power.items() if k.startswith('power_')]
    if power_vals:
        labels7 = [f"Poder {v['months']:.0f}m\n(n={v['n_required']})"
                   for k, v in power_vals]
        colors7 = [GREEN if v['months'] < 18 else GOLD if v['months'] < 36 else RED
                   for k, v in power_vals]
        ax7.bar(labels7,
                [v['months'] for k, v in power_vals],
                color=colors7, alpha=0.85)
        ax7.axhline(12, color=GOLD, lw=1.5, ls='--', label='12 meses')
        ax7.axhline(24, color=RED, lw=1, ls=':', label='24 meses')
        ax7.legend(facecolor=BG, labelcolor='white', fontsize=8)
    ax_style(ax7, "Poder Estadístico: Meses de Paper Trading para Confirmar p<0.05")
    ax7.set_ylabel("Meses requeridos", color=GRAY)

    # 8. Spread real en la señal
    ax8 = fig.add_subplot(gs[2, 2])
    if spread_stats:
        spread_pts = [spread_stats['p25'], spread_stats['median'],
                      spread_stats['p75'],  spread_stats['p95']]
        labels8 = ['P25', 'Med', 'P75', 'P95']
        assumed = 4.0  # pts NQ asumidos (2 pts spread × 2)
        colors8 = [GREEN if v < assumed else RED for v in spread_pts]
        ax8.bar(labels8, spread_pts, color=colors8, alpha=0.85)
        ax8.axhline(assumed, color=GOLD, lw=2, ls='--', label=f'Asumido={assumed} pts')
        ax8.legend(facecolor=BG, labelcolor='white', fontsize=8)
    ax_style(ax8, "Spread Real en Momento de Señal\n(puntos NQ equivalente)")
    ax8.set_ylabel("Spread real (pts NQ)", color=GRAY)

    fig.suptitle(
        f"H3v2: VALIDACIÓN DE EJECUCIÓN REALISTA — FASE 6.6\n"
        f"MC p-val={mc.get('mc_pvalue',1):.4f}  |  "
        f"Sharpe IS CI=[{mc.get('ci_lo',0):.2f},{mc.get('ci_hi',0):.2f}]  |  "
        f"OOS Equity +{(np.array(oos_eq.get('equity',[1]))[-1]-1)*100:.1f}%",
        color='white', fontsize=12, fontweight='bold', y=0.99
    )

    out = ARTIFACTS_DIR / "nq_h3_execution.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"\n  ✅ Gráfico: {out}")


# ════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════

def main():
    logger.info("╔" + "═"*70 + "╗")
    logger.info("║   FASE 6.6 — EJECUCIÓN REALISTA: H3v2 Previo Bajista          ║")
    logger.info("║   Verificando que el edge no asume ejecución perfecta          ║")
    logger.info("╚" + "═"*70 + "╝")

    parquet = PROJECT_ROOT / "quant_bot" / "data" / "processed" / "USATECHIDXUSD_M1.parquet"
    logger.info(f"\n  Cargando {parquet.name}...")
    df = pd.read_parquet(parquet, engine='pyarrow')
    logger.info(f"  → {len(df):,} barras")

    if 'session' not in df.columns:
        from quant_bot.data.nq_loader import add_session_labels
        df = add_session_labels(df)

    logger.info("\n  Construyendo señales de ejecución granulares...")
    sigs_all = build_execution_signals(df)
    sigs_is  = sigs_all[sigs_all['year'] < OOS_YEAR]
    sigs_oos = sigs_all[sigs_all['year'] >= OOS_YEAR]

    logger.info(f"  IS: {len(sigs_is)} días | OOS: {len(sigs_oos)} días")

    COST_PTS = 2.0

    lat    = entry_latency_test(sigs_is, cost_pts=COST_PTS)
    spread = real_spread_analysis(sigs_all)
    slip   = slippage_impact(sigs_is)
    mc     = monte_carlo_h3v2(sigs_is, n_runs=5000, cost_pts=COST_PTS)
    power  = statistical_power_analysis(sigs_is, cost_pts=COST_PTS)
    oos_eq = oos_equity_detailed(sigs_oos, cost_pts=COST_PTS)

    # Clasificación final
    logger.info("\n" + "═"*70)
    logger.info("  CLASIFICACIÓN FASE 6.6 — EJECUCIÓN REALISTA")
    logger.info("═"*70)

    max_latency_sh = lat.iloc[2]['sharpe'] if len(lat) > 2 else 0
    checks = {
        'robusto_latencia_2min':    max_latency_sh > 0.8,
        'spread_real_ok':           spread.get('pct_above_assumption', 1) < 0.25,
        'robusto_slippage_fijo_2pt': slip[slip['scenario'].str.contains('fijo: 2')]['sharpe'].iloc[0] > 1.0
                                      if not slip.empty else False,
        'mc_significativo':         mc.get('mc_pvalue', 1) < 0.10,
        'mc_ruina_baja':            mc.get('ruin_30', 1)  < 0.10,
        'oos_equity_positivo':      (np.array(oos_eq.get('equity', [1]))[-1] - 1) > 0,
        'oos_ci_positivo':          oos_eq.get('ci_lo', 0) > 1.0,
    }

    score = sum(checks.values())
    for k, v in checks.items():
        logger.info(f"  {'✅' if v else '❌'} {k}")

    logger.info(f"\n  SCORE FASE 6.6: {score}/7")

    if score >= 6:
        verdict = "✅ EJECUCIÓN REALISTA VÁLIDA — edge no depende de fills perfectos"
    elif score >= 4:
        verdict = "⚠️  EJECUCIÓN MAYORMENTE VÁLIDA — verificar latencia y spread en vivo"
    else:
        verdict = "❌ EDGE DEPENDE DE EJECUCIÓN PERFECTA — inválido para retail"

    logger.info(f"  VEREDICTO: {verdict}")

    plot_execution(lat, slip, mc, power, oos_eq, spread)

    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.bool_,)): return bool(o)
            if isinstance(o, (np.ndarray,)): return o.tolist()
            return super().default(o)

    out_data = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'latency_test': lat.to_dict('records'),
        'spread_real': spread,
        'slippage_scenarios': slip.to_dict('records'),
        'monte_carlo': {k: v for k, v in mc.items()
                        if k not in ('mc_sharpes', 'mc_totals', 'mc_dds')},
        'statistical_power': power,
        'oos_equity': {k: v for k, v in oos_eq.items() if k not in ('equity', 'dates')},
        'checks': {k: bool(v) for k, v in checks.items()},
        'score': int(score), 'verdict': verdict,
    }
    out_json = ARTIFACTS_DIR / "h3_execution_metrics.json"
    with open(out_json, 'w') as f:
        json.dump(out_data, f, indent=2, cls=NE)

    logger.info(f"\n  Métricas → {out_json}")
    logger.info("  ✅ Fase 6.6 completada")


if __name__ == "__main__":
    main()
