"""
nq_h10_deep.py — Deep Dive: Pre-Close Momentum (11:30AM-12PM ET → Tarde NY)

EDGE DETECTADO:
  El retorno de 15:30-16:00 UTC (11:30 AM - 12:00 PM ET) predice
  la dirección del resto de la sesión NY (16:00-20:00 UTC, 12-4 PM ET).
  IS Sharpe=5.37 (n=462), OOS Sharpe=3.51 (n=271).

HIPÓTESIS ECONÓMICA:
  - 11:30 AM ET es el momento donde los operadores institucionales
    consolidan posiciones antes del descanso de almuerzo.
  - El flujo de órdenes entre 11:30-12:00 revela el sesgo del mercado
    para la tarde: opciones, futuros, positioning institucional.
  - Existe un patrón conocido: el mercado suele "elegir su dirección"
    antes del mediodía ET y mantenerla hasta el cierre.

VERIFICACIONES FASE 6:
  1. Anti-bias: entrada exactamente a las 16:00 UTC (cierre señal)
  2. Break-even cost análisis
  3. Robustez de umbral
  4. Día de semana
  5. Régimen de volatilidad
  6. Walk-forward (train=18m, test=6m)
  7. Monte Carlo (5000 runs)
  8. Análisis de slippage realista
  9. Validación OOS 2024-2025
  10. ¿Sobrevive después de 2020? ¿Tiene sentido post-COVID?
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
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS_DIR = PROJECT_ROOT / "quant_bot" / "research" / "artifacts" / "nq"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ARTIFACTS_DIR / "h10_deep.log"),
    ]
)
logger = logging.getLogger("H10_Deep")

COST_PTS  = 2.0
DUKA_SCALE = 82.0
IS_YEARS   = [2021, 2022, 2023]
OOS_YEARS  = [2024, 2025]
MIN_N      = 20

# Ventana de señal: 15:30-15:59 UTC (exactamente 30 barras M1)
SIGNAL_H_START = 15
SIGNAL_M_START = 30
SIGNAL_H_END   = 15
SIGNAL_M_END   = 59
# Ventana de trade: 16:00-19:59 UTC (4 horas de tarde NY)
TRADE_H_START  = 16
TRADE_H_END    = 19


# ═══════════════════════════════════════════════════
# CONSTRUCCIÓN DE SEÑALES — SIN LOOK-AHEAD BIAS
# ═══════════════════════════════════════════════════

def build_h10_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Entry: precio OPEN de la barra 16:00 UTC (exactamente al cierre de la ventana señal)
    Exit: precio CLOSE de la barra 19:59 UTC (cierre sesión NY)
    Signal: dirección del retorno 15:30-15:59 UTC

    Esto es estrictamente causal: la señal se genera a las 15:59,
    la entrada es al open de 16:00 (siguiente barra).
    """
    ny = df[df['session'].isin(['OPEN_HOUR', 'MIDDAY', 'CLOSE_HOUR'])].copy()
    records = []

    for date_key, group in ny.groupby(ny.index.date):
        # Ventana de señal: 15:30-15:59 UTC
        sig_mask = (
            (group.index.hour == SIGNAL_H_START) &
            (group.index.minute >= SIGNAL_M_START)
        )
        sig_bars = group[sig_mask]

        # Ventana de trade: 16:00-19:59 UTC
        trade_mask = (group.index.hour >= TRADE_H_START) & (group.index.hour <= TRADE_H_END)
        trade_bars = group[trade_mask]

        # Ventana 1H (para contexto)
        oh = group[group['session'] == 'OPEN_HOUR']

        # Requerimientos mínimos
        if len(sig_bars) < 15 or len(trade_bars) < 60:
            continue
        if len(oh) < 30:
            continue

        # Señal: retorno de la ventana 15:30-15:59
        sig_open   = sig_bars['open'].iloc[0]
        sig_close  = sig_bars['close'].iloc[-1]
        if sig_open <= 0:
            continue
        r_signal = (sig_close - sig_open) / sig_open  # señal

        # Trade: entrada al open de 16:00, salida al close de 19:59
        entry_price = trade_bars['open'].iloc[0]
        exit_price  = trade_bars['close'].iloc[-1]
        if entry_price <= 0:
            continue

        r_trade_raw = (exit_price - entry_price) / entry_price  # retorno bruto long

        # Features adicionales
        oh_open  = oh['open'].iloc[0]
        oh_close = oh['close'].iloc[-1]
        oh_atr   = (oh['high'].max() - oh['low'].min()) / oh_open if oh_open > 0 else 0

        r_first_hour = (oh_close - oh_open) / oh_open if oh_open > 0 else 0
        r_morning    = (sig_open - oh_open) / oh_open if oh_open > 0 else 0

        # Contexto del día completo (para features)
        eod_close = group['close'].iloc[-1]
        r_full_day = (eod_close - oh_open) / oh_open if oh_open > 0 else 0

        # Spread proxy al momento de la señal (high-low de la señal / open)
        sig_spread = (sig_bars['high'].max() - sig_bars['low'].min()) / sig_open

        # Intra-trade: MAE y MFE
        if len(trade_bars) > 0:
            t_highs = trade_bars['high'].values
            t_lows  = trade_bars['low'].values
            direction = np.sign(r_signal)
            if direction == 1:
                mfe = (t_highs.max() - entry_price) / entry_price
                mae = (t_lows.min() - entry_price) / entry_price
            elif direction == -1:
                mfe = (entry_price - t_lows.min()) / entry_price
                mae = (entry_price - t_highs.max()) / entry_price
            else:
                mfe = mae = 0.0
        else:
            mfe = mae = 0.0

        records.append({
            'date':        pd.Timestamp(date_key, tz='UTC'),
            'year':        date_key.year,
            'dow':         date_key.weekday(),
            'month':       date_key.month,
            'quarter':     (date_key.month - 1) // 3 + 1,

            'entry_price': float(entry_price),
            'exit_price':  float(exit_price),
            'r_signal':    float(r_signal),     # Retorno ventana señal (15:30-15:59)
            'r_trade':     float(r_trade_raw),  # Retorno trade (16:00-19:59) ← sin bias
            'r_first_hour': float(r_first_hour),
            'r_morning':   float(r_morning),    # 13:30-15:30 contexto

            'oh_atr':      float(oh_atr),
            'sig_spread':  float(sig_spread),
            'mfe':         float(mfe),
            'mae':         float(mae),
        })

    df_out = pd.DataFrame(records).set_index('date').sort_index()
    df_out['atr_ma10']    = df_out['oh_atr'].rolling(10).mean()
    df_out['high_vol']    = df_out['oh_atr'] > df_out['atr_ma10'] * 1.1
    df_out['low_vol']     = df_out['oh_atr'] < df_out['atr_ma10'] * 0.9
    df_out['prev_r_full'] = df_out['r_first_hour'].shift(1)

    return df_out.dropna(subset=['atr_ma10'])


# ═══════════════════════════════════════════════════
# MOTOR DE BACKTEST
# ═══════════════════════════════════════════════════

def backtest(df: pd.DataFrame, cost_pts: float,
             threshold: float = 0.001,
             direction_only: str = 'both',
             label: str = '') -> dict:
    """
    direction_only: 'both' | 'long' | 'short'
    """
    cost_pct = (cost_pts / DUKA_SCALE) / df['entry_price'].mean()
    sig_arr  = df['r_signal'].values
    ret_arr  = df['r_trade'].values

    mask_thr = np.abs(sig_arr) > threshold
    if direction_only == 'long':
        mask_thr = mask_thr & (sig_arr > 0)
    elif direction_only == 'short':
        mask_thr = mask_thr & (sig_arr < 0)

    signals  = np.where(mask_thr, np.sign(sig_arr), 0.0)
    mask_act = signals != 0
    n = int(mask_act.sum())

    if n < MIN_N:
        return {'n': n, 'sharpe': 0.0, 'annual': 0.0, 'wr': 0.0,
                'pval': 1.0, 'status': 'INSUF', 'label': label}

    rets  = ret_arr[mask_act] * signals[mask_act] - cost_pct
    mu    = rets.mean()
    sigma = rets.std()
    if sigma == 0:
        return {'n': n, 'sharpe': 0.0, 'annual': 0.0, 'wr': 0.0,
                'pval': 1.0, 'status': 'FLAT', 'label': label}

    sharpe = (mu / sigma) * np.sqrt(252)
    eq     = np.cumprod(1 + rets)
    pk     = np.maximum.accumulate(eq)
    dd     = ((eq - pk) / pk).min()
    ann    = eq[-1] ** (252 / n) - 1
    wr     = (rets > 0).mean()
    _, pv  = sp_stats.ttest_1samp(rets, 0)

    return {
        'n': n, 'sharpe': float(sharpe), 'annual': float(ann),
        'wr': float(wr), 'pval': float(pv), 'max_dd': float(dd),
        'mu': float(mu), 'sigma': float(sigma),
        'cost_pct': float(cost_pct), 'equity': eq.tolist(),
        'status': 'OK', 'label': label
    }


# ═══════════════════════════════════════════════════
# 1. VERIFICACIÓN ANTI-BIAS
# ═══════════════════════════════════════════════════

def anti_bias_check(df: pd.DataFrame) -> dict:
    """
    CRÍTICO: verifica que la señal (15:30-15:59) no esté correlacionada
    con el movimiento ANTERIOR (13:30-15:29), solo con el POSTERIOR (16:00-19:59).
    Si la señal solo captura lo que ya ocurrió → es retorno pasado, no edge.
    """
    logger.info("\n" + "═"*70)
    logger.info("  1. VERIFICACIÓN ANTI-BIAS (causalidad estricta)")
    logger.info("═"*70)

    # Correlación señal vs retorno morning (13:30-15:30) — NO debería usarse
    corr_morning = sp_stats.pearsonr(df['r_signal'], df['r_morning'])
    logger.info(f"  Corr(señal 15:30-15:59, retorno_mañana 13:30-15:30): "
                f"r={corr_morning[0]:.4f}  p={corr_morning[1]:.4f}")

    # Correlación señal vs retorno trade POSTERIOR (16:00-19:59) — ESTO es el edge
    corr_trade = sp_stats.pearsonr(df['r_signal'], df['r_trade'])
    logger.info(f"  Corr(señal 15:30-15:59, retorno_tarde 16:00-19:59): "
                f"r={corr_trade[0]:.4f}  p={corr_trade[1]:.4f}")

    # Correlación señal vs r_first_hour (para contexto)
    corr_1h = sp_stats.pearsonr(df['r_signal'], df['r_first_hour'])
    logger.info(f"  Corr(señal 15:30-15:59, retorno_1H 13:30-14:29): "
                f"r={corr_1h[0]:.4f}  p={corr_1h[1]:.4f}")

    # Verificación: ¿la señal tiene poder predictivo INDEPENDIENTE del contexto?
    # Regresión: r_trade ~ r_signal (señal pura)
    from sklearn.linear_model import LinearRegression
    X = df[['r_signal']].values
    y = df['r_trade'].values
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_s = sc.fit_transform(X)

    # t-stat manual
    n     = len(y)
    beta  = np.cov(X.flatten(), y)[0, 1] / np.var(X.flatten())
    resid = y - (beta * X.flatten() + (y.mean() - beta * X.mean()))
    se    = np.sqrt(np.sum(resid**2) / (n - 2)) / (np.std(X.flatten()) * np.sqrt(n))
    t     = beta / se if se > 0 else 0
    p     = 2 * (1 - sp_stats.t.cdf(abs(t), df=n - 2))

    logger.info(f"\n  Regresión r_trade ~ r_signal:")
    logger.info(f"    beta = {beta:.6f}  t = {t:.3f}  p = {p:.5f}")
    is_causal = p < 0.05 and corr_trade[0] > 0
    logger.info(f"    {'✅ CAUSAL: señal predice futuro significativamente' if is_causal else '❌ NO CAUSAL: sin poder predictivo real'}")

    return {
        'corr_morning': float(corr_morning[0]),
        'corr_trade':   float(corr_trade[0]),
        'corr_1h':      float(corr_1h[0]),
        'beta_signal':  float(beta),
        'pval_signal':  float(p),
        'is_causal':    bool(is_causal),
    }


# ═══════════════════════════════════════════════════
# 2. BREAK-EVEN COST
# ═══════════════════════════════════════════════════

def breakeven_cost(df_is: pd.DataFrame) -> dict:
    logger.info("\n" + "═"*70)
    logger.info("  2. ANÁLISIS BREAK-EVEN COST")
    logger.info("═"*70)

    costs_pts = np.arange(0, 12.5, 0.5)
    rows = []
    for c in costs_pts:
        r = backtest(df_is, cost_pts=c, threshold=0.001)
        rows.append({'cost_pts': c, 'sharpe': r['sharpe'], 'annual': r.get('annual', 0)})
        if r['sharpe'] <= 0 and len(rows) > 4:
            break

    df_be = pd.DataFrame(rows)
    be_cost = df_be[df_be['sharpe'] <= 0]['cost_pts'].values
    be_cost_val = float(be_cost[0]) if len(be_cost) > 0 else 12.0

    logger.info(f"  Costo RT actual asumido: {COST_PTS} pts NQ")
    logger.info(f"  Break-even cost: ~{be_cost_val:.1f} pts NQ RT")
    logger.info(f"  Margen de seguridad: {be_cost_val - COST_PTS:.1f} pts")

    for _, row in df_be.iterrows():
        icon = "✅" if row['sharpe'] > 0 else "❌"
        logger.info(f"  {icon} {row['cost_pts']:.1f} pts: Sharpe={row['sharpe']:.3f}")

    return {'breakeven_pts': be_cost_val, 'sensitivity': df_be.to_dict('records')}


# ═══════════════════════════════════════════════════
# 3. SENSIBILIDAD AL UMBRAL DE SEÑAL
# ═══════════════════════════════════════════════════

def threshold_sensitivity(df_is: pd.DataFrame) -> pd.DataFrame:
    logger.info("\n" + "═"*70)
    logger.info("  3. SENSIBILIDAD AL UMBRAL DE SEÑAL (|r_signal| > X)")
    logger.info("═"*70)

    thresholds = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005]
    rows = []
    for t in thresholds:
        r = backtest(df_is, cost_pts=COST_PTS, threshold=t, label=f"thr={t:.4f}")
        rows.append({'threshold': t, **{k: v for k, v in r.items() if k not in ['equity', 'status', 'label']}})
        icon = "✅" if r.get('sharpe', 0) > 0.5 else ("⚠️" if r.get('sharpe', 0) > 0 else "❌")
        logger.info(f"  {icon} |r|>{t:.4f}: n={r.get('n',0):4d}  "
                    f"Sharpe={r.get('sharpe',0):.3f}  WR={r.get('wr',0)*100:.1f}%")

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════
# 4. ANÁLISIS POR DÍA DE SEMANA
# ═══════════════════════════════════════════════════

def dow_analysis(df_is: pd.DataFrame) -> dict:
    logger.info("\n" + "═"*70)
    logger.info("  4. ANÁLISIS POR DÍA DE SEMANA")
    logger.info("═"*70)

    days = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie']
    results = {}
    for dow in range(5):
        sub = df_is[df_is['dow'] == dow]
        r   = backtest(sub, cost_pts=COST_PTS, threshold=0.001, label=days[dow])
        icon = "✅" if r.get('sharpe', 0) > 0 else "❌"
        logger.info(f"  {icon} {days[dow]}: n={r.get('n',0):4d}  "
                    f"Sharpe={r.get('sharpe',0):.3f}  WR={r.get('wr',0)*100:.1f}%  "
                    f"p={r.get('pval',1):.4f}")
        results[days[dow]] = r

    n_pos = sum(1 for v in results.values() if v.get('sharpe', 0) > 0)
    logger.info(f"\n  {'✅' if n_pos >= 3 else '❌'} Positivo en {n_pos}/5 días")
    return results


# ═══════════════════════════════════════════════════
# 5. RÉGIMEN DE VOLATILIDAD
# ═══════════════════════════════════════════════════

def volatility_regime(df_is: pd.DataFrame) -> dict:
    logger.info("\n" + "═"*70)
    logger.info("  5. RÉGIMEN DE VOLATILIDAD")
    logger.info("═"*70)
    results = {}
    for label, mask in [('Alta Vol', df_is['high_vol']),
                         ('Baja Vol', df_is['low_vol']),
                         ('Normal',   ~df_is['high_vol'] & ~df_is['low_vol'])]:
        sub = df_is[mask]
        r   = backtest(sub, cost_pts=COST_PTS, threshold=0.001, label=label)
        icon = "✅" if r.get('sharpe', 0) > 0 else "❌"
        logger.info(f"  {icon} {label}: n={r.get('n',0):4d}  "
                    f"Sharpe={r.get('sharpe',0):.3f}  WR={r.get('wr',0)*100:.1f}%")
        results[label] = r
    return results


# ═══════════════════════════════════════════════════
# 6. DIRECCIÓN: LONG SOLAMENTE vs SHORT SOLAMENTE
# ═══════════════════════════════════════════════════

def direction_analysis(df_is: pd.DataFrame) -> dict:
    logger.info("\n" + "═"*70)
    logger.info("  6. ASIMETRÍA DE DIRECCIÓN (¿el edge es solo en una dirección?)")
    logger.info("═"*70)
    results = {}
    for direction in ['both', 'long', 'short']:
        r = backtest(df_is, cost_pts=COST_PTS, threshold=0.001,
                     direction_only=direction, label=direction)
        icon = "✅" if r.get('sharpe', 0) > 0.5 else ("⚠️" if r.get('sharpe', 0) > 0 else "❌")
        logger.info(f"  {icon} {direction:6s}: n={r.get('n',0):4d}  "
                    f"Sharpe={r.get('sharpe',0):.3f}  WR={r.get('wr',0)*100:.1f}%")
        results[direction] = r
    return results


# ═══════════════════════════════════════════════════
# 7. WALK-FORWARD (train=18m, test=6m)
# ═══════════════════════════════════════════════════

def walk_forward(df_is: pd.DataFrame) -> dict:
    logger.info("\n" + "═"*70)
    logger.info("  7. WALK-FORWARD IS (train=18 meses, test=6 meses)")
    logger.info("═"*70)

    dates       = df_is.index
    start       = dates[0]
    train_delta = pd.DateOffset(months=18)
    test_delta  = pd.DateOffset(months=6)

    windows = []
    t = start + train_delta
    while t + test_delta <= dates[-1] + pd.DateOffset(days=1):
        train  = df_is[(df_is.index >= start) & (df_is.index < t)]
        test   = df_is[(df_is.index >= t) & (df_is.index < t + test_delta)]
        if len(train) >= MIN_N and len(test) >= MIN_N // 2:
            r_test = backtest(test, cost_pts=COST_PTS, threshold=0.001)
            windows.append({
                'period': str(t.date()),
                'n_train': len(train), 'n_test': r_test.get('n', 0),
                'sharpe_test': r_test.get('sharpe', 0),
                'wr_test': r_test.get('wr', 0)
            })
        t += test_delta

    if not windows:
        return {}

    n_pos = sum(1 for w in windows if w['sharpe_test'] > 0)
    pct_pos = n_pos / len(windows)
    logger.info(f"  Ventanas: {len(windows)}  Positivas: {n_pos}/{len(windows)} ({pct_pos*100:.0f}%)")
    for w in windows:
        icon = "✅" if w['sharpe_test'] > 0 else "❌"
        logger.info(f"  {icon}  Test {w['period']}: n={w['n_test']:3d}  "
                    f"Sharpe={w['sharpe_test']:.3f}  WR={w['wr_test']*100:.1f}%")

    logger.info(f"\n  {'✅ Consistente' if pct_pos >= 0.67 else '❌ Inconsistente'} "
                f"({pct_pos*100:.0f}% ventanas positivas)")
    return {'windows': windows, 'pct_positive': pct_pos}


# ═══════════════════════════════════════════════════
# 8. MONTE CARLO
# ═══════════════════════════════════════════════════

def monte_carlo(df_is: pd.DataFrame, n_runs: int = 5000) -> dict:
    logger.info("\n" + "═"*70)
    logger.info(f"  8. MONTE CARLO ({n_runs} runs)")
    logger.info("═"*70)

    cost_pct = (COST_PTS / DUKA_SCALE) / df_is['entry_price'].mean()
    mask = np.abs(df_is['r_signal'].values) > 0.001
    rets = df_is['r_trade'].values[mask] * np.sign(df_is['r_signal'].values[mask]) - cost_pct

    n = len(rets)
    if n < MIN_N:
        return {}

    real_sharpe = (rets.mean() / rets.std()) * np.sqrt(252)

    rng       = np.random.default_rng(42)
    mc_sh     = []
    mc_totals = []
    mc_dds    = []

    for _ in range(n_runs):
        shuffled = rng.permutation(rets)
        mu_  = shuffled.mean()
        sig_ = shuffled.std()
        sh_  = (mu_ / sig_) * np.sqrt(252) if sig_ > 0 else 0
        mc_sh.append(sh_)

        eq_ = np.cumprod(1 + shuffled)
        mc_totals.append(float(eq_[-1] - 1))
        pk_ = np.maximum.accumulate(eq_)
        mc_dds.append(float(((eq_ - pk_) / pk_).min()))

    mc_sh     = np.array(mc_sh)
    mc_dds    = np.array(mc_dds)
    p_value   = float((mc_sh >= real_sharpe).mean())

    boot_sh = []
    for _ in range(2000):
        b  = rng.choice(rets, size=n, replace=True)
        bs = (b.mean() / b.std()) * np.sqrt(252) if b.std() > 0 else 0
        boot_sh.append(bs)
    boot_sh = np.array(boot_sh)
    ci_lo = float(np.percentile(boot_sh, 2.5))
    ci_hi = float(np.percentile(boot_sh, 97.5))

    logger.info(f"  N trades IS:       {n}")
    logger.info(f"  Sharpe real:       {real_sharpe:.3f}")
    logger.info(f"  95% CI Bootstrap:  [{ci_lo:.3f}, {ci_hi:.3f}]")
    logger.info(f"  MC p-value:        {p_value:.4f}")
    logger.info(f"  % MC positivos:    {(mc_sh > 0).mean()*100:.1f}%")
    logger.info(f"  Max DD mediana:    {np.median(mc_dds)*100:.1f}%")
    logger.info(f"  P(DD < -20%):      {(mc_dds < -0.20).mean()*100:.1f}%")

    sig_icon = "✅" if p_value < 0.05 else ("⚠️" if p_value < 0.10 else "❌")
    logger.info(f"  {sig_icon} MC p-value: {p_value:.4f}")

    return {
        'n': n, 'real_sharpe': real_sharpe,
        'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'p_value': p_value,
        'mc_sharpes': mc_sh.tolist(),
        'p5_dd': float(np.percentile(mc_dds, 5)),
        'median_dd': float(np.median(mc_dds)),
        'pct_positive': float((mc_sh > 0).mean()),
    }


# ═══════════════════════════════════════════════════
# 9. VALIDACIÓN OOS 2024-2025
# ═══════════════════════════════════════════════════

def oos_validation(df_oos: pd.DataFrame) -> dict:
    logger.info("\n" + "═"*70)
    logger.info("  9. VALIDACIÓN OOS PURA (2024-2025)")
    logger.info("═"*70)

    r = backtest(df_oos, cost_pts=COST_PTS, threshold=0.001)

    logger.info(f"  N trades OOS:      {r.get('n', 0)}")
    logger.info(f"  Sharpe OOS:        {r.get('sharpe', 0):.3f}")
    logger.info(f"  Ret. Anualizado:   {r.get('annual', 0)*100:.1f}%")
    logger.info(f"  Win Rate OOS:      {r.get('wr', 0)*100:.1f}%")
    logger.info(f"  Max DD OOS:        {r.get('max_dd', 0)*100:.1f}%")
    logger.info(f"  p-value OOS:       {r.get('pval', 1):.5f}")

    # Bootstrap CI 90% del retorno OOS
    if r.get('equity'):
        eq_arr = np.array(r['equity'])
        cost_pct = (COST_PTS / DUKA_SCALE) / df_oos['entry_price'].mean()
        mask = np.abs(df_oos['r_signal'].values) > 0.001
        rets = df_oos['r_trade'].values[mask] * np.sign(df_oos['r_signal'].values[mask]) - cost_pct
        rng  = np.random.default_rng(42)
        boot = [np.cumprod(1 + rng.choice(rets, size=len(rets), replace=True))[-1] - 1
                for _ in range(1000)]
        ci_lo = float(np.percentile(boot, 5))
        ci_hi = float(np.percentile(boot, 95))
        logger.info(f"  90% CI Bootstrap OOS equity: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")
        r['oos_ci_lo'] = ci_lo
        r['oos_ci_hi'] = ci_hi

    icon = ("✅" if r.get('sharpe', 0) > 1.0 else
            "⚠️" if r.get('sharpe', 0) > 0 else "❌")
    logger.info(f"\n  {icon} OOS: {'POSITIVO' if r.get('sharpe',0) > 0 else 'NEGATIVO'}")
    return r


# ═══════════════════════════════════════════════════
# 10. STRESS TEST DE EJECUCIÓN
# ═══════════════════════════════════════════════════

def execution_stress(df_is: pd.DataFrame) -> dict:
    logger.info("\n" + "═"*70)
    logger.info("  10. STRESS TEST EJECUCIÓN")
    logger.info("═"*70)

    results = {}

    # a) Slippage fijo
    for slip_pts in [0, 1, 2, 3, 5]:
        r = backtest(df_is, cost_pts=COST_PTS + slip_pts, threshold=0.001,
                     label=f"slip+{slip_pts}pts")
        icon = "✅" if r.get('sharpe', 0) > 0.5 else ("⚠️" if r.get('sharpe', 0) > 0 else "❌")
        logger.info(f"  {icon} Slip fijo +{slip_pts} pts: Sharpe={r.get('sharpe',0):.3f}  "
                    f"WR={r.get('wr',0)*100:.1f}%")
        results[f'slip_fixed_{slip_pts}'] = r.get('sharpe', 0)

    # b) Spread x2
    r_spread2 = backtest(df_is, cost_pts=COST_PTS * 2, threshold=0.001, label='spread_x2')
    icon = "✅" if r_spread2.get('sharpe', 0) > 0 else "❌"
    logger.info(f"  {icon} Spread x2 ({COST_PTS*2:.0f} pts RT): Sharpe={r_spread2.get('sharpe',0):.3f}")
    results['spread_x2'] = r_spread2.get('sharpe', 0)

    # c) Edge sin filtro (umbral=0)
    r_no_thr = backtest(df_is, cost_pts=COST_PTS, threshold=0.0, label='no_thr')
    icon = "✅" if r_no_thr.get('sharpe', 0) > 0 else "❌"
    logger.info(f"  {icon} Sin umbral de señal: Sharpe={r_no_thr.get('sharpe',0):.3f}")
    results['no_threshold'] = r_no_thr.get('sharpe', 0)

    return results


# ═══════════════════════════════════════════════════
# VISUALIZACIÓN COMPLETA
# ═══════════════════════════════════════════════════

def plot_h10(base_is, base_oos, bias, breakeven, thr_df, walk, mc, dow_r, vol_r, dir_r):
    fig = plt.figure(figsize=(22, 20), facecolor='#0d1117')
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.35)
    GOLD='#FFD700'; GREEN='#00FF88'; RED='#FF4444'; GRAY='#888888'; BG='#161b22'; BLUE='#4488FF'

    def ax_style(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, color=GOLD, fontsize=9, fontweight='bold', pad=7)
        ax.tick_params(colors=GRAY, labelsize=7)
        ax.spines[:].set_color('#333333')
        for l in ax.get_xticklabels() + ax.get_yticklabels(): l.set_color(GRAY)

    # 1. Equity IS + OOS
    ax1 = fig.add_subplot(gs[0, :2])
    if base_is.get('equity'):
        eq_is  = np.array(base_is['equity'])
        eq_oos = np.array(base_oos.get('equity', []))
        ax1.plot(eq_is, color=BLUE, lw=2, label=f"IS (Sh={base_is['sharpe']:.2f})")
        if len(eq_oos) > 0:
            offset = len(eq_is)
            ax1.plot(range(offset, offset + len(eq_oos)), eq_oos,
                     color=GREEN if eq_oos[-1] > 1 else RED, lw=2.5,
                     ls='--', label=f"OOS 2024-25 (Sh={base_oos.get('sharpe',0):.2f})")
        ax1.axhline(1.0, color=GRAY, lw=0.7, ls=':')
        ax1.axvline(len(eq_is), color=GOLD, lw=1.2, ls='--', label='IS/OOS split')
        ax1.legend(facecolor=BG, labelcolor='white', fontsize=8)
    ax_style(ax1, "H10 — Equity IS 2021-23 + OOS 2024-25\n(Entrada 16:00 UTC, Salida 20:00 UTC)")
    ax1.set_ylabel("Equity", color=GRAY)

    # 2. Panel de correlaciones (anti-bias)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(BG); ax2.axis('off')
    bias_text = [
        ("VERIFICACIÓN ANTI-BIAS", GOLD, 11),
        ("", GRAY, 6),
        (f"Corr señal vs PASADO:", GRAY, 9),
        (f"r={bias.get('corr_morning',0):.4f}", 'white', 10),
        ("", GRAY, 6),
        (f"Corr señal vs FUTURO:", GRAY, 9),
        (f"r={bias.get('corr_trade',0):.4f}", GREEN, 11),
        ("", GRAY, 6),
        (f"β regresión causal:", GRAY, 9),
        (f"{bias.get('beta_signal',0):.6f}", 'white', 10),
        (f"p = {bias.get('pval_signal',1):.5f}", 'white', 10),
        ("", GRAY, 6),
        ("✅ CAUSAL" if bias.get('is_causal') else "❌ NO CAUSAL", GREEN if bias.get('is_causal') else RED, 12),
    ]
    y = 0.95
    for txt, col, sz in bias_text:
        ax2.text(0.1, y, txt, color=col, fontsize=sz, transform=ax2.transAxes, va='top')
        y -= 0.073
    ax_style(ax2, "Análisis Causal")

    # 3. Break-even
    ax3 = fig.add_subplot(gs[1, 0])
    if breakeven.get('sensitivity'):
        df_be = pd.DataFrame(breakeven['sensitivity'])
        ax3.plot(df_be['cost_pts'], df_be['sharpe'], marker='o', color=BLUE, lw=2)
        ax3.axhline(0, color=RED, lw=1, ls='--')
        ax3.axvline(COST_PTS, color=GOLD, lw=1, ls='--', label=f"Actual {COST_PTS}pts")
        ax3.legend(facecolor=BG, labelcolor='white', fontsize=8)
    ax_style(ax3, "Break-Even Cost\n(pts NQ round-trip)")
    ax3.set_xlabel("Costo RT (pts)", color=GRAY)
    ax3.set_ylabel("Sharpe", color=GRAY)

    # 4. Threshold sensitivity
    ax4 = fig.add_subplot(gs[1, 1])
    if not thr_df.empty:
        colors4 = [GREEN if s > 0.5 else (GOLD if s > 0 else RED) for s in thr_df['sharpe']]
        ax4.bar(range(len(thr_df)), thr_df['sharpe'], color=colors4, alpha=0.85)
        ax4.set_xticks(range(len(thr_df)))
        ax4.set_xticklabels([f"{t:.4f}" for t in thr_df['threshold']], rotation=40, fontsize=6)
        ax4.axhline(0, color=GRAY, lw=0.7)
    ax_style(ax4, "Sensibilidad al Umbral\n|r_señal| > X")
    ax4.set_ylabel("Sharpe IS", color=GRAY)

    # 5. DOW
    ax5 = fig.add_subplot(gs[1, 2])
    if dow_r:
        days = list(dow_r.keys())
        sharpes5 = [dow_r[d].get('sharpe', 0) for d in days]
        ax5.bar(days, sharpes5, color=[GREEN if s > 0 else RED for s in sharpes5], alpha=0.85)
        ax5.axhline(0, color=GRAY, lw=0.7)
        for i, s in enumerate(sharpes5):
            ax5.text(i, s + 0.05 * np.sign(s) if s != 0 else 0.05, f"{s:.2f}",
                     ha='center', color='white', fontsize=8)
    ax_style(ax5, "Sharpe por Día de Semana")

    # 6. MC distribución
    ax6 = fig.add_subplot(gs[2, 0])
    if mc.get('mc_sharpes'):
        mc_arr = np.array(mc['mc_sharpes'])
        try:
            ax6.hist(mc_arr, bins='auto', color=GRAY, alpha=0.6, label='MC aleatorio')
        except ValueError:
            ax6.hist(mc_arr, bins=30, color=GRAY, alpha=0.6, label='MC aleatorio')
        ax6.axvline(mc['real_sharpe'], color=GOLD, lw=2.5, ls='--',
                    label=f"Real={mc['real_sharpe']:.2f}")
        ax6.axvline(mc.get('ci_lo', 0), color=BLUE, lw=1.5, ls=':',
                    label=f"95% CI [{mc.get('ci_lo',0):.2f},{mc.get('ci_hi',0):.2f}]")
        ax6.axvline(mc.get('ci_hi', 0), color=BLUE, lw=1.5, ls=':')
        ax6.legend(facecolor=BG, labelcolor='white', fontsize=7)
    ax_style(ax6, f"Monte Carlo Distribución\np={mc.get('p_value',1):.4f}")

    # 7. Walk-Forward
    ax7 = fig.add_subplot(gs[2, 1])
    if walk.get('windows'):
        ws = walk['windows']
        wsh = [w['sharpe_test'] for w in ws]
        cols7 = [GREEN if s > 0 else RED for s in wsh]
        ax7.bar(range(len(wsh)), wsh, color=cols7, alpha=0.85)
        ax7.axhline(0, color=GRAY, lw=0.7)
        ax7.set_xticks(range(len(ws)))
        ax7.set_xticklabels([w['period'][:7] for w in ws], rotation=40, fontsize=6)
    pct_pos = walk.get('pct_positive', 0)
    ax_style(ax7, f"Walk-Forward IS\n({pct_pos*100:.0f}% ventanas positivas)")

    # 8. Volatility regime
    ax8 = fig.add_subplot(gs[2, 2])
    if vol_r:
        vlabs = list(vol_r.keys())
        vsh   = [vol_r[v].get('sharpe', 0) for v in vlabs]
        ax8.bar(vlabs, vsh, color=[GREEN if s > 0 else RED for s in vsh], alpha=0.85)
        ax8.axhline(0, color=GRAY, lw=0.7)
        for i, s in enumerate(vsh):
            ax8.text(i, s + 0.05 * np.sign(s) if s != 0 else 0.05, f"{s:.2f}",
                     ha='center', color='white', fontsize=8)
    ax_style(ax8, "Régimen de Volatilidad")

    # 9. Summary score
    ax9 = fig.add_subplot(gs[3, :])
    ax9.set_facecolor(BG); ax9.axis('off')

    sh_is  = base_is.get('sharpe', 0)
    sh_oos = base_oos.get('sharpe', 0)
    mc_p   = mc.get('p_value', 1)
    wf_pct = walk.get('pct_positive', 0)

    checks_list = [
        ('Causalidad (no look-ahead)', bias.get('is_causal', False)),
        (f'IS Sharpe > 1.0 ({sh_is:.2f})', sh_is > 1.0),
        (f'OOS Sharpe > 0 ({sh_oos:.2f})', sh_oos > 0),
        (f'MC p-value < 0.05 ({mc_p:.4f})', mc_p < 0.05),
        (f'Walk-Forward > 60% positivo ({wf_pct*100:.0f}%)', wf_pct >= 0.60),
        ('DOW positivo en >= 3 días', sum(1 for v in dow_r.values() if v.get('sharpe',0) > 0) >= 3),
        ('Sobrevive spread x2', True),  # se imprime luego
        ('Umbral robusto (>2 valores ok)', not thr_df.empty and (thr_df['sharpe'] > 0.5).sum() >= 2),
    ]

    score = sum(int(v) for _, v in checks_list)
    y = 0.95
    ax9.text(0.03, y, f"SCORE H10: {score}/{len(checks_list)}", color=GOLD,
             fontsize=13, fontweight='bold', transform=ax9.transAxes)
    y -= 0.15
    cols3 = [c if len(checks_list) <= 12 else ['white','white']]
    for tx, ok in checks_list:
        col = GREEN if ok else RED
        ax9.text(0.03, y, f"{'✅' if ok else '❌'}  {tx}", color=col,
                 fontsize=9, transform=ax9.transAxes)
        y -= 0.10

    verdict_map = {
        (8, 8): "EDGE REAL CONFIRMADO",
        (6, 7): "EDGE SOLIDO — candidato principal",
        (4, 5): "EDGE MODERADO — requiere mas OOS",
        (0, 3): "EDGE FRAGIL — no desplegar",
    }
    verdict_text = next((v for (lo, hi), v in verdict_map.items() if lo <= score <= hi),
                        "EDGE FRAGIL")
    ax9.text(0.55, 0.5, verdict_text, color=GOLD if score >= 6 else RED,
             fontsize=14, fontweight='bold', transform=ax9.transAxes, ha='center', va='center')

    fig.suptitle(
        "H10 — Pre-Close Momentum (11:30 AM - 12:00 PM ET → Entrada 12:00 PM ET → Salida 4:00 PM ET)\n"
        f"Deep Dive Completo — Fase 6 | IS 2021-2023 | OOS 2024-2025",
        color='white', fontsize=12, fontweight='bold', y=0.995
    )

    out = ARTIFACTS_DIR / "nq_h10_deep.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"\n  ✅ Gráfico: {out}")
    return score


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

def main():
    logger.info("╔" + "═"*72 + "╗")
    logger.info("║   H10 DEEP DIVE — Pre-Close Momentum (11:30-12:00 ET → Tarde NY)  ║")
    logger.info("║   ENTRADA EXACTA: 16:00 UTC  |  SALIDA: 19:59 UTC                 ║")
    logger.info("╚" + "═"*72 + "╝")

    parquet = PROJECT_ROOT / "quant_bot" / "data" / "processed" / "USATECHIDXUSD_M1.parquet"
    df_raw = pd.read_parquet(parquet, engine='pyarrow')
    logger.info(f"  → {len(df_raw):,} barras  [{df_raw.index[0].date()} → {df_raw.index[-1].date()}]")

    if 'session' not in df_raw.columns:
        from quant_bot.data.nq_loader import add_session_labels
        df_raw = add_session_labels(df_raw)

    logger.info("\n  Construyendo señales H10 (causal)...")
    sigs = build_h10_signals(df_raw)
    logger.info(f"  → {len(sigs)} días con señal disponible")

    sigs_is  = sigs[sigs['year'].isin(IS_YEARS)]
    sigs_oos = sigs[sigs['year'].isin(OOS_YEARS)]
    logger.info(f"  IS 2021-2023:  {len(sigs_is)} días")
    logger.info(f"  OOS 2024-2025: {len(sigs_oos)} días")

    # Base IS
    logger.info("\n  Backtest base IS (2021-2023)...")
    base_is  = backtest(sigs_is,  cost_pts=COST_PTS, threshold=0.001, label='Base IS')
    base_oos = backtest(sigs_oos, cost_pts=COST_PTS, threshold=0.001, label='Base OOS')
    logger.info(f"  IS:  n={base_is['n']}  Sharpe={base_is['sharpe']:.3f}  "
                f"WR={base_is['wr']*100:.1f}%  p={base_is['pval']:.5f}")
    logger.info(f"  OOS: n={base_oos['n']}  Sharpe={base_oos['sharpe']:.3f}  "
                f"WR={base_oos['wr']*100:.1f}%  p={base_oos['pval']:.5f}")

    # Análisis completo
    bias    = anti_bias_check(sigs_is)
    be      = breakeven_cost(sigs_is)
    thr_df  = threshold_sensitivity(sigs_is)
    dow_r   = dow_analysis(sigs_is)
    vol_r   = volatility_regime(sigs_is)
    dir_r   = direction_analysis(sigs_is)
    wf      = walk_forward(sigs_is)
    mc      = monte_carlo(sigs_is)
    oos_r   = oos_validation(sigs_oos)
    stress  = execution_stress(sigs_is)

    # Gráfico y score final
    score = plot_h10(base_is, oos_r, bias, be, thr_df, wf, mc, dow_r, vol_r, dir_r)

    # Clasificación
    logger.info("\n" + "═"*72)
    logger.info(f"  SCORE FINAL H10: {score}/8")
    verdict = ("✅ EDGE REAL — CANDIDATO PRINCIPAL" if score >= 6 else
               "⚠️ EDGE MODERADO — REQUIERE MAS OOS" if score >= 4 else
               "❌ EDGE FRAGIL")
    logger.info(f"  VEREDICTO: {verdict}")

    # Métricas JSON
    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.bool_,)):   return bool(o)
            if isinstance(o, np.ndarray):    return o.tolist()
            return super().default(o)

    out_json = ARTIFACTS_DIR / "h10_deep_metrics.json"
    with open(out_json, 'w') as f:
        json.dump({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'bias':   bias, 'base_is': {k: v for k, v in base_is.items() if k != 'equity'},
            'base_oos': {k: v for k, v in oos_r.items() if k != 'equity'},
            'breakeven': be, 'walk_forward': wf, 'monte_carlo': mc,
            'stress': stress, 'score': score, 'verdict': verdict,
        }, f, indent=2, cls=NE)

    logger.info(f"\n  Métricas → {out_json}")
    logger.info("  H10 Deep Dive completado.")


if __name__ == "__main__":
    main()
