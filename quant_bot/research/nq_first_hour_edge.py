"""
nq_first_hour_edge.py — Validación del Edge: Primera Hora de NY → Dirección del Día

HIPÓTESIS (H3):
  Si el retorno de los primeros 60 minutos de NY (13:30-14:30 UTC)
  supera un umbral mínimo, predice la dirección del cierre del día.

  Estrategia concreta:
    - Calcular retorno 13:30→14:30 UTC (primera hora NY)
    - Si retorno > +umbral → BUY al cierre de 14:30 UTC
    - Si retorno < -umbral → SELL al cierre de 14:30 UTC
    - Salida: 20:00 UTC (cierre sesión NY) o SL
    - SL fijo en unidades Dukascopy

FILOSOFÍA (Fase 6):
  "Intentar demostrar que NO funciona."
  Solo si sobrevive todos los tests → edge válido.

PREGUNTAS DE DESTRUCCIÓN:
  1. ¿Sigue vivo con spread x2?
  2. ¿Sigue vivo retrasando la entrada 1 vela (15 min)?
  3. ¿Depende de outliers extremos?
  4. ¿Colapsa en 2022 (bear)?
  5. ¿Monte Carlo muestra ruina?
  6. ¿Hay p-hacking en el umbral?
  7. ¿Edge estable año a año?
  8. ¿Sobrevive slippage variable?
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
        logging.FileHandler(ARTIFACTS_DIR / "first_hour_edge.log"),
    ]
)
logger = logging.getLogger("H3_FirstHour")

# ─────────────────────────────────────────────────────────
# Escala Dukascopy: precio ~128-262 = NQ/~82
# ─────────────────────────────────────────────────────────
DUKASCOPY_SCALE = 82.0

# Costos en unidades Dukascopy
SPREAD_ENTRY  = 2.0 / DUKASCOPY_SCALE   # 2 pts NQ spread entrada
SPREAD_EXIT   = 2.0 / DUKASCOPY_SCALE   # 2 pts NQ spread salida
SLIPPAGE      = 1.0 / DUKASCOPY_SCALE   # 1 pt NQ slippage cada lado
COST_RT       = (SPREAD_ENTRY + SPREAD_EXIT + 2 * SLIPPAGE)  # round-trip total

OOS_YEAR = 2025   # ← OOS fijo, nunca tocar

logger.info(f"  Costo round-trip: {COST_RT:.4f} units ({COST_RT * DUKASCOPY_SCALE:.2f} pts NQ)")


# ═══════════════════════════════════════════════════════════
# 1. PREPARAR SEÑALES DIARIAS
# ═══════════════════════════════════════════════════════════

def build_daily_signals(df: pd.DataFrame, threshold_pct: float = 0.002) -> pd.DataFrame:
    """
    Para cada día de trading construye:
      - first_hour_ret: retorno 13:30→14:30 UTC
      - day_ret:        retorno 14:30→20:00 UTC  (parte de la sesión post-1h)
      - signal:         +1 / -1 / 0
      - entry_price:    precio al cierre de 14:30 UTC
      - exit_price:     precio al cierre de 20:00 UTC
      - trade_ret:      retorno neto del trade
    """
    # Solo sesión NY completa
    ny = df[df['session'].isin(['OPEN_HOUR', 'MIDDAY', 'CLOSE_HOUR'])].copy()

    records = []

    for date, group in ny.groupby(ny.index.date):
        # Primera hora: 13:30-14:30 UTC (OPEN_HOUR)
        oh = group[group['session'] == 'OPEN_HOUR']
        if len(oh) < 5:  # mínimo 5 min para señal válida
            continue

        oh_open  = oh['open'].iloc[0]
        oh_close = oh['close'].iloc[-1]
        first_hour_ret = (oh_close - oh_open) / oh_open

        # Post-primera hora: 14:30-20:00 UTC
        post = group[group['session'].isin(['MIDDAY', 'CLOSE_HOUR'])]
        if len(post) < 10:
            continue

        entry_price = oh_close   # entrada al cierre de la primera hora
        exit_price  = post['close'].iloc[-1]  # salida al cierre de NY

        # Señal
        if first_hour_ret > threshold_pct:
            signal = 1
        elif first_hour_ret < -threshold_pct:
            signal = -1
        else:
            signal = 0

        # Retorno bruto del trade (en dirección de la señal)
        if signal == 0:
            trade_ret_gross = 0.0
        else:
            trade_ret_gross = signal * (exit_price - entry_price) / entry_price

        # Retorno neto (descontar costos como % del precio)
        cost_pct = COST_RT / entry_price
        trade_ret_net = trade_ret_gross - cost_pct

        records.append({
            'date':            pd.Timestamp(date, tz='UTC'),
            'first_hour_ret':  first_hour_ret,
            'entry_price':     entry_price,
            'exit_price':      exit_price,
            'day_ret':         (exit_price - entry_price) / entry_price,
            'signal':          signal,
            'trade_ret_gross': trade_ret_gross,
            'trade_ret_net':   trade_ret_net,
            'year':            date.year,
        })

    df_signals = pd.DataFrame(records).set_index('date')
    return df_signals


# ═══════════════════════════════════════════════════════════
# 2. ESTADÍSTICAS BASE (pre-estrategia)
# ═══════════════════════════════════════════════════════════

def analyze_correlation(df_signals: pd.DataFrame, label: str = "FULL") -> dict:
    """Analiza la correlación estadística 1H→Día sin sesgos de costos."""
    active = df_signals[df_signals['signal'] != 0].copy()

    # Correlación entre retorno 1H y retorno post-1H
    r_pearson, p_pearson = stats.pearsonr(
        active['first_hour_ret'],
        active['day_ret']
    )
    r_spearman, p_spearman = stats.spearmanr(
        active['first_hour_ret'],
        active['day_ret']
    )

    # ¿Siguen la misma dirección?
    same_direction = (active['first_hour_ret'] * active['day_ret'] > 0)
    dir_rate = same_direction.mean()
    n = len(active)
    z_dir = (dir_rate - 0.5) / np.sqrt(0.25 / n)
    p_dir = 2 * (1 - stats.norm.cdf(abs(z_dir)))

    # Test binomial
    binom = stats.binomtest(int(same_direction.sum()), n, 0.5, alternative='greater')

    logger.info(f"\n[{label}] ── CORRELACIÓN H3: PRIMERA HORA → DÍA ──")
    logger.info(f"  N días activos (con señal): {n}")
    logger.info(f"  Pearson  r = {r_pearson:.4f}  (p={p_pearson:.6f})")
    logger.info(f"  Spearman r = {r_spearman:.4f}  (p={p_spearman:.6f})")
    logger.info(f"  % días en misma dirección: {dir_rate*100:.1f}%  (Z={z_dir:.2f}, p={p_dir:.6f})")
    logger.info(f"  Test binomial p (>50%): {binom.pvalue:.6f}")

    is_sig = r_pearson > 0.15 and p_pearson < 0.05 and dir_rate > 0.52
    if is_sig:
        logger.info(f"  ✅ CORRELACIÓN SIGNIFICATIVA ({dir_rate*100:.1f}% misma dirección)")
    else:
        logger.info(f"  ❌ CORRELACIÓN NO EXPLOTABLE")

    return {
        'n': n,
        'pearson_r': float(r_pearson),
        'pearson_p': float(p_pearson),
        'spearman_r': float(r_spearman),
        'spearman_p': float(p_spearman),
        'dir_rate': float(dir_rate),
        'z_dir': float(z_dir),
        'p_dir': float(p_dir),
        'binom_p': float(binom.pvalue),
        'is_significant': bool(is_sig),
    }


# ═══════════════════════════════════════════════════════════
# 3. BACKTEST BASE
# ═══════════════════════════════════════════════════════════

def backtest(df_signals: pd.DataFrame, label: str = "FULL",
             use_net: bool = True) -> dict:
    """Simula equity curve de la estrategia H3."""
    active = df_signals[df_signals['signal'] != 0].copy()
    if active.empty:
        return {}

    ret_col = 'trade_ret_net' if use_net else 'trade_ret_gross'
    rets = active[ret_col].values

    equity = np.cumprod(1 + rets)
    total_ret = equity[-1] - 1
    n_days    = (active.index[-1] - active.index[0]).days / 365.25
    ann_ret   = (equity[-1] ** (1 / n_days) - 1) if n_days > 0 else 0
    mean_ret  = rets.mean()
    std_ret   = rets.std()
    sharpe    = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0

    win_mask  = rets > 0
    win_rate  = win_mask.mean()
    avg_win   = rets[win_mask].mean() if win_mask.any() else 0
    avg_loss  = rets[~win_mask].mean() if (~win_mask).any() else 0
    profit_factor = (rets[win_mask].sum() / abs(rets[~win_mask].sum())
                     if (~win_mask).any() and rets[~win_mask].sum() != 0 else np.inf)

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / peak
    max_dd = dd.min()

    # T-test
    t, p = stats.ttest_1samp(rets, 0)

    logger.info(f"\n[{label}] ── BACKTEST {'NETO' if use_net else 'BRUTO'} ──")
    logger.info(f"  N trades:            {len(active)}")
    logger.info(f"  Retorno total:       {total_ret*100:.2f}%")
    logger.info(f"  Retorno anualizado:  {ann_ret*100:.2f}%")
    logger.info(f"  Sharpe:              {sharpe:.3f}")
    logger.info(f"  Win Rate:            {win_rate*100:.1f}%")
    logger.info(f"  Avg Win/Loss:        {avg_win*100:.3f}% / {avg_loss*100:.3f}%")
    logger.info(f"  Profit Factor:       {profit_factor:.3f}")
    logger.info(f"  Max Drawdown:        {max_dd*100:.2f}%")
    logger.info(f"  T-test vs 0:         t={t:.3f}, p={p:.6f}")

    if p < 0.05 and mean_ret > 0:
        logger.info(f"  ✅ EDGE ESTADÍSTICO (p<0.05, retorno>0)")
    else:
        logger.info(f"  ❌ Edge no significativo o negativo")

    return {
        'n': int(len(active)),
        'total_ret': float(total_ret),
        'annual_ret': float(ann_ret),
        'sharpe': float(sharpe),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'max_dd': float(max_dd),
        'pvalue': float(p),
        'equity': equity.tolist(),
        'dates': [str(d.date()) for d in active.index],
        'returns': rets.tolist(),
    }


# ═══════════════════════════════════════════════════════════
# 4. STRESS TESTS (destrucción sistemática)
# ═══════════════════════════════════════════════════════════

def run_stress_tests(df_signals: pd.DataFrame) -> dict:
    """
    Intenta destruir el edge con variaciones realistas.
    Un edge real debe sobrevivir TODOS estos tests.
    """
    logger.info("\n\n── STRESS TESTS — INTENTANDO DESTRUIR EL EDGE ──")
    results = {}

    def _test(label, signals_mod):
        active = signals_mod[signals_mod['signal'] != 0]
        if active.empty:
            return {'label': label, 'win_rate': 0, 'sharpe': -99, 'pvalue': 1, 'survived': False}
        rets = active['trade_ret_net'].values
        if len(rets) < 10:
            return {'label': label, 'win_rate': 0, 'sharpe': -99, 'pvalue': 1, 'survived': False}
        mean_r = rets.mean()
        std_r  = rets.std()
        sharpe = (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0
        wr     = (rets > 0).mean()
        _, p   = stats.ttest_1samp(rets, 0)
        survived = bool(mean_r > 0 and p < 0.10)
        icon = "✅" if survived else "❌"
        logger.info(f"  {icon} {label:35s}: WR={wr*100:.1f}%  Sharpe={sharpe:.3f}  p={p:.4f}")
        return {'label': label, 'win_rate': float(wr), 'sharpe': float(sharpe),
                'pvalue': float(p), 'survived': survived}

    # 1. Spread x2
    mod = df_signals.copy()
    extra_cost_pct = SPREAD_ENTRY / mod['entry_price']
    mod['trade_ret_net'] = mod['trade_ret_net'] - extra_cost_pct
    results['spread_x2'] = _test("Spread x2", mod)

    # 2. Spread x3
    mod2 = df_signals.copy()
    extra_cost_pct2 = 2 * SPREAD_ENTRY / mod2['entry_price']
    mod2['trade_ret_net'] = mod2['trade_ret_net'] - extra_cost_pct2
    results['spread_x3'] = _test("Spread x3", mod2)

    # 3. Entrada retrasada 15 min (usa precio 15min después del cierre 1H)
    #    Aproximamos: gap de mercado medio en NY ≈ 0.05% adicional en contra
    mod3 = df_signals.copy()
    delay_cost = 0.0005  # 0.05% de desplazamiento adverso medio
    mod3['trade_ret_net'] = mod3['trade_ret_net'] - abs(mod3['signal']) * delay_cost
    results['delay_15m'] = _test("Entrada retrasada 15min", mod3)

    # 4. Slippage x3
    mod4 = df_signals.copy()
    extra_slip = 2 * SLIPPAGE / mod4['entry_price']
    mod4['trade_ret_net'] = mod4['trade_ret_net'] - extra_slip
    results['slippage_x3'] = _test("Slippage x3", mod4)

    # 5. Sin top 10% trades
    mod5 = df_signals.copy()
    top10_threshold = np.percentile(mod5[mod5['signal'] != 0]['trade_ret_gross'], 90)
    mod5.loc[mod5['trade_ret_gross'] >= top10_threshold, 'signal'] = 0
    results['no_outliers'] = _test("Sin top 10% trades (sin outliers)", mod5)

    # 6. Solo 2022 (bear market)
    mod6 = df_signals[df_signals['year'] == 2022].copy()
    results['bear_2022'] = _test("Solo 2022 (BEAR market)", mod6)

    # 7. Solo 2021
    mod7 = df_signals[df_signals['year'] == 2021].copy()
    results['year_2021'] = _test("Solo 2021", mod7)

    # 8. Umbral 0 (sin filtro de magnitud)
    mod8 = df_signals.copy()
    mod8['signal'] = np.sign(mod8['first_hour_ret'])
    rets8 = mod8[mod8['signal'] != 0]['trade_ret_net'].values
    if len(rets8) > 0:
        cost_adj = COST_RT / mod8[mod8['signal'] != 0]['entry_price'].values
        mod8.loc[mod8['signal'] != 0, 'trade_ret_net'] -= cost_adj
    results['no_threshold'] = _test("Sin umbral (threshold=0)", mod8)

    # 9. Solo señales LONG
    mod9 = df_signals.copy()
    mod9.loc[mod9['signal'] == -1, 'signal'] = 0
    results['long_only'] = _test("Solo LONG", mod9)

    # 10. Solo señales SHORT
    mod10 = df_signals.copy()
    mod10.loc[mod10['signal'] == 1, 'signal'] = 0
    results['short_only'] = _test("Solo SHORT", mod10)

    survived_count = sum(1 for v in results.values() if v.get('survived', False))
    logger.info(f"\n  Stress tests superados: {survived_count}/{len(results)}")

    return results


# ═══════════════════════════════════════════════════════════
# 5. SENSIBILIDAD DE PARÁMETROS
# ═══════════════════════════════════════════════════════════

def parameter_sensitivity(df_is: pd.DataFrame) -> pd.DataFrame:
    """
    ¿El edge depende de un umbral específico?
    Tests en IS únicamente — si hay un pico estrecho → curve fitting.
    Un edge real debe ser ancho y plano.
    """
    logger.info("\n── SENSIBILIDAD: UMBRAL DE ENTRADA ──")
    thresholds = [0.0, 0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020]
    rows = []

    for thr in thresholds:
        sigs = build_daily_signals(
            df_is.assign(**{c: df_is[c] for c in df_is.columns}),
            threshold_pct=thr
        )
        active = sigs[sigs['signal'] != 0]
        if len(active) < 20:
            continue
        rets = active['trade_ret_net'].values
        mean_r = rets.mean()
        sharpe = (mean_r / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
        wr     = (rets > 0).mean()
        _, p   = stats.ttest_1samp(rets, 0)
        rows.append({
            'threshold': thr,
            'n_trades':  len(active),
            'mean_return': float(mean_r),
            'sharpe':      float(sharpe),
            'win_rate':    float(wr),
            'pvalue':      float(p),
        })
        logger.info(f"  thr={thr:.3f}: n={len(active):4d}  wr={wr*100:.1f}%  "
                    f"sharpe={sharpe:.3f}  p={p:.4f}")

    df_sens = pd.DataFrame(rows)
    return df_sens


# ═══════════════════════════════════════════════════════════
# 6. MONTE CARLO
# ═══════════════════════════════════════════════════════════

def monte_carlo(df_signals: pd.DataFrame, n_runs: int = 2000) -> dict:
    """
    Monte Carlo sobre el orden de los trades.
    ¿Cuántas veces el equity aleatorio supera el real?
    """
    active = df_signals[df_signals['signal'] != 0].copy()
    rets   = active['trade_ret_net'].values
    n      = len(rets)

    if n < 20:
        return {'n_runs': 0}

    real_sharpe = (rets.mean() / rets.std()) * np.sqrt(252)
    real_equity = np.cumprod(1 + rets)[-1] - 1

    rng = np.random.default_rng(42)
    sharpes    = []
    equities   = []
    max_dds    = []

    for _ in range(n_runs):
        shuffled  = rng.permutation(rets)
        eq        = np.cumprod(1 + shuffled)
        peak      = np.maximum.accumulate(eq)
        dd        = (eq - peak) / peak
        s         = (shuffled.mean() / shuffled.std()) * np.sqrt(252) if shuffled.std() > 0 else 0
        sharpes.append(s)
        equities.append(eq[-1] - 1)
        max_dds.append(dd.min())

    sharpes  = np.array(sharpes)
    equities = np.array(equities)
    max_dds  = np.array(max_dds)

    pct_positive   = (equities > 0).mean()
    pct_beat_real  = (equities > real_equity).mean()  # robustez vs suerte
    p5_dd          = np.percentile(max_dds, 5)
    ruin_prob      = (max_dds < -0.50).mean()

    logger.info("\n── MONTE CARLO (orden aleatorio de trades) ──")
    logger.info(f"  N runs: {n_runs}")
    logger.info(f"  Sharpe real: {real_sharpe:.3f}")
    logger.info(f"  Sharpe mediana MC: {np.median(sharpes):.3f}")
    logger.info(f"  % runs positivos: {pct_positive*100:.1f}%")
    logger.info(f"  % runs > retorno real (suerte?): {pct_beat_real*100:.1f}%")
    logger.info(f"  Max DD mediana: {np.median(max_dds)*100:.1f}%")
    logger.info(f"  Max DD P5 (peor 5%): {p5_dd*100:.1f}%")
    logger.info(f"  Probabilidad ruina (DD<-50%): {ruin_prob*100:.1f}%")

    if ruin_prob < 0.05 and pct_positive > 0.60:
        logger.info("  ✅ MC: Edge robusto al orden de trades")
    else:
        logger.info("  ❌ MC: Edge frágil o dependiente del orden")

    return {
        'n_runs': n_runs,
        'n_trades': int(n),
        'real_sharpe': float(real_sharpe),
        'real_equity': float(real_equity),
        'pct_positive': float(pct_positive),
        'pct_beat_real': float(pct_beat_real),
        'median_sharpe': float(np.median(sharpes)),
        'p5_dd': float(p5_dd),
        'ruin_prob': float(ruin_prob),
        'sharpes': sharpes.tolist(),
        'equities': equities.tolist(),
        'max_dds': max_dds.tolist(),
    }


# ═══════════════════════════════════════════════════════════
# 7. ANÁLISIS POR AÑO (estabilidad temporal)
# ═══════════════════════════════════════════════════════════

def yearly_analysis(df_signals: pd.DataFrame) -> pd.DataFrame:
    """¿El edge es estable año a año o depende de un período específico?"""
    logger.info("\n── ANÁLISIS POR AÑO ──")
    rows = []

    for year, group in df_signals.groupby('year'):
        active = group[group['signal'] != 0]
        if len(active) < 10:
            continue
        rets   = active['trade_ret_net'].values
        wr     = (rets > 0).mean()
        mean_r = rets.mean()
        sharpe = (mean_r / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
        total  = np.prod(1 + rets) - 1
        _, p   = stats.ttest_1samp(rets, 0)
        icon   = "✅" if mean_r > 0 else "❌"
        logger.info(f"  {icon} {year}: n={len(active):3d}  WR={wr*100:.1f}%  "
                    f"Ret={total*100:.2f}%  Sharpe={sharpe:.3f}  p={p:.4f}")
        rows.append({
            'year': int(year), 'n': int(len(active)),
            'win_rate': float(wr), 'total_return': float(total),
            'sharpe': float(sharpe), 'pvalue': float(p), 'positive': bool(mean_r > 0),
        })

    df_yearly = pd.DataFrame(rows)
    if not df_yearly.empty:
        pct_pos_years = df_yearly['positive'].mean()
        logger.info(f"\n  Años positivos: {pct_pos_years*100:.0f}%  "
                    f"({df_yearly['positive'].sum()}/{len(df_yearly)})")
        if pct_pos_years >= 0.70:
            logger.info("  ✅ Edge estable en el tiempo")
        else:
            logger.info("  ❌ Edge inestable o dependiente del período")
    return df_yearly


# ═══════════════════════════════════════════════════════════
# 8. VISUALIZACIÓN
# ═══════════════════════════════════════════════════════════

def plot_results(bt_is: dict, bt_oos: dict, mc: dict,
                 yearly: pd.DataFrame, stress: dict,
                 sens: pd.DataFrame, df_all: pd.DataFrame) -> None:

    fig = plt.figure(figsize=(20, 22), facecolor='#0d1117')
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    GOLD   = '#FFD700'
    GREEN  = '#00FF88'
    RED    = '#FF4444'
    BLUE   = '#4488FF'
    GRAY   = '#888888'
    BG     = '#161b22'

    def ax_style(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, color=GOLD, fontsize=11, fontweight='bold', pad=8)
        ax.tick_params(colors=GRAY)
        ax.spines[:].set_color('#333333')
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color(GRAY)

    # ── 1. Equity IS vs OOS ──
    ax1 = fig.add_subplot(gs[0, :2])
    if bt_is.get('equity'):
        eq_is = np.array(bt_is['equity'])
        ax1.plot(eq_is, color=BLUE, lw=2, label=f"IS  Sharpe={bt_is['sharpe']:.2f}")
    if bt_oos.get('equity'):
        eq_oos = np.array(bt_oos['equity'])
        oos_x  = np.arange(len(eq_is) if bt_is.get('equity') else 0,
                           len(eq_is if bt_is.get('equity') else []) + len(eq_oos))
        ax1.plot(oos_x, eq_oos, color=GREEN, lw=2,
                 label=f"OOS Sharpe={bt_oos['sharpe']:.2f}")
    ax1.axhline(1.0, color=GRAY, lw=0.8, ls='--')
    if bt_is.get('equity'):
        ax1.axvline(len(bt_is['equity']), color=RED, lw=1.5, ls=':', label="IS/OOS split")
    ax1.legend(facecolor=BG, labelcolor='white', fontsize=9)
    ax_style(ax1, "H3 — EQUITY CURVE: Correlación 1H → Día")
    ax1.set_ylabel("Equity (base=1.0)", color=GRAY)

    # ── 2. Retorno por año ──
    ax2 = fig.add_subplot(gs[0, 2])
    if not yearly.empty:
        colors = [GREEN if r > 0 else RED for r in yearly['total_return']]
        ax2.bar(yearly['year'].astype(str), yearly['total_return'] * 100, color=colors)
    ax2.axhline(0, color=GRAY, lw=0.8)
    ax_style(ax2, "Retorno por Año (%)")
    ax2.set_ylabel("Retorno %", color=GRAY)
    ax2.tick_params(axis='x', rotation=45)

    # ── 3. Distribución retornos diarios ──
    ax3 = fig.add_subplot(gs[1, 0])
    active_all = df_all[df_all['signal'] != 0]['trade_ret_net'].values * 100
    if len(active_all) > 0:
        ax3.hist(active_all, bins=60, color=BLUE, alpha=0.75, edgecolor='none')
        ax3.axvline(active_all.mean(), color=GOLD, lw=2, ls='--',
                    label=f"μ={active_all.mean():.3f}%")
        ax3.axvline(0, color=RED, lw=1, ls=':')
        ax3.legend(facecolor=BG, labelcolor='white', fontsize=8)
    ax_style(ax3, "Distribución Retornos Diarios")
    ax3.set_xlabel("Retorno neto (%)", color=GRAY)

    # ── 4. Scatter 1H_ret vs day_ret ──
    ax4 = fig.add_subplot(gs[1, 1])
    active_df = df_all[df_all['signal'] != 0]
    if len(active_df) > 0:
        sc = ax4.scatter(active_df['first_hour_ret'] * 100,
                         active_df['day_ret'] * 100,
                         c=active_df['signal'], cmap='RdYlGn',
                         alpha=0.4, s=8)
        # Línea de regresión
        x = active_df['first_hour_ret'].values
        y = active_df['day_ret'].values
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 50)
        ax4.plot(x_line * 100, (m * x_line + b) * 100, color=GOLD, lw=2)
    ax4.axhline(0, color=GRAY, lw=0.5)
    ax4.axvline(0, color=GRAY, lw=0.5)
    ax_style(ax4, f"Scatter: 1H ret vs Día ret  (r={bt_is.get('pearson_r', 0):.3f})")
    ax4.set_xlabel("Primera hora (%)", color=GRAY)
    ax4.set_ylabel("Resto del día (%)", color=GRAY)

    # ── 5. Sensibilidad umbral ──
    ax5 = fig.add_subplot(gs[1, 2])
    if not sens.empty:
        ax5.plot(sens['threshold'] * 100, sens['sharpe'],
                 color=BLUE, lw=2, marker='o', markersize=5)
        ax5.axhline(0, color=GRAY, lw=0.8, ls='--')
    ax_style(ax5, "Sensibilidad: Umbral vs Sharpe")
    ax5.set_xlabel("Umbral 1H (% retorno)", color=GRAY)
    ax5.set_ylabel("Sharpe", color=GRAY)

    # ── 6. Monte Carlo equity distribution ──
    ax6 = fig.add_subplot(gs[2, 0])
    if mc.get('equities'):
        mc_eq = np.array(mc['equities'])
        ax6.hist(mc_eq * 100, bins=60, color=GRAY, alpha=0.5, label='Aleatorio')
        ax6.axvline(mc['real_equity'] * 100, color=GOLD, lw=2.5, ls='--',
                    label=f"Real: {mc['real_equity']*100:.1f}%")
        ax6.legend(facecolor=BG, labelcolor='white', fontsize=8)
    ax_style(ax6, f"MC: Equidad Real vs Aleatoria  (p_pos={mc.get('pct_positive',0)*100:.0f}%)")
    ax6.set_xlabel("Retorno total (%)", color=GRAY)

    # ── 7. Monte Carlo drawdown distribution ──
    ax7 = fig.add_subplot(gs[2, 1])
    if mc.get('max_dds'):
        mc_dd = np.array(mc['max_dds'])
        ax7.hist(mc_dd * 100, bins=60, color=RED, alpha=0.5)
        ax7.axvline(mc.get('p5_dd', 0) * 100, color=GOLD, lw=2, ls='--',
                    label=f"P5 DD: {mc.get('p5_dd',0)*100:.1f}%")
        ax7.legend(facecolor=BG, labelcolor='white', fontsize=8)
    ax_style(ax7, f"MC: Distribución Max Drawdown  (ruina={mc.get('ruin_prob',0)*100:.1f}%)")
    ax7.set_xlabel("Max DD (%)", color=GRAY)

    # ── 8. Stress tests ──
    ax8 = fig.add_subplot(gs[2, 2])
    if stress:
        labels  = [v['label'][:28] for v in stress.values()]
        sharpes = [v['sharpe'] for v in stress.values()]
        colors  = [GREEN if v['survived'] else RED for v in stress.values()]
        ybars   = range(len(labels))
        ax8.barh(list(ybars), sharpes, color=colors, alpha=0.8)
        ax8.set_yticks(list(ybars))
        ax8.set_yticklabels(labels, fontsize=7)
        ax8.axvline(0, color=GRAY, lw=0.8, ls='--')
    ax_style(ax8, "Stress Tests — Sharpe Ratio")
    ax8.tick_params(axis='y', labelcolor=GRAY)

    # ── 9. IS vs OOS comparative ──
    ax9 = fig.add_subplot(gs[3, :])
    metrics = ['sharpe', 'win_rate', 'annual_ret', 'max_dd']
    labels_m = ['Sharpe', 'Win Rate', 'Ann. Return', 'Max DD']
    is_vals  = [bt_is.get(m, 0) for m in metrics]
    oos_vals = [bt_oos.get(m, 0) for m in metrics]
    x_pos = np.arange(len(metrics))
    width = 0.35
    ax9.bar(x_pos - width/2, is_vals,  width, label='IS',  color=BLUE, alpha=0.8)
    ax9.bar(x_pos + width/2, oos_vals, width, label='OOS', color=GREEN, alpha=0.8)
    ax9.set_xticks(x_pos)
    ax9.set_xticklabels(labels_m, color=GRAY)
    ax9.axhline(0, color=GRAY, lw=0.8)
    ax9.legend(facecolor=BG, labelcolor='white', fontsize=9)
    ax_style(ax9, "Comparación IS vs OOS — Métricas Clave")

    # ── TÍTULO PRINCIPAL ──
    stress_survived = sum(1 for v in stress.values() if v.get('survived', False))
    total_stress    = len(stress)
    fig.suptitle(
        f"H3: PRIMERA HORA NY → DIRECCIÓN DEL DÍA\n"
        f"IS Sharpe={bt_is.get('sharpe',0):.2f}  |  OOS Sharpe={bt_oos.get('sharpe',0):.2f}  |  "
        f"Stress: {stress_survived}/{total_stress}  |  MC Ruina={mc.get('ruin_prob',0)*100:.1f}%",
        color='white', fontsize=14, fontweight='bold', y=0.98
    )

    out_path = ARTIFACTS_DIR / "nq_first_hour_edge.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"\n  ✅ Gráfico guardado: {out_path}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║   H3: PRIMERA HORA NY → DIRECCIÓN DEL DÍA                     ║")
    logger.info("║   Validación Fase 6 — Intentar destruir el edge                ║")
    logger.info("╚" + "═" * 68 + "╝")

    # ── Cargar datos ──
    parquet_path = PROJECT_ROOT / "quant_bot" / "data" / "processed" / "USATECHIDXUSD_M1.parquet"
    logger.info(f"\n  Cargando {parquet_path.name}...")
    df = pd.read_parquet(parquet_path, engine='pyarrow')
    logger.info(f"  → {len(df):,} barras M1  ({df.index[0].date()} → {df.index[-1].date()})")

    if 'session' not in df.columns:
        from quant_bot.data.nq_loader import add_session_labels
        df = add_session_labels(df)

    # ── IS / OOS split ──
    df_is  = df[df.index.year < OOS_YEAR]
    df_oos = df[df.index.year >= OOS_YEAR]
    logger.info(f"\n  IS:  {df_is.index[0].date()} → {df_is.index[-1].date()}")
    logger.info(f"  OOS: {df_oos.index[0].date()} → {df_oos.index[-1].date()}")

    # ── Construcción de señales (umbral base = 0.2%) ──
    THRESHOLD = 0.002  # 0.2% minimal move para señal

    logger.info(f"\n  Umbral de señal: {THRESHOLD*100:.1f}% (primera hora NQ)")
    logger.info(f"  Costo round-trip: {COST_RT:.5f} units = {COST_RT*DUKASCOPY_SCALE:.2f} pts NQ equiv.")

    sigs_is  = build_daily_signals(df_is,  threshold_pct=THRESHOLD)
    sigs_oos = build_daily_signals(df_oos, threshold_pct=THRESHOLD)
    sigs_all = pd.concat([sigs_is, sigs_oos])

    logger.info(f"\n  Señales IS:  {(sigs_is['signal']  != 0).sum()} / {len(sigs_is)}")
    logger.info(f"  Señales OOS: {(sigs_oos['signal'] != 0).sum()} / {len(sigs_oos)}")

    # ── ANÁLISIS DE CORRELACIÓN ──
    corr_is  = analyze_correlation(sigs_is,  "IS")
    corr_oos = analyze_correlation(sigs_oos, "OOS")

    # ── BACKTEST IS / OOS ──
    bt_is  = backtest(sigs_is,  "IS",   use_net=True)
    bt_oos = backtest(sigs_oos, "OOS",  use_net=True)

    # Añadir pearson_r al bt_is para el gráfico
    bt_is['pearson_r'] = corr_is['pearson_r']

    # Backtest bruto IS (para ver edge antes de costos)
    bt_is_gross = backtest(sigs_is, "IS BRUTO", use_net=False)

    # ── STRESS TESTS (IS completo) ──
    stress = run_stress_tests(sigs_is)

    # ── SENSIBILIDAD (IS únicamente — no contaminar OOS) ──
    sens = parameter_sensitivity(df_is)

    # ── ESTABILIDAD TEMPORAL ──
    yearly = yearly_analysis(sigs_all)

    # ── MONTE CARLO (IS) ──
    mc = monte_carlo(sigs_is, n_runs=2000)

    # ── CLASIFICACIÓN FINAL ──
    logger.info("\n\n" + "═" * 70)
    logger.info("  CLASIFICACIÓN FINAL — H3: PRIMERA HORA NY")
    logger.info("═" * 70)

    checks = {
        'corr_IS_significativa':    corr_is['is_significant'],
        'corr_OOS_significativa':   corr_oos['is_significant'],
        'backtest_IS_positivo':     bt_is.get('pvalue', 1) < 0.05 and bt_is.get('sharpe', 0) > 0.3,
        'backtest_OOS_positivo':    bt_oos.get('pvalue', 1) < 0.10 and bt_oos.get('sharpe', 0) > 0,
        'stress_spread_x2':         stress.get('spread_x2', {}).get('survived', False),
        'stress_no_outliers':       stress.get('no_outliers', {}).get('survived', False),
        'stress_bear_2022':         stress.get('bear_2022', {}).get('survived', False),
        'edge_estable_temporal':    yearly['positive'].mean() >= 0.60 if not yearly.empty else False,
        'mc_robusto':               mc.get('ruin_prob', 1) < 0.10 and mc.get('pct_positive', 0) > 0.55,
        'bruto_positivo':           bt_is_gross.get('sharpe', 0) > 0.5,
    }

    score = sum(checks.values())
    max_score = len(checks)

    for k, v in checks.items():
        icon = "✅" if v else "❌"
        logger.info(f"  {icon} {k}")

    logger.info(f"\n  SCORE: {score}/{max_score}")

    if score >= 8:
        verdict = "🏆 EDGE ROBUSTO REAL — candidato para trading"
    elif score >= 6:
        verdict = "✅ EDGE PROMETEDOR — necesita validación tick"
    elif score >= 4:
        verdict = "⚠️  EDGE DÉBIL — no implementar aún"
    else:
        verdict = "❌ ILUSIÓN ESTADISTICA — no explotable"

    logger.info(f"  VEREDICTO: {verdict}")

    # ── GRÁFICO ──
    plot_results(bt_is, bt_oos, mc, yearly, stress, sens, sigs_all)

    # ── GUARDAR MÉTRICAS ──
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.bool_,)): return bool(obj)
            if isinstance(obj, (np.ndarray,)): return obj.tolist()
            return super().default(obj)

    output = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'threshold': THRESHOLD,
        'oos_year': OOS_YEAR,
        'cost_rt_duck': float(COST_RT),
        'cost_rt_nq_pts': float(COST_RT * DUKASCOPY_SCALE),
        'correlation_IS': corr_is,
        'correlation_OOS': corr_oos,
        'backtest_IS': {k: v for k, v in bt_is.items()
                        if k not in ('equity', 'dates', 'returns')},
        'backtest_OOS': {k: v for k, v in bt_oos.items()
                         if k not in ('equity', 'dates', 'returns')},
        'stress_tests': {k: {kk: vv for kk, vv in v.items() if kk != 'label'}
                         for k, v in stress.items()},
        'monte_carlo': {k: v for k, v in mc.items()
                        if k not in ('sharpes', 'equities', 'max_dds')},
        'checks': {k: bool(v) for k, v in checks.items()},
        'score': int(score),
        'max_score': int(max_score),
        'verdict': verdict,
    }

    out_json = ARTIFACTS_DIR / "h3_first_hour_metrics.json"
    with open(out_json, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\n  Métricas guardadas: {out_json}")
    logger.info("  ✅ Script H3 completado")


if __name__ == "__main__":
    main()
