"""
nq_short_overnight.py — Validación del Edge: SHORT Overnight NQ100

HIPÓTESIS (H4):
  El retorno overnight del NQ100 (cierre NY → apertura NY) es
  sistemáticamente NEGATIVO. Vender al cierre de NY (20:00 UTC)
  y recomprar a la apertura (13:30 UTC siguiente día) genera retorno
  positivo neto de costos.

  El análisis previo mostró:
    - T-stat = -5.09  (p < 0.0001) → negativo y significativo
    - Tendencia confirmada en IS, OOS y todos los regímenes

ESTRATEGIA:
    - Entrada SELL: último minuto de CLOSE_HOUR (precio cierre ~20:00 UTC)
    - Salida  BUY:  primer minuto de OPEN_HOUR (precio apertura ~13:30 UTC)
    - Dirección: SIEMPRE SHORT (no filtro de dirección)
    - SL: ninguno en posición raw (se estudia el riesgo de gap)

COSTOS OVERNIGHT (más altos por horario extendido):
    - Spread extendido: ~8 pts NQ = 8/82 ≈ 0.098 Dukascopy
    - Slippage: ~3 pts NQ = 3/82 ≈ 0.037 Dukascopy
    - Round-trip total: ~22 pts NQ ≈ 0.268 Dukascopy

FILOSOFÍA (Fase 6):
    "El sistema NO intenta demostrar que funciona.
     Intenta demostrar que NO funciona."
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

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
        logging.FileHandler(ARTIFACTS_DIR / "short_overnight.log"),
    ]
)
logger = logging.getLogger("H4_ShortOvernight")

# ─────────────────────────────────────────────────────────
# Escala Dukascopy: precio ~128-262 = NQ real / ~82
# ─────────────────────────────────────────────────────────
DUKASCOPY_SCALE = 82.0

# Costos OVERNIGHT (spread más ancho en horario extendido)
SPREAD_ON_ENTRY  = 8.0 / DUKASCOPY_SCALE   # 8 pts NQ = 0.098
SPREAD_ON_EXIT   = 8.0 / DUKASCOPY_SCALE   # 8 pts NQ = 0.098
SLIP_ON          = 3.0 / DUKASCOPY_SCALE   # 3 pts NQ = 0.037
COST_RT_ON       = SPREAD_ON_ENTRY + SPREAD_ON_EXIT + 2 * SLIP_ON  # ~0.268

OOS_YEAR = 2025

logger.info(f"  Costos overnight round-trip:")
logger.info(f"    Spread entry+exit: {(SPREAD_ON_ENTRY+SPREAD_ON_EXIT)*DUKASCOPY_SCALE:.1f} pts NQ")
logger.info(f"    Slippage total:    {2*SLIP_ON*DUKASCOPY_SCALE:.1f} pts NQ")
logger.info(f"    TOTAL RT:          {COST_RT_ON*DUKASCOPY_SCALE:.1f} pts NQ = {COST_RT_ON:.4f} Dukascopy")


# ═══════════════════════════════════════════════════════════
# 1. CONSTRUIR TRADES OVERNIGHT
# ═══════════════════════════════════════════════════════════

def build_overnight_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada transición día-a-día construye un trade SHORT overnight:
      - entry_price:  último precio de CLOSE_HOUR del día D   (~20:00 UTC)
      - exit_price:   primer precio de OPEN_HOUR del día D+1  (~13:30 UTC)
      - overnight_ret BRUTO: (entry - exit) / entry  [short = ganamos si baja]
      - overnight_ret NETO:  bruto - cost_rt_pct

    Incluye métricas de riesgo:
      - gap_abs:     magnitud del gap en % al abrir
      - adverse_move: máximo movimiento adverso durante la noche (si tenemos datos)
    """
    # Sesiones relevantes
    close_hour = df[df['session'] == 'CLOSE_HOUR'].copy()
    open_hour  = df[df['session'] == 'OPEN_HOUR'].copy()

    # Precio cierre diario: último bar de CLOSE_HOUR
    close_prices = close_hour.groupby(close_hour.index.date)['close'].last()
    close_prices.index = pd.to_datetime(close_prices.index, utc=True)

    # Precio apertura diario: primer bar de OPEN_HOUR
    open_prices = open_hour.groupby(open_hour.index.date)['open'].first()
    open_prices.index = pd.to_datetime(open_prices.index, utc=True)

    # Spread promedio al cierre y apertura
    close_spread = close_hour.groupby(close_hour.index.date)['spread_avg'].mean()
    close_spread.index = pd.to_datetime(close_spread.index, utc=True)
    open_spread = open_hour.groupby(open_hour.index.date)['spread_avg'].mean()
    open_spread.index = pd.to_datetime(open_spread.index, utc=True)

    # Movimiento overnight: desde CLOSE_HOUR hasta OPEN_HOUR (zona overnight)
    overnight_sess = df[df['session'] == 'OVERNIGHT'].copy()
    overnight_high = overnight_sess.groupby(overnight_sess.index.date)['high'].max()
    overnight_low  = overnight_sess.groupby(overnight_sess.index.date)['low'].min()
    overnight_high.index = pd.to_datetime(overnight_high.index, utc=True)
    overnight_low.index  = pd.to_datetime(overnight_low.index,  utc=True)

    records = []

    dates = sorted(close_prices.index)
    for i, date in enumerate(dates):
        # Buscamos el siguiente día de trading con apertura
        next_date = date + pd.Timedelta(days=1)
        # Máximo 4 días adelante (por fin de semana/festivos)
        found_next = None
        for delta in range(1, 5):
            candidate = date + pd.Timedelta(days=delta)
            if candidate in open_prices.index:
                found_next = candidate
                break

        if found_next is None:
            continue

        entry = close_prices[date]
        exit_ = open_prices[found_next]

        if pd.isna(entry) or pd.isna(exit_) or entry <= 0:
            continue

        # Retorno SHORT bruto: ganamos si el precio baja overnight
        short_ret_gross = (entry - exit_) / entry

        # Costo como % del precio de entrada
        cost_pct = COST_RT_ON / entry

        short_ret_net = short_ret_gross - cost_pct

        # Gap magnitude (entrada vs salida, sin dirección)
        gap_pct = abs((exit_ - entry) / entry)

        # Spread real observado
        real_spread_entry = close_spread.get(date, SPREAD_ON_ENTRY)
        real_spread_exit  = open_spread.get(found_next, SPREAD_ON_EXIT)

        # Overnight range (proxy de riesgo nocturno)
        on_date = (date + pd.Timedelta(days=1)).date()
        on_date_ts = pd.Timestamp(on_date, tz='UTC')
        overnight_range = 0.0
        if on_date_ts in overnight_high.index and on_date_ts in overnight_low.index:
            h = overnight_high[on_date_ts]
            l = overnight_low[on_date_ts]
            overnight_range = (h - l) / entry

        # Días calendario entre cierre y apertura
        gap_days = (found_next - date).days

        records.append({
            'date':             date,
            'next_date':        found_next,
            'gap_days':         int(gap_days),
            'entry_price':      float(entry),
            'exit_price':       float(exit_),
            'spread_entry':     float(real_spread_entry),
            'spread_exit':      float(real_spread_exit),
            'gap_abs_pct':      float(gap_pct),
            'overnight_range':  float(overnight_range),
            'ret_gross':        float(short_ret_gross),
            'cost_pct':         float(cost_pct),
            'ret_net':          float(short_ret_net),
            'year':             int(date.year),
            'month':            int(date.month),
        })

    df_trades = pd.DataFrame(records).set_index('date')
    logger.info(f"\n  Trades overnight construidos: {len(df_trades)}")
    logger.info(f"  Weekend gaps (>2 días): {(df_trades['gap_days'] > 2).sum()}")
    logger.info(f"  Gap promedio (abs): {df_trades['gap_abs_pct'].mean()*100:.3f}%")
    logger.info(f"  Gap max (abs):      {df_trades['gap_abs_pct'].max()*100:.3f}%")
    return df_trades


# ═══════════════════════════════════════════════════════════
# 2. BACKTEST
# ═══════════════════════════════════════════════════════════

def backtest_short_on(trades: pd.DataFrame, label: str = "FULL") -> dict:
    """Simula el equity curve del SHORT Overnight."""
    df = trades.dropna(subset=['ret_net']).copy()
    if df.empty:
        return {}

    rets_net   = df['ret_net'].values
    rets_gross = df['ret_gross'].values

    # Equity curves
    eq_net   = np.cumprod(1 + rets_net)
    eq_gross = np.cumprod(1 + rets_gross)

    total_net   = eq_net[-1]   - 1
    total_gross = eq_gross[-1] - 1

    n_days     = (df.index[-1] - df.index[0]).days / 365.25
    ann_net    = (eq_net[-1]   ** (1 / n_days) - 1) if n_days > 0 else 0
    ann_gross  = (eq_gross[-1] ** (1 / n_days) - 1) if n_days > 0 else 0

    mean_net = rets_net.mean()
    std_net  = rets_net.std()
    sharpe   = (mean_net / std_net) * np.sqrt(252) if std_net > 0 else 0

    win_mask = rets_net > 0
    win_rate = win_mask.mean()
    avg_win  = rets_net[win_mask].mean()  if win_mask.any()  else 0
    avg_loss = rets_net[~win_mask].mean() if (~win_mask).any() else 0
    pf       = (rets_net[win_mask].sum() / abs(rets_net[~win_mask].sum())
                if (~win_mask).any() and rets_net[~win_mask].sum() != 0 else np.inf)

    peak   = np.maximum.accumulate(eq_net)
    dd     = (eq_net - peak) / peak
    max_dd = dd.min()

    # Calmar
    calmar = ann_net / abs(max_dd) if max_dd < 0 else 0

    t, p = stats.ttest_1samp(rets_net, 0)

    logger.info(f"\n[{label}] ── SHORT OVERNIGHT BACKTEST ──")
    logger.info(f"  N trades:             {len(df)}")
    logger.info(f"  Período:              {df.index[0].date()} → {df.index[-1].date()}")
    logger.info(f"  Retorno bruto total:  {total_gross*100:.2f}%")
    logger.info(f"  Retorno neto total:   {total_net*100:.2f}%")
    logger.info(f"  Retorno anualizado NETO: {ann_net*100:.2f}%")
    logger.info(f"  Sharpe (neto):        {sharpe:.3f}")
    logger.info(f"  Calmar:               {calmar:.3f}")
    logger.info(f"  Win Rate:             {win_rate*100:.1f}%")
    logger.info(f"  Avg Win / Loss:       {avg_win*100:.3f}% / {avg_loss*100:.3f}%")
    logger.info(f"  Profit Factor:        {pf:.3f}")
    logger.info(f"  Max Drawdown:         {max_dd*100:.2f}%")
    logger.info(f"  T-test vs 0:          t={t:.3f}, p={p:.6f}")

    costo_por_trade = df['cost_pct'].mean() * 100
    logger.info(f"  Costo medio/trade:    {costo_por_trade:.4f}%")

    if mean_net > 0 and p < 0.05:
        logger.info(f"  ✅ EDGE SIGNIFICATIVO (p={p:.4f})")
    elif mean_net > 0 and p < 0.10:
        logger.info(f"  ⚠️  Edge marginal (p={p:.4f}) — necesita más datos")
    else:
        logger.info(f"  ❌ No significativo (p={p:.4f})")

    return {
        'n': int(len(df)),
        'total_gross': float(total_gross),
        'total_net': float(total_net),
        'annual_gross': float(ann_gross),
        'annual_net': float(ann_net),
        'sharpe': float(sharpe),
        'calmar': float(calmar),
        'win_rate': float(win_rate),
        'profit_factor': float(pf),
        'max_dd': float(max_dd),
        'pvalue': float(p),
        'equity_net': eq_net.tolist(),
        'equity_gross': eq_gross.tolist(),
        'dates': [str(d.date()) for d in df.index],
    }


# ═══════════════════════════════════════════════════════════
# 3. ANÁLISIS DE RIESGO DE GAP
# ═══════════════════════════════════════════════════════════

def analyze_gap_risk(trades: pd.DataFrame) -> dict:
    """
    El mayor riesgo del SHORT overnight es un gap alcista enorme.
    Cuantificamos la distribución de gaps adversos.
    """
    logger.info("\n── ANÁLISIS DE RIESGO DE GAP (SHORT) ──")

    # Short pierde cuando precio SUBE overnight (gap adverso)
    adverse  = trades[trades['ret_gross'] < 0]['gap_abs_pct'].values
    gaps_all = trades['ret_gross'].values  # positivo = short ganó

    # Percentiles de pérdida máxima
    losses = -gaps_all[gaps_all < 0]  # magnitud de pérdidas

    if len(losses) == 0:
        logger.info("  No hay días con pérdida bruta (raro)")
        return {}

    p95_loss = np.percentile(losses, 95)
    p99_loss = np.percentile(losses, 99)
    max_loss = losses.max()

    pct_days_loss = (gaps_all < 0).mean()
    logger.info(f"  % días con pérdida bruta (precio sube): {pct_days_loss*100:.1f}%")
    logger.info(f"  Pérdida P95 (1 en 20 días):  {p95_loss*100:.3f}%")
    logger.info(f"  Pérdida P99 (1 en 100 días): {p99_loss*100:.3f}%")
    logger.info(f"  Pérdida máxima histórica:    {max_loss*100:.3f}%")

    # ¿Cuántos días el gap adverso supera X pts NQ?
    for pts in [10, 25, 50, 100]:
        pct_duck = pts / DUKASCOPY_SCALE
        over_thresh = (trades['ret_gross'] < -pct_duck / trades['entry_price']).mean()
        logger.info(f"  Gap adverso > {pts:3d} pts NQ: {over_thresh*100:.2f}% días")

    # Weekend gaps (más peligrosos)
    weekend = trades[trades['gap_days'] > 2]
    weekday = trades[trades['gap_days'] <= 2]
    logger.info(f"\n  Weekend gap stats:")
    logger.info(f"    N weekends:    {len(weekend)}")
    logger.info(f"    WR weekends:   {(weekend['ret_net'] > 0).mean()*100:.1f}%")
    logger.info(f"    Sharpe wknd:   {(weekend['ret_net'].mean()/weekend['ret_net'].std())*np.sqrt(52):.3f}")
    logger.info(f"  Weekday gap stats:")
    logger.info(f"    N weekdays:    {len(weekday)}")
    logger.info(f"    WR weekdays:   {(weekday['ret_net'] > 0).mean()*100:.1f}%")

    return {
        'pct_loss_days': float(pct_days_loss),
        'p95_loss': float(p95_loss),
        'p99_loss': float(p99_loss),
        'max_loss': float(max_loss),
        'n_weekend': int(len(weekend)),
        'wr_weekend': float((weekend['ret_net'] > 0).mean()),
        'n_weekday': int(len(weekday)),
        'wr_weekday': float((weekday['ret_net'] > 0).mean()),
    }


# ═══════════════════════════════════════════════════════════
# 4. STRESS TESTS
# ═══════════════════════════════════════════════════════════

def stress_tests(trades: pd.DataFrame) -> dict:
    """Destrucción sistemática del edge."""
    logger.info("\n── STRESS TESTS — DESTRUYENDO EL EDGE ──")
    results = {}

    def _bt(label, t):
        t = t.dropna(subset=['ret_net'])
        if len(t) < 15:
            return {'label': label, 'annual_net': 0, 'sharpe': -99, 'pvalue': 1, 'survived': False}
        rets = t['ret_net'].values
        mean_r = rets.mean()
        std_r  = rets.std()
        sharpe = (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0
        wr     = (rets > 0).mean()
        _, p   = stats.ttest_1samp(rets, 0)
        eq     = np.cumprod(1 + rets)
        ann    = eq[-1] ** (252 / len(rets)) - 1
        survived = bool(mean_r > 0 and p < 0.10)
        icon = "✅" if survived else "❌"
        logger.info(f"  {icon} {label:38s}: WR={wr*100:.1f}%  Sharpe={sharpe:.3f}"
                    f"  Ann={ann*100:.1f}%  p={p:.4f}")
        return {
            'label': label, 'win_rate': float(wr), 'sharpe': float(sharpe),
            'annual_net': float(ann), 'pvalue': float(p), 'survived': survived,
        }

    # 1. Spread overnight x2
    t1 = trades.copy()
    t1['ret_net'] = t1['ret_gross'] - 2 * t1['cost_pct']
    results['spread_x2'] = _bt("Spread overnight x2", t1)

    # 2. Spread overnight x3
    t2 = trades.copy()
    t2['ret_net'] = t2['ret_gross'] - 3 * t2['cost_pct']
    results['spread_x3'] = _bt("Spread overnight x3", t2)

    # 3. Sin top 10% trades (sin noches más rentables)
    t3 = trades.copy()
    p90 = np.percentile(t3['ret_gross'], 90)
    t3 = t3[t3['ret_gross'] < p90]
    results['no_outliers'] = _bt("Sin top 10% noches (sin outliers)", t3)

    # 4. Solo 2022 (bear market)
    results['year_2022'] = _bt("Solo 2022 (BEAR)", trades[trades['year'] == 2022].copy())

    # 5. Solo 2023 (bull lateral)
    results['year_2023'] = _bt("Solo 2023 (BULL)", trades[trades['year'] == 2023].copy())

    # 6. Solo 2024
    results['year_2024'] = _bt("Solo 2024", trades[trades['year'] == 2024].copy())

    # 7. Solo weekdays (excluir complejidad fin de semana)
    results['weekdays_only'] = _bt("Solo weekdays (sin weekend)",
                                    trades[trades['gap_days'] <= 2].copy())

    # 8. Solo weekends
    results['weekends_only'] = _bt("Solo weekends",
                                    trades[trades['gap_days'] > 2].copy())

    # 9. Con SL del 1% (eliminamos noches con >1% loss)
    t9 = trades.copy()
    sl_level = 0.01  # 1% SL
    t9['ret_net'] = np.where(
        t9['ret_gross'] < -sl_level,
        -sl_level - t9['cost_pct'],  # SL activado
        t9['ret_net']
    )
    results['with_sl_1pct'] = _bt("Con Stop Loss 1%", t9)

    # 10. Volatilidad alta únicamente (ATR > mediana)
    t10 = trades.copy()
    # Proxy: usar overnight_range como filtro de volatilidad
    median_range = t10['overnight_range'].median()
    t10_vol = t10[t10['overnight_range'] > median_range]
    results['high_vol_only'] = _bt("Alta volatilidad overnight", t10_vol)

    survived = sum(1 for v in results.values() if v.get('survived', False))
    logger.info(f"\n  Tests superados: {survived}/{len(results)}")
    return results


# ═══════════════════════════════════════════════════════════
# 5. ANÁLISIS POR AÑO Y RÉGIMEN
# ═══════════════════════════════════════════════════════════

def yearly_regime_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    """Estabilidad temporal y por régimen."""
    logger.info("\n── ANÁLISIS POR AÑO ──")
    rows = []

    for year, g in trades.groupby('year'):
        if len(g) < 10:
            continue
        rets = g['ret_net'].values
        wr   = (rets > 0).mean()
        mean = rets.mean()
        std  = rets.std()
        eq   = np.cumprod(1 + rets)
        ann  = eq[-1] ** (252 / len(rets)) - 1
        sh   = (mean / std) * np.sqrt(252) if std > 0 else 0
        _, p = stats.ttest_1samp(rets, 0)
        icon = "✅" if mean > 0 else "❌"
        logger.info(f"  {icon} {year}: n={len(g):3d}  WR={wr*100:.1f}%  "
                    f"Ann={ann*100:.1f}%  Sharpe={sh:.3f}  p={p:.4f}")
        rows.append({
            'year': int(year), 'n': int(len(g)),
            'win_rate': float(wr), 'annual_return': float(ann),
            'sharpe': float(sh), 'pvalue': float(p),
            'positive': bool(mean > 0),
        })

    df_yr = pd.DataFrame(rows)
    if not df_yr.empty:
        pct_pos = df_yr['positive'].mean()
        logger.info(f"\n  Años positivos: {df_yr['positive'].sum()}/{len(df_yr)} "
                    f"({pct_pos*100:.0f}%)")
        if pct_pos >= 0.70:
            logger.info("  ✅ Edge temporalmente estable")
        else:
            logger.info("  ❌ Edge inestable — depende del período")
    return df_yr


# ═══════════════════════════════════════════════════════════
# 6. MONTE CARLO
# ═══════════════════════════════════════════════════════════

def monte_carlo(trades: pd.DataFrame, n_runs: int = 3000) -> dict:
    """Monte Carlo sobre el orden de trades."""
    rets = trades['ret_net'].dropna().values
    n    = len(rets)

    if n < 20:
        return {}

    real_eq     = np.cumprod(1 + rets)
    real_total  = real_eq[-1] - 1
    real_sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
    peak        = np.maximum.accumulate(real_eq)
    real_dd     = ((real_eq - peak) / peak).min()

    rng = np.random.default_rng(42)
    totals  = []
    dds     = []

    for _ in range(n_runs):
        sh = rng.permutation(rets)
        eq = np.cumprod(1 + sh)
        pk = np.maximum.accumulate(eq)
        dd = ((eq - pk) / pk).min()
        totals.append(eq[-1] - 1)
        dds.append(dd)

    totals = np.array(totals)
    dds    = np.array(dds)

    pct_pos      = (totals > 0).mean()
    pct_beat     = (totals > real_total).mean()
    p5_dd        = np.percentile(dds, 5)
    ruin_50      = (dds < -0.50).mean()
    ruin_30      = (dds < -0.30).mean()

    logger.info("\n── MONTE CARLO (3000 runs, orden aleatorio) ──")
    logger.info(f"  N trades: {n}")
    logger.info(f"  Retorno real: {real_total*100:.2f}%")
    logger.info(f"  Sharpe real:  {real_sharpe:.3f}")
    logger.info(f"  Max DD real:  {real_dd*100:.2f}%")
    logger.info(f"  % runs positivos:       {pct_pos*100:.1f}%")
    logger.info(f"  % runs > retorno real:  {pct_beat*100:.1f}%  (mide suerte vs señal)")
    logger.info(f"  Max DD mediana:         {np.median(dds)*100:.1f}%")
    logger.info(f"  Max DD P5 (peor 5%):    {p5_dd*100:.1f}%")
    logger.info(f"  P(ruina DD<-50%):       {ruin_50*100:.1f}%")
    logger.info(f"  P(DD<-30%):             {ruin_30*100:.1f}%")

    if ruin_50 < 0.05 and pct_pos > 0.65:
        logger.info("  ✅ MC: Edge robusto")
    else:
        logger.info("  ❌ MC: Edge frágil o alta probabilidad de ruina")

    return {
        'n_runs': int(n_runs), 'n_trades': int(n),
        'real_total': float(real_total),
        'real_sharpe': float(real_sharpe),
        'real_dd': float(real_dd),
        'pct_positive': float(pct_pos),
        'pct_beat_real': float(pct_beat),
        'median_dd': float(np.median(dds)),
        'p5_dd': float(p5_dd),
        'ruin_50': float(ruin_50),
        'ruin_30': float(ruin_30),
        'totals': totals.tolist(),
        'dds': dds.tolist(),
    }


# ═══════════════════════════════════════════════════════════
# 7. ANÁLISIS DE COSTO BREAK-EVEN
# ═══════════════════════════════════════════════════════════

def breakeven_analysis(trades: pd.DataFrame) -> dict:
    """
    ¿Cuántos puntos NQ de costo máximo puede absorber la estrategia
    antes de que el edge desaparezca?
    """
    logger.info("\n── ANÁLISIS BREAK-EVEN DE COSTOS ──")
    rets_gross = trades['ret_gross'].values
    entry_mean = trades['entry_price'].mean()

    results = {}
    for pts_nq in [0, 2, 4, 6, 8, 10, 12, 15, 18, 22, 25, 30]:
        cost_duck = pts_nq / DUKASCOPY_SCALE
        cost_pct  = cost_duck / entry_mean
        rets_net  = rets_gross - cost_pct
        mean_r    = rets_net.mean()
        _, p      = stats.ttest_1samp(rets_net, 0)
        sharpe    = (mean_r / rets_net.std()) * np.sqrt(252) if rets_net.std() > 0 else 0
        icon      = "✅" if mean_r > 0 else "❌"
        logger.info(f"  {icon} {pts_nq:3d} pts NQ RT: mean={mean_r*100:.4f}%  Sharpe={sharpe:.3f}  p={p:.4f}")
        results[pts_nq] = {'mean': float(mean_r), 'sharpe': float(sharpe), 'pvalue': float(p)}

    return results


# ═══════════════════════════════════════════════════════════
# 8. VISUALIZACIÓN
# ═══════════════════════════════════════════════════════════

def plot_results(bt_full: dict, bt_is: dict, bt_oos: dict,
                 mc: dict, yearly: pd.DataFrame, stress: dict,
                 gap_risk: dict, trades_all: pd.DataFrame) -> None:

    fig = plt.figure(figsize=(22, 20), facecolor='#0d1117')
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)
    GOLD = '#FFD700'; GREEN = '#00FF88'; RED = '#FF4444'
    BLUE = '#4488FF'; GRAY = '#888888'; BG = '#161b22'

    def ax_style(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, color=GOLD, fontsize=10, fontweight='bold', pad=8)
        ax.tick_params(colors=GRAY)
        ax.spines[:].set_color('#333333')
        for l in ax.get_xticklabels() + ax.get_yticklabels(): l.set_color(GRAY)

    # ── 1. Equity FULL + IS/OOS ──
    ax1 = fig.add_subplot(gs[0, :2])
    if bt_is.get('equity_net'):
        eq_is = np.array(bt_is['equity_net'])
        ax1.plot(eq_is, color=BLUE, lw=2, label=f"IS (Sharpe={bt_is['sharpe']:.2f})")
    if bt_oos.get('equity_net'):
        eq_oos = np.array(bt_oos['equity_net'])
        off = len(bt_is.get('equity_net', []))
        ax1.plot(range(off, off + len(eq_oos)), eq_oos, color=GREEN, lw=2,
                 label=f"OOS (Sharpe={bt_oos['sharpe']:.2f})")
        ax1.axvline(off, color=RED, lw=1.5, ls=':', label="IS/OOS split")
    ax1.axhline(1.0, color=GRAY, lw=0.8, ls='--')
    ax1.legend(facecolor=BG, labelcolor='white', fontsize=9)
    ax_style(ax1, "SHORT OVERNIGHT — EQUITY CURVE (neto de costos)")
    ax1.set_ylabel("Equity (base=1.0)", color=GRAY)

    # ── 2. Retorno por año ──
    ax2 = fig.add_subplot(gs[0, 2])
    if not yearly.empty:
        colors = [GREEN if r > 0 else RED for r in yearly['annual_return']]
        bars = ax2.bar(yearly['year'].astype(str), yearly['annual_return'] * 100, color=colors)
        for bar, row in zip(bars, yearly.itertuples()):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + (0.5 if row.annual_return >= 0 else -1.5),
                     f"{row.annual_return*100:.1f}%",
                     ha='center', va='bottom', color='white', fontsize=7)
    ax2.axhline(0, color=GRAY, lw=0.8)
    ax_style(ax2, "Retorno Anual Neto (%)")
    ax2.set_ylabel("%", color=GRAY)
    ax2.tick_params(axis='x', rotation=45)

    # ── 3. Distribución retornos overnight ──
    ax3 = fig.add_subplot(gs[1, 0])
    rets_net = trades_all['ret_net'].dropna().values * 100
    ax3.hist(rets_net, bins=80, color=BLUE, alpha=0.7, edgecolor='none')
    ax3.axvline(rets_net.mean(), color=GOLD, lw=2, ls='--',
                label=f"μ={rets_net.mean():.3f}%")
    ax3.axvline(0, color=RED, lw=1, ls=':')
    ax3.legend(facecolor=BG, labelcolor='white', fontsize=8)
    ax_style(ax3, f"Distribución Retornos Netos\nWR={bt_full.get('win_rate',0)*100:.1f}%")
    ax3.set_xlabel("Retorno neto %", color=GRAY)

    # ── 4. Retorno SHORT por día de la semana ──
    ax4 = fig.add_subplot(gs[1, 1])
    t = trades_all.copy()
    t['dow'] = pd.to_datetime([str(d.date()) for d in t.index]).dayofweek
    dow_means = t.groupby('dow')['ret_net'].agg(['mean', 'count'])
    days = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie']
    colors4 = [GREEN if v > 0 else RED for v in dow_means['mean']]
    ax4.bar([days[i] for i in dow_means.index if i < 5],
            [v for i, v in dow_means['mean'].items() if i < 5],
            color=[c for i, c in enumerate(colors4) if i < 5])
    ax4.axhline(0, color=GRAY, lw=0.8)
    ax_style(ax4, "Retorno Neto Promedio por Día de Semana")
    ax4.set_ylabel("Retorno %", color=GRAY)

    # ── 5. Scatter: overnight ret bruto vs neto ──
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(trades_all['ret_gross'] * 100, trades_all['ret_net'] * 100,
                c=np.sign(trades_all['ret_net']), cmap='RdYlGn',
                alpha=0.3, s=5)
    ax5.axhline(0, color=GRAY, lw=0.5, ls=':')
    ax5.axvline(0, color=GRAY, lw=0.5, ls=':')
    ax_style(ax5, "Bruto vs Neto (impacto de costos)")
    ax5.set_xlabel("Retorno bruto %", color=GRAY)
    ax5.set_ylabel("Retorno neto %", color=GRAY)

    # ── 6. Monte Carlo equity distribution ──
    ax6 = fig.add_subplot(gs[2, 0])
    if mc.get('totals'):
        ax6.hist(np.array(mc['totals']) * 100, bins=80, color=GRAY, alpha=0.5,
                 label='MC aleatorio')
        ax6.axvline(mc['real_total'] * 100, color=GOLD, lw=2.5, ls='--',
                    label=f"Real: {mc['real_total']*100:.1f}%")
        ax6.legend(facecolor=BG, labelcolor='white', fontsize=8)
    ax_style(ax6, f"Monte Carlo: Equity total  "
                  f"(pos={mc.get('pct_positive', 0)*100:.0f}%)")
    ax6.set_xlabel("Retorno total %", color=GRAY)

    # ── 7. Monte Carlo drawdown ──
    ax7 = fig.add_subplot(gs[2, 1])
    if mc.get('dds'):
        ax7.hist(np.array(mc['dds']) * 100, bins=80, color=RED, alpha=0.5)
        ax7.axvline(mc.get('p5_dd', 0) * 100, color=GOLD, lw=2, ls='--',
                    label=f"P5: {mc.get('p5_dd', 0)*100:.1f}%")
        ax7.axvline(mc['real_dd'] * 100, color=BLUE, lw=2,
                    label=f"Real: {mc['real_dd']*100:.1f}%")
        ax7.legend(facecolor=BG, labelcolor='white', fontsize=8)
    ax_style(ax7, f"MC: Max Drawdown  "
                  f"(P(ruina)={mc.get('ruin_50', 0)*100:.1f}%)")
    ax7.set_xlabel("Max DD %", color=GRAY)

    # ── 8. Stress tests ──
    ax8 = fig.add_subplot(gs[2, 2])
    if stress:
        labels  = [v['label'][:32] for v in stress.values()]
        sharpes = [v['sharpe'] for v in stress.values()]
        colors8 = [GREEN if v['survived'] else RED for v in stress.values()]
        ax8.barh(range(len(labels)), sharpes, color=colors8, alpha=0.8)
        ax8.set_yticks(range(len(labels)))
        ax8.set_yticklabels(labels, fontsize=7)
        ax8.axvline(0, color=GRAY, lw=0.8, ls='--')
    ax_style(ax8, "Stress Tests — Sharpe")
    ax8.tick_params(axis='y', labelcolor=GRAY)

    # ── 9. IS vs OOS métricas ──
    ax9 = fig.add_subplot(gs[3, :])
    if bt_is and bt_oos:
        metrics  = ['sharpe', 'win_rate', 'annual_net', 'max_dd', 'profit_factor']
        labels_m = ['Sharpe', 'Win Rate', 'Ann. Net %', 'Max DD', 'Profit Factor']
        vals_is  = [bt_is.get(m, 0) * (100 if m in ('win_rate', 'annual_net', 'max_dd') else 1)
                    for m in metrics]
        vals_oos = [bt_oos.get(m, 0) * (100 if m in ('win_rate', 'annual_net', 'max_dd') else 1)
                    for m in metrics]
        x = np.arange(len(metrics))
        w = 0.35
        ax9.bar(x - w/2, vals_is,  w, label='IS',  color=BLUE,  alpha=0.8)
        ax9.bar(x + w/2, vals_oos, w, label='OOS', color=GREEN, alpha=0.8)
        ax9.set_xticks(x)
        ax9.set_xticklabels(labels_m, color=GRAY, fontsize=9)
        ax9.axhline(0, color=GRAY, lw=0.8)
        ax9.legend(facecolor=BG, labelcolor='white', fontsize=9)
    ax_style(ax9, "Comparación IS vs OOS — Métricas Clave")

    # Score
    stress_surv = sum(1 for v in stress.values() if v.get('survived', False))
    fig.suptitle(
        f"H4: SHORT OVERNIGHT NQ100\n"
        f"Full Sharpe={bt_full.get('sharpe',0):.2f}  |  IS Sharpe={bt_is.get('sharpe',0):.2f}"
        f"  |  OOS Sharpe={bt_oos.get('sharpe',0):.2f}  |  "
        f"Stress={stress_surv}/{len(stress)}  |  MC Ruina={mc.get('ruin_50',0)*100:.1f}%",
        color='white', fontsize=13, fontweight='bold', y=0.98
    )

    out_path = ARTIFACTS_DIR / "nq_short_overnight.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"\n  ✅ Gráfico guardado: {out_path}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║   H4: SHORT OVERNIGHT USATECHIDXUSD (NQ100)                   ║")
    logger.info("║   Fase 6 — Si no sobrevive → no hay edge                       ║")
    logger.info("╚" + "═" * 68 + "╝")

    # ── Cargar datos ──
    parquet = PROJECT_ROOT / "quant_bot" / "data" / "processed" / "USATECHIDXUSD_M1.parquet"
    logger.info(f"\n  Cargando {parquet.name}...")
    df = pd.read_parquet(parquet, engine='pyarrow')
    logger.info(f"  → {len(df):,} barras  ({df.index[0].date()} → {df.index[-1].date()})")

    if 'session' not in df.columns:
        from quant_bot.data.nq_loader import add_session_labels
        df = add_session_labels(df)

    # ── Construir trades ──
    all_trades = build_overnight_trades(df)

    # ── Split IS / OOS ──
    trades_is  = all_trades[all_trades['year'] < OOS_YEAR]
    trades_oos = all_trades[all_trades['year'] >= OOS_YEAR]
    logger.info(f"\n  IS trades:  {len(trades_is)}")
    logger.info(f"  OOS trades: {len(trades_oos)}")

    # ── BACKTESTS ──
    bt_full = backtest_short_on(all_trades, "FULL")
    bt_is   = backtest_short_on(trades_is,  "IS")
    bt_oos  = backtest_short_on(trades_oos, "OOS")

    # ── RIESGO DE GAP ──
    gap = analyze_gap_risk(all_trades)

    # ── STRESS TESTS ──
    stress = stress_tests(trades_is)

    # ── BREAK-EVEN de costos ──
    be = breakeven_analysis(all_trades)

    # ── ANÁLISIS POR AÑO ──
    yearly = yearly_regime_analysis(all_trades)

    # ── MONTE CARLO ──
    mc = monte_carlo(trades_is, n_runs=3000)

    # ── CLASIFICACIÓN FINAL ──
    logger.info("\n" + "═" * 70)
    logger.info("  CLASIFICACIÓN FINAL — H4: SHORT OVERNIGHT")
    logger.info("═" * 70)

    pct_pos_years = float(yearly['positive'].mean()) if not yearly.empty else 0

    checks = {
        'backtest_full_positivo':   bt_full.get('pvalue', 1) < 0.05 and bt_full.get('annual_net', 0) > 0,
        'backtest_IS_significativo': bt_is.get('pvalue', 1) < 0.05 and bt_is.get('sharpe', 0) > 0.3,
        'backtest_OOS_positivo':    bt_oos.get('annual_net', 0) > 0,
        'stress_spread_x2':         stress.get('spread_x2', {}).get('survived', False),
        'stress_no_outliers':       stress.get('no_outliers', {}).get('survived', False),
        'stress_bear_2022':         stress.get('year_2022', {}).get('survived', False),
        'stress_con_sl':            stress.get('with_sl_1pct', {}).get('survived', False),
        'estabilidad_temporal':     pct_pos_years >= 0.70,
        'mc_robusto':               mc.get('ruin_50', 1) < 0.05 and mc.get('pct_positive', 0) > 0.65,
        'gap_riesgo_manejable':     gap.get('p99_loss', 1) < 0.03,  # <3% pérdida P99
    }

    score = sum(checks.values())
    for k, v in checks.items():
        logger.info(f"  {'✅' if v else '❌'} {k}")

    logger.info(f"\n  SCORE: {score}/10")

    if score >= 8:
        verdict = "🏆 EDGE ROBUSTO REAL — implementar con gestión de riesgo"
    elif score >= 6:
        verdict = "✅ EDGE PROMETEDOR — validar en datos tick + paper trading"
    elif score >= 4:
        verdict = "⚠️  EDGE DÉBIL — no implementar aún"
    else:
        verdict = "❌ ILUSIÓN ESTADÍSTICA — no tiene edge real"

    logger.info(f"  VEREDICTO: {verdict}")

    # ── GRÁFICO ──
    plot_results(bt_full, bt_is, bt_oos, mc, yearly, stress, gap, all_trades)

    # ── GUARDAR JSON ──
    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.bool_,)): return bool(o)
            if isinstance(o, (np.ndarray,)): return o.tolist()
            return super().default(o)

    output = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'oos_year': OOS_YEAR,
        'cost_rt_pts_nq': float(COST_RT_ON * DUKASCOPY_SCALE),
        'backtest_full': {k: v for k, v in bt_full.items()
                          if k not in ('equity_net', 'equity_gross', 'dates')},
        'backtest_is': {k: v for k, v in bt_is.items()
                        if k not in ('equity_net', 'equity_gross', 'dates')},
        'backtest_oos': {k: v for k, v in bt_oos.items()
                         if k not in ('equity_net', 'equity_gross', 'dates')},
        'gap_risk': gap,
        'stress_tests': {k: {kk: vv for kk, vv in v.items() if kk != 'label'}
                         for k, v in stress.items()},
        'monte_carlo': {k: v for k, v in mc.items()
                        if k not in ('totals', 'dds')},
        'breakeven': be,
        'checks': {k: bool(v) for k, v in checks.items()},
        'score': int(score),
        'verdict': verdict,
    }

    out_json = ARTIFACTS_DIR / "h4_short_overnight_metrics.json"
    with open(out_json, 'w') as f:
        json.dump(output, f, indent=2, cls=NE)

    logger.info(f"\n  Métricas: {out_json}")
    logger.info("  ✅ Script H4 completado")


if __name__ == "__main__":
    main()
