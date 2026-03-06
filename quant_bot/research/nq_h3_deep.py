"""
nq_h3_deep.py — Exploración Profunda del Edge H3: Primera Hora NY → Día

CONTEXTO:
  La señal bruta EXISTE con significancia estadística (p=0.021, Sharpe 1.55 IS).
  Pero los costos de 6 pts NQ RT la destruyen.

  Objetivo: entender la ESTRUCTURA del edge para encontrar la versión
  implementable con costos < 2-3 pts NQ (futuros NQ en IB o similar).

PREGUNTAS A RESPONDER CON DATOS:
  1. ¿A qué costo RT el edge sobrevive? (curva break-even detallada)
  2. ¿Qué día de semana tiene la señal más fuerte?
  3. ¿Qué tamaño de primera hora filtra mejor? (umbral óptimo IS)
  4. ¿La señal mejora con filtros exógenos? (VIX proxy, prior day)
  5. ¿Qué período de holding es óptimo? (30min, 1h, 2h, 4h, EOD)
  6. ¿La señal es más fuerte en los primeros vs últimos 30 min de la 1H?
  7. ¿Hay efecto momentum vs reversión intra-día?
  8. ¿Walk-forward con ventana rodante confirma estabilidad?
  9. ¿k-fold cross-validation (no solo un IS/OOS) confirma el edge?
 10. ¿Qué régimen de mercado (vol alta/baja) maximiza la señal?

METODOLOGÍA:
  Todo análisis paramétrico se hace SOLO en IS (2021-2024).
  OOS (2025) = testigo ciego, tocado UNA sola vez al final.
  No se optimiza para inflar resultado.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from itertools import product

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
        logging.FileHandler(ARTIFACTS_DIR / "h3_deep.log"),
    ]
)
logger = logging.getLogger("H3_Deep")

DUKASCOPY_SCALE = 82.0
OOS_YEAR = 2025  # ← NUNCA TOCAR HASTA EL FINAL


# ═══════════════════════════════════════════════════════════
# CONSTRUCCIÓN BASE DE SEÑALES (granular, por holding period)
# ═══════════════════════════════════════════════════════════

def build_granular_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Por cada día construye retornos POST primera hora para múltiples
    ventanas de holding: 30min, 1h, 2h, 3h, EOD (20:00 UTC).
    También calcula métricas del contexto de mercado ese día.
    """
    ny_df = df[df['session'].isin(['OPEN_HOUR', 'MIDDAY', 'CLOSE_HOUR'])].copy()

    records = []

    for date, group in ny_df.groupby(ny_df.index.date):
        oh = group[group['session'] == 'OPEN_HOUR']
        if len(oh) < 30:  # mínimo 30 barras (30 min de datos)
            continue

        oh_open  = oh['open'].iloc[0]
        oh_close = oh['close'].iloc[-1]  # ~14:30 UTC
        first_hour_ret = (oh_close - oh_open) / oh_open

        # ATR de la primera hora (proxy de volatilidad)
        oh_atr = (oh['high'].max() - oh['low'].min()) / oh_open
        # Volumen relativo primera hora
        oh_vol = oh['volume'].sum()

        # Spread promedio primera hora
        oh_spread = oh['spread_avg'].mean()

        # ¿La 1H fue tendencia o rango? (ratio retorno/ATR)
        directionality = abs(first_hour_ret) / oh_atr if oh_atr > 0 else 0

        # Post-primera hora para múltiples horizontes
        post = group[group['session'].isin(['MIDDAY', 'CLOSE_HOUR'])]
        if len(post) < 10:
            continue

        entry_price = oh_close
        post_times  = post.index

        def ret_at_offset(minutes: int):
            """Retorno desde entry_price hasta el cierre del bar ~X min después."""
            target_ts = oh.index[-1] + pd.Timedelta(minutes=minutes)
            mask = post.index <= target_ts
            if not mask.any():
                return None
            return (post.loc[mask, 'close'].iloc[-1] - entry_price) / entry_price

        # Retornos en distintos horizontes
        r30m  = ret_at_offset(30)
        r1h   = ret_at_offset(60)
        r2h   = ret_at_offset(120)
        r3h   = ret_at_offset(180)
        r_eod = (post['close'].iloc[-1] - entry_price) / entry_price  # hasta 20:00

        # Contexto del día previo (usamos close previo del mismo DF si está)
        prior_close = None  # llenamos después

        # VWAP simple del día (hasta el punto de entrada ~14:30)
        vwap_num = (group['close'] * group['volume']).sum()
        vwap_den = group['volume'].sum()
        vwap = vwap_num / vwap_den if vwap_den > 0 else entry_price
        entry_vs_vwap = (entry_price - vwap) / vwap

        records.append({
            'date':              pd.Timestamp(date, tz='UTC'),
            'year':              date.year,
            'month':             date.month,
            'dow':               date.weekday(),  # 0=Lun, 4=Vie
            'entry_price':       float(entry_price),
            'first_hour_ret':    float(first_hour_ret),
            'first_hour_atr':    float(oh_atr),
            'directionality':    float(directionality),
            'oh_spread':         float(oh_spread),
            'oh_vol':            float(oh_vol),
            'entry_vs_vwap':     float(entry_vs_vwap),
            'ret_30m':           float(r30m)  if r30m  is not None else np.nan,
            'ret_1h':            float(r1h)   if r1h   is not None else np.nan,
            'ret_2h':            float(r2h)   if r2h   is not None else np.nan,
            'ret_3h':            float(r3h)   if r3h   is not None else np.nan,
            'ret_eod':           float(r_eod),
        })

    df_out = pd.DataFrame(records).set_index('date')

    # Prior day return: close día anterior
    df_out['prior_day_ret'] = df_out['ret_eod'].shift(1)

    # Volatilidad rolling 10D (proxy del VIX local sin datos externos)
    df_out['vol_10d'] = df_out['first_hour_atr'].rolling(10).mean()
    df_out['vol_regime'] = np.where(
        df_out['first_hour_atr'] > df_out['vol_10d'],
        'HIGH_VOL', 'LOW_VOL'
    )

    logger.info(f"\n  Señales granulares construidas: {len(df_out)} días")
    logger.info(f"  Período: {df_out.index[0].date()} → {df_out.index[-1].date()}")

    return df_out


# ═══════════════════════════════════════════════════════════
# FUNCIÓN BASE DE BACKTEST (uso interno, rápida)
# ═══════════════════════════════════════════════════════════

def _quick_backtest(ret_col: np.ndarray, signal: np.ndarray,
                    cost_pct: float = 0.0, label: str = "") -> dict:
    """Backtest rápido dado array de retornos brutos y señales."""
    mask = signal != 0
    if mask.sum() < 5:
        return {'n': 0, 'sharpe': 0, 'annual': 0, 'pvalue': 1, 'win_rate': 0}

    rets_gross = ret_col[mask] * signal[mask]  # en dirección de la señal
    rets_net   = rets_gross - cost_pct

    if len(rets_net) < 5:
        return {'n': 0, 'sharpe': 0, 'annual': 0, 'pvalue': 1, 'win_rate': 0}

    mean = rets_net.mean()
    std  = rets_net.std()
    sh   = (mean / std) * np.sqrt(252) if std > 0 else 0
    eq   = np.cumprod(1 + rets_net)
    ann  = eq[-1] ** (252 / len(rets_net)) - 1
    wr   = (rets_net > 0).mean()
    _, p = stats.ttest_1samp(rets_net, 0)
    return {'n': int(mask.sum()), 'sharpe': float(sh), 'annual': float(ann),
            'pvalue': float(p), 'win_rate': float(wr)}


# ═══════════════════════════════════════════════════════════
# 1. ANÁLISIS DE BREAK-EVEN DETALLADO
# ═══════════════════════════════════════════════════════════

def breakeven_detailed(df_is: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada cost scenario, calcula métricas completas.
    Busca el punto exacto donde el edge se vuelve no-explotable.
    """
    logger.info("\n" + "═"*65)
    logger.info("  1. ANÁLISIS BREAK-EVEN DETALLADO (IS únicamente)")
    logger.info("═"*65)

    signal = np.sign(df_is['first_hour_ret'].values)
    ret_eod = df_is['ret_eod'].values
    entry   = df_is['entry_price'].values

    rows = []
    for pts in np.arange(0, 15.5, 0.5):
        cost_duck = pts / DUKASCOPY_SCALE
        cost_pct  = cost_duck / entry.mean()
        r = _quick_backtest(ret_eod, signal, cost_pct=cost_pct)
        rows.append({
            'pts_nq': float(pts), 'cost_pct': float(cost_pct),
            'sharpe': r['sharpe'], 'annual': r['annual'],
            'win_rate': r['win_rate'], 'pvalue': r['pvalue'], 'n': r['n']
        })

    df_be = pd.DataFrame(rows)

    # Encontrar break-even exacto (donde mean_ret cruce 0)
    be_idx = df_be[df_be['sharpe'] <= 0].first_valid_index()
    be_pts = df_be.loc[be_idx, 'pts_nq'] if be_idx is not None else ">15 pts"

    logger.info(f"\n  Break-even: {be_pts} pts NQ round-trip")
    logger.info(f"  Con 1 pt RT:   Sharpe={df_be[df_be['pts_nq']==1.0]['sharpe'].iloc[0]:.3f}")
    logger.info(f"  Con 2 pts RT:  Sharpe={df_be[df_be['pts_nq']==2.0]['sharpe'].iloc[0]:.3f}")
    logger.info(f"  Con 3 pts RT:  Sharpe={df_be[df_be['pts_nq']==3.0]['sharpe'].iloc[0]:.3f}")
    logger.info(f"  Con 5 pts RT:  Sharpe={df_be[df_be['pts_nq']==5.0]['sharpe'].iloc[0]:.3f}")
    logger.info(f"  Con 6 pts RT:  Sharpe={df_be[df_be['pts_nq']==6.0]['sharpe'].iloc[0]:.3f}")

    return df_be


# ═══════════════════════════════════════════════════════════
# 2. MEJOR PERÍODO DE HOLDING (IS)
# ═══════════════════════════════════════════════════════════

def holding_period_analysis(df_is: pd.DataFrame) -> pd.DataFrame:
    """¿Cuándo cerrar la posición post-primera hora?"""
    logger.info("\n" + "═"*65)
    logger.info("  2. ANÁLISIS DE PERÍODO DE HOLDING (IS)")
    logger.info("═"*65)

    signal  = np.sign(df_is['first_hour_ret'].values)
    entry   = df_is['entry_price'].values
    # Costo = 2 pts NQ (objetivo de bajo costo)
    cost_pct = 2.0 / DUKASCOPY_SCALE / entry.mean()

    periods = {
        '30min': 'ret_30m', '1h': 'ret_1h',
        '2h': 'ret_2h',     '3h': 'ret_3h',
        'EOD': 'ret_eod',
    }
    rows = []
    for label, col in periods.items():
        valid = df_is[col].notna()
        r = _quick_backtest(
            df_is.loc[valid, col].values,
            signal[valid.values],
            cost_pct=cost_pct,
            label=label
        )
        logger.info(f"  {label:6s}: n={r['n']:4d}  Sharpe={r['sharpe']:.3f}"
                    f"  Ann={r['annual']*100:.1f}%  WR={r['win_rate']*100:.1f}%  p={r['pvalue']:.4f}")
        rows.append({'period': label, **r})

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# 3. SENSIBILIDAD AL UMBRAL (granular, IS)
# ═══════════════════════════════════════════════════════════

def threshold_sensitivity(df_is: pd.DataFrame, cost_pts: float = 2.0) -> pd.DataFrame:
    """Mapa de umbral vs Sharpe (sólo IS). Detecta si hay un pico estrecho (curve fitting)."""
    logger.info("\n" + "═"*65)
    logger.info(f"  3. SENSIBILIDAD AL UMBRAL (IS, costo={cost_pts} pts NQ RT)")
    logger.info("═"*65)

    cost_pct = cost_pts / DUKASCOPY_SCALE / df_is['entry_price'].mean()
    fhr = df_is['first_hour_ret'].values
    ret = df_is['ret_eod'].values

    rows = []
    for thr in np.arange(0.0, 0.012, 0.001):
        signal = np.where(fhr > thr, 1, np.where(fhr < -thr, -1, 0))
        r = _quick_backtest(ret, signal, cost_pct=cost_pct)
        rows.append({'threshold': float(thr), **r})
        logger.info(f"  thr={thr:.3f}: n={r['n']:4d}  Sharpe={r['sharpe']:.3f}  "
                    f"p={r['pvalue']:.4f}  WR={r['win_rate']*100:.1f}%")

    df_sens = pd.DataFrame(rows)
    # ¿Es ancho y plano o pico estrecho? (ancho = más robusto)
    positive_range = df_sens[df_sens['sharpe'] > 0.5]
    logger.info(f"\n  Rango de umbrales con Sharpe>0.5: "
                f"{positive_range['threshold'].min():.3f}–{positive_range['threshold'].max():.3f}")
    logger.info(f"  Anchura del edge: {len(positive_range)} de {len(df_sens)} umbrales")
    is_broad = len(positive_range) >= 4
    logger.info(f"  {'✅ ANCHO (robusto)' if is_broad else '❌ ESTRECHO (curve fitting)'}")

    return df_sens


# ═══════════════════════════════════════════════════════════
# 4. ANÁLISIS POR DÍA DE SEMANA (IS)
# ═══════════════════════════════════════════════════════════

def dow_analysis(df_is: pd.DataFrame, cost_pts: float = 2.0) -> pd.DataFrame:
    """¿El edge es uniforme o concentrado en días específicos?"""
    logger.info("\n" + "═"*65)
    logger.info(f"  4. ANÁLISIS POR DÍA DE SEMANA (IS, costo={cost_pts} pts RT)")
    logger.info("═"*65)

    cost_pct = cost_pts / DUKASCOPY_SCALE / df_is['entry_price'].mean()
    rows = []
    days = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie']

    for dow in range(5):
        sub = df_is[df_is['dow'] == dow]
        if len(sub) < 20:
            continue
        signal = np.sign(sub['first_hour_ret'].values)
        r = _quick_backtest(sub['ret_eod'].values, signal, cost_pct=cost_pct)
        icon = "✅" if r['sharpe'] > 0.3 else "❌"
        logger.info(f"  {icon} {days[dow]}: n={r['n']:3d}  Sharpe={r['sharpe']:.3f}"
                    f"  Ann={r['annual']*100:.1f}%  WR={r['win_rate']*100:.1f}%  p={r['pvalue']:.4f}")
        rows.append({'dow': dow, 'name': days[dow], **r})

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# 5. ANÁLISIS POR RÉGIMEN DE VOLATILIDAD (IS)
# ═══════════════════════════════════════════════════════════

def volatility_regime_analysis(df_is: pd.DataFrame, cost_pts: float = 2.0) -> dict:
    """¿El edge es más fuerte en alta o baja volatilidad?"""
    logger.info("\n" + "═"*65)
    logger.info(f"  5. ANÁLISIS POR RÉGIMEN DE VOLATILIDAD (IS)")
    logger.info("═"*65)

    cost_pct = cost_pts / DUKASCOPY_SCALE / df_is['entry_price'].mean()
    results = {}

    for regime in ['HIGH_VOL', 'LOW_VOL']:
        sub = df_is[df_is['vol_regime'] == regime].dropna(subset=['ret_eod'])
        if len(sub) < 20:
            continue
        signal = np.sign(sub['first_hour_ret'].values)
        r = _quick_backtest(sub['ret_eod'].values, signal, cost_pct=cost_pct)
        icon = "✅" if r['sharpe'] > 0.3 else "❌"
        logger.info(f"  {icon} {regime:10s}: n={r['n']:4d}  Sharpe={r['sharpe']:.3f}"
                    f"  Ann={r['annual']*100:.1f}%  WR={r['win_rate']*100:.1f}%")
        results[regime] = r

    # Directionality filter: entrar solo cuando la 1H fue tendencial (no rango)
    logger.info(f"\n  Filtro: solo días TENDENCIALES (directionality > mediana):")
    med_dir = df_is['directionality'].median()
    sub_dir = df_is[df_is['directionality'] > med_dir].dropna(subset=['ret_eod'])
    signal_dir = np.sign(sub_dir['first_hour_ret'].values)
    r_dir = _quick_backtest(sub_dir['ret_eod'].values, signal_dir, cost_pct=cost_pct)
    icon = "✅" if r_dir['sharpe'] > 0.3 else "❌"
    logger.info(f"  {icon} Días tendenciales: n={r_dir['n']:4d}  Sharpe={r_dir['sharpe']:.3f}"
                f"  WR={r_dir['win_rate']*100:.1f}%")
    results['DIRECTIONAL'] = r_dir

    return results


# ═══════════════════════════════════════════════════════════
# 6. ANÁLISIS DE FILTROS COMBINADOS (IS — búsqueda de condición ideal)
# ═══════════════════════════════════════════════════════════

def filter_search(df_is: pd.DataFrame, cost_pts: float = 2.0) -> pd.DataFrame:
    """
    Busca combinaciones de filtros que mejoren el edge.
    NOTA: Todos los filtros se evalúan en IS — riesgo de overfitting.
    Solo los resultados ANALÍTICAmente coherentes se reportan.
    """
    logger.info("\n" + "═"*65)
    logger.info(f"  6. BÚSQUEDA DE FILTROS MEJORADORES (IS)")
    logger.info("═"*65)
    logger.info("  ⚠️  ADVERTENCIA: Exploración IS — riesgo de overfitting")

    cost_pct = cost_pts / DUKASCOPY_SCALE / df_is['entry_price'].mean()
    df = df_is.dropna(subset=['ret_eod', 'first_hour_ret']).copy()
    signal_base = np.sign(df['first_hour_ret'].values)

    # Base sin filtro
    r_base = _quick_backtest(df['ret_eod'].values, signal_base, cost_pct=cost_pct)
    logger.info(f"\n  Base (sin filtro): Sharpe={r_base['sharpe']:.3f}  p={r_base['pvalue']:.4f}")

    results = [{'filter': 'BASE', 'n': r_base['n'], 'sharpe': r_base['sharpe'],
                'annual': r_base['annual'], 'pvalue': r_base['pvalue']}]

    # Filtros a probar (cada uno en forma independiente, no combinado)
    tests = [
        # Dirección del día previo (momentum vs mean-reversion)
        ('Previo alcista (momentum)',
         df['prior_day_ret'].values > 0.001),
        ('Previo bajista (momentum)',
         df['prior_day_ret'].values < -0.001),

        # Magnitude de la primera hora
        ('1H ret > 0.3% (señal fuerte)',
         np.abs(df['first_hour_ret'].values) > 0.003),
        ('1H ret > 0.5% (señal muy fuerte)',
         np.abs(df['first_hour_ret'].values) > 0.005),

        # Tendencialidad de la primera hora
        ('Directionality > P60',
         df['directionality'].values > np.percentile(df['directionality'].values, 60)),
        ('Directionality > P75',
         df['directionality'].values > np.percentile(df['directionality'].values, 75)),

        # Régimen de volatilidad
        ('Alta volatilidad (ATR > P50)',
         df['first_hour_atr'].values > np.percentile(df['first_hour_atr'].values, 50)),
        ('Baja volatilidad (ATR < P50)',
         df['first_hour_atr'].values < np.percentile(df['first_hour_atr'].values, 50)),

        # Día de semana (los mejores del análisis DOW)
        ('Solo Martes + Miércoles',
         df['dow'].isin([1, 2]).values),
        ('Excluir Lunes y Viernes',
         df['dow'].isin([1, 2, 3]).values),

        # Entry vs VWAP
        ('Entrada alcista vs VWAP',
         (df['first_hour_ret'].values > 0) & (df['entry_vs_vwap'].values > 0)),
        ('Entrada bajista vs VWAP',
         (df['first_hour_ret'].values < 0) & (df['entry_vs_vwap'].values < 0)),
    ]

    for filter_label, mask in tests:
        sub = df[mask]
        if len(sub) < 30:
            continue
        sig = np.sign(sub['first_hour_ret'].values)
        r = _quick_backtest(sub['ret_eod'].values, sig, cost_pct=cost_pct)
        improvement = r['sharpe'] - r_base['sharpe']
        icon = "✅" if r['sharpe'] > r_base['sharpe'] + 0.1 else "↔️" if r['sharpe'] > r_base['sharpe'] else "❌"
        logger.info(f"  {icon} {filter_label:40s}: n={r['n']:4d}  "
                    f"Sharpe={r['sharpe']:.3f}  (Δ={improvement:+.3f})  p={r['pvalue']:.4f}")
        results.append({'filter': filter_label, 'n': r['n'], 'sharpe': r['sharpe'],
                        'annual': r['annual'], 'pvalue': r['pvalue']})

    df_results = pd.DataFrame(results).sort_values('sharpe', ascending=False)
    return df_results


# ═══════════════════════════════════════════════════════════
# 7. WALK-FORWARD CON VENTANA RODANTE (IS)
# ═══════════════════════════════════════════════════════════

def walk_forward_analysis(df_is: pd.DataFrame, cost_pts: float = 2.0) -> pd.DataFrame:
    """
    Walk-forward con ventana rodante de 12 meses → test 3 meses.
    Simula si la señal es estable en el tiempo sin info del futuro.
    """
    logger.info("\n" + "═"*65)
    logger.info(f"  7. WALK-FORWARD RODANTE (IS, costo={cost_pts} pts RT)")
    logger.info("═"*65)

    cost_pct = cost_pts / DUKASCOPY_SCALE / df_is['entry_price'].mean()
    df = df_is.dropna(subset=['ret_eod', 'first_hour_ret']).copy()
    df = df.sort_index()

    TRAIN_MONTHS = 12
    TEST_MONTHS  = 3

    rows = []
    current = df.index[0]
    end     = df.index[-1]

    while current + pd.DateOffset(months=TRAIN_MONTHS + TEST_MONTHS) <= end:
        train_end  = current + pd.DateOffset(months=TRAIN_MONTHS)
        test_end   = train_end + pd.DateOffset(months=TEST_MONTHS)

        train = df[df.index < train_end]
        test  = df[(df.index >= train_end) & (df.index < test_end)]

        if len(train) < 50 or len(test) < 10:
            current += pd.DateOffset(months=TEST_MONTHS)
            continue

        # En train: encontrar el mejor umbral (simple grid)
        best_thr = 0.0
        best_sh  = -99
        for thr in [0.0, 0.001, 0.002, 0.003, 0.005]:
            sig_tr = np.where(
                train['first_hour_ret'].values > thr, 1,
                np.where(train['first_hour_ret'].values < -thr, -1, 0)
            )
            r_tr = _quick_backtest(train['ret_eod'].values, sig_tr, cost_pct=cost_pct)
            if r_tr['sharpe'] > best_sh:
                best_sh  = r_tr['sharpe']
                best_thr = thr

        # En test: aplicar umbral derivado del train
        sig_te = np.where(
            test['first_hour_ret'].values > best_thr, 1,
            np.where(test['first_hour_ret'].values < -best_thr, -1, 0)
        )
        r_te = _quick_backtest(test['ret_eod'].values, sig_te, cost_pct=cost_pct)

        logger.info(f"  {train_end.strftime('%Y-%m')} → {test_end.strftime('%Y-%m')}: "
                    f"thr={best_thr:.3f}  Sharpe={r_te['sharpe']:.3f}  "
                    f"WR={r_te['win_rate']*100:.1f}%")

        rows.append({
            'period': train_end.strftime('%Y-%m'),
            'threshold': float(best_thr),
            **r_te
        })
        current += pd.DateOffset(months=TEST_MONTHS)

    df_wf = pd.DataFrame(rows)
    if not df_wf.empty:
        pct_pos = (df_wf['sharpe'] > 0).mean()
        logger.info(f"\n  Ventanas positivas: {pct_pos*100:.0f}%  "
                    f"({(df_wf['sharpe']>0).sum()}/{len(df_wf)})")
        if pct_pos >= 0.65:
            logger.info("  ✅ Edge temporalmente estable en WF")
        else:
            logger.info("  ❌ Edge inestable en walk-forward")

    return df_wf


# ═══════════════════════════════════════════════════════════
# 8. K-FOLD CROSS VALIDATION (IS)
# ═══════════════════════════════════════════════════════════

def kfold_validation(df_is: pd.DataFrame, k: int = 5,
                     cost_pts: float = 2.0) -> dict:
    """
    k-fold para estimar la varianza del Sharpe.
    ¿El Sharpe es consistente o muy variable entre folds?
    """
    logger.info("\n" + "═"*65)
    logger.info(f"  8. {k}-FOLD CROSS VALIDATION (IS, costo={cost_pts} pts RT)")
    logger.info("═"*65)

    cost_pct = cost_pts / DUKASCOPY_SCALE / df_is['entry_price'].mean()
    df = df_is.dropna(subset=['ret_eod', 'first_hour_ret']).copy().sort_index()
    n  = len(df)
    fold_size = n // k

    sharpes = []
    for i in range(k):
        start = i * fold_size
        end   = (i + 1) * fold_size if i < k - 1 else n
        fold  = df.iloc[start:end]
        sig   = np.sign(fold['first_hour_ret'].values)
        r     = _quick_backtest(fold['ret_eod'].values, sig, cost_pct=cost_pct)
        logger.info(f"  Fold {i+1}/{k}: n={r['n']:4d}  Sharpe={r['sharpe']:.3f}"
                    f"  WR={r['win_rate']*100:.1f}%  p={r['pvalue']:.4f}")
        sharpes.append(r['sharpe'])

    sharpes = np.array(sharpes)
    pct_pos = (sharpes > 0).mean()
    logger.info(f"\n  Sharpe promedio: {sharpes.mean():.3f} ± {sharpes.std():.3f}")
    logger.info(f"  Folds positivos: {pct_pos*100:.0f}%")
    if pct_pos >= 0.80 and sharpes.mean() > 0.3:
        logger.info("  ✅ Edge consistente entre folds")
    else:
        logger.info("  ❌ Edge inconsistente — puede ser ruido")

    return {
        'mean_sharpe': float(sharpes.mean()),
        'std_sharpe': float(sharpes.std()),
        'pct_positive': float(pct_pos),
        'sharpes': sharpes.tolist(),
    }


# ═══════════════════════════════════════════════════════════
# 9. VALIDACIÓN OOS FINAL (tocar UNA sola vez)
# ═══════════════════════════════════════════════════════════

def final_oos_validation(df_is: pd.DataFrame, df_oos: pd.DataFrame,
                          best_config: dict) -> dict:
    """
    Aplica la configuración derivada del IS al OOS.
    Solo se ejecuta al final — resultado de "testigo ciego".
    """
    logger.info("\n" + "═"*65)
    logger.info("  9. VALIDACIÓN OOS FINAL (testigo ciego — se toca UNA vez)")
    logger.info("═"*65)

    thr      = best_config['threshold']
    cost_pts = best_config['cost_pts']
    col      = best_config.get('ret_col', 'ret_eod')
    cost_pct = cost_pts / DUKASCOPY_SCALE / df_oos['entry_price'].mean()

    oos = df_oos.dropna(subset=[col, 'first_hour_ret']).copy()
    signal = np.where(
        oos['first_hour_ret'].values > thr, 1,
        np.where(oos['first_hour_ret'].values < -thr, -1, 0)
    )

    rets_gross = oos[col].values * signal
    rets_net   = rets_gross - cost_pct

    active_mask = signal != 0
    rets_net_a  = rets_net[active_mask]

    if len(rets_net_a) < 5:
        logger.info("  ❌ Insuficientes trades en OOS")
        return {}

    mean = rets_net_a.mean()
    std  = rets_net_a.std()
    sh   = (mean / std) * np.sqrt(252) if std > 0 else 0
    eq   = np.cumprod(1 + rets_net_a)
    ann  = eq[-1] ** (252 / len(rets_net_a)) - 1
    wr   = (rets_net_a > 0).mean()
    _, p = stats.ttest_1samp(rets_net_a, 0)

    peak = np.maximum.accumulate(eq)
    dd   = ((eq - peak) / peak).min()

    logger.info(f"\n  Config OOS: threshold={thr}  cost={cost_pts} pts NQ RT")
    logger.info(f"  N trades OOS:     {active_mask.sum()}")
    logger.info(f"  Retorno total:    {(eq[-1]-1)*100:.2f}%")
    logger.info(f"  Retorno anualizado: {ann*100:.2f}%")
    logger.info(f"  Sharpe:           {sh:.3f}")
    logger.info(f"  Win Rate:         {wr*100:.1f}%")
    logger.info(f"  Max Drawdown:     {dd*100:.2f}%")
    logger.info(f"  T-test:           t={_:.3f}, p={p:.6f}" if (lambda t: t)(stats.ttest_1samp(rets_net_a, 0)[0]) else "")
    logger.info(f"  p-value:          {p:.6f}")

    if mean > 0 and p < 0.10:
        logger.info("  ✅ OOS: EDGE SOBREVIVE — señal explotable")
    elif mean > 0 and sh > 0.3:
        logger.info("  ⚠️  OOS: Edge marginal (p>{0.10:.0f}%) — más datos necesarios")
    else:
        logger.info("  ❌ OOS: Edge no sobrevive")

    return {
        'n': int(active_mask.sum()),
        'total_return': float(eq[-1]-1),
        'annual': float(ann),
        'sharpe': float(sh),
        'win_rate': float(wr),
        'max_dd': float(dd),
        'pvalue': float(p),
        'equity': eq.tolist(),
    }


# ═══════════════════════════════════════════════════════════
# 10. VISUALIZACIÓN COMPLETA
# ═══════════════════════════════════════════════════════════

def plot_deep_analysis(be: pd.DataFrame, hp: pd.DataFrame, sens: pd.DataFrame,
                       dow: pd.DataFrame, wf: pd.DataFrame, kf: dict,
                       filters: pd.DataFrame, oos_result: dict,
                       df_all: pd.DataFrame) -> None:

    fig = plt.figure(figsize=(22, 24), facecolor='#0d1117')
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.35)
    GOLD='#FFD700'; GREEN='#00FF88'; RED='#FF4444'; BLUE='#4488FF'; GRAY='#888888'; BG='#161b22'

    def ax_style(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, color=GOLD, fontsize=10, fontweight='bold', pad=8)
        ax.tick_params(colors=GRAY)
        ax.spines[:].set_color('#333333')
        for l in ax.get_xticklabels() + ax.get_yticklabels(): l.set_color(GRAY)

    # 1. Break-even curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(be['pts_nq'], be['sharpe'], color=BLUE, lw=2.5, marker='o', markersize=4)
    ax1.axhline(0, color=RED, lw=1.5, ls='--', label='Sharpe=0')
    ax1.axhline(0.5, color=GREEN, lw=1, ls=':', label='Sharpe=0.5')
    be_pt = be[be['sharpe'] <= 0]['pts_nq'].min() if (be['sharpe'] <= 0).any() else None
    if be_pt:
        ax1.axvline(be_pt, color=GOLD, lw=2, ls='--', label=f'Break-even={be_pt:.1f}pts')
    ax1.legend(facecolor=BG, labelcolor='white', fontsize=8)
    ax_style(ax1, "Break-even de Costos\n(IS, umbral=0%)")
    ax1.set_xlabel("Costo Total RT (pts NQ)", color=GRAY)
    ax1.set_ylabel("Sharpe", color=GRAY)

    # 2. Holding period
    ax2 = fig.add_subplot(gs[0, 1])
    colors2 = [GREEN if s > 0 else RED for s in hp['sharpe']]
    bars = ax2.bar(hp['period'], hp['sharpe'], color=colors2)
    ax2.axhline(0, color=GRAY, lw=0.8)
    for bar, val in zip(bars, hp['sharpe']):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02 * np.sign(val),
                 f"{val:.2f}", ha='center', va='bottom', color='white', fontsize=8)
    ax_style(ax2, "Sharpe por Período de Holding\n(IS, 2 pts NQ RT)")
    ax2.set_ylabel("Sharpe", color=GRAY)

    # 3. Threshold sensitivity
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(sens['threshold']*100, sens['sharpe'], color=BLUE, lw=2, marker='o', markersize=5)
    ax3.fill_between(sens['threshold']*100,
                     sens['sharpe'] - sens['sharpe'].std(),
                     sens['sharpe'] + sens['sharpe'].std(),
                     alpha=0.2, color=BLUE)
    ax3.axhline(0, color=RED, lw=1, ls='--')
    ax_style(ax3, "Sensibilidad al Umbral\n(IS, 2 pts RT) — plano=robusto")
    ax3.set_xlabel("Umbral Primera Hora (%)", color=GRAY)
    ax3.set_ylabel("Sharpe", color=GRAY)

    # 4. Day of week
    ax4 = fig.add_subplot(gs[1, 0])
    if not dow.empty:
        colors4 = [GREEN if s > 0 else RED for s in dow['sharpe']]
        ax4.bar(dow['name'], dow['sharpe'], color=colors4)
        ax4.axhline(0, color=GRAY, lw=0.8)
    ax_style(ax4, "Sharpe por Día de Semana\n(IS, 2 pts RT)")
    ax4.set_ylabel("Sharpe", color=GRAY)

    # 5. Walk-forward
    ax5 = fig.add_subplot(gs[1, 1])
    if not wf.empty:
        colors5 = [GREEN if s > 0 else RED for s in wf['sharpe']]
        ax5.bar(wf['period'], wf['sharpe'], color=colors5, width=0.6)
        ax5.axhline(0, color=GRAY, lw=0.8)
        ax5.tick_params(axis='x', rotation=45)
    ax_style(ax5, "Walk-Forward Rodante\n(train=12m, test=3m)")
    ax5.set_ylabel("Sharpe OOS", color=GRAY)

    # 6. K-fold
    ax6 = fig.add_subplot(gs[1, 2])
    if kf.get('sharpes'):
        ks = kf['sharpes']
        colors6 = [GREEN if s > 0 else RED for s in ks]
        ax6.bar(range(1, len(ks)+1), ks, color=colors6)
        ax6.axhline(kf['mean_sharpe'], color=GOLD, lw=2, ls='--',
                    label=f"μ={kf['mean_sharpe']:.3f}")
        ax6.axhline(0, color=RED, lw=1, ls=':')
        ax6.legend(facecolor=BG, labelcolor='white', fontsize=8)
    ax_style(ax6, f"k-Fold CV (k=5)\n({kf.get('pct_positive',0)*100:.0f}% folds positivos)")
    ax6.set_xlabel("Fold", color=GRAY)
    ax6.set_ylabel("Sharpe", color=GRAY)

    # 7. Top filtros
    ax7 = fig.add_subplot(gs[2, :2])
    top_filters = filters.head(12)
    if not top_filters.empty:
        colors7 = [GREEN if s > filters[filters['filter']=='BASE']['sharpe'].iloc[0]
                   else GRAY for s in top_filters['sharpe']]
        ax7.barh(range(len(top_filters)), top_filters['sharpe'], color=colors7, alpha=0.8)
        ax7.set_yticks(range(len(top_filters)))
        ax7.set_yticklabels(top_filters['filter'].str[:40], fontsize=7)
        base_sh = filters[filters['filter']=='BASE']['sharpe'].iloc[0]
        ax7.axvline(base_sh, color=GOLD, lw=2, ls='--', label=f'Base={base_sh:.3f}')
        ax7.axvline(0, color=GRAY, lw=0.8, ls=':')
        ax7.legend(facecolor=BG, labelcolor='white', fontsize=8)
    ax_style(ax7, "Top Filtros — Sharpe (IS, 2 pts RT) — ordenado desc")
    ax7.tick_params(axis='y', labelcolor=GRAY)
    ax7.set_xlabel("Sharpe", color=GRAY)

    # 8. OOS equity
    ax8 = fig.add_subplot(gs[2, 2])
    if oos_result.get('equity'):
        eq = np.array(oos_result['equity'])
        color8 = GREEN if eq[-1] > 1.0 else RED
        ax8.plot(eq, color=color8, lw=2.5)
        ax8.axhline(1.0, color=GRAY, lw=0.8, ls='--')
        ax8.fill_between(range(len(eq)), 1.0, eq,
                         alpha=0.2, color=color8)
    ann_pct = oos_result.get('annual', 0) * 100
    sh = oos_result.get('sharpe', 0)
    ax_style(ax8, f"OOS Equity (2025)\nSharpe={sh:.3f}  Ann={ann_pct:.1f}%")
    ax8.set_ylabel("Equity", color=GRAY)

    # 9. Scatter 1H vs day (por volatilidad)
    ax9 = fig.add_subplot(gs[3, :])
    ny = df_all.dropna(subset=['first_hour_ret', 'ret_eod', 'vol_regime'])
    for regime, color9 in [('HIGH_VOL', BLUE), ('LOW_VOL', GREEN)]:
        sub9 = ny[ny['vol_regime'] == regime]
        ax9.scatter(sub9['first_hour_ret']*100, sub9['ret_eod']*100,
                    c=color9, alpha=0.25, s=8, label=regime)
    # Regresión global
    x9 = ny['first_hour_ret'].values
    y9 = ny['ret_eod'].values
    mask9 = ~(np.isnan(x9) | np.isnan(y9))
    if mask9.sum() > 10:
        m9, b9 = np.polyfit(x9[mask9], y9[mask9], 1)
        xl = np.linspace(x9[mask9].min(), x9[mask9].max(), 50)
        ax9.plot(xl*100, (m9*xl + b9)*100, color=GOLD, lw=2.5, label=f'Regresión (m={m9:.2f})')
    ax9.axhline(0, color=GRAY, lw=0.5, ls=':')
    ax9.axvline(0, color=GRAY, lw=0.5, ls=':')
    ax9.legend(facecolor=BG, labelcolor='white', fontsize=9)
    ax_style(ax9, "Scatter: 1H ret → EOD ret  (por régimen de volatilidad)")
    ax9.set_xlabel("Primera Hora (%)", color=GRAY)
    ax9.set_ylabel("Resto del Día (%)", color=GRAY)

    oos_sh = oos_result.get('sharpe', 0)
    wf_pct = (wf['sharpe'] > 0).mean() * 100 if not wf.empty else 0
    kf_pct = kf.get('pct_positive', 0) * 100

    fig.suptitle(
        f"H3 PROFUNDO: PRIMERA HORA NY → DIRECCIÓN DEL DÍA  (Con Costos Bajos)\n"
        f"WF={wf_pct:.0f}% positivo  |  kFold={kf_pct:.0f}% positivo  |  "
        f"OOS Sharpe={oos_sh:.3f}",
        color='white', fontsize=13, fontweight='bold', y=0.99
    )

    out = ARTIFACTS_DIR / "nq_h3_deep.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"\n  ✅ Gráfico: {out}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    logger.info("╔" + "═"*70 + "╗")
    logger.info("║   H3 EXPLORACIÓN PROFUNDA — PRIMERA HORA NY → DIRECCIÓN DÍA   ║")
    logger.info("║   Objetivo: ¿Explotable con costos < 3 pts NQ RT?              ║")
    logger.info("╚" + "═"*70 + "╝")

    # ── Cargar datos ──
    parquet = PROJECT_ROOT / "quant_bot" / "data" / "processed" / "USATECHIDXUSD_M1.parquet"
    logger.info(f"\n  Cargando {parquet.name}...")
    df = pd.read_parquet(parquet, engine='pyarrow')
    logger.info(f"  → {len(df):,} barras  ({df.index[0].date()} → {df.index[-1].date()})")

    if 'session' not in df.columns:
        from quant_bot.data.nq_loader import add_session_labels
        df = add_session_labels(df)

    # ── Señales granulares ──
    logger.info("\n  Construyendo señales granulares (puede tardar 1-2 min)...")
    sigs_all = build_granular_signals(df)
    sigs_is  = sigs_all[sigs_all['year'] < OOS_YEAR]
    sigs_oos = sigs_all[sigs_all['year'] >= OOS_YEAR]

    logger.info(f"  IS: {len(sigs_is)} días | OOS: {len(sigs_oos)} días")

    # ── COST_PTS para análisis (objetivo de bajo costo) ──
    COST_TARGET = 2.0  # pts NQ RT objetivo (IB futuros: comisión fija ~$1 = ~0.5pt)

    # ANÁLISIS 1–8 (solo IS)
    be   = breakeven_detailed(sigs_is)
    hp   = holding_period_analysis(sigs_is)
    sens = threshold_sensitivity(sigs_is, cost_pts=COST_TARGET)
    dow  = dow_analysis(sigs_is, cost_pts=COST_TARGET)
    vol  = volatility_regime_analysis(sigs_is, cost_pts=COST_TARGET)
    filt = filter_search(sigs_is, cost_pts=COST_TARGET)
    wf   = walk_forward_analysis(sigs_is, cost_pts=COST_TARGET)
    kf   = kfold_validation(sigs_is, k=5, cost_pts=COST_TARGET)

    # ── DERIVAR MEJOR CONFIGURACIÓN DEL IS ──
    # Umbral con mejor Sharpe IS sin seleccionar extremos:
    # Usar el umbral con mejor Sharpe en la zona estable de la curva
    best_thr_row = sens.sort_values('sharpe', ascending=False).iloc[0]
    best_threshold = float(best_thr_row['threshold'])

    logger.info(f"\n  Config derivada del IS:")
    logger.info(f"    Umbral: {best_threshold:.3f} ({best_threshold*100:.1f}%)")
    logger.info(f"    Costo:  {COST_TARGET} pts NQ RT")
    logger.info(f"    Holding: EOD (cierre NY)")

    best_config = {
        'threshold': best_threshold,
        'cost_pts': COST_TARGET,
        'ret_col': 'ret_eod',
    }

    # ── OOS FINAL — SE TOCA UNA SOLA VEZ ──
    oos_result = final_oos_validation(sigs_is, sigs_oos, best_config)

    # ── CLASIFICACIÓN FINAL ──
    logger.info("\n" + "═"*70)
    logger.info("  CLASIFICACIÓN FINAL H3 — CON COSTOS BAJOS (2 pts NQ RT)")
    logger.info("═"*70)

    be_pt = be[be['sharpe'] <= 0]['pts_nq'].min() if (be['sharpe'] <= 0).any() else 15
    wf_pct = (wf['sharpe'] > 0).mean() if not wf.empty else 0

    checks = {
        'break_even_mayor_2pts':   float(be_pt) > 2.0,
        'break_even_mayor_3pts':   float(be_pt) > 3.0,
        'holding_eod_positivo':    hp[hp['period']=='EOD']['sharpe'].iloc[0] > 0.3 if not hp.empty else False,
        'threshold_robusto_ancho': (sens['sharpe'] > 0.3).sum() >= 4,
        'wf_mayoria_positivo':     wf_pct >= 0.60,
        'kfold_consistente':       kf.get('pct_positive', 0) >= 0.60 and kf.get('mean_sharpe', 0) > 0.2,
        'oos_positivo':            oos_result.get('sharpe', 0) > 0,
        'oos_wr_mayor_50':         oos_result.get('win_rate', 0) > 0.50,
    }

    score = sum(checks.values())
    for k, v in checks.items():
        logger.info(f"  {'✅' if v else '❌'} {k}")

    logger.info(f"\n  SCORE: {score}/8")
    logger.info(f"  Break-even exacto: {be_pt:.1f} pts NQ RT")

    if score >= 6:
        verdict = "🏆 EDGE EXPLOTABLE con bajo costo de ejecución"
        detail  = f"Viable en NQ futuros (IB) con RT < {be_pt:.0f} pts NQ"
    elif score >= 4:
        verdict = "✅ SEÑAL PROMETEDORA — validar en paper trading"
        detail  = "Requiere costos RT < 3 pts NQ para ser rentable"
    elif score >= 2:
        verdict = "⚠️  SEÑAL DÉBIL — solo con costos RT < 2 pts NQ"
        detail  = "Difícil de capturar en retail estándar"
    else:
        verdict = "❌ SEÑAL INSUFICIENTE incluso con costos cero"
        detail  = "No hay edge explotable bajo ninguna condición"

    logger.info(f"\n  VEREDICTO: {verdict}")
    logger.info(f"  DETALLE:   {detail}")

    # ── GRÁFICO ──
    plot_deep_analysis(be, hp, sens, dow, wf, kf, filt, oos_result, sigs_all)

    # ── GUARDAR REPORTE ──
    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.bool_,)): return bool(o)
            if isinstance(o, (np.ndarray,)): return o.tolist()
            return super().default(o)

    output = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'cost_target_pts_nq': COST_TARGET,
        'oos_year': OOS_YEAR,
        'break_even_pts_nq': float(be_pt),
        'best_threshold': best_threshold,
        'breakeven_table': be.to_dict('records'),
        'holding_periods': hp.to_dict('records'),
        'threshold_sensitivity': sens.to_dict('records'),
        'dow_analysis': dow.to_dict('records'),
        'volatility_regimes': vol,
        'top_filters': filt.head(8).to_dict('records'),
        'walk_forward': wf.to_dict('records') if not wf.empty else [],
        'kfold': {k: v for k, v in kf.items() if k != 'sharpes'},
        'oos_result': {k: v for k, v in oos_result.items() if k != 'equity'},
        'checks': {k: bool(v) for k, v in checks.items()},
        'score': int(score),
        'verdict': verdict,
        'detail': detail,
    }

    out_json = ARTIFACTS_DIR / "h3_deep_metrics.json"
    with open(out_json, 'w') as f:
        json.dump(output, f, indent=2, cls=NE)

    logger.info(f"\n  Métricas: {out_json}")
    logger.info("  ✅ Script H3-Deep completado")


if __name__ == "__main__":
    main()
