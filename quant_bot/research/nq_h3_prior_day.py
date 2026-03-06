"""
nq_h3_prior_day.py — Exploración del Filtro: Día Previo Bajista + H3

CONTEXTO:
  En el análisis H3 Deep se encontró que el filtro "día previo bajista"
  mejora el Sharpe de 0.986 → 2.07 (Δ+1.08) con p=0.012 en IS.

  Esto puede indicar un efecto de MOMENTUM CONDICIONAL:
  - Si ayer el NQ cerró en negativo
  - Y hoy la primera hora también baja (señal SHORT)
    → La presión vendedora continúa hasta el cierre con mayor probabilidad

  O alternativamente (hipótesis más interesante):
  - Si ayer bajó y hoy la primera hora SUBE (señal LONG)
  - El mercado está rebotando y la señal de compra es más fuerte

PREGUNTAS A DESTRUIR:
  1. ¿El filtro funciona por MOMENTUM (primera hora va en misma dirección que ayer)?
  2. ¿O funciona por REVERSIÓN (primera hora va en dirección opuesta a ayer)?
  3. ¿Qué magnitud de "día previo bajista" maximiza el edge?
  4. ¿Cuántos días de "momentum previo" se acumulan? (1d, 2d, 3d)
  5. ¿El efecto depende de la magnitud del retorno previo?
  6. ¿Sobrevive OOS (2025)?
  7. ¿Walk-forward confirma estabilidad?
  8. ¿k-Fold confirma consistencia?
  9. ¿Es el filtro realmente el driver o solo correlaciona con algo más?
 10. ¿El filtro combinado Day Previo + Directionality > P75 es viable?

METODOLOGÍA ESTRICTA:
  - Todo análisis exploratorio: IS únicamente (2021-2024)
  - OOS (2025): tocado UNA sola vez al final
  - No optimizar parámetros al p-value — usar rangos amplios
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
        logging.FileHandler(ARTIFACTS_DIR / "h3_prior_day.log"),
    ]
)
logger = logging.getLogger("H3v2_PriorDay")

DUKASCOPY_SCALE = 82.0
COST_TARGET_PTS = 2.0   # objetivo de bajo costo
OOS_YEAR        = 2025


# ════════════════════════════════════════════════════════════
# REUTILIZAR SEÑALES GRANULARES DE H3 DEEP
# ════════════════════════════════════════════════════════════

def build_enriched_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye señales H3 con contexto enriquecido del día previo.
    """
    ny_df = df[df['session'].isin(['OPEN_HOUR', 'MIDDAY', 'CLOSE_HOUR'])].copy()
    records = []

    for date, group in ny_df.groupby(ny_df.index.date):
        oh = group[group['session'] == 'OPEN_HOUR']
        if len(oh) < 30:
            continue

        oh_open  = oh['open'].iloc[0]
        oh_close = oh['close'].iloc[-1]
        if oh_open <= 0:
            continue

        first_hour_ret = (oh_close - oh_open) / oh_open
        oh_atr = (oh['high'].max() - oh['low'].min()) / oh_open
        directionality = abs(first_hour_ret) / oh_atr if oh_atr > 0 else 0

        post = group[group['session'].isin(['MIDDAY', 'CLOSE_HOUR'])]
        if len(post) < 10:
            continue

        entry_price = oh_close
        ret_eod = (post['close'].iloc[-1] - entry_price) / entry_price

        def ret_at_offset(m):
            tgt = oh.index[-1] + pd.Timedelta(minutes=m)
            m2  = post.index <= tgt
            return (post.loc[m2, 'close'].iloc[-1] - entry_price) / entry_price if m2.any() else np.nan

        records.append({
            'date':           pd.Timestamp(date, tz='UTC'),
            'year':           date.year,
            'month':          date.month,
            'dow':            date.weekday(),
            'entry_price':    float(entry_price),
            'first_hour_ret': float(first_hour_ret),
            'oh_atr':         float(oh_atr),
            'directionality': float(directionality),
            'ret_eod':        float(ret_eod),
            'ret_1h':         ret_at_offset(60),
            'ret_2h':         ret_at_offset(120),
            # Full day: apertura de sesión a cierre (para "day return")
            'day_open':       float(oh_open),
            'day_close':      float(post['close'].iloc[-1]),
        })

    df_out = pd.DataFrame(records).set_index('date')

    # Day return (para calcular prior_day)
    df_out['day_return_full'] = (df_out['day_close'] - df_out['day_open']) / df_out['day_open']

    # Prior day returns
    df_out['prior_1d_ret']   = df_out['day_return_full'].shift(1)
    df_out['prior_2d_ret']   = df_out['day_return_full'].shift(2)
    df_out['prior_3d_ret']   = df_out['day_return_full'].shift(3)
    df_out['prior_5d_ret']   = df_out['day_return_full'].rolling(5).mean().shift(1)
    df_out['prior_2d_cum']   = (1 + df_out['day_return_full'].shift(1)) * \
                                (1 + df_out['day_return_full'].shift(2)) - 1

    # Clasificación del día previo
    df_out['prior_bearish']  = df_out['prior_1d_ret'] < -0.001  # bajó > 0.1%
    df_out['prior_bullish']  = df_out['prior_1d_ret'] >  0.001  # subió > 0.1%
    df_out['prior_neutral']  = ~(df_out['prior_bearish'] | df_out['prior_bullish'])

    # Momentum de la señal H3 vs día previo
    df_out['momentum_signal'] = (  # misma dirección que ayer
        (df_out['first_hour_ret'] > 0) == (df_out['prior_1d_ret'] > 0)
    )

    # Régimen de volatilidad rolling
    df_out['vol_10d'] = df_out['oh_atr'].rolling(10).mean()
    df_out['high_vol'] = df_out['oh_atr'] > df_out['vol_10d']

    return df_out.dropna(subset=['prior_1d_ret', 'ret_eod', 'first_hour_ret'])


# ════════════════════════════════════════════════════════════
# HELPER: quickbt
# ════════════════════════════════════════════════════════════

def qbt(rets_gross: np.ndarray, signals: np.ndarray,
        cost_pct: float = 0.0, label: str = "") -> dict:
    mask = signals != 0
    if mask.sum() < 8:
        return {'n': 0, 'sharpe': 0.0, 'annual': 0.0, 'pvalue': 1.0,
                'win_rate': 0.0, 'max_dd': 0.0, 'total': 0.0}
    rn = rets_gross[mask] * signals[mask] - cost_pct
    if rn.std() == 0:
        return {'n': int(mask.sum()), 'sharpe': 0.0, 'annual': 0.0, 'pvalue': 1.0,
                'win_rate': float((rn > 0).mean()), 'max_dd': 0.0, 'total': 0.0}
    sh = (rn.mean() / rn.std()) * np.sqrt(252)
    eq = np.cumprod(1 + rn)
    ann = eq[-1] ** (252 / len(rn)) - 1
    pk = np.maximum.accumulate(eq); dd = ((eq - pk) / pk).min()
    wr = (rn > 0).mean()
    _, p = stats.ttest_1samp(rn, 0)
    return {'n': int(mask.sum()), 'sharpe': float(sh), 'annual': float(ann),
            'pvalue': float(p), 'win_rate': float(wr), 'max_dd': float(dd),
            'total': float(eq[-1] - 1)}


def cost_pct_from_pts(pts: float, entry_mean: float) -> float:
    return (pts / DUKASCOPY_SCALE) / entry_mean


# ════════════════════════════════════════════════════════════
# 1. ANATOMÍA DEL FILTRO PREVIO BAJISTA
# ════════════════════════════════════════════════════════════

def anatomy_prior_day(df_is: pd.DataFrame) -> dict:
    """
    Descompone el filtro para entender QUÉ lo impulsa:
    ¿Momentum (1H sigue bajando tras día bajista) o reversión?
    """
    logger.info("\n" + "═"*68)
    logger.info("  1. ANATOMÍA DEL FILTRO 'DÍA PREVIO BAJISTA'")
    logger.info("═"*68)

    cost = cost_pct_from_pts(COST_TARGET_PTS, df_is['entry_price'].mean())
    ret  = df_is['ret_eod'].values
    sig  = np.sign(df_is['first_hour_ret'].values)

    # ── A: Base sin filtro ──
    r_base = qbt(ret, sig, cost_pct=cost)
    logger.info(f"\n  BASE (sin filtro):       n={r_base['n']:4d}  Sharpe={r_base['sharpe']:.3f}"
                f"  WR={r_base['win_rate']*100:.1f}%  p={r_base['pvalue']:.4f}")

    # ── B: Solo días con previo bajista ──
    mask_bear = df_is['prior_bearish'].values
    r_bear = qbt(ret[mask_bear], sig[mask_bear], cost_pct=cost)
    icon = "✅" if r_bear['sharpe'] > r_base['sharpe'] else "❌"
    logger.info(f"  {icon} PREVIO BAJISTA:          n={r_bear['n']:4d}  Sharpe={r_bear['sharpe']:.3f}"
                f"  WR={r_bear['win_rate']*100:.1f}%  p={r_bear['pvalue']:.4f}")

    # ── C: Solo días con previo alcista ──
    mask_bull = df_is['prior_bullish'].values
    r_bull = qbt(ret[mask_bull], sig[mask_bull], cost_pct=cost)
    icon = "✅" if r_bull['sharpe'] > r_base['sharpe'] else "❌"
    logger.info(f"  {icon} PREVIO ALCISTA:          n={r_bull['n']:4d}  Sharpe={r_bull['sharpe']:.3f}"
                f"  WR={r_bull['win_rate']*100:.1f}%  p={r_bull['pvalue']:.4f}")

    # ── D: Descomponer: momentum vs reversión dentro del subconjunto bajista ──
    logger.info("\n  ─── Dentro del subconjunto DÍA PREVIO BAJISTA ───")
    df_bear = df_is[mask_bear]

    # Momentum: 1H y día previo van en la misma dirección
    mom_mask = df_bear['momentum_signal'].values
    r_mom = qbt(df_bear['ret_eod'].values[mom_mask],
                np.sign(df_bear['first_hour_ret'].values[mom_mask]),
                cost_pct=cost)
    icon = "✅" if r_mom['sharpe'] > 0.5 else "↔️"
    logger.info(f"  {icon}   → MOMENTUM (1H = dir ayer): n={r_mom['n']:4d}  Sharpe={r_mom['sharpe']:.3f}"
                f"  WR={r_mom['win_rate']*100:.1f}%  p={r_mom['pvalue']:.4f}")

    # Reversión: 1H va en dirección opuesta al día previo bajista
    rev_mask = ~mom_mask
    r_rev = qbt(df_bear['ret_eod'].values[rev_mask],
                np.sign(df_bear['first_hour_ret'].values[rev_mask]),
                cost_pct=cost)
    icon = "✅" if r_rev['sharpe'] > 0.5 else "↔️"
    logger.info(f"  {icon}   → REVERSIÓN (1H ≠ dir ayer): n={r_rev['n']:4d}  Sharpe={r_rev['sharpe']:.3f}"
                f"  WR={r_rev['win_rate']*100:.1f}%  p={r_rev['pvalue']:.4f}")

    # ── E: 2 días previos bajistas (momentum acumulado) ──
    mask_2bear = (df_is['prior_1d_ret'].values < -0.001) & \
                 (df_is['prior_2d_ret'].values < -0.001)
    r_2bear = qbt(ret[mask_2bear], sig[mask_2bear], cost_pct=cost)
    icon = "✅" if r_2bear['sharpe'] > r_base['sharpe'] else "❌"
    logger.info(f"\n  {icon} 2 DÍAS PREVIOS BAJISTAS: n={r_2bear['n']:4d}  Sharpe={r_2bear['sharpe']:.3f}"
                f"  WR={r_2bear['win_rate']*100:.1f}%  p={r_2bear['pvalue']:.4f}")

    # ── F: 3 días previos bajistas ──
    mask_3bear = (df_is['prior_1d_ret'].values < -0.001) & \
                 (df_is['prior_2d_ret'].values < -0.001) & \
                 (df_is['prior_3d_ret'].values < -0.001)
    r_3bear = qbt(ret[mask_3bear], sig[mask_3bear], cost_pct=cost)
    icon = "✅" if r_3bear['sharpe'] > r_base['sharpe'] else "❌"
    logger.info(f"  {icon} 3 DÍAS PREVIOS BAJISTAS: n={r_3bear['n']:4d}  Sharpe={r_3bear['sharpe']:.3f}"
                f"  WR={r_3bear['win_rate']*100:.1f}%  p={r_3bear['pvalue']:.4f}")

    return {
        'base': r_base, 'bear': r_bear, 'bull': r_bull,
        'momentum': r_mom, 'reversal': r_rev,
        '2bear': r_2bear, '3bear': r_3bear,
    }


# ════════════════════════════════════════════════════════════
# 2. SENSIBILIDAD AL UMBRAL DE "PREVIO BAJISTA"
# ════════════════════════════════════════════════════════════

def prior_day_threshold_sensitivity(df_is: pd.DataFrame) -> pd.DataFrame:
    """
    ¿Qué tan bajista tiene que haber sido ayer para que el filtro funcione?
    Testea desde -0.05% hasta -2.0% como umbral de 'bajista'.
    """
    logger.info("\n" + "═"*68)
    logger.info("  2. SENSIBILIDAD AL UMBRAL 'PREVIO BAJISTA'")
    logger.info("═"*68)

    cost = cost_pct_from_pts(COST_TARGET_PTS, df_is['entry_price'].mean())
    ret  = df_is['ret_eod'].values
    sig  = np.sign(df_is['first_hour_ret'].values)
    prior = df_is['prior_1d_ret'].values

    rows = []
    for thr_pct in [0.0, -0.001, -0.002, -0.003, -0.005, -0.007,
                    -0.010, -0.015, -0.020, -0.030]:
        mask = prior < thr_pct
        r = qbt(ret[mask], sig[mask], cost_pct=cost)
        icon = "✅" if r['sharpe'] > 0.8 and r['n'] >= 50 else "↔️"
        logger.info(f"  {icon} prior<{thr_pct*100:.1f}%: n={r['n']:4d}  Sharpe={r['sharpe']:.3f}"
                    f"  Ann={r['annual']*100:.1f}%  WR={r['win_rate']*100:.1f}%  p={r['pvalue']:.4f}")
        rows.append({'threshold_pct': float(thr_pct), **r})

    # También testear "previo alcista" con umbrales simétricos
    logger.info("  [Simétrico — previo alcista:]")
    for thr_pct in [0.001, 0.003, 0.005, 0.010]:
        mask = prior > thr_pct
        r = qbt(ret[mask], sig[mask], cost_pct=cost)
        icon = "✅" if r['sharpe'] > 0.8 and r['n'] >= 50 else "❌"
        logger.info(f"  {icon} prior>{thr_pct*100:.1f}%: n={r['n']:4d}  Sharpe={r['sharpe']:.3f}"
                    f"  WR={r['win_rate']*100:.1f}%  p={r['pvalue']:.4f}")

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════
# 3. FILTRO COMBINADO: PREVIO BAJISTA + H3 THRESHOLD + DIRECTIONALITY
# ════════════════════════════════════════════════════════════

def combined_filter_analysis(df_is: pd.DataFrame) -> pd.DataFrame:
    """
    Prueba combinaciones de filtros para encontrar el subconjunto
    con el mejor Sharpe robusto.
    ⚠️ Solo IS — alto riesgo de overfitting si hay demasiadas combinaciones.
    Documentamos bien cada resultado.
    """
    logger.info("\n" + "═"*68)
    logger.info("  3. FILTROS COMBINADOS (⚠️ Solo IS — overfitting risk)")
    logger.info("═"*68)

    cost  = cost_pct_from_pts(COST_TARGET_PTS, df_is['entry_price'].mean())
    ret   = df_is['ret_eod'].values
    sig   = np.sign(df_is['first_hour_ret'].values)
    prior = df_is['prior_1d_ret'].values
    fhr   = df_is['first_hour_ret'].values
    dr    = df_is['directionality'].values
    vol   = df_is['oh_atr'].values
    vol10 = df_is['vol_10d'].values

    # Percentiles para umbrales no optimizados
    p60_dir = np.nanpercentile(dr, 60)
    p75_dir = np.nanpercentile(dr, 75)
    p50_vol = np.nanpercentile(vol, 50)

    combos = [
        ("Base (sin filtro)",
         np.ones(len(df_is), dtype=bool)),

        ("PREVIO BAJISTA (>0.1%)",
         prior < -0.001),

        ("PREVIO BAJISTA (>0.3%)",
         prior < -0.003),

        ("PREVIO BAJISTA + 1H>0.2%",
         (prior < -0.001) & (np.abs(fhr) > 0.002)),

        ("PREVIO BAJISTA + 1H>0.3%",
         (prior < -0.001) & (np.abs(fhr) > 0.003)),

        ("PREVIO BAJISTA + Dirn>P60",
         (prior < -0.001) & (dr > p60_dir)),

        ("PREVIO BAJISTA + Dirn>P75",
         (prior < -0.001) & (dr > p75_dir)),

        ("PREVIO BAJISTA + High Vol",
         (prior < -0.001) & (vol > p50_vol)),

        ("PREVIO BAJISTA + 1H>0.3% + Dirn>P60",
         (prior < -0.001) & (np.abs(fhr) > 0.003) & (dr > p60_dir)),

        ("PREVIO BAJISTA + 1H>0.3% + Dirn>P75",
         (prior < -0.001) & (np.abs(fhr) > 0.003) & (dr > p75_dir)),

        ("2 DÍAS PREVIOS BAJISTAS",
         (prior < -0.001) & (df_is['prior_2d_ret'].values < -0.001)),

        ("2 DÍAS BAJ + 1H>0.3%",
         (prior < -0.001) & (df_is['prior_2d_ret'].values < -0.001) & (np.abs(fhr) > 0.003)),

        ("Solo Mar+Mié + PREVIO BAJ",
         (prior < -0.001) & df_is['dow'].isin([1, 2]).values),
    ]

    rows = []
    base_sharpe = 0
    for label, mask in combos:
        r = qbt(ret[mask], sig[mask], cost_pct=cost)
        if label == "Base (sin filtro)":
            base_sharpe = r['sharpe']
        delta = r['sharpe'] - base_sharpe
        icon  = "🏆" if r['sharpe'] > 1.5 and r['n'] >= 40 and r['pvalue'] < 0.10 else \
                "✅" if r['sharpe'] > base_sharpe + 0.2 else \
                "❌" if r['sharpe'] < base_sharpe - 0.2 else "↔️"
        logger.info(f"  {icon} {label:42s}: n={r['n']:4d}  Sharpe={r['sharpe']:.3f}"
                    f"  (Δ={delta:+.3f})  Ann={r['annual']*100:.1f}%  WR={r['win_rate']*100:.1f}%"
                    f"  p={r['pvalue']:.4f}")
        rows.append({'filter': label, 'delta': float(delta), **r})

    df_res = pd.DataFrame(rows).sort_values('sharpe', ascending=False)
    return df_res


# ════════════════════════════════════════════════════════════
# 4. DISTRIBUCIÓN DE RETORNOS POR CUARTIL DEL DÍA PREVIO
# ════════════════════════════════════════════════════════════

def prior_day_quartile_analysis(df_is: pd.DataFrame) -> pd.DataFrame:
    """
    Descompone el edge por cuartiles del retorno del día previo.
    ¿El efecto es monotónico? (a mayor caída previa, mayor edge)
    """
    logger.info("\n" + "═"*68)
    logger.info("  4. ANÁLISIS POR CUARTIL DEL DÍA PREVIO")
    logger.info("═"*68)

    cost  = cost_pct_from_pts(COST_TARGET_PTS, df_is['entry_price'].mean())
    ret   = df_is['ret_eod'].values
    sig   = np.sign(df_is['first_hour_ret'].values)
    prior = df_is['prior_1d_ret'].values

    quantiles = np.nanpercentile(prior, [0, 25, 50, 75, 100])
    rows = []

    for i in range(4):
        lo, hi = quantiles[i], quantiles[i+1]
        mask = (prior >= lo) & (prior < hi)
        q_label = f"Q{i+1} [{lo*100:.2f}%→{hi*100:.2f}%]"
        r = qbt(ret[mask], sig[mask], cost_pct=cost)
        icon = "✅" if r['sharpe'] > 0.5 else "❌"
        logger.info(f"  {icon} {q_label:35s}: n={r['n']:4d}  Sharpe={r['sharpe']:.3f}"
                    f"  WR={r['win_rate']*100:.1f}%  Ann={r['annual']*100:.1f}%")
        rows.append({'quartile': q_label, 'q_num': i+1, 'lo': float(lo), 'hi': float(hi), **r})

    df_q = pd.DataFrame(rows)
    is_monotonic = all(df_q['sharpe'].iloc[i] >= df_q['sharpe'].iloc[i+1]
                       for i in range(len(df_q)-1))
    logger.info(f"\n  ¿Relación monotónica (más bajista previo = más edge)?")
    logger.info(f"  {'✅ SÍ — efecto REAL y coherente' if is_monotonic else '❌ NO — puede ser artefacto'}")

    return df_q


# ════════════════════════════════════════════════════════════
# 5. WALK-FORWARD CON FILTRO
# ════════════════════════════════════════════════════════════

def walk_forward_filtered(df_is: pd.DataFrame,
                           best_filter_mask_fn,
                           cost_pts: float = COST_TARGET_PTS) -> pd.DataFrame:
    """Walk-forward 12m train / 3m test con el filtro que mejor funcionó."""
    logger.info("\n" + "═"*68)
    logger.info("  5. WALK-FORWARD CON FILTRO 'DÍA PREVIO BAJISTA + 1H>0.3%'")
    logger.info("═"*68)

    cost = cost_pct_from_pts(cost_pts, df_is['entry_price'].mean())
    df   = df_is.sort_index()
    rows = []

    TRAIN_M = 12
    TEST_M  = 3
    current = df.index[0]
    end     = df.index[-1]

    while current + pd.DateOffset(months=TRAIN_M + TEST_M) <= end:
        train_end = current + pd.DateOffset(months=TRAIN_M)
        test_end  = train_end + pd.DateOffset(months=TEST_M)

        test = df[(df.index >= train_end) & (df.index < test_end)]
        if len(test) < 8:
            current += pd.DateOffset(months=TEST_M)
            continue

        # Aplicar filtro en el test
        f_mask = best_filter_mask_fn(test)
        if f_mask.sum() < 5:
            current += pd.DateOffset(months=TEST_M)
            continue

        r = qbt(test['ret_eod'].values[f_mask],
                np.sign(test['first_hour_ret'].values[f_mask]),
                cost_pct=cost)

        icon = "✅" if r['sharpe'] > 0 else "❌"
        logger.info(f"  {icon} Test {train_end.strftime('%Y-%m')}→{test_end.strftime('%Y-%m')}: "
                    f"n={r['n']:3d}  Sharpe={r['sharpe']:.3f}  WR={r['win_rate']*100:.1f}%")

        rows.append({
            'period': train_end.strftime('%Y-%m'),
            **r,
        })
        current += pd.DateOffset(months=TEST_M)

    df_wf = pd.DataFrame(rows)
    if not df_wf.empty:
        pct_pos = (df_wf['sharpe'] > 0).mean()
        logger.info(f"\n  Ventanas positivas: {pct_pos*100:.0f}%  ({(df_wf['sharpe']>0).sum()}/{len(df_wf)})")

    return df_wf


# ════════════════════════════════════════════════════════════
# 6. K-FOLD CON EL MEJOR FILTRO
# ════════════════════════════════════════════════════════════

def kfold_filtered(df_is: pd.DataFrame, best_filter_mask_fn,
                   k: int = 5, cost_pts: float = COST_TARGET_PTS) -> dict:
    logger.info("\n" + "═"*68)
    logger.info(f"  6. {k}-FOLD CV CON FILTRO")
    logger.info("═"*68)

    cost = cost_pct_from_pts(cost_pts, df_is['entry_price'].mean())
    df   = df_is.sort_index()
    n    = len(df)
    fold_size = n // k
    sharpes   = []

    for i in range(k):
        s = i * fold_size
        e = (i + 1) * fold_size if i < k - 1 else n
        fold      = df.iloc[s:e]
        f_mask    = best_filter_mask_fn(fold)
        if f_mask.sum() < 5:
            continue
        r = qbt(fold['ret_eod'].values[f_mask],
                np.sign(fold['first_hour_ret'].values[f_mask]),
                cost_pct=cost)
        logger.info(f"  Fold {i+1}/{k}: n={r['n']:4d}  Sharpe={r['sharpe']:.3f}"
                    f"  WR={r['win_rate']*100:.1f}%  p={r['pvalue']:.4f}")
        sharpes.append(r['sharpe'])

    sharpes = np.array(sharpes)
    pct_pos = (sharpes > 0).mean()
    logger.info(f"\n  Promedio: {sharpes.mean():.3f} ± {sharpes.std():.3f}")
    logger.info(f"  Positivos: {pct_pos*100:.0f}%")
    if pct_pos >= 0.80:
        logger.info("  ✅ Consistente entre folds")
    else:
        logger.info("  ❌ Inconsistente")

    return {
        'mean_sharpe': float(sharpes.mean()),
        'std_sharpe': float(sharpes.std()),
        'pct_positive': float(pct_pos),
        'sharpes': sharpes.tolist(),
    }


# ════════════════════════════════════════════════════════════
# 7. VALIDACIÓN OOS CON EL MEJOR FILTRO
# ════════════════════════════════════════════════════════════

def oos_validation_filtered(df_oos: pd.DataFrame, best_filter_mask_fn,
                              cost_pts: float = COST_TARGET_PTS) -> dict:
    logger.info("\n" + "═"*68)
    logger.info("  7. VALIDACIÓN OOS (2025) — TESTIGO CIEGO — SE TOCA UNA VEZ")
    logger.info("═"*68)

    cost   = cost_pct_from_pts(cost_pts, df_oos['entry_price'].mean())
    f_mask = best_filter_mask_fn(df_oos)

    if f_mask.sum() < 5:
        logger.info("  ❌ Insuficientes trades OOS con el filtro")
        return {}

    rets_gross = df_oos['ret_eod'].values[f_mask]
    sigs       = np.sign(df_oos['first_hour_ret'].values[f_mask])
    rets_net   = rets_gross * sigs - cost

    mean = rets_net.mean()
    std  = rets_net.std()
    sh   = (mean / std) * np.sqrt(252) if std > 0 else 0
    eq   = np.cumprod(1 + rets_net)
    ann  = eq[-1] ** (252 / len(rets_net)) - 1
    wr   = (rets_net > 0).mean()
    pk   = np.maximum.accumulate(eq)
    dd   = ((eq - pk) / pk).min()
    _, p = stats.ttest_1samp(rets_net, 0)

    logger.info(f"\n  Filtro: Día previo bajista (>0.1%) + 1H ret > 0.3%")
    logger.info(f"  Costo RT: {cost_pts} pts NQ ({cost*df_oos['entry_price'].mean()*DUKASCOPY_SCALE:.2f} pts NQ)")
    logger.info(f"  N trades OOS:      {f_mask.sum()}")
    logger.info(f"  Retorno total:     {(eq[-1]-1)*100:.2f}%")
    logger.info(f"  Retorno anual:     {ann*100:.2f}%")
    logger.info(f"  Sharpe:            {sh:.3f}")
    logger.info(f"  Win Rate:          {wr*100:.1f}%")
    logger.info(f"  Max Drawdown:      {dd*100:.2f}%")
    logger.info(f"  p-value:           {p:.6f}")

    if mean > 0 and p < 0.10:
        logger.info("  ✅ OOS SIGNIFICATIVO — edge robusto con filtro")
    elif mean > 0 and sh > 0.5:
        logger.info("  ⚠️  OOS positivo pero no significativo (n pequeño)")
    else:
        logger.info("  ❌ OOS negativo — el filtro no sobrevive fuera de muestra")

    return {
        'n': int(f_mask.sum()),
        'total': float(eq[-1]-1),
        'annual': float(ann),
        'sharpe': float(sh),
        'win_rate': float(wr),
        'max_dd': float(dd),
        'pvalue': float(p),
        'equity': eq.tolist(),
    }


# ════════════════════════════════════════════════════════════
# 8. VISUALIZACIÓN COMPLETA
# ════════════════════════════════════════════════════════════

def plot_prior_day(anatomy: dict, quartiles: pd.DataFrame,
                   thr_sens: pd.DataFrame, combos: pd.DataFrame,
                   wf: pd.DataFrame, kf: dict, oos: dict,
                   df_all: pd.DataFrame) -> None:

    fig = plt.figure(figsize=(22, 22), facecolor='#0d1117')
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.48, wspace=0.35)
    GOLD='#FFD700'; GREEN='#00FF88'; RED='#FF4444'; BLUE='#4488FF'; GRAY='#888888'; BG='#161b22'

    def ax_style(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, color=GOLD, fontsize=10, fontweight='bold', pad=8)
        ax.tick_params(colors=GRAY)
        ax.spines[:].set_color('#333333')
        for l in ax.get_xticklabels() + ax.get_yticklabels(): l.set_color(GRAY)

    # 1. Anatomy bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    labels1 = ['Base', 'Previo\nBajista', 'Previo\nAlcista', 'Momentum', 'Reversión', '2d Bajista']
    vals1 = [anatomy[k]['sharpe'] for k in ['base','bear','bull','momentum','reversal','2bear']]
    colors1 = [GREEN if v > anatomy['base']['sharpe']+0.05 else RED if v < anatomy['base']['sharpe']-0.05 else GRAY for v in vals1]
    bars1 = ax1.bar(labels1, vals1, color=colors1, alpha=0.85)
    ax1.axhline(anatomy['base']['sharpe'], color=GOLD, lw=2, ls='--', label=f"Base={anatomy['base']['sharpe']:.2f}")
    ax1.axhline(0, color=GRAY, lw=0.8, ls=':')
    for bar, val in zip(bars1, vals1):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha='center', color='white', fontsize=8)
    ax1.legend(facecolor=BG, labelcolor='white', fontsize=8)
    ax_style(ax1, "Anatomía del Filtro\n'Día Previo Bajista'")
    ax1.set_ylabel("Sharpe (IS)", color=GRAY)

    # 2. Quartile analysis
    ax2 = fig.add_subplot(gs[0, 1])
    if not quartiles.empty:
        colors2 = [GREEN if s > 0 else RED for s in quartiles['sharpe']]
        bars2 = ax2.bar(quartiles['quartile'].str[:12], quartiles['sharpe'], color=colors2)
        for bar, val in zip(bars2, quartiles['sharpe']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                     f"{val:.2f}", ha='center', color='white', fontsize=8)
    ax2.axhline(0, color=GRAY, lw=0.8)
    ax_style(ax2, "Sharpe por Cuartil\nde Retorno Previo")
    ax2.set_ylabel("Sharpe", color=GRAY)
    ax2.tick_params(axis='x', rotation=25)

    # 3. Threshold sensitivity prior day
    ax3 = fig.add_subplot(gs[0, 2])
    if not thr_sens.empty:
        ax3.plot(thr_sens['threshold_pct']*100, thr_sens['sharpe'],
                 color=BLUE, lw=2, marker='o', markersize=6)
        ax3.fill_between(thr_sens['threshold_pct']*100, 0, thr_sens['sharpe'],
                         where=thr_sens['sharpe'] > 0, alpha=0.2, color=GREEN)
        ax3.axhline(anatomy['base']['sharpe'], color=GOLD, lw=1.5, ls='--',
                    label=f"Base={anatomy['base']['sharpe']:.2f}")
        ax3.axhline(0, color=RED, lw=1, ls=':')
        ax3.legend(facecolor=BG, labelcolor='white', fontsize=8)
    ax_style(ax3, "Sensibilidad al Umbral\n'Día Previo Bajista' (%)")
    ax3.set_xlabel("Umbral retorno previo (%)", color=GRAY)
    ax3.set_ylabel("Sharpe", color=GRAY)

    # 4. Top combined filters
    ax4 = fig.add_subplot(gs[1, :2])
    top = combos.head(10) if not combos.empty else pd.DataFrame()
    if not top.empty:
        base_sh = combos[combos['filter']=='Base (sin filtro)']['sharpe'].iloc[0] \
                  if 'Base (sin filtro)' in combos['filter'].values else 0
        colors4 = ['#FFD700' if s > base_sh + 0.3 else GREEN if s > base_sh else RED for s in top['sharpe']]
        ax4.barh(range(len(top)), top['sharpe'], color=colors4, alpha=0.85)
        ax4.set_yticks(range(len(top)))
        ax4.set_yticklabels(top['filter'].str[:45], fontsize=8)
        ax4.axvline(base_sh, color=GOLD, lw=2, ls='--', label=f"Base={base_sh:.2f}")
        ax4.axvline(0, color=GRAY, lw=0.8, ls=':')
        ax4.legend(facecolor=BG, labelcolor='white', fontsize=8)
    ax_style(ax4, "Top Filtros Combinados — Sharpe IS\n(ordenados descendente)")
    ax4.tick_params(axis='y', labelcolor=GRAY)
    ax4.set_xlabel("Sharpe (IS)", color=GRAY)

    # 5. Walk-forward
    ax5 = fig.add_subplot(gs[1, 2])
    if not wf.empty:
        colors5 = [GREEN if s > 0 else RED for s in wf['sharpe']]
        ax5.bar(wf['period'], wf['sharpe'], color=colors5, width=0.6)
        ax5.axhline(0, color=GRAY, lw=0.8)
        ax5.tick_params(axis='x', rotation=45)
    ax_style(ax5, "Walk-Forward\n(Filtro + test 3m)")
    ax5.set_ylabel("Sharpe OOS", color=GRAY)

    # 6. kFold
    ax6 = fig.add_subplot(gs[2, 0])
    if kf.get('sharpes'):
        ks = kf['sharpes']
        colors6 = [GREEN if s > 0 else RED for s in ks]
        ax6.bar(range(1, len(ks)+1), ks, color=colors6)
        ax6.axhline(kf['mean_sharpe'], color=GOLD, lw=2, ls='--',
                    label=f"μ={kf['mean_sharpe']:.2f}")
        ax6.axhline(0, color=RED, lw=1, ls=':')
        ax6.legend(facecolor=BG, labelcolor='white', fontsize=8)
    ax_style(ax6, f"k-Fold CV (k=5)\n{kf.get('pct_positive',0)*100:.0f}% folds positivos")
    ax6.set_xlabel("Fold", color=GRAY)
    ax6.set_ylabel("Sharpe", color=GRAY)

    # 7. OOS equity
    ax7 = fig.add_subplot(gs[2, 1])
    if oos.get('equity'):
        eq = np.array(oos['equity'])
        color7 = GREEN if eq[-1] > 1.0 else RED
        ax7.plot(eq, color=color7, lw=2.5)
        ax7.axhline(1.0, color=GRAY, lw=0.8, ls='--')
        ax7.fill_between(range(len(eq)), 1.0, eq, alpha=0.2, color=color7)
    ax_style(ax7, f"OOS 2025 — Equity\nSharpe={oos.get('sharpe',0):.3f}"
                  f"  Ann={oos.get('annual',0)*100:.1f}%")
    ax7.set_ylabel("Equity (base=1.0)", color=GRAY)

    # 8. Scatter con color por día previo
    ax8 = fig.add_subplot(gs[2, 2])
    df_plot = df_all.dropna(subset=['first_hour_ret', 'ret_eod', 'prior_1d_ret'])
    bear_plt = df_plot[df_plot['prior_bearish']]
    bull_plt = df_plot[~df_plot['prior_bearish']]
    ax8.scatter(bull_plt['first_hour_ret']*100, bull_plt['ret_eod']*100,
                c=GRAY, alpha=0.2, s=5, label='Previo neutro/alcista')
    ax8.scatter(bear_plt['first_hour_ret']*100, bear_plt['ret_eod']*100,
                c=BLUE, alpha=0.5, s=8, label='Previo BAJISTA')
    # Regresion solo días bajistas previos
    x_b = bear_plt['first_hour_ret'].values; y_b = bear_plt['ret_eod'].values
    valid = ~(np.isnan(x_b) | np.isnan(y_b))
    if valid.sum() > 10:
        m, b = np.polyfit(x_b[valid], y_b[valid], 1)
        xl = np.linspace(x_b[valid].min(), x_b[valid].max(), 50)
        ax8.plot(xl*100, (m*xl+b)*100, color=GOLD, lw=2, label=f"Regresión bear (m={m:.2f})")
    ax8.axhline(0, color=GRAY, lw=0.5, ls=':'); ax8.axvline(0, color=GRAY, lw=0.5, ls=':')
    ax8.legend(facecolor=BG, labelcolor='white', fontsize=7)
    ax_style(ax8, "Scatter 1H→EOD\n(color = día previo bajista)")
    ax8.set_xlabel("Primera hora (%)", color=GRAY)
    ax8.set_ylabel("EOD ret (%)", color=GRAY)

    # 9. Métricas IS vs OOS con filtro (barchart comparativo)
    ax9 = fig.add_subplot(gs[3, :])
    best_is = combos[combos['filter'].str.contains('PREVIO BAJISTA \\+ 1H>0.3%', regex=True)].head(1)
    if not best_is.empty and oos:
        metrics  = ['sharpe', 'win_rate', 'annual', 'max_dd']
        labels_m = ['Sharpe', 'Win Rate (%)', 'Ann. Ret (%)', 'Max DD (%)']
        m_scale  = {'sharpe': 1, 'win_rate': 100, 'annual': 100, 'max_dd': 100}
        is_vals  = [best_is.iloc[0][m] * m_scale[m] for m in metrics]
        oos_vals = [oos.get(m, 0) * m_scale[m] for m in metrics]
        x = np.arange(len(metrics)); w = 0.35
        ax9.bar(x - w/2, is_vals,  w, label='IS',  color=BLUE,  alpha=0.8)
        ax9.bar(x + w/2, oos_vals, w, label='OOS', color=GREEN, alpha=0.8)
        ax9.set_xticks(x); ax9.set_xticklabels(labels_m, color=GRAY)
        ax9.axhline(0, color=GRAY, lw=0.8)
        ax9.legend(facecolor=BG, labelcolor='white', fontsize=9)
    ax_style(ax9, "IS vs OOS — Mejor Filtro Combinado")

    bear_sharpe = anatomy.get('bear', {}).get('sharpe', 0)
    oos_sharpe  = oos.get('sharpe', 0)
    wf_pct      = (wf['sharpe'] > 0).mean() * 100 if not wf.empty else 0

    fig.suptitle(
        f"H3v2: PRIMERA HORA + DÍA PREVIO BAJISTA\n"
        f"IS (Bajista) Sharpe={bear_sharpe:.2f}  |  WF={wf_pct:.0f}% positivo  |  "
        f"OOS Sharpe={oos_sharpe:.3f}",
        color='white', fontsize=13, fontweight='bold', y=0.99
    )

    out = ARTIFACTS_DIR / "nq_h3_prior_day.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"\n  ✅ Gráfico: {out}")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    logger.info("╔" + "═"*70 + "╗")
    logger.info("║   H3v2: PRIMERA HORA + FILTRO 'DÍA PREVIO BAJISTA'             ║")
    logger.info("║   Investigación con paciencia — los grandes fondos tardan meses ║")
    logger.info("╚" + "═"*70 + "╝")

    # Cargar datos
    parquet = PROJECT_ROOT / "quant_bot" / "data" / "processed" / "USATECHIDXUSD_M1.parquet"
    logger.info(f"\n  Cargando {parquet.name}...")
    df = pd.read_parquet(parquet, engine='pyarrow')
    logger.info(f"  → {len(df):,} barras  ({df.index[0].date()} → {df.index[-1].date()})")

    if 'session' not in df.columns:
        from quant_bot.data.nq_loader import add_session_labels
        df = add_session_labels(df)

    # Señales enriquecidas
    logger.info("\n  Construyendo señales enriquecidas...")
    sigs_all = build_enriched_signals(df)
    sigs_is  = sigs_all[sigs_all['year'] < OOS_YEAR]
    sigs_oos = sigs_all[sigs_all['year'] >= OOS_YEAR]

    logger.info(f"  IS: {len(sigs_is)} días | OOS: {len(sigs_oos)} días")
    logger.info(f"  Días con previo bajista (IS): {sigs_is['prior_bearish'].sum()}")
    logger.info(f"  Días con previo bajista (OOS): {sigs_oos['prior_bearish'].sum()}")

    # Definir el mejor filtro: día previo bajista + 1H > 0.3%
    def best_filter(df_):
        return ((df_['prior_bearish'].values) &
                (np.abs(df_['first_hour_ret'].values) > 0.003))

    # Análisis
    anatomy  = anatomy_prior_day(sigs_is)
    thr_sens = prior_day_threshold_sensitivity(sigs_is)
    quartiles = prior_day_quartile_analysis(sigs_is)
    combos   = combined_filter_analysis(sigs_is)
    wf       = walk_forward_filtered(sigs_is, best_filter)
    kf       = kfold_filtered(sigs_is, best_filter)
    oos      = oos_validation_filtered(sigs_oos, best_filter)

    # Clasificación final
    logger.info("\n" + "═"*70)
    logger.info("  CLASIFICACIÓN FINAL — H3v2: FILTRO DÍA PREVIO BAJISTA")
    logger.info("═"*70)

    wf_pct = (wf['sharpe'] > 0).mean() if not wf.empty else 0

    checks = {
        'bear_sharpe_mejor_que_base':  anatomy['bear']['sharpe'] > anatomy['base']['sharpe'] + 0.2,
        'bear_p_value_significativo':  anatomy['bear']['pvalue'] < 0.10,
        'efecto_monotónico':           quartiles['sharpe'].iloc[0] > quartiles['sharpe'].iloc[-1] if len(quartiles) > 1 else False,
        'filtro_combo_sharpe_gt_1.5':  combos['sharpe'].max() > 1.5,
        'wf_mayoria_positivo':         wf_pct >= 0.55,
        'kfold_consistente':           kf.get('pct_positive', 0) >= 0.60,
        'oos_positivo':                oos.get('sharpe', 0) > 0,
        'oos_ann_gt_10pct':            oos.get('annual', 0) > 0.10,
    }

    score = sum(checks.values())
    for k, v in checks.items():
        logger.info(f"  {'✅' if v else '❌'} {k}")

    logger.info(f"\n  SCORE: {score}/8")

    if score >= 6:
        verdict = "🏆 FILTRO VÁLIDO — H3v2 tiene edge robusto con bajos costos"
    elif score >= 4:
        verdict = "✅ FILTRO PROMETEDOR — continuar validación en más datos"
    elif score >= 2:
        verdict = "⚠️  FILTRO DÉBIL — posible artefacto de muestreo"
    else:
        verdict = "❌ FILTRO NO VÁLIDO — overfitting o artefacto"

    logger.info(f"  VEREDICTO: {verdict}")

    # Gráfico
    plot_prior_day(anatomy, quartiles, thr_sens, combos, wf, kf, oos, sigs_all)

    # Guardar métricas
    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.bool_,)): return bool(o)
            if isinstance(o, (np.ndarray,)): return o.tolist()
            return super().default(o)

    out_data = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'cost_pts_nq': COST_TARGET_PTS,
        'n_is': int(len(sigs_is)),
        'n_oos': int(len(sigs_oos)),
        'n_bear_is': int(sigs_is['prior_bearish'].sum()),
        'anatomy': anatomy,
        'combined_filters': combos.to_dict('records'),
        'walk_forward': wf.to_dict('records') if not wf.empty else [],
        'kfold': kf,
        'oos_result': {k: v for k, v in oos.items() if k != 'equity'},
        'checks': {k: bool(v) for k, v in checks.items()},
        'score': int(score),
        'verdict': verdict,
    }

    out_json = ARTIFACTS_DIR / "h3v2_prior_day_metrics.json"
    with open(out_json, 'w') as f:
        json.dump(out_data, f, indent=2, cls=NE)

    logger.info(f"\n  Métricas → {out_json}")
    logger.info("  ✅ Script H3v2 completado")


if __name__ == "__main__":
    main()
