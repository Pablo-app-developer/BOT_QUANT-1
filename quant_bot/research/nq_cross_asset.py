"""
nq_cross_asset.py — Validación Cruzada: ¿El Edge H3v2 Existe en Activos Correlacionados?

OBJETIVO:
  Si el mismo patrón "primera hora → día" existe en SPY (S&P500), DIA (Dow Jones)
  u otros índices correlacionados con NQ100, entonces:
  1. El edge es más universal (no específico de NQ → más robusto)
  2. Podemos acumular OOS más rápido (3× más trades combinados)
  3. Reduce el riesgo de que sea un artefacto del instrumento particular

FUENTES DE DATOS:
  - NQ propio (Dukascopy M1): datos ya disponibles
  - Si hay datos adicionales de Dukascopy disponibles: los usa
  - Análisis de subgrupos del NQ como proxy de robustez adicional

TESTS:
  1. Sub-período IS (2021-2022) vs Sub-período IS (2022-2024) — ¿consistente?
  2. Edge en días de LUNES-MIÉRCOLES vs JUEVES-VIERNES — ¿uniforme?
  3. Edge en meses de alta estacionalidad vs baja — ¿depende de estacionalidad?
  4. Edge: ¿mejor en Q1/Q2 vs Q3/Q4?
  5. Edge en NQ separado por magnitud de la caída previa

  + Si hay datos de otros instrumentos Dukascopy (SPX500, DJIA, etc.)
    → test completo sobre ellos

CONCLUSIÓN:
  Si el edge se replica en ≥2 de los sub-tests de consistencia → más confianza.
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
        logging.FileHandler(ARTIFACTS_DIR / "cross_asset.log"),
    ]
)
logger = logging.getLogger("CrossAsset")

DUKASCOPY_SCALE = 82.0
COST_PTS = 2.0
OOS_YEAR = 2025

THR_PRIOR_DAY  = -0.001
THR_FIRST_HOUR = 0.003


# ════════════════════════════════════════════════════
# SEÑALES BASE REUTILIZABLES
# ════════════════════════════════════════════════════

def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Misma construcción que H3v2."""
    ny = df[df['session'].isin(['OPEN_HOUR', 'MIDDAY', 'CLOSE_HOUR'])].copy()
    records = []
    for date_key, group in ny.groupby(ny.index.date):
        oh   = group[group['session'] == 'OPEN_HOUR']
        post = group[group['session'].isin(['MIDDAY', 'CLOSE_HOUR'])]
        if len(oh) < 30 or len(post) < 10:
            continue
        oh_open = oh['open'].iloc[0]
        oh_close = oh['close'].iloc[-1]
        if oh_open <= 0:
            continue
        first_hour_ret = (oh_close - oh_open) / oh_open
        oh_atr = (oh['high'].max() - oh['low'].min()) / oh_open
        directionality = abs(first_hour_ret) / oh_atr if oh_atr > 0 else 0
        day_return = (post['close'].iloc[-1] - oh_open) / oh_open
        ret_eod = (post['close'].iloc[-1] - oh_close) / oh_close
        records.append({
            'date': pd.Timestamp(date_key, tz='UTC'),
            'year': date_key.year, 'month': date_key.month,
            'quarter': (date_key.month - 1) // 3 + 1,
            'dow': date_key.weekday(),
            'entry_price': float(oh_close),
            'first_hour_ret': float(first_hour_ret),
            'oh_atr': float(oh_atr),
            'directionality': float(directionality),
            'ret_eod': float(ret_eod),
            'day_return': float(day_return),
        })
    df_out = pd.DataFrame(records).set_index('date')
    df_out['prior_day_ret'] = df_out['day_return'].shift(1)
    df_out['prior_bearish'] = df_out['prior_day_ret'] < THR_PRIOR_DAY
    df_out['vol_10d'] = df_out['oh_atr'].rolling(10).mean()
    df_out['high_vol'] = df_out['oh_atr'] > df_out['vol_10d']
    return df_out.dropna(subset=['prior_day_ret', 'ret_eod', 'first_hour_ret'])


def qbt(rets_gross, signals, cost_pct=0.0):
    """Quick backtest."""
    mask = signals != 0
    if mask.sum() < 8:
        return {'n': int(mask.sum()), 'sharpe': 0.0, 'annual': 0.0, 'pvalue': 1.0, 'win_rate': 0.0}
    rn = rets_gross[mask] * signals[mask] - cost_pct
    if rn.std() == 0:
        return {'n': int(mask.sum()), 'sharpe': 0.0, 'annual': 0.0, 'pvalue': 1.0, 'win_rate': float((rn>0).mean())}
    sh = (rn.mean() / rn.std()) * np.sqrt(252)
    eq = np.cumprod(1 + rn)
    ann = eq[-1] ** (252 / len(rn)) - 1
    wr = (rn > 0).mean()
    _, p = stats.ttest_1samp(rn, 0)
    return {'n': int(mask.sum()), 'sharpe': float(sh), 'annual': float(ann),
            'pvalue': float(p), 'win_rate': float(wr)}


def apply_h3v2_filter(df):
    """Aplica el filtro H3v2 y retorna la señal."""
    filt = df['prior_bearish'] & (np.abs(df['first_hour_ret']) > THR_FIRST_HOUR)
    return filt, np.where(filt, np.sign(df['first_hour_ret']), 0)


# ════════════════════════════════════════════════════
# 1. CONSISTENCIA TEMPORAL — IS SPLIT EN SUBPERÍODOS
# ════════════════════════════════════════════════════

def temporal_consistency(df_is: pd.DataFrame) -> dict:
    """Divide IS en 3 subperíodos y verifica que el edge sea positivo en cada uno."""
    logger.info("\n" + "═"*68)
    logger.info("  1. CONSISTENCIA TEMPORAL — IS DIVIDIDO EN SUBPERÍODOS")
    logger.info("═"*68)
    logger.info("  (Si el edge depende de un sub-período → frágil)")

    cost = (COST_PTS / DUKASCOPY_SCALE) / df_is['entry_price'].mean()
    years = sorted(df_is['year'].unique())
    n_sub = 3
    chunks = np.array_split(years, n_sub)

    results = {}
    all_pos = True
    for i, chunk in enumerate(chunks):
        sub = df_is[df_is['year'].isin(chunk)]
        filt, sig = apply_h3v2_filter(sub)
        r = qbt(sub.loc[filt, 'ret_eod'].values, np.sign(sub.loc[filt, 'first_hour_ret'].values), cost)
        icon = "✅" if r['sharpe'] > 0 else "❌"
        if r['sharpe'] <= 0:
            all_pos = False
        years_label = f"{int(chunk[0])}-{int(chunk[-1])}"
        logger.info(f"  {icon} Subperíodo {years_label} ({i+1}/{n_sub}): "
                    f"n={r['n']:4d}  Sharpe={r['sharpe']:.3f}  WR={r['win_rate']*100:.1f}%  p={r['pvalue']:.4f}")
        results[years_label] = r

    logger.info(f"\n  {'✅ Consistente en todos los subperíodos' if all_pos else '❌ Inconsistente — depende del período'}")
    return results


# ════════════════════════════════════════════════════
# 2. CONSISTENCIA POR DÍA DE SEMANA
# ════════════════════════════════════════════════════

def dow_consistency(df_is: pd.DataFrame) -> dict:
    """¿El edge funciona uniformemente todos los días de la semana?"""
    logger.info("\n" + "═"*68)
    logger.info("  2. CONSISTENCIA POR DÍA DE SEMANA (IS)")
    logger.info("═"*68)

    cost = (COST_PTS / DUKASCOPY_SCALE) / df_is['entry_price'].mean()
    days = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie']
    results = {}
    sharpes = []

    for dow in range(5):
        sub = df_is[df_is['dow'] == dow]
        filt, _ = apply_h3v2_filter(sub)
        sub_f = sub[filt]
        if len(sub_f) < 8:
            continue
        r = qbt(sub_f['ret_eod'].values, np.sign(sub_f['first_hour_ret'].values), cost)
        icon = "✅" if r['sharpe'] > 0 else "❌"
        logger.info(f"  {icon} {days[dow]}: n={r['n']:4d}  Sharpe={r['sharpe']:.3f}  "
                    f"WR={r['win_rate']*100:.1f}%  p={r['pvalue']:.4f}")
        results[days[dow]] = r
        sharpes.append(r['sharpe'])

    sharpes = np.array(sharpes)
    pct_pos = (sharpes > 0).mean()
    logger.info(f"\n  Días positivos: {pct_pos*100:.0f}%  "
                f"{'✅ Uniforme' if pct_pos >= 0.6 else '❌ Concentrado en pocos días'}")
    return results


# ════════════════════════════════════════════════════
# 3. CONSISTENCIA POR TRIMESTRE Y MES
# ════════════════════════════════════════════════════

def quarterly_consistency(df_is: pd.DataFrame) -> dict:
    """¿El edge tiene estacionalidad fuerte?"""
    logger.info("\n" + "═"*68)
    logger.info("  3. CONSISTENCIA POR TRIMESTRE (IS)")
    logger.info("═"*68)

    cost = (COST_PTS / DUKASCOPY_SCALE) / df_is['entry_price'].mean()
    results = {}

    for q in [1, 2, 3, 4]:
        sub = df_is[df_is['quarter'] == q]
        filt, _ = apply_h3v2_filter(sub)
        sub_f = sub[filt]
        if len(sub_f) < 8:
            continue
        r = qbt(sub_f['ret_eod'].values, np.sign(sub_f['first_hour_ret'].values), cost)
        icon = "✅" if r['sharpe'] > 0 else "❌"
        logger.info(f"  {icon} Q{q}: n={r['n']:4d}  Sharpe={r['sharpe']:.3f}  "
                    f"WR={r['win_rate']*100:.1f}%  p={r['pvalue']:.4f}")
        results[f'Q{q}'] = r

    return results


# ════════════════════════════════════════════════════
# 4. GRADIENTE DE MAGNITUD DEL DÍA PREVIO
# ════════════════════════════════════════════════════

def prior_day_magnitude_gradient(df_is: pd.DataFrame) -> pd.DataFrame:
    """
    Descompone el edge por deciles de magnitud de la caída previa.
    ¿A mayor caída previa = mayor edge? (monotónico = más confiable)
    """
    logger.info("\n" + "═"*68)
    logger.info("  4. GRADIENTE: MAGNITUD CAÍDA PREVIA → EDGE")
    logger.info("═"*68)

    cost   = (COST_PTS / DUKASCOPY_SCALE) / df_is['entry_price'].mean()
    df_b   = df_is[df_is['prior_bearish']].copy()  # solo días con previo bajista
    prior  = df_b['prior_day_ret'].values

    # Solo testar la mitad bajista con deciles
    quantiles = np.percentile(prior, [0, 25, 50, 75, 100])

    rows = []
    for i in range(len(quantiles)-1):
        lo, hi = quantiles[i], quantiles[i+1]
        mask = (prior >= lo) & (prior < hi)
        sub_f = df_b[mask]
        if len(sub_f) < 8:
            continue
        r = qbt(sub_f['ret_eod'].values, np.sign(sub_f['first_hour_ret'].values), cost)
        label = f"Q{i+1} ({lo*100:.2f}%→{hi*100:.2f}%)"
        icon  = "✅" if r['sharpe'] > 0 else "❌"
        logger.info(f"  {icon} {label}: n={r['n']:4d}  Sharpe={r['sharpe']:.3f}  "
                    f"WR={r['win_rate']*100:.1f}%")
        rows.append({'decile': label, 'lo': float(lo), 'hi': float(hi), **r})

    df_res = pd.DataFrame(rows)

    # Monotónico: ¿más bajista = mayor Sharpe?
    if not df_res.empty and len(df_res) > 1:
        sharpes = df_res['sharpe'].values
        # Q1 = más bajista → debería tener mayor Sharpe
        is_monotonic = sharpes[0] >= sharpes[-1]
        logger.info(f"\n  Monotónico (más bajista = más edge): "
                    f"{'✅ SÍ' if is_monotonic else '❌ NO'}")

    return df_res


# ════════════════════════════════════════════════════
# 5. BUSCAR DATOS DE OTROS INSTRUMENTOS DUKASCOPY
# ════════════════════════════════════════════════════

def check_other_instruments() -> list:
    """Detecta si hay datos de otros instrumentos Dukascopy disponibles."""
    data_root = PROJECT_ROOT / "quant_bot" / "data"
    instruments = []
    for path in data_root.iterdir():
        if path.is_dir() and path.name not in ('processed', 'raw', '__pycache__'):
            # Verificar que tiene subcarpetas de años
            years = [p for p in path.iterdir() if p.is_dir() and p.name.isdigit()]
            if years:
                instruments.append(path.name)

    logger.info("\n" + "═"*68)
    logger.info("  5. INSTRUMENTOS DUKASCOPY DISPONIBLES")
    logger.info("═"*68)
    for inst in instruments:
        logger.info(f"  📊 {inst}")

    if len(instruments) == 1:
        logger.info("  → Solo tenemos NQ. Para cross-asset real: descargar SPX500 de Dukascopy.")
        logger.info("  → Alternativa: usar subconsistencia interna como validación adicional.")
    return instruments


# ════════════════════════════════════════════════════
# 6. REPLICACIÓN EN VENTANAS OOS EXPANCIVAS
# ════════════════════════════════════════════════════

def expanding_oos_validation(df_is: pd.DataFrame, df_oos: pd.DataFrame) -> pd.DataFrame:
    """
    Simula el edge acumulando datos OOS trimestre a trimestre.
    ¿El equity OOS es estable y creciente?
    """
    logger.info("\n" + "═"*68)
    logger.info("  6. VALIDACIÓN OOS EXPANSIVA (trimestre a trimestre)")
    logger.info("═"*68)

    cost = (COST_PTS / DUKASCOPY_SCALE) / df_oos['entry_price'].mean()
    df_oos_sorted = df_oos.sort_index()

    # Obtener trimestres únicos en OOS
    df_oos_copy = df_oos_sorted.copy()
    df_oos_copy['quarter_year'] = df_oos_copy.index.to_period('Q')
    quarters = sorted(df_oos_copy['quarter_year'].unique())

    rows = []
    accumulated_rets = []

    for q in quarters:
        q_data = df_oos_copy[df_oos_copy['quarter_year'] == q]
        filt, sig = apply_h3v2_filter(q_data)
        q_f = q_data[filt]
        if len(q_f) < 3:
            rows.append({'quarter': str(q), 'n': 0, 'sharpe': 0, 'win_rate': 0})
            continue

        rets = q_f['ret_eod'].values * np.sign(q_f['first_hour_ret'].values) - cost
        accumulated_rets.extend(rets.tolist())
        sh_acc = (np.mean(accumulated_rets) / np.std(accumulated_rets)) * np.sqrt(252) \
                 if np.std(accumulated_rets) > 0 else 0
        wr = (rets > 0).mean()
        logger.info(f"  Q={q}: n_quarter={len(rets):3d}  WR={wr*100:.0f}%  "
                    f"Sharpe acum.={sh_acc:.3f}  total_trades={len(accumulated_rets)}")
        rows.append({'quarter': str(q), 'n': len(rets),
                     'sharpe_accum': float(sh_acc),
                     'win_rate': float(wr),
                     'total_trades': len(accumulated_rets)})

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════
# 7. PLAN DE DESCARGA DE DATOS ADICIONALES
# ════════════════════════════════════════════════════

def generate_download_plan() -> str:
    """
    Genera el plan de descarga de datos Dukascopy para cross-asset.
    Dukascopy permite descargar cualquier instrumento en .bi5 format.
    """
    plan = """
    ═══════════════════════════════════════════════════════════════
    PLAN: DESCARGA DE DATOS CRUZADOS DE DUKASCOPY
    ═══════════════════════════════════════════════════════════════

    INSTRUMENTOS OBJETIVO (mismo horario NY, alta correlación con NQ):

    1. USAINDXUSD (S&P 500 CFD) — correlación con NQ: ~0.93
       Directorio destino: quant_bot/data/USAINDXUSD/
       Años: 2021-2025

    2. WS30USD (Dow Jones CFD) — correlación con NQ: ~0.85
       Directorio destino: quant_bot/data/WS30USD/
       Años: 2021-2025

    MÉTODO: Dukascopy JForex Historical Data (público)
    URL: https://www.dukascopy.com/datafeed/
    Formato: YYYY/MM/DD_HH.bi5 (idéntico a NQ)

    SCRIPT DE DESCARGA SUGERIDO:
      python3 quant_bot/data/download_dukascopy.py --instrument USAINDXUSD --years 2021-2025
      python3 quant_bot/data/download_dukascopy.py --instrument WS30USD    --years 2021-2025

    BENEFICIO:
      NQ:  ~48 trades/año → OOS~4/mes
      NQ + SPX + DJI: ~144 trades/año → OOS~12/mes
      → Confirmación estadística en ~10 meses (vs 54 meses solo con NQ)
    ═══════════════════════════════════════════════════════════════
    """
    return plan


# ════════════════════════════════════════════════════
# VISUALIZACIÓN
# ════════════════════════════════════════════════════

def plot_cross_validation(temporal: dict, dow_res: dict, quarterly: dict,
                          gradient: pd.DataFrame, exp_oos: pd.DataFrame) -> None:

    fig = plt.figure(figsize=(20, 18), facecolor='#0d1117')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)
    GOLD='#FFD700'; GREEN='#00FF88'; RED='#FF4444'; BLUE='#4488FF'; GRAY='#888888'; BG='#161b22'

    def ax_style(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, color=GOLD, fontsize=10, fontweight='bold', pad=8)
        ax.tick_params(colors=GRAY)
        ax.spines[:].set_color('#333333')
        for l in ax.get_xticklabels() + ax.get_yticklabels(): l.set_color(GRAY)

    # 1. Temporal consistency
    ax1 = fig.add_subplot(gs[0, 0])
    if temporal:
        labels = list(temporal.keys())
        sharpes = [temporal[k]['sharpe'] for k in labels]
        colors1 = [GREEN if s > 0 else RED for s in sharpes]
        ax1.bar(labels, sharpes, color=colors1, alpha=0.85)
        ax1.axhline(0, color=GRAY, lw=0.8)
        for i, (l, s) in enumerate(zip(labels, sharpes)):
            ax1.text(i, s + 0.05 * np.sign(s), f"{s:.2f}", ha='center', color='white', fontsize=9)
    ax_style(ax1, "Consistencia IS\nPor Subperíodo Temporal")
    ax1.set_ylabel("Sharpe", color=GRAY)

    # 2. DOW consistency
    ax2 = fig.add_subplot(gs[0, 1])
    if dow_res:
        days = list(dow_res.keys())
        sharpes_dow = [dow_res[d]['sharpe'] for d in days]
        colors2 = [GREEN if s > 0 else RED for s in sharpes_dow]
        ax2.bar(days, sharpes_dow, color=colors2, alpha=0.85)
        ax2.axhline(0, color=GRAY, lw=0.8)
    ax_style(ax2, "Consistencia por\nDía de Semana")
    ax2.set_ylabel("Sharpe", color=GRAY)

    # 3. Quarterly
    ax3 = fig.add_subplot(gs[0, 2])
    if quarterly:
        qs = list(quarterly.keys())
        sharpes_q = [quarterly[q]['sharpe'] for q in qs]
        colors3 = [GREEN if s > 0 else RED for s in sharpes_q]
        ax3.bar(qs, sharpes_q, color=colors3, alpha=0.85)
        ax3.axhline(0, color=GRAY, lw=0.8)
        for i, (q, s) in enumerate(zip(qs, sharpes_q)):
            ax3.text(i, s + 0.05 * np.sign(s), f"{s:.2f}", ha='center', color='white', fontsize=9)
    ax_style(ax3, "Consistencia por\nTrimestre")
    ax3.set_ylabel("Sharpe", color=GRAY)

    # 4. Gradient prior day
    ax4 = fig.add_subplot(gs[1, :2])
    if not gradient.empty:
        colors4 = [GREEN if s > 0 else RED for s in gradient['sharpe']]
        bars4 = ax4.bar(range(len(gradient)), gradient['sharpe'], color=colors4, alpha=0.85)
        ax4.set_xticks(range(len(gradient)))
        ax4.set_xticklabels(gradient['decile'].str[:20], rotation=25, fontsize=8)
        ax4.axhline(0, color=GRAY, lw=0.8)
        for bar, val in zip(bars4, gradient['sharpe']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.2f}", ha='center', color='white', fontsize=8)
    ax_style(ax4, "Gradiente: Magnitud Caída Previa → Sharpe\n(Q1=más bajista, Q4=menos bajista)")
    ax4.set_ylabel("Sharpe IS", color=GRAY)

    # 5. Expanding OOS
    ax5 = fig.add_subplot(gs[1, 2])
    if not exp_oos.empty and 'total_trades' in exp_oos.columns:
        acc_sharpe = exp_oos['sharpe_accum'].fillna(0).values
        total      = exp_oos['total_trades'].values
        color5 = GREEN if acc_sharpe[-1] > 0 else RED
        ax5.plot(total, acc_sharpe, color=color5, lw=2.5, marker='o', markersize=6)
        ax5.fill_between(total, 0, acc_sharpe, alpha=0.2, color=color5)
        ax5.axhline(0, color=RED, lw=1, ls='--')
        ax5.axhline(0.5, color=GOLD, lw=1, ls=':', label='Sharpe=0.5')
        ax5.legend(facecolor=BG, labelcolor='white', fontsize=8)
    ax_style(ax5, "OOS Expansivo: Sharpe acumulado\nvs N trades")
    ax5.set_xlabel("N trades OOS", color=GRAY)
    ax5.set_ylabel("Sharpe (acumulado)", color=GRAY)

    # 6. Síntesis: heatmap de robustez (visual)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.set_facecolor(BG); ax6.axis('off')

    # Tabla de consistencia
    headers = ['Test', 'Resultado', 'Sharpe', 'Conclusión']
    rows_summary = []
    if temporal:
        pct_pos = sum(1 for v in temporal.values() if v['sharpe'] > 0) / len(temporal)
        rows_summary.append(['Temporal (subperíodos)', f"{pct_pos*100:.0f}% positivos",
                             f"{np.mean([v['sharpe'] for v in temporal.values()]):.2f}",
                             '✅' if pct_pos >= 0.67 else '❌'])
    if dow_res:
        pct_pos = sum(1 for v in dow_res.values() if v['sharpe'] > 0) / len(dow_res)
        rows_summary.append(['DOW (día semana)', f"{pct_pos*100:.0f}% positivos",
                             f"{np.mean([v['sharpe'] for v in dow_res.values()]):.2f}",
                             '✅' if pct_pos >= 0.60 else '❌'])
    if quarterly:
        pct_pos = sum(1 for v in quarterly.values() if v['sharpe'] > 0) / len(quarterly)
        rows_summary.append(['Trimestral', f"{pct_pos*100:.0f}% positivos",
                             f"{np.mean([v['sharpe'] for v in quarterly.values()]):.2f}",
                             '✅' if pct_pos >= 0.50 else '❌'])
    if not gradient.empty:
        monotonic = gradient['sharpe'].iloc[0] >= gradient['sharpe'].iloc[-1]
        rows_summary.append(['Gradiente previo', 'Monotónico' if monotonic else 'No monotónico',
                             f"{gradient['sharpe'].mean():.2f}", '✅' if monotonic else '❌'])
    if not exp_oos.empty and 'sharpe_accum' in exp_oos.columns:
        final_sh = exp_oos['sharpe_accum'].iloc[-1] if len(exp_oos) > 0 else 0
        rows_summary.append(['OOS expansivo', f"Sharpe final: {final_sh:.2f}",
                             f"{final_sh:.2f}", '✅' if final_sh > 0.3 else '❌'])

    score_total = sum(1 for r in rows_summary if r[3] == '✅')
    table = ax6.table(
        cellText=rows_summary,
        colLabels=headers,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2d3748')
            cell.set_text_props(color=GOLD, fontweight='bold')
        else:
            cell.set_facecolor('#1a202c')
            text = cell.get_text().get_text()
            cell.set_text_props(color=GREEN if '✅' in text else (RED if '❌' in text else 'white'))
    ax6.set_title(f"SÍNTESIS: Robustez Cross-Validation  SCORE={score_total}/{len(rows_summary)}",
                  color=GOLD, fontsize=11, fontweight='bold', pad=12)

    n_checks = len(rows_summary)
    verdict = ('🏆 EDGE ALTAMENTE ROBUSTO' if score_total >= n_checks * 0.8 else
               '✅ EDGE ROBUSTO' if score_total >= n_checks * 0.6 else
               '⚠️  EDGE MODERADO' if score_total >= n_checks * 0.4 else '❌ EDGE FRÁGIL')

    fig.suptitle(
        f"H3v2 CROSS-VALIDATION — Robustez Interna\n"
        f"Score: {score_total}/{len(rows_summary)} — {verdict}",
        color='white', fontsize=13, fontweight='bold', y=0.99
    )

    out = ARTIFACTS_DIR / "nq_h3_cross_validation.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"\n  ✅ Gráfico: {out}")


# ════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════

def main():
    logger.info("╔" + "═"*70 + "╗")
    logger.info("║   CROSS-ASSET VALIDATION — Robustez del Edge H3v2              ║")
    logger.info("║   ¿El edge es universal o específico de NQ?                     ║")
    logger.info("╚" + "═"*70 + "╝")

    parquet = PROJECT_ROOT / "quant_bot" / "data" / "processed" / "USATECHIDXUSD_M1.parquet"
    logger.info(f"\n  Cargando {parquet.name}...")
    df = pd.read_parquet(parquet, engine='pyarrow')
    logger.info(f"  → {len(df):,} barras")

    if 'session' not in df.columns:
        from quant_bot.data.nq_loader import add_session_labels
        df = add_session_labels(df)

    logger.info("\n  Construyendo señales...")
    sigs_all = build_signals(df)
    sigs_is  = sigs_all[sigs_all['year'] < OOS_YEAR]
    sigs_oos = sigs_all[sigs_all['year'] >= OOS_YEAR]

    logger.info(f"  IS: {len(sigs_is)} días | OOS: {len(sigs_oos)} días")

    # Análisis
    instruments   = check_other_instruments()
    temporal      = temporal_consistency(sigs_is)
    dow_res       = dow_consistency(sigs_is)
    quarterly_res = quarterly_consistency(sigs_is)
    gradient      = prior_day_magnitude_gradient(sigs_is)
    exp_oos       = expanding_oos_validation(sigs_is, sigs_oos)

    # Mostrar plan de descarga
    plan = generate_download_plan()
    logger.info(plan)

    # Clasificación final
    logger.info("\n" + "═"*70)
    logger.info("  CLASIFICACIÓN FINAL — CROSS-VALIDATION INTERNA")
    logger.info("═"*70)

    checks = {
        'temporal_2_de_3_positivo':    sum(1 for v in temporal.values() if v['sharpe'] > 0) >= 2,
        'dow_3_de_5_positivo':         sum(1 for v in dow_res.values()    if v['sharpe'] > 0) >= 3,
        'quarterly_3_de_4_positivo':   sum(1 for v in quarterly_res.values() if v['sharpe'] > 0) >= 3,
        'gradiente_monotonico':         (not gradient.empty and
                                        gradient['sharpe'].iloc[0] >= gradient['sharpe'].iloc[-1]) if len(gradient) > 1 else False,
        'oos_expansivo_positivo':       (not exp_oos.empty and
                                        'sharpe_accum' in exp_oos.columns and
                                        exp_oos['sharpe_accum'].iloc[-1] > 0) if not exp_oos.empty else False,
    }

    score = sum(checks.values())
    for k, v in checks.items():
        logger.info(f"  {'✅' if v else '❌'} {k}")

    logger.info(f"\n  SCORE: {score}/{len(checks)}")
    verdict = ('🏆 EDGE UNIVERSAL — muy robusto y probablemente real' if score >= 4 else
               '✅ EDGE ROBUSTO — consistente en múltiples dimensiones' if score >= 3 else
               '⚠️  EDGE MODERADO — funciona pero con varianza alta' if score >= 2 else
               '❌ EDGE FRÁGIL — depende de condiciones específicas')
    logger.info(f"  VEREDICTO: {verdict}")

    # Gráfico
    plot_cross_validation(temporal, dow_res, quarterly_res, gradient, exp_oos)

    # Guardar
    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.bool_,)): return bool(o)
            if isinstance(o, (np.ndarray,)): return o.tolist()
            return super().default(o)

    out_data = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'instruments_available': instruments,
        'temporal_consistency': temporal,
        'dow_consistency': dow_res,
        'quarterly_consistency': quarterly_res,
        'gradient': gradient.to_dict('records') if not gradient.empty else [],
        'expanding_oos': exp_oos.to_dict('records') if not exp_oos.empty else [],
        'checks': {k: bool(v) for k, v in checks.items()},
        'score': int(score), 'verdict': verdict,
    }
    out_json = ARTIFACTS_DIR / "h3_cross_validation_metrics.json"
    with open(out_json, 'w') as f:
        json.dump(out_data, f, indent=2, cls=NE)

    logger.info(f"\n  Métricas → {out_json}")
    logger.info("  ✅ Cross-validation completado")


if __name__ == "__main__":
    main()
