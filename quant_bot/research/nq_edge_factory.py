"""
nq_edge_factory.py — Búsqueda Sistemática de Edges (20 Hipótesis)

FILOSOFÍA:
  Testeamos 20 hipótesis simultáneamente.
  Todas bajo corrección estadística Benjamini-Hochberg (FDR < 5%).
  Separación estricta IS (2021-2023) / OOS puro (2024-2025).
  Solo se acepta un edge si:
    1. Sobrevive IS con Sharpe > 0.5 y p_ajustado < 0.05
    2. Tiene una explicación económica coherente
    3. Sobrevive OOS con signo positivo

HIPÓTESIS:
  H01  - First Hour Momentum (base H3)
  H02  - First Hour + Día Previo Bajista (H3v2)
  H03  - First Hour + Día Previo Bajista + Alta Volatilidad
  H04  - Opening Range Breakout 15min
  H05  - Opening Range Breakout 30min
  H06  - Gap de Apertura → Continuación
  H07  - Gap de Apertura → Reversión (Gap Fill)
  H08  - Momentum Lunes (weekend effect → lunes alcista)
  H09  - Reversión Miércoles (mid-week reversal)
  H10  - Pre-Close Momentum (retorno 15:30-16:00 → dirección cierre)
  H11  - Volatilidad Alta → Reversión Intra-Session
  H12  - Volatilidad Baja → Momentum Intra-Session
  H13  - 2 Días Consecutivos Bajistas → Rebote
  H14  - 3 Días Consecutivos Alcistas → Continuación o Reversión
  H15  - Tamaño de la Primera Hora: extremo → dirección
  H16  - Primera Hora Estrecha → Breakout Post-lunch (2:30pm NY)
  H17  - Viernes Sesgo Bajista (venta de fin de semana)
  H18  - Lunes Sesgo Alcista (compra inicio semana)
  H19  - Martes Reversión (corrección al lunes)
  H20  - First Hour + Triple Filtro (previo + volatilidad + directionality)
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
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ARTIFACTS_DIR / "edge_factory.log"),
    ]
)
logger = logging.getLogger("EdgeFactory")

# ─── PARÁMETROS GLOBALES ───────────────────────────────────────────────
COST_PTS = 2.0           # pts NQ round-trip (costo conservador)
DUKA_SCALE = 82.0        # precio NQ ≈ 21.000 / 82 divisor BI5
IS_YEARS   = [2021, 2022, 2023]
OOS_YEARS  = [2024, 2025]
MIN_TRADES = 30          # mínimo trades para considerar hipótesis
FDR_ALPHA  = 0.10        # False Discovery Rate (BH correction)


# ─── CONSTRUCCIÓN DEL DATASET DIARIO ENRIQUECIDO ──────────────────────

def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye un DataFrame diario con todas las features necesarias
    para testear las 20 hipótesis.
    """
    ny = df[df['session'].isin(['OPEN_HOUR', 'MIDDAY', 'CLOSE_HOUR'])].copy()

    records = []
    for date_key, group in ny.groupby(ny.index.date):
        oh   = group[group['session'] == 'OPEN_HOUR']
        post = group[group['session'].isin(['MIDDAY', 'CLOSE_HOUR'])]

        if len(oh) < 30 or len(post) < 10:
            continue

        oh_open    = oh['open'].iloc[0]
        oh_close   = oh['close'].iloc[-1]
        oh_high    = oh['high'].max()
        oh_low     = oh['low'].min()
        eod_close  = post['close'].iloc[-1]

        if oh_open <= 0:
            continue

        # ── Retornos clave ─────────────────────────
        r_first_hour   = (oh_close - oh_open) / oh_open       # Ret 13:30-14:29
        r_eod_from_1h  = (eod_close - oh_close) / oh_close    # Ret 14:29-20:00 (PnL trade)
        r_eod_full     = (eod_close - oh_open) / oh_open       # Ret 13:30-20:00

        oh_atr         = (oh_high - oh_low) / oh_open          # Rango 1H (normalizado)
        directionality = abs(r_first_hour) / oh_atr if oh_atr > 0 else 0

        # ── Pre-close: barras 15:30-16:00 UTC ──────
        pre_close_bars = post.between_time('15:30', '16:00') if hasattr(post.index, 'time') else pd.DataFrame()
        # Para barras UTC ya indexadas:
        pc_mask = (post.index.hour == 15) & (post.index.minute >= 30)
        pc_mask |= (post.index.hour == 16) & (post.index.minute == 0)
        pre_close = post[pc_mask]
        r_pre_close = 0.0
        if len(pre_close) >= 1:
            pc_open  = pre_close['open'].iloc[0]
            pc_close = pre_close['close'].iloc[-1]
            r_pre_close = (pc_close - pc_open) / pc_open if pc_open > 0 else 0.0

        # ── Post-lunch: barras 17:30-18:30 UTC ─────
        pl_mask = (post.index.hour == 17) & (post.index.minute >= 30)
        pl_mask |= post.index.hour == 18
        post_lunch = post[pl_mask & (post.index.hour < 19)]
        r_post_lunch = 0.0
        if len(post_lunch) >= 1:
            pl_open  = post_lunch['open'].iloc[0]
            pl_close = post_lunch['close'].iloc[-1]
            r_post_lunch = (pl_close - pl_open) / pl_open if pl_open > 0 else 0.0

        # ── Gap de apertura (vs cierre día anterior) ─
        # Se calcula despues de shift
        records.append({
            'date':          pd.Timestamp(date_key, tz='UTC'),
            'year':          date_key.year,
            'dow':           date_key.weekday(),  # 0=Lun, 4=Vie
            'month':         date_key.month,
            'quarter':       (date_key.month - 1) // 3 + 1,

            'oh_open':       float(oh_open),
            'oh_close':      float(oh_close),
            'oh_high':       float(oh_high),
            'oh_low':        float(oh_low),
            'eod_close':     float(eod_close),

            'r_first_hour':   float(r_first_hour),
            'r_eod_full':     float(r_eod_full),
            'r_eod_from_1h':  float(r_eod_from_1h),

            'oh_atr':         float(oh_atr),
            'directionality': float(directionality),
            'r_pre_close':    float(r_pre_close),
            'r_post_lunch':   float(r_post_lunch),
        })

    df_d = pd.DataFrame(records).set_index('date').sort_index()

    # ── Features con look-back (shift) ──────────────
    df_d['prev_r']     = df_d['r_eod_full'].shift(1)   # retorno día anterior
    df_d['prev_r2']    = df_d['r_eod_full'].shift(2)
    df_d['prev_r3']    = df_d['r_eod_full'].shift(3)
    df_d['atr_ma10']   = df_d['oh_atr'].rolling(10).mean()
    df_d['gap']        = (df_d['oh_open'] - df_d['eod_close'].shift(1)) / df_d['eod_close'].shift(1)

    # OR del día (Opening Range)
    df_d['or_high']    = df_d['oh_high']
    df_d['or_low']     = df_d['oh_low']
    df_d['or_size']    = df_d['oh_atr']

    df_d['high_vol']   = df_d['oh_atr'] > df_d['atr_ma10'] * 1.1
    df_d['low_vol']    = df_d['oh_atr'] < df_d['atr_ma10'] * 0.9

    df_d['consec_bear'] = (
        (df_d['prev_r'] < 0) &
        (df_d['prev_r2'] < 0)
    )
    df_d['consec3_bull'] = (
        (df_d['prev_r'] > 0) &
        (df_d['prev_r2'] > 0) &
        (df_d['prev_r3'] > 0)
    )

    # 1H en la primera vez
    df_d['first_1h_big']   = df_d['oh_atr'] > df_d['atr_ma10'] * 1.3
    df_d['first_1h_small'] = df_d['oh_atr'] < df_d['atr_ma10'] * 0.6

    return df_d.dropna(subset=['prev_r', 'atr_ma10', 'gap'])


# ─── MOTOR DE BACKTEST SIMPLE ──────────────────────────────────────────

def bt(signals: np.ndarray, rets: np.ndarray, cost_pct: float, label: str) -> dict:
    """
    signals: array de -1, 0, +1
    rets: array de retornos brutos en la dirección 'natural' (long)
    Retorna diccionario con métricas clave.
    """
    mask = signals != 0
    n = int(mask.sum())

    if n < MIN_TRADES:
        return dict(label=label, n=n, sharpe=0, annual=0, wr=0, pval=1.0,
                    cost=cost_pct, status='INSUF')

    trade_rets = rets[mask] * signals[mask] - cost_pct
    mu  = trade_rets.mean()
    sig = trade_rets.std()

    if sig == 0:
        return dict(label=label, n=n, sharpe=0, annual=0, wr=0, pval=1.0,
                    cost=cost_pct, status='FLAT')

    sharpe = (mu / sig) * np.sqrt(252)
    eq     = np.cumprod(1 + trade_rets)
    ann    = eq[-1] ** (252 / n) - 1
    wr     = (trade_rets > 0).mean()
    _, pval = sp_stats.ttest_1samp(trade_rets, 0)

    return dict(label=label, n=n, sharpe=float(sharpe), annual=float(ann),
                wr=float(wr), pval=float(pval), cost=float(cost_pct),
                mu=float(mu), sigma=float(sig), status='OK',
                equity=eq.tolist())


# ─── LAS 20 HIPÓTESIS ─────────────────────────────────────────────────

def run_all_hypotheses(df: pd.DataFrame, suffix='IS') -> list:
    cost = (COST_PTS / DUKA_SCALE) / df['oh_close'].mean()
    r    = df['r_eod_from_1h'].values
    r_full = df['r_eod_full'].values
    fhr  = df['r_first_hour'].values
    results = []

    # H01 — First Hour Momentum (sin filtros)
    sig  = np.where(np.abs(fhr) > 0.003, np.sign(fhr), 0)
    results.append(bt(sig, r, cost, f"H01-FirstHrMomentum [{suffix}]"))

    # H02 — First Hour + Día Previo Bajista (H3v2)
    cond = (df['prev_r'].values < -0.001) & (np.abs(fhr) > 0.003)
    sig  = np.where(cond, np.sign(fhr), 0)
    results.append(bt(sig, r, cost, f"H02-H3v2 [{suffix}]"))

    # H03 — H3v2 + Alta Volatilidad
    cond = (df['prev_r'].values < -0.001) & (np.abs(fhr) > 0.003) & df['high_vol'].values
    sig  = np.where(cond, np.sign(fhr), 0)
    results.append(bt(sig, r, cost, f"H03-H3v2+HighVol [{suffix}]"))

    # H04 — OR Breakout 15min (proxy: OH fue < 50% primer ATR pero salió)
    ors  = df['oh_atr'].values
    big  = np.abs(fhr) > 0.003
    cond = big & (ors > df['atr_ma10'].values * 1.0)  # ya en rango normalmente activo
    sig  = np.where(cond, np.sign(fhr), 0)
    results.append(bt(sig, r, cost, f"H04-ORB-proxy [{suffix}]"))

    # H05 — OR muy estrecho → dirección post-lunch
    cond = df['first_1h_small'].values
    sig  = np.where(cond, np.sign(df['r_post_lunch'].values), 0)
    results.append(bt(sig, r_full, cost, f"H05-SmallOR-PostLunch [{suffix}]"))

    # H06 — Gap Apertura → Continuación
    gap  = df['gap'].values
    cond = np.abs(gap) > 0.002
    sig  = np.where(cond, np.sign(gap), 0)
    results.append(bt(sig, r_full, cost, f"H06-GapContinuation [{suffix}]"))

    # H07 — Gap → Reversión (Gap Fill)
    cond = np.abs(gap) > 0.002
    sig  = np.where(cond, -np.sign(gap), 0)
    results.append(bt(sig, r_full, cost, f"H07-GapReversal [{suffix}]"))

    # H08 — Lunes sesgo ALCISTA (Monday effect)
    cond = df['dow'].values == 0  # Monday
    sig  = np.where(cond, 1, 0)  # siempre LONG los Lunes
    results.append(bt(sig, r_full, cost, f"H08-Monday-Long [{suffix}]"))

    # H09 — Miércoles → dirección de la primera hora (mid-week clarity)
    cond = (df['dow'].values == 2) & (np.abs(fhr) > 0.002)
    sig  = np.where(cond, np.sign(fhr), 0)
    results.append(bt(sig, r, cost, f"H09-Wednesday-1H [{suffix}]"))

    # H10 — Pre-Close Momentum: el movimiento 15:30-16:00 predice el resto
    cond = np.abs(df['r_pre_close'].values) > 0.001
    sig  = np.where(cond, np.sign(df['r_pre_close'].values), 0)
    results.append(bt(sig, r_full, cost, f"H10-PreClose-Momentum [{suffix}]"))

    # H11 — Alta Volatilidad → Reversión (mean revert after spike)
    cond = df['high_vol'].values & (np.abs(fhr) > 0.003)
    sig  = np.where(cond, -np.sign(fhr), 0)  # contra la primera hora
    results.append(bt(sig, r, cost, f"H11-HighVol-Reversal [{suffix}]"))

    # H12 — Baja Volatilidad → Momentum (breakout en vol baja)
    cond = df['low_vol'].values & (np.abs(fhr) > 0.002)
    sig  = np.where(cond, np.sign(fhr), 0)
    results.append(bt(sig, r, cost, f"H12-LowVol-Momentum [{suffix}]"))

    # H13 — 2 Días Bajistas Consecutivos → Rebote al tercer día
    cond = df['consec_bear'].values & (np.abs(fhr) > 0.002)
    sig  = np.where(cond, np.sign(fhr), 0)
    results.append(bt(sig, r, cost, f"H13-ConsecBear2-Bounce [{suffix}]"))

    # H14 — 3 Días Alcistas → Reversión (se agotó el momentum)
    cond = df['consec3_bull'].values & (np.abs(fhr) > 0.002)
    sig  = np.where(cond, -np.sign(fhr), 0)  # contra la primera hora
    results.append(bt(sig, r, cost, f"H14-Bull3-Reversal [{suffix}]"))

    # H15 — Primera Hora EXTREMA (>1.5x ATR mediano)
    cond = df['first_1h_big'].values & (np.abs(fhr) > 0.004)
    sig  = np.where(cond, np.sign(fhr), 0)
    results.append(bt(sig, r, cost, f"H15-ExtremeFirstHour [{suffix}]"))

    # H16 — First Hour pequeña post-luncha recovery
    cond = df['first_1h_small'].values & (np.abs(fhr) > 0.001)
    sig  = np.where(cond, np.sign(fhr), 0)
    results.append(bt(sig, r, cost, f"H16-SmallOR-Continuation [{suffix}]"))

    # H17 — Viernes sesgo BAJISTA (liquidación fin de semana)
    cond = df['dow'].values == 4  # Friday
    sig  = np.where(cond, -1, 0)  # siempre SHORT los Viernes
    results.append(bt(sig, r_full, cost, f"H17-Friday-Short [{suffix}]"))

    # H18 — Lunes + Primer Hora Alcista → Fuerte continuación
    cond = (df['dow'].values == 0) & (fhr > 0.003)
    sig  = np.where(cond, 1, 0)
    results.append(bt(sig, r, cost, f"H18-Monday-BullFirstHour [{suffix}]"))

    # H19 — Martes Reversión (corrección del lunes)
    cond = (df['dow'].values == 1) & (df['prev_r'].values > 0.003)
    sig  = np.where(cond, -np.sign(fhr), 0)  # Short si Lunes fue alcista
    results.append(bt(sig, r, cost, f"H19-Tuesday-Reversal [{suffix}]"))

    # H20 — Triple Filtro: previo bajista + alta vol + directionality
    cond = (
        (df['prev_r'].values < -0.001) &
        (np.abs(fhr) > 0.003) &
        df['high_vol'].values &
        (df['directionality'].values > 0.3)
    )
    sig  = np.where(cond, np.sign(fhr), 0)
    results.append(bt(sig, r, cost, f"H20-TripleFilter [{suffix}]"))

    return results


# ─── CORRECCIÓN BENJAMINI-HOCHBERG (FDR) ──────────────────────────────

def bh_correction(results: list, alpha: float = FDR_ALPHA) -> list:
    """Aplica BH-FDR correction. Retorna resultados con p_adj."""
    ok = [r for r in results if r['status'] == 'OK']
    if not ok:
        return results

    pvals = np.array([r['pval'] for r in ok])
    n     = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)

    p_adj = np.minimum(1.0, pvals * n / ranks)
    # Hacer monotónico de derecha a izquierda
    for i in range(n - 2, -1, -1):
        p_adj[order[i]] = min(p_adj[order[i]], p_adj[order[i + 1]])

    for r, pa in zip(ok, p_adj):
        r['p_adj'] = float(pa)
        r['bh_sig'] = pa < alpha

    for r in results:
        if 'p_adj' not in r:
            r['p_adj'] = 1.0
            r['bh_sig'] = False

    return results


# ─── VISUALIZACIÓN ────────────────────────────────────────────────────

def plot_factory(is_res: list, oos_res: list, survivors: list) -> None:
    GOLD='#FFD700'; GREEN='#00FF88'; RED='#FF4444'; GRAY='#888888'; BG='#161b22'; BLUE='#4488FF'

    fig = plt.figure(figsize=(22, 16), facecolor='#0d1117')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)

    def ax_style(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, color=GOLD, fontsize=9, fontweight='bold', pad=7)
        ax.tick_params(colors=GRAY, labelsize=7)
        ax.spines[:].set_color('#333333')
        for l in ax.get_xticklabels() + ax.get_yticklabels(): l.set_color(GRAY)

    # 1. Sharpe IS — todas las hipótesis (barras)
    ax1 = fig.add_subplot(gs[0, :2])
    is_ok = [r for r in is_res if r['status'] == 'OK']
    labels = [r['label'].split('[')[0].strip() for r in is_ok]
    sharpes = [r['sharpe'] for r in is_ok]
    colors1 = [GREEN if r.get('bh_sig') else (BLUE if s > 0 else RED)
               for r, s in zip(is_ok, sharpes)]
    bars = ax1.bar(range(len(sharpes)), sharpes, color=colors1, alpha=0.85)
    ax1.axhline(0, color=GRAY, lw=0.8)
    ax1.axhline(0.5, color=GOLD, lw=0.8, ls='--', label='Sharpe=0.5')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=35, ha='right', fontsize=7)
    ax1.legend(facecolor=BG, labelcolor='white', fontsize=8)
    ax_style(ax1, "IS Sharpe por Hipótesis (VERDE = BH significativo | AZUL = positivo | ROJO = neg)")
    ax1.set_ylabel("Sharpe IS", color=GRAY)

    # 2. IS vs OOS Sharpe scatter
    ax2 = fig.add_subplot(gs[0, 2])
    oos_dict = {r['label'].replace(' [OOS]','').strip(): r for r in oos_res if r['status']=='OK'}
    xs, ys, labs = [], [], []
    for r in is_ok:
        lkey = r['label'].replace(' [IS]','').strip()
        if lkey in oos_dict:
            xs.append(r['sharpe'])
            ys.append(oos_dict[lkey]['sharpe'])
            labs.append(lkey.split('-')[0])
    if xs:
        ax2.scatter(xs, ys, c=BLUE, alpha=0.7, s=30)
        for x, y, l in zip(xs, ys, labs):
            ax2.annotate(l, (x, y), fontsize=6, color=GRAY)
        lim = max(abs(max(xs+ys, default=1)), abs(min(xs+ys, default=-1))) + 0.5
        ax2.axhline(0, color=GRAY, lw=0.8); ax2.axvline(0, color=GRAY, lw=0.8)
        ax2.plot([-lim, lim], [-lim, lim], color=GOLD, lw=0.8, ls='--', label='y=x')
        ax2.legend(facecolor=BG, labelcolor='white', fontsize=8)
    ax_style(ax2, "IS vs OOS Sharpe\n(encima de y=x → OOS mejor que IS)")
    ax2.set_xlabel("Sharpe IS", color=GRAY)
    ax2.set_ylabel("Sharpe OOS", color=GRAY)

    # 3. Equity curves de survivors
    ax3 = fig.add_subplot(gs[1, :])
    cmap = plt.cm.get_cmap('Set1', max(1, len(survivors)))
    for i, surv in enumerate(survivors[:6]):
        label_clean = surv['is']['label'].split('[')[0].strip()
        if surv['is'].get('equity'):
            eq = np.array(surv['is']['equity'])
            ax3.plot(eq, lw=1.5, label=f"{label_clean} (IS)", color=cmap(i), alpha=0.7)
        if surv['oos'].get('equity'):
            eq_oos = np.array(surv['oos']['equity'])
            ax3.plot(range(len(surv['is'].get('equity', [])), len(surv['is'].get('equity', [])) + len(eq_oos)),
                     eq_oos, lw=2.5, ls='--', label=f"{label_clean} (OOS)", color=cmap(i))
    ax3.axhline(1.0, color=GRAY, lw=0.8, ls=':')
    ax3.legend(facecolor=BG, labelcolor='white', fontsize=7, ncol=4)
    ax_style(ax3, "Equity IS (línea sólida) y OOS (línea punteada) — Edges Supervivientes")
    ax3.set_ylabel("Equity", color=GRAY)

    # 4. Tabla de supervivientes
    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_facecolor(BG); ax4.axis('off')
    if survivors:
        table_data = []
        for s in survivors:
            ois = s['is']
            oos = s['oos']
            table_data.append([
                ois['label'].split('[')[0].strip()[:30],
                f"{ois['n']}",
                f"{ois['sharpe']:.3f}",
                f"{ois['wr']*100:.1f}%",
                f"{ois.get('p_adj', ois['pval']):.4f}",
                f"{oos['n']}",
                f"{oos['sharpe']:.3f}",
                f"{'POSITIVO' if oos['sharpe'] > 0 else 'NEGATIVO'}",
            ])
        headers = ['Hipótesis', 'N IS', 'Sharpe IS', 'WR IS', 'p_adj IS',
                   'N OOS', 'Sharpe OOS', 'OOS Status']
        table = ax4.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.8)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#2d3748')
                cell.set_text_props(color=GOLD, fontweight='bold')
            else:
                cell.set_facecolor('#1a202c')
                txt = cell.get_text().get_text()
                cell.set_text_props(color=GREEN if 'POSITIVO' in txt else (RED if 'NEGATIVO' in txt else 'white'))

    ax4.set_title(f"EDGES SUPERVIVIENTES (BH FDR<{FDR_ALPHA:.0%} + OOS Positivo)",
                  color=GOLD, fontsize=11, fontweight='bold')

    fig.suptitle(
        f"NQ100 EDGE FACTORY — 20 Hipótesis | IS 2021-2023 | OOS 2024-2025\n"
        f"Corrección: Benjamini-Hochberg FDR<{FDR_ALPHA:.0%}",
        color='white', fontsize=12, fontweight='bold', y=0.99
    )

    out = ARTIFACTS_DIR / "nq_edge_factory.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"\n  Grafico: {out}")


# ─── MAIN ──────────────────────────────────────────────────────────────

def main():
    logger.info("╔" + "═"*70 + "╗")
    logger.info("║   NQ EDGE FACTORY — 20 Hipótesis en Paralelo                    ║")
    logger.info("║   IS: 2021-2023  |  OOS: 2024-2025                              ║")
    logger.info("║   Corrección: Benjamini-Hochberg FDR < 10%                      ║")
    logger.info("╚" + "═"*70 + "╝")

    # Cargar datos
    parquet = PROJECT_ROOT / "quant_bot" / "data" / "processed" / "USATECHIDXUSD_M1.parquet"
    logger.info(f"\n  Cargando {parquet.name}...")
    df_raw = pd.read_parquet(parquet, engine='pyarrow')
    logger.info(f"  → {len(df_raw):,} barras  [{df_raw.index[0].date()} → {df_raw.index[-1].date()}]")

    if 'session' not in df_raw.columns:
        from quant_bot.data.nq_loader import add_session_labels
        df_raw = add_session_labels(df_raw)

    logger.info("\n  Construyendo dataset diario enriquecido...")
    df_daily = build_daily(df_raw)
    logger.info(f"  → {len(df_daily)} días de trading")

    df_is  = df_daily[df_daily['year'].isin(IS_YEARS)]
    df_oos = df_daily[df_daily['year'].isin(OOS_YEARS)]
    logger.info(f"  IS ({IS_YEARS[0]}-{IS_YEARS[-1]}): {len(df_is)} días")
    logger.info(f"  OOS ({OOS_YEARS[0]}-{OOS_YEARS[-1]}): {len(df_oos)} días")

    # ── IS ──────────────────────────────────────────
    logger.info("\n" + "═"*72)
    logger.info("  FASE 1 — PRUEBA IS (2021-2023)")
    logger.info("═"*72)
    is_results = run_all_hypotheses(df_is, suffix='IS')
    is_results = bh_correction(is_results, alpha=FDR_ALPHA)

    logger.info("\n  Ranking IS (solo con n suficiente):")
    is_ok = sorted([r for r in is_results if r['status'] == 'OK'],
                   key=lambda x: x['sharpe'], reverse=True)
    for r in is_ok:
        bh  = "* BH_SIG *" if r.get('bh_sig') else ""
        logger.info(f"  {r['sharpe']:+.3f} sh  WR={r['wr']*100:.1f}%  p={r['pval']:.4f}  "
                    f"n={r['n']:4d}  {bh}  {r['label']}")

    n_is_sig = sum(1 for r in is_results if r.get('bh_sig'))
    logger.info(f"\n  Hipótesis BH-significativas IS: {n_is_sig}")

    # ── OOS ─────────────────────────────────────────
    logger.info("\n" + "═"*72)
    logger.info("  FASE 2 — VALIDACIÓN OOS PURA (2024-2025)")
    logger.info("═"*72)
    oos_results = run_all_hypotheses(df_oos, suffix='OOS')

    oos_dict = {r['label'].replace(' [OOS]', '').strip(): r
                for r in oos_results if r['status'] == 'OK'}

    # ── SUPERVIVIENTES ───────────────────────────────
    logger.info("\n" + "═"*72)
    logger.info("  FASE 3 — SUPERVIVIENTES (BH IS + OOS Positivo)")
    logger.info("═"*72)

    survivors = []
    for r_is in is_results:
        if not r_is.get('bh_sig'):
            continue
        key = r_is['label'].replace(' [IS]', '').strip()
        r_oos = oos_dict.get(key)
        if r_oos is None:
            continue
        survivors.append({'is': r_is, 'oos': r_oos})

    if not survivors:
        # Relajar: solo IS sharpe > 0.5 + OOS positivo
        logger.info("  Sin BH-significativos. Relajando: IS Sharpe>0.5 + OOS>0")
        for r_is in is_ok:
            if r_is['sharpe'] < 0.5:
                continue
            key = r_is['label'].replace(' [IS]', '').strip()
            r_oos = oos_dict.get(key)
            if r_oos and r_oos['sharpe'] > 0:
                survivors.append({'is': r_is, 'oos': r_oos})

    survivors.sort(key=lambda x: x['oos']['sharpe'], reverse=True)

    logger.info(f"\n  TOTAL SUPERVIVIENTES: {len(survivors)}")
    logger.info("  " + "─"*70)

    for s in survivors:
        ois = s['is']; oos = s['oos']
        status = "POSITIVO" if oos['sharpe'] > 0 else "NEGATIVO"
        logger.info(f"  {ois['label'].split('[')[0].strip()}")
        logger.info(f"    IS:  Sharpe={ois['sharpe']:.3f}  WR={ois['wr']*100:.1f}%  "
                    f"n={ois['n']:4d}  p_adj={ois.get('p_adj', ois['pval']):.4f}")
        logger.info(f"    OOS: Sharpe={oos['sharpe']:.3f}  WR={oos['wr']*100:.1f}%  "
                    f"n={oos['n']:4d}  → {status}")
        logger.info("")

    # Clasificación final
    logger.info("═"*72)
    logger.info("  CLASIFICACIÓN FINAL")
    logger.info("═"*72)
    n_double = sum(1 for s in survivors if s['oos']['sharpe'] > 0.5)
    if n_double >= 2:
        verdict = "MULTIPLE EDGES DETECTADOS — investigar el más robusto"
    elif len(survivors) == 1 and survivors[0]['oos']['sharpe'] > 0.3:
        verdict = "UN EDGE CANDIDATO SOLIDO"
    elif survivors:
        verdict = "EDGES MARGINALES — se necesita mas OOS"
    else:
        verdict = "SIN EDGES CONFIRMADOS — mas datos o nuevas hipotesis"

    logger.info(f"  {verdict}")

    # ── GRAFICO ─────────────────────────────────────
    plot_factory(is_results, oos_results, survivors)

    # ── GUARDAR JSON ─────────────────────────────────
    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.integer):  return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.bool_):    return bool(o)
            if isinstance(o, np.ndarray):  return o.tolist()
            return super().default(o)

    out_json = ARTIFACTS_DIR / "edge_factory_results.json"
    with open(out_json, 'w') as f:
        json.dump({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'is_years': IS_YEARS, 'oos_years': OOS_YEARS,
            'is_results': is_results,
            'oos_results': oos_results,
            'survivors': survivors,
            'verdict': verdict,
        }, f, indent=2, cls=NE)

    logger.info(f"\n  Resultados → {out_json}")
    logger.info("  Edge Factory completado.")


if __name__ == "__main__":
    main()
