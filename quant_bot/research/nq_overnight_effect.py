"""
nq_overnight_effect.py — H2: Overnight Drift Effect

HIPÓTESIS:
  El Nasdaq 100 tiene un drift positivo estadísticamente significativo
  durante las horas fuera de mercado (21:00 → 13:30 UTC). Este drift
  puede usarse como filtro de dirección para estrategias intradía.

PROTOCOLO CIENTÍFICO (Fase 6):
  1. Calcular retorno overnight puro (sin costos)
  2. Aplicar costos realistas overnight (spread extendido = 10 pts)
  3. Separar por régimen de mercado (bull/bear)
  4. Walk-forward de 6 meses
  5. Análisis de dependencia de eventos macro
  6. Generar regla de filtro para uso en otras estrategias
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

from quant_bot.data.nq_loader import load_nq_m1, add_session_labels, get_daily_summary

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NQ_Overnight")

ARTIFACTS_DIR = PROJECT_ROOT / "quant_bot" / "research" / "artifacts" / "nq"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# NOTA IMPORTANTE: Escala de precios Dukascopy
# El USATECHIDXUSD en Dukascopy está normalizado por ~82:
#   Precio Dukascopy ~ 128-262  ←→  NQ real ~ 10,500 - 21,500
#   SCALE_FACTOR ≈ 82 (NQ real / Dukascopy)
#
# spread_avg real en datos ≈ 0.026 Dukascopy units ≈ 2.1 puntos NQ
# Costo overnight extendido real: ~10 pts NQ = 10/82 ≈ 0.122 Dukascopy
# ─────────────────────────────────────────────────────────────────────

# Factor de escala Dukascopy → NQ real (calibrado empiricamente)
DUKASCOPY_SCALE = 82.0

# Costos en UNIDADES DUKASCOPY (no en puntos NQ reales)
OVERNIGHT_SPREAD_DUCK  = 10.0 / DUKASCOPY_SCALE   # ~10 pts NQ extendido = 0.122
OVERNIGHT_SLIPPAGE_DUCK = 2.0 / DUKASCOPY_SCALE   # ~2 pts NQ slippage  = 0.024
OVERNIGHT_TOTAL_COST    = (OVERNIGHT_SPREAD_DUCK + OVERNIGHT_SLIPPAGE_DUCK) * 2  # entrada+salida

logger.info(f"  Costos overnight en unidades Dukascopy:")
logger.info(f"    Spread: {OVERNIGHT_SPREAD_DUCK:.4f} | Slippage: {OVERNIGHT_SLIPPAGE_DUCK:.4f}")
logger.info(f"    Total round-trip: {OVERNIGHT_TOTAL_COST:.4f}")


# ═══════════════════════════════════════════════════════════
# 1. ANÁLISIS OVERNIGHT DRIFT PURO
# ═══════════════════════════════════════════════════════════

def compute_overnight_trades(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula los retornos del overnight drift con y sin costos.

    Una operación overnight = comprar al close NY (20:00 UTC), vender al open NY (13:30 UTC)
    del día siguiente.

    Como trabajamos con datos diarios de la sesión NY:
      - Entry = close del día anterior
      - Exit  = open del día actual
      - Return = (open_t - close_{t-1}) / close_{t-1}
    """
    daily = df_daily.copy()

    # Retorno bruto overnight
    daily['on_return_gross'] = daily['overnight_return']  # ya calculado en get_daily_summary

    # Retorno neto tras costos (en % del capital, aproximado)
    # Cost en puntos → convertir a % del precio
    avg_price = daily['close'].mean()
    cost_pct = OVERNIGHT_TOTAL_COST / avg_price

    daily['on_return_net'] = daily['on_return_gross'] - cost_pct

    # Dirección del drift (para generar el filtro)
    daily['on_direction'] = np.where(daily['on_return_gross'] > 0, 'UP', 'DOWN')

    # Retorno del día siguiente tras el overnight (¿predice continuación o reversión?)
    daily['next_day_return'] = daily['day_return'].shift(-1)

    return daily


# ═══════════════════════════════════════════════════════════
# 2. BACKTEST DEL DRIFT PURO
# ═══════════════════════════════════════════════════════════

def backtest_overnight_drift(daily_trades: pd.DataFrame, label: str = "Full") -> dict:
    """
    Simula el equity del overnight drift (comprar siempre al cierre NY).
    """
    returns_gross = daily_trades['on_return_gross'].dropna()
    returns_net   = daily_trades['on_return_net'].dropna()

    if len(returns_gross) < 20:
        logger.warning(f"  [{label}] Insuficientes datos.")
        return {}

    # Equity acumulada
    equity_gross = (1 + returns_gross).cumprod()
    equity_net   = (1 + returns_net).cumprod()

    # Métricas
    n = len(returns_net)
    mean_ret = returns_net.mean()
    std_ret  = returns_net.std()
    sharpe   = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0

    roll_max = equity_net.cummax()
    dd = (equity_net - roll_max) / roll_max
    max_dd = dd.min()

    pct_positive = (returns_net > 0).mean()

    t_stat, p_value = stats.ttest_1samp(returns_net, popmean=0)

    # Retorno anualizado estimado
    total_return = equity_net.iloc[-1] - 1
    n_years = n / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    logger.info(f"\n[{label}] ── OVERNIGHT DRIFT BACKTEST ──")
    logger.info(f"  N días: {n}")
    logger.info(f"  Retorno anualizado (neto): {annual_return*100:.2f}%")
    logger.info(f"  Sharpe (neto): {sharpe:.3f}")
    logger.info(f"  Max Drawdown: {max_dd*100:.2f}%")
    logger.info(f"  % días positivos (neto): {pct_positive*100:.1f}%")
    logger.info(f"  T-test vs 0: t={t_stat:.2f}, p={p_value:.4f}")

    if p_value < 0.05 and mean_ret > 0:
        logger.info(f"  ✅ DRIFT POSITIVO Y SIGNIFICATIVO")
    elif p_value < 0.05 and mean_ret < 0:
        logger.info(f"  ⚠️  Drift NEGATIVO significativo (podría explorarse short)")
    else:
        logger.info(f"  ❌ Drift NO significativo (ruido estadístico)")

    return {
        'label': label,
        'n_days': n,
        'mean_return_net': float(mean_ret),
        'annual_return_net': float(annual_return),
        'sharpe_net': float(sharpe),
        'max_dd': float(max_dd),
        'pct_positive': float(pct_positive),
        'pvalue': float(p_value),
        'equity_gross': equity_gross,
        'equity_net': equity_net,
        'daily_trades': daily_trades,
    }


# ═══════════════════════════════════════════════════════════
# 3. ANÁLISIS POR RÉGIMEN
# ═══════════════════════════════════════════════════════════

def analyze_regime_performance(daily_trades: pd.DataFrame) -> dict:
    """
    Separa los resultados por régimen de mercado:
    - Bull:   retorno de 60 días positivo
    - Bear:   retorno de 60 días negativo
    - 2022:   bear market severo (análisis específico)
    """
    logger.info("\n── ANÁLISIS POR RÉGIMEN ──")

    dt = daily_trades.copy()

    # Definir régimen usando MA de 60 días del retorno del cierre
    dt['regime'] = 'BULL'
    close_60d = dt['close'].pct_change(60)
    dt.loc[close_60d < 0, 'regime'] = 'BEAR'

    results = {}
    for regime in ['BULL', 'BEAR']:
        subset = dt[dt['regime'] == regime]
        if len(subset) < 10:
            continue
        net = subset['on_return_net']
        results[regime] = {
            'n': len(subset),
            'mean': float(net.mean()),
            'pct_positive': float((net > 0).mean()),
        }
        t, p = stats.ttest_1samp(net.dropna(), 0)
        results[regime]['pvalue'] = float(p)
        logger.info(f"  {regime}: n={len(subset)}, mean={net.mean()*100:.4f}%, pos={net.gt(0).mean()*100:.1f}%, p={p:.4f}")

    # 2022 específico
    bear_2022 = dt[dt.index.year == 2022]
    if len(bear_2022) > 0:
        net_2022 = bear_2022['on_return_net']
        total_2022 = net_2022.sum()
        logger.info(f"\n  2022 (BEAR MARKET):")
        logger.info(f"    Retorno total overnight 2022: {total_2022*100:.2f}%")
        logger.info(f"    % días positivos: {(net_2022 > 0).mean()*100:.1f}%")
        results['2022'] = {
            'n': len(net_2022),
            'total_return': float(total_2022),
            'pct_positive': float((net_2022 > 0).mean()),
        }

    return results


# ═══════════════════════════════════════════════════════════
# 4. WALK-FORWARD ANALYSIS
# ═══════════════════════════════════════════════════════════

def walk_forward_overnight(daily_trades: pd.DataFrame,
                            train_months: int = 12,
                            test_months: int = 3) -> pd.DataFrame:
    """
    Walk-forward: ventanas deslizantes para evaluar estabilidad temporal.
    """
    logger.info(f"\n── WALK-FORWARD (train={train_months}m, test={test_months}m) ──")

    dt = daily_trades.dropna(subset=['on_return_net']).copy()
    results = []

    start = dt.index[0]
    end   = dt.index[-1]

    current = start + pd.DateOffset(months=train_months)

    while current + pd.DateOffset(months=test_months) <= end:
        test_end = current + pd.DateOffset(months=test_months)
        test = dt.loc[current:test_end]

        if len(test) < 10:
            current += pd.DateOffset(months=test_months)
            continue

        test_net = test['on_return_net']
        results.append({
            'period_start': current,
            'period_end': test_end,
            'n': len(test_net),
            'mean_return': float(test_net.mean()),
            'pct_positive': float((test_net > 0).mean()),
            'total_return': float(test_net.sum()),
        })

        current += pd.DateOffset(months=test_months)

    wf_df = pd.DataFrame(results)

    if len(wf_df) > 0:
        pct_positive_windows = (wf_df['total_return'] > 0).mean()
        logger.info(f"  Ventanas analizadas: {len(wf_df)}")
        logger.info(f"  % ventanas positivas: {pct_positive_windows*100:.1f}%")
        logger.info(f"  Retorno medio por ventana: {wf_df['mean_return'].mean()*100:.4f}%")

    return wf_df


# ═══════════════════════════════════════════════════════════
# 5. FILTRO OVERNIGHT PARA ESTRATEGIAS INTRADÍA
# ═══════════════════════════════════════════════════════════

def generate_overnight_filter(daily_trades: pd.DataFrame) -> pd.Series:
    """
    Genera la señal filtro para estrategias intradía:
    - 1: Overnight fue positivo → operar solo LONG intradía
    - -1: Overnight fue negativo → operar solo SHORT intradía
    - 0: Overnight neutral → no operar

    IMPORTANTE: Esta es una señal ex-post. Para uso real, necesita
    estar disponible al inicio de la sesión NY (13:30 UTC).

    Retorna: Pd.Series con index de fechas y valores {-1, 0, 1}
    """
    NEUTRAL_THRESHOLD = 0.05  # overnight < ±0.05% = neutral

    filter_signal = pd.Series(0, index=daily_trades.index)
    filter_signal[daily_trades['on_return_gross'] > NEUTRAL_THRESHOLD / 100]  = 1
    filter_signal[daily_trades['on_return_gross'] < -NEUTRAL_THRESHOLD / 100] = -1

    logger.info(f"\n── FILTRO OVERNIGHT ──")
    logger.info(f"  Señales LONG: {(filter_signal == 1).sum()} días")
    logger.info(f"  Señales SHORT: {(filter_signal == -1).sum()} días")
    logger.info(f"  Señales NEUTRAL: {(filter_signal == 0).sum()} días")

    return filter_signal


# ═══════════════════════════════════════════════════════════
# 6. VISUALIZACIÓN
# ═══════════════════════════════════════════════════════════

def plot_overnight_results(
    bt_full: dict,
    bt_is: dict,
    bt_oos: dict,
    regime_results: dict,
    wf_df: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(20, 20), facecolor='#0d1117')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    tc = '#c9d1d9'; gc = '#21262d'; c1 = '#58a6ff'; c2 = '#3fb950'; c3 = '#ff7b72'

    fig.suptitle('H2: Overnight Drift Effect — Análisis Completo',
                 fontsize=16, color=tc, fontweight='bold', y=0.98)

    # Plot 1: Equity gross vs net ──
    ax1 = fig.add_subplot(gs[0, :])
    if 'equity_gross' in bt_full:
        ax1.plot(bt_full['equity_gross'].index, bt_full['equity_gross'].values,
                 color=c1, alpha=0.6, linewidth=1, label='Bruto (sin costos)')
        ax1.plot(bt_full['equity_net'].index, bt_full['equity_net'].values,
                 color=c2, linewidth=1.5, label='Neto (con spread/slippage)')
    ax1.axhline(y=1, color=tc, linewidth=0.5, alpha=0.5)
    ax1.set_facecolor('#161b22')
    ax1.set_title('Equity: Overnight Drift (Comprar cierre NY, vender apertura NY)', color=tc)
    ax1.set_ylabel('Equity acumulada', color=tc)
    ax1.legend(facecolor='#21262d', labelcolor=tc)
    ax1.tick_params(colors=tc)
    ax1.grid(True, color=gc, alpha=0.5)
    for s in ax1.spines.values(): s.set_color(gc)

    # Plot 2: IS vs OOS equity ──
    ax2 = fig.add_subplot(gs[1, 0])
    if 'equity_net' in bt_is:
        ax2.plot(bt_is['equity_net'].index, bt_is['equity_net'].values,
                 color=c1, linewidth=1.5, label=f"IS (Sharpe={bt_is.get('sharpe_net', 0):.2f})")
    if 'equity_net' in bt_oos:
        ax2.plot(bt_oos['equity_net'].index, bt_oos['equity_net'].values,
                 color=c2, linewidth=2, label=f"OOS (Sharpe={bt_oos.get('sharpe_net', 0):.2f})",
                 linestyle='--')
    ax2.axhline(y=1, color=tc, linewidth=0.5, alpha=0.5)
    ax2.set_facecolor('#161b22')
    ax2.set_title('Equity IS vs OOS', color=tc, fontsize=11)
    ax2.legend(facecolor='#21262d', labelcolor=tc)
    ax2.tick_params(colors=tc)
    ax2.grid(True, color=gc, alpha=0.5)
    for s in ax2.spines.values(): s.set_color(gc)

    # Plot 3: Retorno por régimen ──
    ax3 = fig.add_subplot(gs[1, 1])
    if regime_results:
        regimes = [k for k in regime_results if k != '2022']
        means = [regime_results[k]['mean'] * 100 for k in regimes]
        colors_r = [c2 if m > 0 else c3 for m in means]
        ax3.bar(regimes, means, color=colors_r, alpha=0.8)
    ax3.axhline(y=0, color=tc, linewidth=0.5)
    ax3.set_facecolor('#161b22')
    ax3.set_title('Retorno Overnight por Régimen (%/día)', color=tc, fontsize=11)
    ax3.tick_params(colors=tc)
    ax3.grid(True, color=gc, alpha=0.5, axis='y')
    for s in ax3.spines.values(): s.set_color(gc)

    # Plot 4: Walk-forward results ──
    ax4 = fig.add_subplot(gs[2, :])
    if len(wf_df) > 0:
        colors_wf = [c2 if t > 0 else c3 for t in wf_df['total_return']]
        ax4.bar(range(len(wf_df)), wf_df['total_return'] * 100,
                color=colors_wf, alpha=0.8)
        pct_pos = (wf_df['total_return'] > 0).mean()
        ax4.axhline(y=0, color=tc, linewidth=0.8)
        ax4.set_title(f'Walk-Forward (ventanas 3m): {pct_pos*100:.0f}% positivas', color=tc, fontsize=11)
        ax4.set_xlabel('Ventana', color=tc)
        ax4.set_ylabel('Retorno período (%)', color=tc)
    ax4.set_facecolor('#161b22')
    ax4.tick_params(colors=tc)
    ax4.grid(True, color=gc, alpha=0.5, axis='y')
    for s in ax4.spines.values(): s.set_color(gc)

    out_path = ARTIFACTS_DIR / "nq_overnight_effect.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)
    logger.info(f"\n  ✅ Gráfico guardado: {out_path}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("  H2: OVERNIGHT DRIFT EFFECT — VALIDACIÓN COMPLETA")
    logger.info("=" * 70)
    logger.info(f"  Costo overnight asumido: {OVERNIGHT_TOTAL_COST:.1f} pts (spread+slippage)")

    # ── Cargar datos ──
    logger.info("\n📂 Cargando datos M1...")
    df = load_nq_m1(use_cache=True)
    df = add_session_labels(df)
    df_daily = get_daily_summary(df)

    logger.info(f"  Días de mercado: {len(df_daily)}")
    logger.info(f"  Período: {df_daily.index[0].date()} → {df_daily.index[-1].date()}")

    # ── Preparar trades ──
    daily_trades = compute_overnight_trades(df_daily)
    daily_trades = daily_trades.dropna(subset=['on_return_net'])

    # ── Separar IS/OOS ──
    OOS_YEAR = 2025
    dt_is  = daily_trades[daily_trades.index.year < OOS_YEAR]
    dt_oos = daily_trades[daily_trades.index.year >= OOS_YEAR]

    logger.info(f"\n  IS:  {len(dt_is)} días | OOS: {len(dt_oos)} días")

    # ── Backtest completo ──
    logger.info("\n📈 Backtesting overnight drift...")
    bt_full = backtest_overnight_drift(daily_trades, "FULL")
    bt_is   = backtest_overnight_drift(dt_is, "IS")
    bt_oos  = backtest_overnight_drift(dt_oos, "OOS")

    # ── Análisis por régimen ──
    regime_results = analyze_regime_performance(daily_trades)

    # ── Walk-forward ──
    wf_df = walk_forward_overnight(daily_trades)

    # ── Filtro para otras estrategias ──
    overnight_filter = generate_overnight_filter(daily_trades)
    filter_path = ARTIFACTS_DIR / "overnight_filter.parquet"
    overnight_filter.to_frame(name='direction').to_parquet(filter_path)
    logger.info(f"\n  Filtro overnight guardado: {filter_path}")

    # ── Visualización ──
    logger.info("\n📊 Generando gráficos...")
    plot_overnight_results(bt_full, bt_is, bt_oos, regime_results, wf_df)

    # ── VEREDICTO FINAL ──
    logger.info("\n" + "=" * 70)
    logger.info("  VEREDICTO FINAL — OVERNIGHT DRIFT")
    logger.info("=" * 70)

    def check(cond, text):
        logger.info(f"  {'✅' if cond else '❌'} {text}")
        return cond

    if not bt_oos:
        logger.warning("  Insuficientes datos OOS para veredicto.")
        return bt_full

    checks_passed = sum([
        check(bt_oos.get('pvalue', 1) < 0.05 and bt_oos.get('mean_return_net', 0) > 0,
              f"Drift OOS positivo y significativo: p={bt_oos.get('pvalue', 1):.4f}"),
        check(bt_oos.get('sharpe_net', 0) > 0.5,
              f"Sharpe OOS neto > 0.5: {bt_oos.get('sharpe_net', 0):.3f}"),
        check(bt_oos.get('pct_positive', 0) > 0.52,
              f"% días positivos OOS > 52%: {bt_oos.get('pct_positive', 0)*100:.1f}%"),
        check(regime_results.get('BEAR', {}).get('mean', -1) > 0,
              "Drift positivo en régimen BEAR"),
        check(len(wf_df) > 0 and (wf_df['total_return'] > 0).mean() > 0.65,
              f"Walk-forward 65%+ positivo: {(wf_df['total_return'] > 0).mean()*100:.1f}%"),
    ])

    logger.info(f"\n  SCORE: {checks_passed}/5 checks passed")

    classifications = {
        5: "🏆 OVERNIGHT DRIFT ROBUSTO — Usar como filtro de dirección",
        4: "✅ DRIFT DÉBIL — Usar como señal auxiliar solo",
        3: "⚠️  DRIFT MARGINAL — No confiable",
        2: "❌ DRIFT FRÁGIL — Probable sesgo de datos",
        1: "❌ SIN DRIFT — No usar",
        0: "❌ SIN DRIFT — No usar",
    }
    logger.info(f"\n  CLASIFICACIÓN: {classifications.get(checks_passed, '❌ SIN DRIFT')}")

    return bt_full


if __name__ == "__main__":
    main()
