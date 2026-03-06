"""
nq_session_analysis.py — Análisis base de microestructura por sesiones del Nasdaq 100.

OBJETIVO:
  1. Confirmar patrón U-Shape de volatilidad (picos apertura/cierre, valle mediodía)
  2. Calcular estadísticas base por sesión y día de semana
  3. Detectar y cuantificar el Overnight Drift
  4. Calcular correlación primera hora vs cierre del día
  5. Análisis gap fill statistics

FILOSOFÍA (Fase 6):
  - Este script solo DESCRIBE, no optimiza.
  - Los resultados son input para los scripts de hipótesis específicas.
  - Si los resultados son aburridos / insignificantes → NO HAY EDGE BASE.
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

# ── Setup paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant_bot.data.nq_loader import (
    load_nq_m1,
    add_session_labels,
    get_daily_summary,
    save_nq_parquet,
    NQ_PROCESSED,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NQ_Session")

ARTIFACTS_DIR = PROJECT_ROOT / "quant_bot" / "research" / "artifacts" / "nq"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# 1. U-SHAPE VOLATILITY — Perfil horario de volatilidad
# ═══════════════════════════════════════════════════════════

def analyze_u_shape_volatility(df: pd.DataFrame) -> dict:
    """
    Analiza el perfil de volatilidad por hora UTC.
    Calcula: ATR promedio por hora, retorno absoluto promedio por hora.
    """
    logger.info("\n── ANÁLISIS U-SHAPE VOLATILITY ──")

    df = df.copy()
    df['bar_range_pct'] = (df['high'] - df['low']) / df['close'] * 100
    df['abs_return_pct'] = df['close'].pct_change().abs() * 100

    # Solo sesión NY principal (13:00 - 21:00 UTC)
    ny_hours = df[(df['hour_utc'] >= 13) & (df['hour_utc'] <= 20)].copy()

    vol_by_hour = ny_hours.groupby('hour_utc').agg({
        'bar_range_pct': ['mean', 'median', 'std'],
        'abs_return_pct': ['mean', 'median'],
        'volume': 'mean',
    }).round(4)

    logger.info(f"\nVolatilidad por hora UTC (NY Session):")
    logger.info(vol_by_hour.to_string())

    # Test estadístico: ¿La volatilidad de la apertura (13-14) es
    # significativamente mayor que el mediodía (15-17)?
    open_vol = ny_hours[ny_hours['hour_utc'].isin([13, 14])]['bar_range_pct'].dropna()
    midday_vol = ny_hours[ny_hours['hour_utc'].isin([15, 16, 17])]['bar_range_pct'].dropna()

    t_stat, p_value = stats.ttest_ind(open_vol, midday_vol)
    logger.info(f"\nTest apertura vs mediodía (volatilidad):")
    logger.info(f"  t-stat: {t_stat:.2f}  | p-value: {p_value:.6f}")
    logger.info(f"  {'✅ DIFERENCIA SIGNIFICATIVA' if p_value < 0.01 else '❌ No significativo'}")

    return {
        'vol_by_hour': vol_by_hour,
        'open_vs_midday_pvalue': p_value,
        'open_mean_range': float(open_vol.mean()),
        'midday_mean_range': float(midday_vol.mean()),
    }


# ═══════════════════════════════════════════════════════════
# 2. OVERNIGHT DRIFT
# ═══════════════════════════════════════════════════════════

def analyze_overnight_drift(df_daily: pd.DataFrame) -> dict:
    """
    Análisis del overnight drift:
    - Retorno medio de comprar al cierre NY y vender a la apertura siguiente
    - Separación por año para detectar régimen-dependencia
    """
    logger.info("\n── ANÁLISIS OVERNIGHT DRIFT ──")

    on_returns = df_daily['overnight_return'].dropna() * 100  # en %

    logger.info(f"  N días: {len(on_returns)}")
    logger.info(f"  Retorno overnight medio: {on_returns.mean():.4f}%")
    logger.info(f"  Retorno overnight mediana: {on_returns.median():.4f}%")
    logger.info(f"  Std dev: {on_returns.std():.4f}%")
    logger.info(f"  % días positivos: {(on_returns > 0).mean() * 100:.1f}%")

    # T-test: ¿El drift es significativamente > 0?
    t_stat, p_value = stats.ttest_1samp(on_returns.dropna(), popmean=0)
    logger.info(f"\n  T-test vs media=0:")
    logger.info(f"  t-stat: {t_stat:.2f}  | p-value: {p_value:.4f}")
    logger.info(f"  {'✅ DRIFT SIGNIFICATIVO' if p_value < 0.05 else '❌ Drift NO significativo (puede ser ruido)'}")

    # Por año
    yearly = on_returns.groupby(on_returns.index.year).agg(['mean', 'std', 'count', lambda x: (x > 0).mean()])
    yearly.columns = ['mean_return', 'std', 'count', 'pct_positive']
    logger.info(f"\n  Drift por año:")
    logger.info(yearly.round(4).to_string())

    # Equity acumulada del drift
    equity = (1 + on_returns / 100).cumprod()

    return {
        'mean_return': float(on_returns.mean()),
        'pct_positive': float((on_returns > 0).mean()),
        'pvalue_vs_zero': float(p_value),
        'is_significant': p_value < 0.05,
        'yearly': yearly,
        'equity_series': equity,
    }


# ═══════════════════════════════════════════════════════════
# 3. CORRELACIÓN PRIMERA HORA vs CIERRE DÍA
# ═══════════════════════════════════════════════════════════

def analyze_first_hour_correlation(df_daily: pd.DataFrame) -> dict:
    """
    Analiza si el retorno de la primera hora de NY predice el cierre del día.
    """
    logger.info("\n── CORRELACIÓN PRIMERA HORA vs CIERRE DÍA ──")

    valid = df_daily[['first_hour_return', 'day_return']].dropna() * 100

    pearson_r, p_pearson = stats.pearsonr(valid['first_hour_return'], valid['day_return'])
    spearman_r, p_spearman = stats.spearmanr(valid['first_hour_return'], valid['day_return'])

    logger.info(f"  N días: {len(valid)}")
    logger.info(f"  Pearson r: {pearson_r:.4f}  (p={p_pearson:.4f})")
    logger.info(f"  Spearman r: {spearman_r:.4f}  (p={p_spearman:.4f})")

    # Test hipótesis: si primera hora > +0.5%, ¿cierra positivo?
    threshold = 0.5
    strong_open = valid[valid['first_hour_return'] > threshold]
    if len(strong_open) > 0:
        pct_pos_after_strong_open = (strong_open['day_return'] > 0).mean()
        logger.info(f"\n  Días con primera hora > +{threshold}%: {len(strong_open)}")
        logger.info(f"  % días que cierran positivo: {pct_pos_after_strong_open * 100:.1f}%")

        # Z-score vs azar (H0: 50%)
        z_score = (pct_pos_after_strong_open - 0.5) / np.sqrt(0.5 * 0.5 / len(strong_open))
        logger.info(f"  Z-Score vs azar (50%): {z_score:.2f}")
        logger.info(f"  {'✅ EDGE ESTADÍSTICO' if abs(z_score) > 2 else '❌ Dentro del ruido estadístico'}")
    else:
        pct_pos_after_strong_open = None
        z_score = None

    # Por direcciones
    pos_open = valid[valid['first_hour_return'] > 0]
    neg_open = valid[valid['first_hour_return'] < 0]

    logger.info(f"\n  Cuando primera hora es POSITIVA ({len(pos_open)} días):")
    logger.info(f"    Cierra positivo: {(pos_open['day_return'] > 0).mean()*100:.1f}%")
    logger.info(f"  Cuando primera hora es NEGATIVA ({len(neg_open)} días):")
    logger.info(f"    Cierra positivo: {(neg_open['day_return'] > 0).mean()*100:.1f}%")

    return {
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r),
        'p_pearson': float(p_pearson),
        'pct_positive_after_strong_open': float(pct_pos_after_strong_open) if pct_pos_after_strong_open else None,
        'z_score_vs_random': float(z_score) if z_score else None,
    }


# ═══════════════════════════════════════════════════════════
# 4. ANÁLISIS DE GAPS Y GAP FILL
# ═══════════════════════════════════════════════════════════

def analyze_gap_fill(df_daily: pd.DataFrame, df_m1: pd.DataFrame) -> dict:
    """
    Analiza probabilidad de gap fill:
    - Si hay un gap de apertura, ¿cuántas veces se llena durante la sesión?
    - ¿Influye la dirección del gap?
    """
    logger.info("\n── ANÁLISIS GAP FILL ──")

    # Gap = overnight_return en términos de precio
    daily = df_daily.copy()
    daily['gap_pct'] = daily['overnight_return'] * 100
    daily['gap_direction'] = np.where(daily['gap_pct'] > 0, 'UP', 'DOWN')
    daily['gap_filled'] = False

    # Verificar fill: ¿el precio del día vuelve al cierre del día anterior?
    prev_close = daily['close'].shift(1)
    daily['day_low'] = daily['low']
    daily['day_high'] = daily['high']

    # Gap UP filled si low del día ≤ close previo
    gap_up = daily['gap_direction'] == 'UP'
    gap_down = daily['gap_direction'] == 'DOWN'

    daily.loc[gap_up, 'gap_filled'] = daily.loc[gap_up, 'day_low'] <= prev_close[gap_up]
    daily.loc[gap_down, 'gap_filled'] = daily.loc[gap_down, 'day_high'] >= prev_close[gap_down]

    # Solo gaps significativos (> 0.1%)
    sig_gaps = daily[daily['gap_pct'].abs() > 0.1]

    if len(sig_gaps) > 0:
        fill_rate = sig_gaps['gap_filled'].mean()
        fill_up = sig_gaps[sig_gaps['gap_direction'] == 'UP']['gap_filled'].mean()
        fill_down = sig_gaps[sig_gaps['gap_direction'] == 'DOWN']['gap_filled'].mean()

        logger.info(f"  N días con gap > 0.1%: {len(sig_gaps)}")
        logger.info(f"  Fill rate total:   {fill_rate * 100:.1f}%")
        logger.info(f"  Fill rate UP gaps: {fill_up * 100:.1f}%")
        logger.info(f"  Fill rate DOWN gaps: {fill_down * 100:.1f}%")

        # Test chi-cuadrado: ¿fill_rate difiere de 50%?
        _, p_chi = stats.chisquare(
            [sig_gaps['gap_filled'].sum(), len(sig_gaps) - sig_gaps['gap_filled'].sum()],
            f_exp=[len(sig_gaps)/2, len(sig_gaps)/2]
        )
        logger.info(f"  Chi-cuadrado vs 50%: p={p_chi:.4f}")
        logger.info(f"  {'✅ DISTRIBUCIÓN SESGADA (edge potencial)' if p_chi < 0.05 else '❌ No difiere del 50%'}")

        return {
            'n_gaps': len(sig_gaps),
            'fill_rate': float(fill_rate),
            'fill_rate_up': float(fill_up),
            'fill_rate_down': float(fill_down),
            'pvalue_vs_random': float(p_chi),
        }

    return {'n_gaps': 0, 'fill_rate': None}


# ═══════════════════════════════════════════════════════════
# 5. ESTADÍSTICAS POR DÍA DE SEMANA
# ═══════════════════════════════════════════════════════════

def analyze_day_of_week(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Analiza si hay estacionalidad por día de semana.
    """
    logger.info("\n── ESTADÍSTICAS POR DÍA DE SEMANA ──")

    daily = df_daily.copy()
    daily['dow'] = daily.index.dayofweek
    day_names = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes'}
    daily['dow_name'] = daily['dow'].map(day_names)

    stats_dow = daily.groupby('dow_name').agg(
        n=('day_return', 'count'),
        mean_return_pct=('day_return', lambda x: x.mean() * 100),
        pct_positive=('day_return', lambda x: (x > 0).mean() * 100),
        std=('day_return', lambda x: x.std() * 100),
    ).round(3)

    logger.info(f"\n{stats_dow.to_string()}")

    return stats_dow


# ═══════════════════════════════════════════════════════════
# 6. GENERACIÓN DE REPORTE Y GRÁFICOS
# ═══════════════════════════════════════════════════════════

def generate_session_report(results: dict, df_daily: pd.DataFrame) -> None:
    """
    Genera el reporte visual y el archivo Markdown con todos los hallazgos.
    """
    fig = plt.figure(figsize=(20, 24), facecolor='#0d1117')
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    text_color = '#c9d1d9'
    grid_color = '#21262d'
    accent1 = '#58a6ff'
    accent2 = '#3fb950'
    accent3 = '#ff7b72'

    fig.suptitle('USATECHIDXUSD (Nasdaq 100) — Análisis Base de Microestructura',
                 fontsize=18, color=text_color, fontweight='bold', y=0.98)

    # ── Plot 1: Volatilidad por hora ──
    ax1 = fig.add_subplot(gs[0, :])
    vol_data = results.get('u_shape', {}).get('vol_by_hour')
    if vol_data is not None:
        hours = vol_data.index
        ranges = vol_data[('bar_range_pct', 'mean')]
        bars = ax1.bar(hours, ranges, color=accent1, alpha=0.8, edgecolor='none')
        ax1.set_facecolor('#161b22')
        ax1.set_title('U-Shape Volatility — Rango promedio por hora UTC (NY Session)',
                      color=text_color, fontsize=12)
        ax1.set_xlabel('Hora UTC', color=text_color)
        ax1.set_ylabel('Rango promedio (%)', color=text_color)
        ax1.tick_params(colors=text_color)
        ax1.grid(True, color=grid_color, alpha=0.5)
        ax1.spines['bottom'].set_color(grid_color)
        ax1.spines['left'].set_color(grid_color)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

    # ── Plot 2: Overnight Drift equity ──
    ax2 = fig.add_subplot(gs[1, 0])
    on_equity = results.get('overnight', {}).get('equity_series')
    if on_equity is not None:
        ax2.plot(on_equity.index, on_equity.values, color=accent2, linewidth=1.5)
        ax2.axhline(y=1, color=grid_color, linestyle='--', linewidth=0.8)
        ax2.set_facecolor('#161b22')
        ax2.set_title('Equity — Overnight Drift\n(Comprar cierre, vender apertura)', color=text_color, fontsize=11)
        ax2.set_ylabel('Equity acumulada', color=text_color)
        ax2.tick_params(colors=text_color)
        ax2.grid(True, color=grid_color, alpha=0.5)
        for spine in ax2.spines.values():
            spine.set_color(grid_color)

    # ── Plot 3: Distribución primera hora vs retorno día ──
    ax3 = fig.add_subplot(gs[1, 1])
    if 'first_hour_corr' in results:
        # Scatter primer hora vs retorno día (sample 500 pts para no saturar)
        on_equity_data = df_daily[['first_hour_return', 'day_return']].dropna() * 100
        sample = on_equity_data.sample(min(500, len(on_equity_data)), random_state=42)
        ax3.scatter(sample['first_hour_return'], sample['day_return'],
                    alpha=0.3, color=accent1, s=8, edgecolors='none')
        ax3.axhline(y=0, color=grid_color, linewidth=0.8)
        ax3.axvline(x=0, color=grid_color, linewidth=0.8)
        ax3.set_facecolor('#161b22')
        r = results['first_hour_corr']['pearson_r']
        ax3.set_title(f'Primera Hora vs Retorno Día\n(Pearson r = {r:.3f})', color=text_color, fontsize=11)
        ax3.set_xlabel('Retorno 1ª Hora (%)', color=text_color)
        ax3.set_ylabel('Retorno Día (%)', color=text_color)
        ax3.tick_params(colors=text_color)
        ax3.grid(True, color=grid_color, alpha=0.5)
        for spine in ax3.spines.values():
            spine.set_color(grid_color)

    # ── Plot 4: Performance por día de semana ──
    ax4 = fig.add_subplot(gs[2, 0])
    dow_stats = results.get('day_of_week')
    if dow_stats is not None:
        colors = [accent2 if x >= 0 else accent3 for x in dow_stats['mean_return_pct']]
        bars = ax4.bar(dow_stats.index, dow_stats['mean_return_pct'], color=colors, alpha=0.8)
        ax4.axhline(y=0, color=text_color, linewidth=0.5)
        ax4.set_facecolor('#161b22')
        ax4.set_title('Retorno promedio por día de semana (%)', color=text_color, fontsize=11)
        ax4.set_ylabel('Retorno medio (%)', color=text_color)
        ax4.tick_params(colors=text_color)
        ax4.grid(True, color=grid_color, alpha=0.5, axis='y')
        for spine in ax4.spines.values():
            spine.set_color(grid_color)

    # ── Plot 5: Distribución overnight returns ──
    ax5 = fig.add_subplot(gs[2, 1])
    on_ret = results.get('overnight', {})
    if 'equity_series' in on_ret:
        on_series = df_daily['overnight_return'].dropna() * 100
        ax5.hist(on_series, bins=80, color=accent1, alpha=0.7, edgecolor='none', density=True)
        ax5.axvline(x=0, color=accent3, linewidth=1.5, linestyle='--', label='Zero')
        ax5.axvline(x=on_series.mean(), color=accent2, linewidth=1.5,
                    linestyle='-', label=f'Media={on_series.mean():.3f}%')
        ax5.set_facecolor('#161b22')
        ax5.set_title('Distribución Retornos Overnight', color=text_color, fontsize=11)
        ax5.set_xlabel('Retorno overnight (%)', color=text_color)
        ax5.legend(facecolor='#21262d', labelcolor=text_color)
        ax5.tick_params(colors=text_color)
        for spine in ax5.spines.values():
            spine.set_color(grid_color)

    # ── Plot 6: Retorno anual overnight ──
    ax6 = fig.add_subplot(gs[3, :])
    yearly_on = results.get('overnight', {}).get('yearly')
    if yearly_on is not None:
        cols_pos = [accent2 if x >= 0 else accent3 for x in yearly_on['mean_return']]
        ax6.bar(yearly_on.index.astype(str), yearly_on['mean_return'] * 100,
                color=cols_pos, alpha=0.8)
        ax6.axhline(y=0, color=text_color, linewidth=0.5)
        ax6.set_facecolor('#161b22')
        ax6.set_title('Retorno Overnight Medio por Año (%)', color=text_color, fontsize=11)
        ax6.set_ylabel('Retorno medio (%)', color=text_color)
        ax6.tick_params(colors=text_color)
        ax6.grid(True, color=grid_color, alpha=0.5, axis='y')
        for spine in ax6.spines.values():
            spine.set_color(grid_color)

    plot_path = ARTIFACTS_DIR / "nq_session_analysis.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)
    logger.info(f"\n  ✅ Gráfico guardado: {plot_path}")


def generate_markdown_report(results: dict, df_daily: pd.DataFrame) -> None:
    """Genera el reporte Markdown con todos los hallazgos."""
    report_path = ARTIFACTS_DIR / "nasdaq_anomaly_report.md"

    overnight = results.get('overnight', {})
    fhc = results.get('first_hour_corr', {})
    gap_fill = results.get('gap_fill', {})
    ushape = results.get('u_shape', {})

    # Determinar veredictos
    drift_verdict = "✅ DRIFT POSITIVO SIGNIFICATIVO" if overnight.get('is_significant') and overnight.get('mean_return', 0) > 0 else "❌ Drift no significativo o negativo"
    corr_verdict = "✅ CORRELACIÓN DETECTADA" if abs(fhc.get('pearson_r', 0)) > 0.3 and fhc.get('p_pearson', 1) < 0.05 else "⚠️ Correlación débil"
    gap_verdict = "✅ GAP FILL SESGADO" if gap_fill.get('pvalue_vs_random', 1) < 0.05 else "❌ Gap fill = azar"
    ushape_verdict = "✅ U-SHAPE CONFIRMADA" if ushape.get('open_vs_midday_pvalue', 1) < 0.01 else "❌ U-Shape no confirmada"

    report = f"""# NASDAQ 100 (USATECHIDXUSD) — Análisis de Microestructura Base
**Generado**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} UTC

---

## RESUMEN EJECUTIVO

| Hipótesis | Veredicto | Métrica Clave |
|-----------|-----------|---------------|
| U-Shape Volatility | {ushape_verdict} | p={ushape.get('open_vs_midday_pvalue', 'N/A'):.4f} |
| Overnight Drift | {drift_verdict} | μ={overnight.get('mean_return', 0):.4f}% / día |
| Correlación 1ª Hora | {corr_verdict} | r={fhc.get('pearson_r', 0):.4f} |
| Gap Fill Statistics | {gap_verdict} | Fill rate={gap_fill.get('fill_rate', 0)*100 if gap_fill.get('fill_rate') else 'N/A':.1f}% |

---

## 1. U-SHAPE VOLATILITY

**Veredicto**: {ushape_verdict}

- Rango medio apertura (13-14h UTC): **{ushape.get('open_mean_range', 0):.4f}%**
- Rango medio mediodía (15-17h UTC): **{ushape.get('midday_mean_range', 0):.4f}%**
- T-test p-value: **{ushape.get('open_vs_midday_pvalue', 'N/A'):.6f}**

**Implicación táctica**: La volatilidad alta en apertura sugiere potencial para estrategias de captación de movimiento. El valle del mediodía sugiere estrategias de reversión.

---

## 2. OVERNIGHT DRIFT

**Veredicto**: {drift_verdict}

- Retorno overnight medio diario: **{overnight.get('mean_return', 0):.4f}%**
- % días positivos: **{overnight.get('pct_positive', 0)*100:.1f}%**
- P-value vs media=0: **{overnight.get('pvalue_vs_zero', 'N/A'):.4f}**
- ¿Significativo estadísticamente? **{'SÍ' if overnight.get('is_significant') else 'NO'}**

**Performance por año**:
```
{results.get('overnight', {}).get('yearly', pd.DataFrame()).round(4).to_string() if results.get('overnight', {}).get('yearly') is not None else 'N/A'}
```

**PREGUNTAS DE DESTRUCCIÓN (Fase 6)**:
- ¿El drift es capturable con spread real overnight? → PENDIENTE validar
- ¿Desaparece en 2022 (bear market)? → Ver tabla por año
- ¿Depende de eventos macro específicos? → PENDIENTE
- ¿Con spread extendido (10+ puntos), sigue positivo? → PENDIENTE

---

## 3. CORRELACIÓN PRIMERA HORA vs CIERRE DÍA

**Veredicto**: {corr_verdict}

- Pearson r: **{fhc.get('pearson_r', 0):.4f}** (p={fhc.get('p_pearson', 1):.4f})
- Spearman r: **{fhc.get('spearman_r', 0):.4f}**
- Días con primera hora > +0.5%: **{fhc.get('pct_positive_after_strong_open', 'N/A')}** cierran positivo
- Z-Score vs azar: **{fhc.get('z_score_vs_random', 'N/A')}**

---

## 4. GAP FILL STATISTICS

**Veredicto**: {gap_verdict}

- N gaps analizados: **{gap_fill.get('n_gaps', 0)}**
- Fill rate total: **{gap_fill.get('fill_rate', 0)*100 if gap_fill.get('fill_rate') else 'N/A':.1f}%**
- Fill rate UP gaps: **{gap_fill.get('fill_rate_up', 0)*100 if gap_fill.get('fill_rate_up') else 'N/A':.1f}%**
- Fill rate DOWN gaps: **{gap_fill.get('fill_rate_down', 0)*100 if gap_fill.get('fill_rate_down') else 'N/A':.1f}%**
- P-value vs 50%: **{gap_fill.get('pvalue_vs_random', 'N/A'):.4f}**

---

## PRÓXIMOS PASOS

Basado en los resultados de este análisis base:

1. **Si Overnight Drift es significativo** → Ejecutar `nq_overnight_effect.py` (backtest completo)
2. **Si U-Shape está confirmada** → Ejecutar `nq_whipsaw_reversal.py` (H1)
3. **Si correlación primera hora es > 0.3** → Usar como filtro de dirección en ORB
4. **Si gap fill > 60%** → Desarrollar estrategia de gap fill específica

---

*Análisis generado automáticamente por NQ Session Analysis v1.0*
*Regla Fase 6: Este análisis DESCRIBE. La validación es el paso siguiente.*
"""

    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"  ✅ Reporte guardado: {report_path}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("  NQ SESSION ANALYSIS — MICROESTRUCTURA BASE")
    logger.info("=" * 60)

    # ── Cargar datos ──
    logger.info("\n📂 Cargando datos M1...")
    try:
        df = load_nq_m1(use_cache=True)
    except FileNotFoundError:
        logger.info("  Cache no encontrado. Generando desde BI5...")
        df = load_nq_m1(use_cache=False)
        save_nq_parquet(df)

    # ── Añadir etiquetas de sesión ──
    logger.info("\n  Añadiendo etiquetas de sesión...")
    df = add_session_labels(df)

    # ── Resumen diario ──
    logger.info("\n  Calculando resumen diario...")
    df_daily = get_daily_summary(df)
    logger.info(f"  Días de mercado: {len(df_daily)}")
    logger.info(f"  Período: {df_daily.index[0].date()} → {df_daily.index[-1].date()}")

    # ── Análisis ──
    results = {}

    results['u_shape'] = analyze_u_shape_volatility(df)
    results['overnight'] = analyze_overnight_drift(df_daily)
    results['first_hour_corr'] = analyze_first_hour_correlation(df_daily)
    results['gap_fill'] = analyze_gap_fill(df_daily, df)
    results['day_of_week'] = analyze_day_of_week(df_daily)

    # ── Generar artefactos ──
    logger.info("\n📊 Generando artefactos visuales...")
    generate_session_report(results, df_daily)

    logger.info("\n📝 Generando reporte Markdown...")
    generate_markdown_report(results, df_daily)

    # ── Resumen final ──
    logger.info("\n" + "=" * 60)
    logger.info("  RESUMEN FINAL — SEÑALES BASE DETECTADAS")
    logger.info("=" * 60)

    overnight = results['overnight']
    corr = results['first_hour_corr']
    gap = results['gap_fill']
    ushape = results['u_shape']

    signals_found = sum([
        overnight.get('is_significant', False) and overnight.get('mean_return', 0) > 0,
        abs(corr.get('pearson_r', 0)) > 0.3 and corr.get('p_pearson', 1) < 0.05,
        gap.get('pvalue_vs_random', 1) < 0.05,
        ushape.get('open_vs_midday_pvalue', 1) < 0.01,
    ])

    logger.info(f"\n  Anomalías detectadas: {signals_found} / 4")
    if signals_found >= 2:
        logger.info("  ✅ Suficiente estructura para hipótesis específicas")
        logger.info("  → Proceder con nq_whipsaw_reversal.py y nq_overnight_effect.py")
    else:
        logger.info("  ⚠️  Pocas anomalías. Validar si el dataset tiene calidad suficiente.")

    logger.info(f"\n  Artefactos en: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
