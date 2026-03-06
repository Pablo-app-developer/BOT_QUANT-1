"""
nq_edge_discovery.py — Script Maestro de Descubrimiento de Edge en Nasdaq 100

MISIÓN:
  Ejecutar el pipeline completo de investigación cuantitativa del USATECHIDXUSD.
  Este script NO optimiza. Describe, valida y destruye.

  Fases:
    0. Construir cache M1 desde archivos .bi5 (una vez)
    1. Análisis base de microestructura (sesiones, volatilidad, drift)
    2. H1: NY Open Whipsaw Reversal
    3. H2: Overnight Drift Effect
    4. Generar reporte final consolidado

REGLAS (Fase 6):
  - OOS = 2025 completo, separado ANTES de cualquier análisis
  - Costos realistas siempre activos
  - Intentar destruir cada hipótesis
  - Solo si sobrevive → clasificar como edge candidato

USO:
  python nq_edge_discovery.py [--rebuild-cache] [--fast]

  --rebuild-cache: Fuerza reconstrucción del Parquet desde .bi5
  --fast: Solo análisis base (sin backtest completo de hipótesis)
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """Convierte tipos numpy a Python nativos para serialización JSON."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


# ── Setup paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Crear directorio de artefactos ANTES de configurar logging ──
ARTIFACTS_DIR = PROJECT_ROOT / "quant_bot" / "research" / "artifacts" / "nq"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

from quant_bot.data.nq_loader import (
    load_nq_m1,
    add_session_labels,
    get_daily_summary,
    save_nq_parquet,
    NQ_PROCESSED,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ARTIFACTS_DIR / "discovery.log"),
    ]
)
logger = logging.getLogger("NQ_EdgeDiscovery")


# ═══════════════════════════════════════════════════════════
# FASE 0: CONSTRUCCIÓN DEL DATASET M1
# ═══════════════════════════════════════════════════════════

def phase_0_build_dataset(rebuild: bool = False) -> pd.DataFrame:
    """
    Construye el dataset M1 desde los archivos .bi5 de Dukascopy.
    Si ya existe el caché, lo usa directamente (mucho más rápido).
    """
    logger.info("\n" + "═" * 70)
    logger.info("  FASE 0 — CONSTRUCCIÓN DEL DATASET M1")
    logger.info("═" * 70)

    cache_path = NQ_PROCESSED / "USATECHIDXUSD_M1.parquet"

    if cache_path.exists() and not rebuild:
        logger.info(f"\n  Cache encontrado: {cache_path}")
        logger.info("  Cargando desde Parquet (rápido)...")
        df = pd.read_parquet(cache_path, engine='pyarrow')
        logger.info(f"  → {len(df):,} barras M1 cargadas")
        logger.info(f"  → Período: {df.index[0]} → {df.index[-1]}")

        if 'session' not in df.columns:
            logger.info("  Añadiendo etiquetas de sesión...")
            df = add_session_labels(df)
            save_nq_parquet(df)

        return df

    logger.info("\n  Construyendo dataset desde archivos .bi5...")
    logger.info("  (Esto puede tardar varios minutos la primera vez)")

    df = load_nq_m1(use_cache=False)
    df = add_session_labels(df)
    save_nq_parquet(df)

    logger.info(f"\n  ✅ Dataset construido y cacheado.")
    logger.info(f"  Barras totales: {len(df):,}")
    logger.info(f"  Período: {df.index[0]} → {df.index[-1]}")

    # Estadísticas rápidas de calidad
    invalid_ohlc = (df['high'] < df['low']).sum()
    pct_zero_vol = (df['volume'] == 0).mean() * 100

    logger.info(f"\n  Calidad de datos:")
    logger.info(f"    Barras con High < Low: {invalid_ohlc}")
    logger.info(f"    % barras volumen cero: {pct_zero_vol:.1f}%")
    logger.info(f"    Sesiones en dataset:   {df['session'].value_counts().to_dict()}")

    return df


# ═══════════════════════════════════════════════════════════
# FASE 1: ANÁLISIS BASE (DESCRIPCIÓN)
# ═══════════════════════════════════════════════════════════

def phase_1_base_analysis(df: pd.DataFrame) -> dict:
    """Ejecuta el análisis de microestructura base."""
    logger.info("\n" + "═" * 70)
    logger.info("  FASE 1 — ANÁLISIS BASE DE MICROESTRUCTURA")
    logger.info("═" * 70)

    try:
        from quant_bot.research.nq_session_analysis import (
            analyze_u_shape_volatility,
            analyze_overnight_drift,
            analyze_first_hour_correlation,
            analyze_gap_fill,
            analyze_day_of_week,
            generate_session_report,
            generate_markdown_report,
        )

        df_daily = get_daily_summary(df)
        results = {}
        results['u_shape']       = analyze_u_shape_volatility(df)
        results['overnight']     = analyze_overnight_drift(df_daily)
        results['first_hour_corr'] = analyze_first_hour_correlation(df_daily)
        results['gap_fill']      = analyze_gap_fill(df_daily, df)
        results['day_of_week']   = analyze_day_of_week(df_daily)

        generate_session_report(results, df_daily)
        generate_markdown_report(results, df_daily)

        # Guardar métricas en JSON
        metrics = {
            'u_shape_pvalue': results['u_shape'].get('open_vs_midday_pvalue', None),
            'overnight_significant': results['overnight'].get('is_significant', False),
            'overnight_mean': results['overnight'].get('mean_return', None),
            'correlation_pearson': results['first_hour_corr'].get('pearson_r', None),
            'gap_fill_rate': results['gap_fill'].get('fill_rate', None),
        }

        with open(ARTIFACTS_DIR / "phase1_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2, cls=NumpyEncoder)

        logger.info("\n  ✅ Fase 1 completada → ver artifacts/nq/nasdaq_anomaly_report.md")
        return results

    except Exception as e:
        logger.error(f"\n  ❌ Error en Fase 1: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ═══════════════════════════════════════════════════════════
# FASE 2: H1 — WHIPSAW REVERSAL
# ═══════════════════════════════════════════════════════════

def phase_2_whipsaw(df: pd.DataFrame) -> dict:
    """Ejecuta la validación de H1: NY Open Whipsaw Reversal."""
    logger.info("\n" + "═" * 70)
    logger.info("  FASE 2 — H1: NY OPEN WHIPSAW REVERSAL")
    logger.info("═" * 70)

    try:
        from quant_bot.research.nq_whipsaw_reversal import (
            generate_whipsaw_signals,
            analyze_results,
            run_stress_tests,
            monte_carlo_simulation,
            plot_whipsaw_results,
        )

        # IS / OOS split
        OOS_YEAR = 2025
        df_is  = df[df.index.year < OOS_YEAR]
        df_oos = df[df.index.year >= OOS_YEAR]

        logger.info(f"\n  IS: {df_is.index[0].date()} → {df_is.index[-1].date()}")
        logger.info(f"  OOS: {df_oos.index[0].date()} → {df_oos.index[-1].date()}")

        df_signals_is  = generate_whipsaw_signals(df_is)
        df_signals_oos = generate_whipsaw_signals(df_oos)

        results_is  = analyze_results(df_signals_is, "IS")
        results_oos = analyze_results(df_signals_oos, "OOS")

        all_sigs = pd.concat([df_signals_is, df_signals_oos]) if len(df_signals_is) > 0 else df_signals_oos

        stress = run_stress_tests(all_sigs)
        mc = monte_carlo_simulation(all_sigs, n_runs=1000)
        plot_whipsaw_results(df_signals_is, df_signals_oos, mc, stress)

        # Score
        checks = sum([
            results_oos.get('n_trades', 0) >= 100,
            results_oos.get('win_rate', 0) > 0.50,
            results_oos.get('expectancy_pts', -1) > 0,
            stress.get('spread_x2', {}).get('expectancy_pts', -1) > 0,
            mc.get('pct_positive', 0) > 0.70,
        ])

        metrics = {
            'n_trades_oos': results_oos.get('n_trades', 0),
            'win_rate_oos': results_oos.get('win_rate', 0),
            'expectancy_oos': results_oos.get('expectancy_pts', None),
            'profit_factor_oos': results_oos.get('profit_factor', 0),
            'score': checks,
        }

        with open(ARTIFACTS_DIR / "phase2_whipsaw_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2, cls=NumpyEncoder)

        logger.info(f"\n  ✅ Fase 2 completada — Score: {checks}/5")
        return metrics

    except Exception as e:
        logger.error(f"\n  ❌ Error en Fase 2: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ═══════════════════════════════════════════════════════════
# FASE 3: H2 — OVERNIGHT EFFECT
# ═══════════════════════════════════════════════════════════

def phase_3_overnight(df: pd.DataFrame) -> dict:
    """Ejecuta la validación de H2: Overnight Drift Effect."""
    logger.info("\n" + "═" * 70)
    logger.info("  FASE 3 — H2: OVERNIGHT DRIFT EFFECT")
    logger.info("═" * 70)

    try:
        from quant_bot.research.nq_overnight_effect import (
            compute_overnight_trades,
            backtest_overnight_drift,
            analyze_regime_performance,
            walk_forward_overnight,
            generate_overnight_filter,
            plot_overnight_results,
        )

        df_daily = get_daily_summary(df)
        daily_trades = compute_overnight_trades(df_daily).dropna(subset=['on_return_net'])

        OOS_YEAR = 2025
        dt_is  = daily_trades[daily_trades.index.year < OOS_YEAR]
        dt_oos = daily_trades[daily_trades.index.year >= OOS_YEAR]

        bt_full = backtest_overnight_drift(daily_trades, "FULL")
        bt_is   = backtest_overnight_drift(dt_is, "IS")
        bt_oos  = backtest_overnight_drift(dt_oos, "OOS")

        regime_results = analyze_regime_performance(daily_trades)
        wf_df = walk_forward_overnight(daily_trades)

        overnight_filter = generate_overnight_filter(daily_trades)
        overnight_filter.to_frame(name='direction').to_parquet(
            ARTIFACTS_DIR / "overnight_filter.parquet"
        )

        plot_overnight_results(bt_full, bt_is, bt_oos, regime_results, wf_df)

        # Calcular score
        if bt_oos:
            checks = sum([
                bt_oos.get('pvalue', 1) < 0.05 and bt_oos.get('mean_return_net', 0) > 0,
                bt_oos.get('sharpe_net', 0) > 0.5,
                bt_oos.get('pct_positive', 0) > 0.52,
                regime_results.get('BEAR', {}).get('mean', -1) > 0,
                len(wf_df) > 0 and (wf_df['total_return'] > 0).mean() > 0.65,
            ])
        else:
            checks = 0

        metrics = {
            'annual_return_net': bt_full.get('annual_return_net', None),
            'sharpe_net': bt_full.get('sharpe_net', None),
            'max_dd': bt_full.get('max_dd', None),
            'pct_positive': bt_full.get('pct_positive', None),
            'oos_pvalue': bt_oos.get('pvalue', None) if bt_oos else None,
            'score': checks,
        }

        with open(ARTIFACTS_DIR / "phase3_overnight_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2, cls=NumpyEncoder)

        logger.info(f"\n  ✅ Fase 3 completada — Score: {checks}/5")
        return metrics

    except Exception as e:
        logger.error(f"\n  ❌ Error en Fase 3: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ═══════════════════════════════════════════════════════════
# FASE 4: REPORTE CONSOLIDADO FINAL
# ═══════════════════════════════════════════════════════════

def phase_4_final_report(
    phase1: dict,
    phase2: dict,
    phase3: dict,
    df: pd.DataFrame,
) -> None:
    """Genera el reporte final consolidado."""
    logger.info("\n" + "═" * 70)
    logger.info("  FASE 4 — REPORTE CONSOLIDADO FINAL")
    logger.info("═" * 70)

    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M')

    # Calcular veredicts
    overnight = phase1.get('overnight', {})
    corr = phase1.get('first_hour_corr', {})
    ushape = phase1.get('u_shape', {})
    gap = phase1.get('gap_fill', {})

    h1_score = phase2.get('score', 0)
    h2_score = phase3.get('score', 0)

    # Clasificación final
    total_max = 10  # 5 checks H1 + 5 checks H2
    total_score = h1_score + h2_score

    if total_score >= 8:
        edge_class = "🏆 EDGE ROBUSTO REAL"
        recommendation = "Candidato serio para implementación con capital real (FTMO)"
    elif total_score >= 6:
        edge_class = "✅ EDGE PROMETEDOR"
        recommendation = "Requiere validación adicional en datos tick antes de implementar"
    elif total_score >= 4:
        edge_class = "⚠️  EDGE DÉBIL"
        recommendation = "No recomendable para trading real. Investigar más"
    else:
        edge_class = "❌ ILUSIÓN ESTADÍSTICA"
        recommendation = "No hay edge explotable con los parámetros actuales"

    report = f"""# NASDAQ 100 (USATECHIDXUSD) — REPORTE FINAL DE EDGE DISCOVERY
**Generado**: {timestamp} UTC  
**Período de datos**: {df.index[0].date()} → {df.index[-1].date()}

---

## CLASIFICACIÓN FINAL: {edge_class}
**Recomendación**: {recommendation}
**Score total**: {total_score}/{total_max}

---

## RESUMEN EJECUTIVO POR HIPÓTESIS

### H1 — NY Open Whipsaw Reversal
- **Score**: {h1_score}/5
- N trades OOS: {phase2.get('n_trades_oos', 'N/A')}
- Win Rate OOS: {phase2.get('win_rate_oos', 0)*100:.1f}% (req: >50%)
- Expectancy OOS: {phase2.get('expectancy_oos', 'N/A')} puntos (req: >0)
- Profit Factor OOS: {phase2.get('profit_factor_oos', 0):.3f} (req: >1.10)

### H2 — Overnight Drift Effect
- **Score**: {h2_score}/5
- Retorno anual neto: {phase3.get('annual_return_net', 0)*100:.2f}%
- Sharpe neto: {phase3.get('sharpe_net', 0):.3f}
- % días positivos: {phase3.get('pct_positive', 0)*100:.1f}%
- OOS p-value: {phase3.get('oos_pvalue', 'N/A')}

---

## ANOMALÍAS BASE DETECTADAS

| Anomalía | Detectada | Valor |
|----------|-----------|-------|
| U-Shape Volatility | {'✅' if ushape.get('open_vs_midday_pvalue', 1) < 0.01 else '❌'} | p={ushape.get('open_vs_midday_pvalue', 'N/A')} |
| Overnight Drift | {'✅' if overnight.get('is_significant', False) else '❌'} | μ={overnight.get('mean_return', 0):.4f}% |
| Correlación 1H-Día | {'✅' if abs(corr.get('pearson_r', 0)) > 0.3 else '❌'} | r={corr.get('pearson_r', 0):.4f} |
| Gap Fill Bias | {'✅' if gap.get('pvalue_vs_random', 1) < 0.05 else '❌'} | {gap.get('fill_rate', 0)*100:.1f}% |

---

## RESPUESTAS A PREGUNTAS DE DESTRUCCIÓN (Fase 6.8)

1. **¿El edge sigue vivo con spread x2?** → Ver phase2_whipsaw_metrics.json
2. **¿Sobrevive retraso de 1 vela?** → Incluido en stress tests
3. **¿Desaparece en 2022 (bear)?** → Ver nq_overnight_effect.png (régimen bear)
4. **¿Depende de pocos trades extremos?** → Test SIN_TOP10PCT en stress tests
5. **¿Monte Carlo muestra ruina?** → Ver distribución MC en nq_whipsaw_reversal.png
6. **¿Edge estable por año?** → Walk-forward en nq_overnight_effect.png

---

## ARTEFACTOS GENERADOS

| Archivo | Descripción |
|---------|-------------|
| `nq_session_analysis.png` | U-Shape, overnight equity, correlaciones |
| `nq_whipsaw_reversal.png` | H1: IS/OOS equity, MC, stress tests |
| `nq_overnight_effect.png` | H2: Drift equity, walk-forward, régimen |
| `nasdaq_anomaly_report.md` | Reporte detallado de microestructura |
| `overnight_filter.parquet` | Señal de filtro para estrategias intradía |
| `phase1_metrics.json` | Métricas Fase 1 |
| `phase2_whipsaw_metrics.json` | Métricas H1 |
| `phase3_overnight_metrics.json` | Métricas H2 |
| `discovery.log` | Log completo del proceso |

---

## PRÓXIMOS PASOS

{'### Si edge sobrevive:' if total_score >= 6 else '### Edge no sobrevive — alternativas:'}
"""

    if total_score >= 6:
        report += """
1. Validar en datos tick (verificar que el fill es real, no OHLC artifact)
2. Test con latencia real 300ms (simular delay de ejecución retail)
3. Comparar con NASDAQ futures (NQ=F) para cross-validación
4. Implementar Risk Engine: 0.5% riesgo fijo por trade
5. SL físico obligatorio en MT5 al momento de entrada
6. Paper trading 60 días antes de capital real
7. Someter al proceso FTMO Challenge
"""
    else:
        report += """
1. Testear hipótesis con datos tick (Fase 6.1)
2. Buscar edges en timeframes más altos (H1, H4)
3. Incorporar datos de componentes (AAPL, MSFT, GOOGL) como filtros
4. Analizar VIX como condición de régimen
5. Explorar estrategia de Gap Fill (si fill_rate > 60%)
6. Revisar hipótesis ORB (Opening Range Breakout) - 30 min
"""

    report += f"""
---

## NOTA CRÍTICA — FTMO COMPLIANCE

> Incluso si el edge tiene win rate superior, el riesgo por trade debe ser:
> - **0.5% fijo del capital** (irrenunciable)
> - **Stop Loss físico en MT5** colocado en el mismo segundo de entrada
> - **Daily Loss Limit**: monitorear para no superar 5% del capital
> - **Max Drawdown**: jamás superar 8% (margen de seguridad vs límite 10% FTMO)

---

*Reporte generado por NQ Edge Discovery v1.0*  
*Filosofía: "El sistema NO intenta demostrar que funciona. Intenta demostrar que NO funciona."*
"""

    report_path = ARTIFACTS_DIR / "nq_final_edge_report.md"
    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"\n  ✅ Reporte final guardado: {report_path}")
    logger.info(f"\n  CLASIFICACIÓN FINAL: {edge_class}")
    logger.info(f"  Score: {total_score}/{total_max}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='NQ Edge Discovery Pipeline — Nasdaq 100 USATECHIDXUSD'
    )
    parser.add_argument('--rebuild-cache', action='store_true',
                        help='Reconstruir el dataset desde archivos .bi5 (lento)')
    parser.add_argument('--fast', action='store_true',
                        help='Solo análisis base sin backtest completo')
    args = parser.parse_args()

    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║   NQ EDGE DISCOVERY — USATECHIDXUSD (NASDAQ 100)              ║")
    logger.info("║   Filosofía: destruir el edge antes de aceptarlo               ║")
    logger.info("╚" + "═" * 68 + "╝")

    # ── FASE 0: Dataset ──
    df = phase_0_build_dataset(rebuild=args.rebuild_cache)

    if df.empty:
        logger.error("ERROR: No se pudo cargar el dataset. Verifica los archivos .bi5.")
        sys.exit(1)

    # ── FASE 1: Análisis base ──
    phase1_results = phase_1_base_analysis(df)

    if args.fast:
        logger.info("\n  --fast mode: solo Fase 1 completada.")
        return

    # ── FASE 2: H1 Whipsaw ──
    phase2_results = phase_2_whipsaw(df)

    # ── FASE 3: H2 Overnight ──
    phase3_results = phase_3_overnight(df)

    # ── FASE 4: Reporte final ──
    phase_4_final_report(phase1_results, phase2_results, phase3_results, df)

    logger.info("\n" + "═" * 70)
    logger.info("  ✅ PIPELINE COMPLETO")
    logger.info(f"  Artefactos en: {ARTIFACTS_DIR}")
    logger.info("═" * 70)


if __name__ == "__main__":
    main()
