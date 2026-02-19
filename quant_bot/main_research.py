"""
main_research.py — Entry point del pipeline de investigación cuantitativa.

Fase 1: Carga, limpieza y validación del dataset EURUSD M1.

Uso:
    python main_research.py              # Ejecuta Phase 1
    python main_research.py --reload     # Fuerza re-extracción de ZIPs
"""

import sys
import argparse
import logging
from pathlib import Path

import pandas as pd

# ── Setup path ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from data import loader

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s │ %(levelname)-7s │ %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("main")


def phase_1(force_reload: bool = False) -> None:
    """
    FASE 1 — Dataset Profesional.

    Pipeline:
      1. Extraer _all.csv de cada ZIP anual
      2. Cargar y combinar todos los CSVs
      3. Limpiar (dedup, NaN, tipos)
      4. Validar integridad
      5. Guardar como Parquet
      6. Verificar round-trip (Parquet → DataFrame)
    """
    logger.info("=" * 55)
    logger.info("  FASE 1 — DATASET PROFESIONAL")
    logger.info("=" * 55)

    # ── Check si ya existe procesado ──
    parquet_path = settings.PROCESSED_DIR / "EURUSD_M1.parquet"
    if parquet_path.exists() and not force_reload:
        logger.info("Parquet procesado ya existe. Cargando...")
        df = loader.load_processed()
        report = loader.validate(df)
        _print_summary(df, report)
        return df

    # ── 1. Extraer ZIPs ──
    logger.info("\n[1/6] Extrayendo ZIPs...")
    loader.extract_zips(force=force_reload)

    # ── 2. Cargar y combinar ──
    logger.info("\n[2/6] Cargando CSVs...")
    df_raw = loader.load_and_combine_all()

    # ── 3. Limpiar ──
    logger.info("\n[3/6] Limpiando dataset...")
    df = loader.clean(df_raw)

    # ── 4. Validar ──
    logger.info("\n[4/6] Validando integridad...")
    report = loader.validate(df)

    # ── 5. Guardar ──
    logger.info("\n[5/6] Guardando Parquet...")
    loader.save_processed(df)

    # ── 6. Verificar round-trip ──
    logger.info("\n[6/6] Verificando round-trip...")
    df_check = loader.load_processed()
    assert len(df_check) == len(df), "Round-trip FALLÓ: tamaños distintos"
    assert (df_check.index == df.index).all(), "Round-trip FALLÓ: índices distintos"
    assert (df_check.values == df.values).all(), "Round-trip FALLÓ: valores distintos"
    logger.info("  ✅ Round-trip verificado: Parquet es idéntico.")

    # ── Resumen ──
    _print_summary(df, report)


def _print_summary(df, report: dict) -> None:
    """Imprime resumen final del dataset."""
    print("\n" + "=" * 55)
    print("  RESUMEN DEL DATASET")
    print("=" * 55)
    print(f"  Total barras:     {report['total_bars']:>12,}")
    print(f"  Período:          {report['start']}")
    print(f"                 →  {report['end']}")
    print(f"  Días calendario:  {report['calendar_days']:>12,}")
    print(f"  High < Low:       {report['high_lt_low']:>12,}")
    print(f"  Gaps intraday:    {report['gaps_intraday']:>12,}")
    print(f"  Gaps weekend:     {report['gaps_weekend']:>12,}")
    print(f"  Rango precio:     {report['price_min']} → {report['price_max']}")
    print("=" * 55)
    print("\nPrimeras barras:")
    print(df.head(10))
    print("\nÚltimas barras:")
    print(df.tail(5))
    print("\nEstadísticas:")
    print(df.describe())
    print("\n✅ FASE 1 COMPLETA. Dataset listo.\n")


# ═══════════════════════════════════════════════
# FASE 2 — BACKTEST ENGINE (SMOKE TEST)
# ═══════════════════════════════════════════════

def phase_2(data: pd.DataFrame) -> None:
    """
    FASE 2 + 3 — Smoke Test del motor de backtest + métricas.

    Principio: si señales RANDOM producen profit después de costes,
    el motor tiene un bug. Expectancy debe ser ≈ 0 o negativa.
    """
    import numpy as np
    import pandas as pd
    from backtest.engine import run as run_backtest
    from backtest.execution_model import ExecConfig
    from backtest.metrics import full_report, print_report

    logger.info("=" * 55)
    logger.info("  FASE 2+3 — BACKTEST ENGINE SMOKE TEST")
    logger.info("=" * 55)

    # Usar subsample para smoke test (más rápido que 3.7M barras)
    sample = data.iloc[:100_000].copy()
    logger.info(f"  Usando subsample: {len(sample):,} barras")

    # ── Señales RANDOM ──
    rng = np.random.default_rng(42)
    signals = pd.Series(
        rng.choice([-1, 0, 1], size=len(sample), p=[0.1, 0.8, 0.1]),
        index=sample.index,
    )
    logger.info(f"  Señales: {(signals == 1).sum()} long, "
                f"{(signals == -1).sum()} short, "
                f"{(signals == 0).sum()} flat")

    # ── Ejecutar backtest ──
    cfg = ExecConfig(spread_pips=1.0, slippage_pips=0.5)
    result = run_backtest(
        sample, signals,
        cfg=cfg,
        sl_pips=20,
        tp_pips=40,
    )

    # ── Métricas ──
    pnls = np.array([t.pnl_net for t in result.trades])
    report = full_report(result.equity_curve, result.returns, pnls)
    print_report(report)

    # ── SANITY CHECK ──
    print("\n  ── SANITY CHECK ──")
    if report['total_return'] > 0.05:
        logger.warning("  ⚠️  Random strategy profitable >5% → POSIBLE BUG!")
        print("  ⚠️  ALERTA: señales random generan profit.")
        print("  Revisar engine antes de avanzar.")
    elif report['total_return'] < -0.20:
        logger.info("  ✅ Random negativo (costes dominan) — correcto.")
        print("  ✅ Expectancy negativa con random + costes = motor correcto.")
    else:
        logger.info("  ✅ Random ≈ breakeven/ligeramente negativo — correcto.")
        print("  ✅ Random ≈ breakeven — motor funciona correctamente.")

    print(f"\n  Trades generados:  {len(result.trades)}")
    print(f"  Equity final:      ${result.final_equity:,.2f}")
    print(f"  Expectancy:        ${report['expectancy']:.2f}/trade")
    print(f"\n✅ FASE 2+3 COMPLETA. Motor y métricas listos.\n")


# ═══════════════════════════════════════════════
# FASE 4 — EDGE DISCOVERY
# ═══════════════════════════════════════════════

def phase_4(data: pd.DataFrame) -> list[dict]:
    """
    FASE 4 — Framework de investigación de edges.

    Ejecuta 8 tests de hipótesis sobre los datos,
    rankea los resultados por viabilidad,
    e imprime reporte completo.
    """
    from research.hypothesis_tests import run_all
    from research.edge_analysis import rank_edges, print_edge_report

    logger.info("=" * 55)
    logger.info("  FASE 4 — EDGE DISCOVERY")
    logger.info("=" * 55)

    # Ejecutar todos los tests
    results = run_all(data)

    # Rankear
    ranked = rank_edges(results)

    # Reportar
    print_edge_report(ranked)

    return ranked


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant Research — EURUSD M1")
    parser.add_argument('--phase', type=int, default=1,
                        help='Ejecutar hasta esta fase (1=data, 2=backtest, 4=edge)')
    parser.add_argument('--reload', action='store_true',
                        help='Forzar re-extracción de ZIPs y reproceso')
    args = parser.parse_args()

    # Fase 1
    data = phase_1(force_reload=args.reload)

    # Fase 2+3
    if args.phase >= 2:
        phase_2(data)

    # Fase 4
    if args.phase >= 4:
        ranked = phase_4(data)
