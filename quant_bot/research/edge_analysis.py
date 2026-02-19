"""
edge_analysis.py — Ranking y filtrado de edges descubiertos.

Sistema de scoring:
  Stability:    50%  → ¿se mantiene en el tiempo?
  Significance: 30%  → ¿es estadísticamente real?
  Effect Size:  20%  → ¿es grande el efecto?

Principio: Stability > Significance > Effect Size.
Un edge pequeño pero estable vale más que uno grande pero intermitente.

Criterios de viabilidad:
  - p_value < 0.05
  - CI excluye cero
  - stability >= 0.60 (≥60% de ventanas con mismo signo)
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


# ── Pesos del scoring ──
W_STABILITY    = 0.50
W_SIGNIFICANCE = 0.30
W_EFFECT       = 0.20


def score_edge(r: dict) -> float:
    """
    Calcula score compuesto para un resultado de test.

    Componentes (todos normalizados a [0, 1]):
      - stability:    directamente [0, 1]
      - significance: 1 - p_value (clipped to [0, 1])
      - effect_size:  normalizado por tanh para acotar
    """
    # Stability: ya está en [0, 1]
    s_stab = r.get('stability', 0.0)

    # Significance: 1 - p_value
    p = r.get('p_value', 1.0)
    s_sig = max(0, min(1, 1 - p))

    # Effect size: usar |effect_size_bps| normalizado con tanh
    # tanh(x/10) → 1 bps = 0.1, 10 bps = 0.76, 50 bps = 1.0
    e = abs(r.get('effect_size_bps', 0.0))
    s_eff = float(np.tanh(e / 10.0))

    return W_STABILITY * s_stab + W_SIGNIFICANCE * s_sig + W_EFFECT * s_eff


def is_viable(r: dict) -> bool:
    """¿Cumple los criterios mínimos para considerarse edge?"""
    return (
        r.get('p_value', 1.0) < 0.05 and
        r.get('ci_excludes_zero', False) and
        r.get('stability', 0.0) >= 0.60
    )


def rank_edges(results: list[dict]) -> list[dict]:
    """
    Agrega score y ordena todos los resultados.
    Marca cuáles son viables y cuáles no.
    """
    for r in results:
        r['score'] = score_edge(r)
        r['viable'] = is_viable(r)

    ranked = sorted(results, key=lambda x: x['score'], reverse=True)
    return ranked


def print_edge_report(ranked: list[dict]) -> None:
    """Imprime reporte completo de edges descubiertos."""
    viable = [r for r in ranked if r['viable']]
    non_viable = [r for r in ranked if not r['viable']]

    print("\n" + "=" * 70)
    print("    REPORTE DE EDGE DISCOVERY")
    print("=" * 70)

    # ── Edges viables ──
    if viable:
        print(f"\n  ✅ EDGES VIABLES: {len(viable)}")
        print(f"  {'#':<3} {'Score':<7} {'Nombre':<30} {'Effect(bps)':<12} "
              f"{'p-value':<10} {'Stab':<6} {'CI'}")
        print("  " + "-" * 85)
        for i, r in enumerate(viable, 1):
            ci = f"[{r['ci_low']*10000:.2f}, {r['ci_high']*10000:.2f}]"
            print(f"  {i:<3} {r['score']:<7.3f} {r['name']:<30} "
                  f"{r['effect_size_bps']:<12.3f} "
                  f"{r['p_value']:<10.6f} {r['stability']:<6.2f} {ci}")
    else:
        print("\n  ❌ NO SE ENCONTRARON EDGES VIABLES")
        print("  Esto es NORMAL. La mayoría de hipótesis no producen edge.")
        print("  Que no haya edge ≠ fracaso. Es ciencia rigurosa.")

    # ── Top 10 no viables (informativos) ──
    print(f"\n  ── Top 10 no viables (informativos) ──")
    print(f"  {'#':<3} {'Score':<7} {'Nombre':<30} {'Effect(bps)':<12} "
          f"{'p-value':<10} {'Stab':<6} {'¿Por qué falla?'}")
    print("  " + "-" * 90)
    for i, r in enumerate(non_viable[:10], 1):
        reasons = []
        if r.get('p_value', 1) >= 0.05:
            reasons.append("p≥0.05")
        if not r.get('ci_excludes_zero', False):
            reasons.append("CI∋0")
        if r.get('stability', 0) < 0.60:
            reasons.append(f"stab={r['stability']:.2f}<0.60")
        reason_str = ", ".join(reasons) if reasons else "?"

        print(f"  {i:<3} {r['score']:<7.3f} {r['name']:<30} "
              f"{r['effect_size_bps']:<12.3f} "
              f"{r['p_value']:<10.6f} {r['stability']:<6.2f} {reason_str}")

    print("\n" + "=" * 70)

    # Resumen
    print(f"\n  Total tests:    {len(ranked)}")
    print(f"  Viable edges:   {len(viable)}")
    print(f"  Non-viable:     {len(non_viable)}")
    if viable:
        print(f"  Best score:     {viable[0]['score']:.3f} "
              f"({viable[0]['name']})")
        print(f"\n  Siguiente paso: Fase 5 — Validar edges viables.\n")
    else:
        print(f"\n  Siguiente paso: Refinar hipótesis o probar nuevas.\n")
