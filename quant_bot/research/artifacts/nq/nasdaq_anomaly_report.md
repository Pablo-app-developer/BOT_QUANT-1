# NASDAQ 100 (USATECHIDXUSD) — Análisis de Microestructura Base
**Generado**: 2026-03-04 21:39 UTC

---

## RESUMEN EJECUTIVO

| Hipótesis | Veredicto | Métrica Clave |
|-----------|-----------|---------------|
| U-Shape Volatility | ✅ U-SHAPE CONFIRMADA | p=0.0000 |
| Overnight Drift | ❌ Drift no significativo o negativo | μ=0.0344% / día |
| Correlación 1ª Hora | ✅ CORRELACIÓN DETECTADA | r=0.4223 |
| Gap Fill Statistics | ✅ GAP FILL SESGADO | Fill rate=56.6% |

---

## 1. U-SHAPE VOLATILITY

**Veredicto**: ✅ U-SHAPE CONFIRMADA

- Rango medio apertura (13-14h UTC): **0.0767%**
- Rango medio mediodía (15-17h UTC): **0.0642%**
- T-test p-value: **0.000000**

**Implicación táctica**: La volatilidad alta en apertura sugiere potencial para estrategias de captación de movimiento. El valle del mediodía sugiere estrategias de reversión.

---

## 2. OVERNIGHT DRIFT

**Veredicto**: ❌ Drift no significativo o negativo

- Retorno overnight medio diario: **0.0344%**
- % días positivos: **55.0%**
- P-value vs media=0: **0.2097**
- ¿Significativo estadísticamente? **NO**

**Performance por año**:
```
      mean_return     std  count  pct_positive
2021       0.0418  0.6731    232        0.5862
2022      -0.1311  1.2206    232        0.4698
2023       0.1127  0.9991    229        0.5371
2024       0.1116  0.7948    233        0.5708
2025       0.0456  0.9269    232        0.5862
2026      -0.0863  0.5169     15        0.5333
```

**PREGUNTAS DE DESTRUCCIÓN (Fase 6)**:
- ¿El drift es capturable con spread real overnight? → PENDIENTE validar
- ¿Desaparece en 2022 (bear market)? → Ver tabla por año
- ¿Depende de eventos macro específicos? → PENDIENTE
- ¿Con spread extendido (10+ puntos), sigue positivo? → PENDIENTE

---

## 3. CORRELACIÓN PRIMERA HORA vs CIERRE DÍA

**Veredicto**: ✅ CORRELACIÓN DETECTADA

- Pearson r: **0.4223** (p=0.0000)
- Spearman r: **0.4122**
- Días con primera hora > +0.5%: **0.7976190476190477** cierran positivo
- Z-Score vs azar: **7.715167498104597**

---

## 4. GAP FILL STATISTICS

**Veredicto**: ✅ GAP FILL SESGADO

- N gaps analizados: **1003**
- Fill rate total: **56.6%**
- Fill rate UP gaps: **55.8%**
- Fill rate DOWN gaps: **57.7%**
- P-value vs 50%: **0.0000**

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
