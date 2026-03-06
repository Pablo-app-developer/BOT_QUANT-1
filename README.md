# BOT_QUANT-1 — Nasdaq 100 Quantitative Edge Discovery & Deployment

## 🎯 Proyecto

Sistema de investigación cuantitativa para descubrir, validar y desplegar edges estadísticos en el **USATECHIDXUSD (Nasdaq 100)**. Diseñado para cuentas de fondeo (FTMO / Prop Firms).

**Edge Validado:** H3v2 — First Hour Conditional Momentum  
**Estado:** ✅ Listo para Paper Trading  
**Cuenta objetivo:** FTMO Challenge $10,000

---

## 📊 Resultados del Edge H3v2

| Métrica | In-Sample (2021-2023) | Out-of-Sample (2024-2025) |
|---------|----------------------|--------------------------|
| N trades | 154 | 88 |
| Sharpe Ratio | 1.67 | **3.60** |
| Win Rate | 61.0% | 61.4% |
| p-value | 0.19 | **0.037** |
| Ret. Anualizado | +33% | +113% |

> El OOS Sharpe (3.60) es **mayor** que el IS Sharpe (1.67) — lo opuesto al overfitting.

---

## 🏗️ Estructura del Proyecto

```
BOT_QUANT-1/
├── DEPLOYMENT_PLAN.md          # Plan de deployment completo (3 fases)
├── RESEARCH_JOURNAL.md         # Diario de investigación con todos los hallazgos
├── README.md                   # Este archivo
│
├── quant_bot/
│   ├── data/                   # Capa de datos
│   │   ├── nq_loader.py        # Parser nativo de archivos .bi5 (Dukascopy)
│   │   ├── loader.py           # Loader genérico
│   │   └── download_dukascopy.py  # Descarga paralela de datos históricos
│   │
│   ├── research/               # Capa de investigación (Fase 6)
│   │   ├── nq_edge_factory.py      # 🏭 20 hipótesis, corrección BH-FDR
│   │   ├── nq_edge_discovery.py    # Pipeline principal de descubrimiento
│   │   ├── nq_h3_prior_day.py      # H3v2: análisis del filtro día previo
│   │   ├── nq_h3_deep.py           # Deep dive completo del edge H3v2
│   │   ├── nq_h3_execution.py      # Fase 6.6: validación ejecución realista
│   │   ├── nq_h3_mae_mfe.py        # MAE/MFE: análisis SL óptimo
│   │   ├── nq_cross_asset.py       # Cross-validation interna
│   │   ├── nq_signal_monitor.py    # 📡 Monitor diario de señales
│   │   ├── nq_h10_deep.py          # Deep dive H10 (descartado: look-ahead)
│   │   ├── nq_first_hour_edge.py   # Análisis de primera hora
│   │   ├── nq_session_analysis.py  # Análisis por sesión
│   │   ├── nq_overnight_effect.py  # Efecto overnight
│   │   ├── nq_whipsaw_reversal.py  # Whipsaw reversal
│   │   ├── nq_short_overnight.py   # Short overnight
│   │   ├── statistical_tools.py    # Herramientas estadísticas
│   │   └── artifacts/nq/           # Gráficos, métricas y logs
│   │
│   ├── execution/              # Capa de ejecución (deployment)
│   │   ├── nq_h3v2_risk_engine.py  # 🛡️ Motor de riesgo FTMO
│   │   └── mt5_h3_bot.py           # 🤖 Daemon de ejecución MT5
│   │
│   ├── backtest/               # Motor de backtesting
│   │   ├── engine.py
│   │   ├── execution_model.py
│   │   └── metrics.py
│   │
│   ├── config/                 # Configuración
│   │   └── settings.py
│   │
│   └── requirements.txt        # Dependencias Python
```

---

## 🚀 Inicio Rápido

### 1. Instalar dependencias
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r quant_bot/requirements.txt
```

### 2. Descargar datos históricos (Dukascopy)
```bash
# NQ100 — 2021 a 2026
python3 quant_bot/data/download_dukascopy.py \
  --instrument USATECHIDXUSD --years 2021-2026 --workers 20
```

### 3. Construir dataset M1
```bash
python3 quant_bot/research/nq_edge_discovery.py --rebuild-cache
```

### 4. Ejecutar el Edge Factory (20 hipótesis)
```bash
python3 quant_bot/research/nq_edge_factory.py
```

### 5. Monitor diario de señales
```bash
# Ejecutar cada noche al cierre NY (~20:10 UTC):
python3 quant_bot/research/nq_signal_monitor.py --signal-only

# Registrar un trade:
python3 quant_bot/research/nq_signal_monitor.py \
  --add-trade 2026-03-05 LONG 21050 21200 0.004
```

---

## 📋 Reglas del Sistema (INAMOVIBLES)

```
SI prev_day_return < -0.1%         → Filtro activo
Y  |first_hour_return| > 0.3%     → Señal confirmada
ENTONCES:
  Dirección  = SIGN(first_hour_return)
  Entrada    = 14:30 UTC (cierre primera hora NY)
  Stop Loss  = 1.5 × ATR(primera hora)  ← FÍSICO INMEDIATO
  Riesgo     = 0.5% del balance
  Salida     = 19:59 UTC (cierre sesión NY)
```

---

## 🔬 Filosofía de Validación

> *"El sistema NO intenta hacer que funcione. Intenta demostrar que NO funciona. Solo si sobrevive → considerar edge real."*

- 20 hipótesis testeadas simultáneamente
- Corrección estadística Benjamini-Hochberg (FDR < 10%)
- Split IS/OOS estricto sin contaminación
- Verificación anti look-ahead bias
- Monte Carlo, Walk-Forward, Stress Tests
- Solo 1 de 20 hipótesis sobrevivió → H3v2

---

## 📄 Documentos Clave

| Documento | Descripción |
|-----------|-------------|
| [DEPLOYMENT_PLAN.md](DEPLOYMENT_PLAN.md) | Plan de deployment en 3 fases |
| [RESEARCH_JOURNAL.md](RESEARCH_JOURNAL.md) | Diario completo de investigación |

---

## ⚠️ Disclaimer

Este proyecto es exclusivamente para investigación cuantitativa y educación. El trading de instrumentos financieros conlleva riesgo de pérdida. Los resultados pasados no garantizan resultados futuros. Nunca arriesgues capital que no puedes permitirte perder.
