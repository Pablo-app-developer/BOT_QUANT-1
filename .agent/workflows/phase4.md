---
description: Fallas y puntos ciegos
---

. Overfitting casi seguro
Con datos de 1 minuto desde 2016 tienes millones de velas. Eso permite ajustar cualquier cosa hasta que “funcione”.
Pregunta incómoda: ¿tu edge sobrevive cuando cambias ligeramente parámetros, activo, spread o timeframe? Si se rompe → no hay edge.

2. Costos reales vs backtest
En 1m el enemigo no es la dirección, es:

spread variable

slippage

ejecución VPS/broker

comisiones
Pregunta: ¿tu backtest usa spread histórico realista o fijo falso? La mayoría miente aquí sin darse cuenta.

3. Regime change (2016 ≠ hoy)
Mercado 2016–2019 ≠ COVID ≠ 2022 tightening ≠ hoy.
Si tu edge solo gana en un régimen, está muerto.
Pregunta: ¿segmentaste por periodos macro o solo corriste todo junto?

4. Data leakage / look-ahead
Muy común en ML + trading. Cosas como:

usar cierre de vela no disponible en tiempo real

indicadores recalculados

normalización con datos futuros
Pregunta dura: ¿puedes demostrar que tu pipeline es 100% causal?

5. Multiple testing / curve fitting
¿Cuántas ideas probaste antes de “encontrar edge”? 5? 50? 500?
Mientras más pruebas, más probabilidad de falso positivo.
Pregunta: ¿ajustaste p-value / probabilidad de edge por número de tests? Casi nadie lo hace.

6. Walk-forward real o falso
Si hiciste:

Train: 2016-2022

Test: 2023-2024
Eso NO es suficiente.

Necesitas:

rolling walk-forward

re-training por ventanas

forward sin tocar parámetros

Si no → optimización disfrazada.

7. Edge ≠ rentable
Necesitas responder esto con números:

Expectancy por trade después de costos

Sharpe realista (<1.5 probablemente)

Max DD

Risk of ruin

Distribución de rachas

Si no sabes esas 5 → no estás en fase 4, estás en fase 2.

8. 1 minuto = microestructura
En M1 compites contra:

HFT

market makers

arbitraje estadístico

Si tu edge depende de precisión de entrada → probablemente no escala en real.