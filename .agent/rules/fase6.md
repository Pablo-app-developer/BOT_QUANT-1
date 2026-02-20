---
trigger: always_on
---

OBJETIVO PRINCIPAL
Construir, validar y demostrar un edge cuantitativo REAL, robusto y explotable en condiciones de mercado reales, eliminando cualquier ventaja ilusoria producida por overfitting, artefactos de datos, simulaciones optimistas o ejecución irreal.

La prioridad NO es optimizar rendimiento en backtest, sino demostrar supervivencia en condiciones adversas y ejecución realista.

FILOSOFÍA DE VALIDACIÓN
El sistema debe asumir que:

Todo edge pequeño probablemente es falso

Toda simulación es optimista por defecto

Toda ejecución real será peor que el backtest

Todo parámetro optimizado es sospechoso

Todo resultado debe intentar ser destruido antes de ser aceptado

El sistema debe actuar como auditor cuantitativo escéptico.

FASE 6 — VALIDACIÓN TOTAL DEL EDGE Y EXPLOTABILIDAD

FASE 6.1 — AUDITORÍA DEL EDGE

Validar si el edge NO proviene de:

Artefactos OHLC

Microestructura sintética

Gap artificial por agregación

Fill implícito irreal

Spread constante falso

Slippage subestimado

Curve fitting

Sesgo de régimen

Acciones:

Recalcular con datos tick si es posible

Comparar OHLC vs Tick

Verificar si el gap microestructural es real

Invalidar hipótesis si desaparece en tick

FASE 6.2 — VALIDACIÓN ESTADÍSTICA

El edge debe sobrevivir:

Out-of-sample puro

Walk-forward

Monte Carlo

Bootstrap

Ruido microestructural

Calcular:

Expectancy neta

Profit factor realista

Max DD

Rolling Sharpe

Edge por año

Riesgo de ruina

Edge con spread x2

Si falla → NO ROBUSTO.

FASE 6.3 — DESTRUCCIÓN DE PARÁMETROS

El edge debe sobrevivir:

Sigma 3.2 / 3.5 / 3.8

Holding 10 / 15 / 20

Sin filtro M5

Entrada retrasada 1 vela

Spread variable

Slippage dinámico

Cambio threshold volatilidad

Si muere → curve fitting.

Generar mapa de sensibilidad.

FASE 6.4 — MICROESTRUCTURA

Analizar:

Duración real spikes

Probabilidad reversión completa

Tiempo hasta reversión

Distribución post-entrada

Impacto spread

Si no se puede ejecutar → edge no capturable.

FASE 6.5 — ROBUSTEZ DE RÉGIMEN

Separar resultados en:

Alta volatilidad

Baja volatilidad

Eventos macro

Tendencia fuerte

Lateral

Si colapsa → edge frágil.

FASE 6.6 — EJECUCIÓN REALISTA

Simular con:

Spread dinámico histórico

Slippage dependiente volatilidad

Probabilidad real de fill

Latencia: 100 / 300 / 600 ms

Wick no ejecutable

Partial fills

Market vs Limit passive vs Limit aggressive

Entrada intra-bar

Entrada en cierre

Si edge neto < 0.5 pip → NO EXPLOTABLE RETAIL.

FASE 6.7 — VERIFICACIÓN CRÍTICA

El sistema debe responder explícitamente:

¿El simulador modela probabilidad realista de fill?
¿Modela latencia end-to-end?
¿O asume ejecución perfecta?

Si asume ejecución perfecta → EDGE INVÁLIDO (TEÓRICO).

FASE 6.8 — PREGUNTAS DE DESTRUCCIÓN DEL EDGE

Responder con datos:

¿El edge sigue vivo si cae 30%?

¿Sigue vivo con spread x2?

¿Sigue vivo con slippage variable?

¿Sobrevive retraso de 1 vela?

¿Sobrevive sin filtro tendencia?

¿Peor año negativo?

¿Desaparece post-2020?

¿Depende de pocos trades extremos?

¿Monte Carlo muestra ruina?

¿Drawdown tolerable para FTMO?

¿Reversión ocurre antes del fill?

¿% reversión completa vs parcial?

¿Depende de fills perfectos?

¿Depende de microestructura OHLC?

¿Sobrevive latencia 600 ms?

¿Edge estable por año?

¿Colapsa en shocks macro?

¿Sobrevive cambio de broker?

¿Edge suficientemente grande para retail?

¿Expectancy neta positiva tras fricción real?

FASE 6.9 — SI SOBREVIVE

Optimizar SOLO:

Ejecución

Captura spike

Reducción fricción

Selección condiciones mercado

NO optimizar parámetros para inflar backtest.

CLASIFICACIÓN FINAL

Ilusión estadística

Edge micro no explotable retail

Edge explotable con ejecución avanzada

Edge robusto real

Explicar por qué.

REQUISITOS TÉCNICOS

Python
Investigación: vectorbt
Ejecución realista: motor manual

No confiar en fills implícitos.

SALIDAS

Reporte validez edge

Reporte ejecución real

Monte Carlo

Sensibilidad parámetros

Robustez

Clasificación final

Diagnóstico

Recomendación

REGLA FINAL

El sistema NO debe intentar hacer que funcione.
Debe intentar demostrar que NO funciona.

Solo si sobrevive → considerar edge real.