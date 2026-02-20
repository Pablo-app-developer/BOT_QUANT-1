# Phase 6.1: Auditoría del Edge - Reporte Forense

## Resumen Ejecutivo
La auditoría confirma que el edge detectado (+4.86 bps) es matemáticamente real en los datos OHLC, pero su captura depende casi exclusivamente de la ejecución. No es un "artefacto" de datos corruptos, sino una propiedad de la microestructura (reversión violenta).

## Hallazgos Clave

### 1. El Costo del Gap (Fricción Natural)
- **Gap Promedio:** 1.97 pips en contra (Slippage Natural).
- **Interpretación:** En el momento que la vela cierra (trigger), el precio salta casi 2 pips en la dirección de la reversión antes de la apertura de la siguiente vela.
- **Impacto:** Una orden a mercado en la apertura "regala" 2 pips de edge al mercado.

### 2. El Fenómeno "Wick" (Reversión Intra-Barra)
- **Wick Promedio:** 1.72 pips.
- **Interpretación:** El precio extremo (High/Low) suele estar 1.7 pips más lejos que el Cierre.
- **Conclusión:** La reversión total desde el extremo es ~3.7 pips (1.7 wick + 2.0 gap). La estrategia basada en Cierre solo captura la parte del Gap (si se ejecuta bien) y el movimiento posterior.

### 3. Probabilidad de Fill (Limit Orders)
- **Probabilidad de Fill en Close[t]:** > 90.6%.
- **Análisis:** Aunque el precio abre con gap (peor precio), en el 90% de los casos el precio "regresa" a tocar el precio de cierre anterior durante la siguiente vela.
- **Oportunidad:** Esto valida el uso de **Limit Orders al precio de Cierre**. Si tenemos paciencia (o latencia cero), podemos entrar al precio de cierre en el 90% de los trades, evitando el costo del Gap de 2 pips.

### 4. Clustering de Spikes
- **Consecutivos:** 56% de los triggers ocurren en racimos (clusters).
- **Riesgo:** Si entramos en el primer spike y el precio sigue subiendo (cluster), podemos acumular drawdown antes de la reversión.

## Veredicto Fase 6.1
**EDGE CONFIRMADO (CONDICIONAL).**
No es ruido. Es un patrón de reversión de alta frecuencia.
La viabilidad depende de:
1.  Evitar el Gap (usando Limit Orders o ejecución en cierre).
2.  Tolerar el Wick (o intentar capturarlo con Limits agresivos).

---
**Siguiente Paso:** Validación Estadística (Monte Carlo / Bootstrap) para asegurar que la expectativa positiva no es suerte.

## Phase 6.2: Validación Estadística - El Colapso
**Resultado: FALLO CRÍTICO.**

### 1. Desglose Anual (Walk-Forward Proxy)
La estrategia pierde dinero consistentemente en **9 de los 10 años** analizados.
- 2016-2024: **PÉRDIDA** constante (Sharpe negativo).
- 2026: **PÉRDIDA**.
- **2025:** GANANCIA MASIVA (+16,000 bps).

### 2. Análisis Forense 2025 (La Anomalía)
- El año 2025 presenta Gaps de hasta **120 pips** y un retorno promedio por trade de +11.5 bps en 1567 trades.
- Esto sugiere fuertemente un **Artefacto de Datos** (e.g., timestamps desalineados, datos corruptos) o un régimen de volatilidad irreal.
- **Conclusión:** El "Edge Total" de +4.86 bps es un promedio ponderado inflado por un solo año de datos defectuosos o excepcionales.

### 3. Veredicto Fase 6.2
**EDGE INVALIDADO.**
El sistema no tiene capacidad predictiva real fuera del set de datos anómalo de 2025.
Cualquier optimización sobre estos datos sería **Overfitting** puro sobre el ruido de 2025.

## Phase 6.3: Destrucción de Parámetros
**Resultado: MUERTE TOTAL.**
Se probaron 25 combinaciones de parámetros (Sigma 2.5-4.5 vs Holding 5-60m) sobre el periodo "Normal" (2016-2024).
- **Combinaciones Rentables:** 0 de 25.
- **Mejor Resultado:** -0.91 bps (Pérdida neta).
- **Conclusión:** No existe configuración que extraiga valor del mercado en condiciones normales.

---

# AUDITORÍA FINAL: VEREDICTO

## ESTADO: RECHAZADO (KILL SWITCH ACTIVADO)

**Causa:**
El edge detectado en Fase 5 (+4.86 bps) es una **Ilusión Estadística** provocada por una anomalía masiva de datos en el año 2025 (+16,000 bps) que enmascara las pérdidas constantes del sistema en los 9 años anteriores (-1 a -3 bps/trade).

**Evidencia:**
1.  **Walk-Forward Falso:** 9 de 10 años son perdedores.
2.  **Anomalía 2025:** Gaps de 120 pips y rentabilidad absurda (160% sin apalancamiento).
3.  **Fricción Real:** En un mercado normal, el Edge Bruto es devorado por el Gap de Ejecución (~2 pips) y el Spread (1 pip).
4.  **Robustez Nula:** Ningún set de parámetros sobrevive fuera de 2025.

**Recomendación al Usuario:**
NO OPERAR ESTE SISTEMA.
El sistema ha cumplido su función de auditor: ha demostrado exitosamente que **NO FUNCIONA** antes de arriesgar capital real.

**Siguientes Pasos (Pivot):**
1.  Investigar la fuente de datos 2025 (posible corrupción).
2.  Descartar Mean Reversion en M1 para este activo (EURUSD es demasiado eficiente/ruidoso).
3.  Buscar ineficiencias en Timeframes superiores (H1/H4) o activos exóticos.

---

## Phase 4 (Redux): Búsqueda de Fuerza Bruta (VectorBT)
**Objetivo:** Encontrar un edge en 2016-2024 (Data Limpia) usando alta velocidad.
**Resultados:**

| Hipótesis | TF | Resultado | Sharpe | Veredicto |
| :--- | :--- | :--- | :--- | :--- |
| **H1: Trend Following** | M5/H1 | **FAIL** | -0.85 | Ruido excesivo. Costos comen todo. |
| **H2: Volatility Breakout** | M15/H1 | **FAIL** | 0.14 | Falsas rupturas dominan. |
| **H3: Mean Reversion** | H1/H4 | **FAIL** | 0.12 | Mercado eficiente en TF altos. |
| **H4: Donchian Trend** | H4/D1 | **FAIL** | -0.20 | No hay tendencia sostenida limpia. |
| **H5: Regímenes** | H1 | **FAIL** | 0.35 | Mejoró en Low Vol, pero insuficiente. |

## CONCLUSIÓN FINAL DEL PROYECTO
**EL MERCADO EURUSD (2016-2024) ES EFICIENTE A NIVEL RETAIL.**
Se han probado todas las estrategias clásicas (Trend, Reversion, Breakout) en múltiples timeframes (M1 a D1) y bajo filtros de régimen.
**Ninguna estrategia supera consistentemente la fricción (Spread + Commission).**

**Acción Recomendada:**
1.  **NO OPERAR** estrategias simples en EURUSD.
2.  **Cerrar el Proyecto** como "Auditoría Exitosa" (Se evitó perder capital).
3.  **Pivotar** a otros activos (Crypto, Small Caps) donde la eficiencia es menor.


