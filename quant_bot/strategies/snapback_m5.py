
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_signals(data: pd.DataFrame, 
                     sigma: float = 3.5, 
                     lookback: int = 15, 
                     hold_period: int = 15) -> pd.Series:
    """
    Genera señales para la estrategia 'SnapBack M5'.
    
    Lógica:
    1. Volatilidad debe ser ALTA (Top 33%).
    2. Tendencia M5 debe estar definida.
    3. Precio M1 debe desviarse > Sigma * DesviaciónEstándar.
    4. Dirección de la desviación opuesta a la Tendencia M5 (Dip in Uptrend).
    
    Salida:
    Series con 1 (Long), -1 (Short), 0 (Flat).
    Nota: La señal indica "ENTRADA". La gestión de salida (Hold 15 min) 
    se maneja en el motor de backtest convirtiendo la señal en trades.
    """
    close = data['close']
    
    # ── 1. Cálculos de Volatilidad ──
    rets = close.pct_change().fillna(0)
    vol_60 = rets.rolling(60).std()
    
    # Umbral de Alta Volatilidad (calculado sobre toda la serie por ahora, 
    # en producción usaríamos ventana rolling grande para evitar lookahead)
    # Para backtest honesto, usamos rolling quantile o valor fijo derivado de Research (Phase 5).
    # En Phase 5 vimos que q66 funciona. Usaremos rolling quantile para no tener lookahead.
    vol_threshold = vol_60.rolling(10_000, min_periods=1000).quantile(0.66)
    
    is_high_vol = vol_60 > vol_threshold
    
    # ── 2. Tendencia M5 ──
    # Aproximación: SMA 5 vs 20 velas de M5 -> 25 vs 100 velas de M1
    sma_fast = close.rolling(25).mean()
    sma_slow = close.rolling(100).mean()
    
    trend_up = sma_fast > sma_slow
    trend_down = sma_fast < sma_slow
    
    # ── 3. Desviación ──
    # Retorno últimos 15 min (que usamos como proxy de desviación)
    # O mejor: precio vs media movil reciente?
    # Phase 5 validó "pct_change(15)". Usaremos eso.
    # Deviation = Retorno acumulado últimos 15 min.
    
    momentum = close.pct_change(lookback)
    
    # El Threshold es dinámico: Sigma * Volatilidad del Momentum
    # Volatilidad del momentum (std de return 15m)
    mom_std = momentum.rolling(1000).std()
    
    threshold_dynamic = sigma * mom_std
    
    # ── 4. Señales de Entrada (Sparse) ──
    entries = pd.Series(0, index=data.index)
    
    # Long: Dip (Negative Mom) > Threshold AND Uptrend AND High Vol
    entries[
        (momentum < -threshold_dynamic) & 
        trend_up & 
        is_high_vol
    ] = 1
    
    # Short: Rally (Positive Mom) > Threshold AND Downtrend AND High Vol
    entries[
        (momentum > threshold_dynamic) & 
        trend_down & 
        is_high_vol
    ] = -1
    
    # ── 5. Gestión de Posición (Time-based Exit y Reversals) ──
    # Convertir entradas dispersas en señal continua de posición
    # para que el engine mantenga el trade durante 'hold_period'.
    
    # Optimización: Iterar sobre indices de entradas en lugar de todo el array
    # Sin embargo, necesitamos llenar los 0s intermedios.
    # Un loop puro en Python para 3.7M filas es lento (10-20s).
    # Usamos un enfoque híbrido: forward fill limitado?
    # Pandas no tiene ffill(limit=N) que respete resets.
    
    # Loop explícito optimizado (usando valores numpy)
    pos_arr = np.zeros(len(data), dtype=int)
    entry_vals = entries.values
    
    current_pos = 0
    bars_left = 0
    
    for i in range(len(entry_vals)):
        sig = entry_vals[i]
        
        if sig != 0:
            current_pos = sig
            bars_left = hold_period
        
        if bars_left > 0:
            pos_arr[i] = current_pos
            bars_left -= 1
        else:
            current_pos = 0
            
    signals = pd.Series(pos_arr, index=data.index)
    
    return signals
