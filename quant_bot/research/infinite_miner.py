"""
infinite_miner.py — Motor de Búsqueda Combinatoria y Validación Científica Continua.

Este script implementa un "minero" infinito optimizado:
1. GPU / CPU Híbrido: Los cálculos se realizan de forma matricial. 
2. Chunking (Memoria RAM Segura): El filtro rápido toma muestras aleatorias de 6 meses
   para evitar llenar la RAM (lo que traba la PC). Si sobrevive 6 meses, se somete a 10 años.
3. Phase 6 (Tortura): Test realista de pips, comisiones y slippage.
"""

import sys
import os
import gc
import logging
import sqlite3
import random
import time
from pathlib import Path
import pandas as pd
import numpy as np
import vectorbt as vbt

# ---------------------------------------------------------
# OPTIMIZADOR DEL SISTEMA (EVITA QUE LA PC SE QUEDE PEGADA)
# ---------------------------------------------------------
# Limita Numba (el motor de vectorbt) para no saturar todos los hilos de tu CPU
os.environ['NUMBA_NUM_THREADS'] = '4' 

try:
    # Si detectamos CuPy, pasamos cálculos masivos a la GPU RTX 3050
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


# Asegurar importe de modules locales
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from data.loader import load_processed

# Configuración
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("InfiniteMiner")
DB_PATH = PROJECT_ROOT / "validated_edges.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS validated_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            hypothesis_desc TEXT,
            regime TEXT,
            trigger TEXT,
            exit_logic TEXT,
            win_rate REAL,
            expectancy REAL,
            total_trades INTEGER,
            profit_factor REAL,
            max_dd REAL,
            metrics_torture TEXT,
            pass_torture BOOLEAN
        )
    ''')
    conn.commit()
    conn.close()

def generate_hypothesis():
    """Generates a random logical configuration bounding our edge search."""
    regimes = ["Vol_Comp", "Vol_Exp", "Trend_Up", "Trend_Down", "Range_Bound"]
    triggers = ["ZScore_Reversion", "ZScore_Momentum", "RSI_Oversold", "RSI_Overbought", "MA_Cross"]
    exits = ["Time_Based"] 
    
    r = random.choice(regimes)
    t = random.choice(triggers)
    e = random.choice(exits)
    
    params = {
        'ma_fast': random.randint(5, 20),
        'ma_slow': random.randint(30, 200),
        'rsi_window': random.choice([7, 10, 14, 21]),
        'z_window': random.choice([20, 50, 100, 200]),
        'z_thresh': round(random.uniform(2.0, 4.0), 2),
        'hold_bars': random.choice([5, 12, 24, 48]),
    }
    
    desc = f"R:{r} | T:{t} | E:{e}"
    return {'regime': r, 'trigger': t, 'exit': e, 'params': params, 'desc': desc}


def rolling_window_gpu(a, window):
    """ (GPU) Efficient sliding window for fast means calculation if CuPy is installed """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return cp.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def gpu_zscore(close_array, window):
    """ Calcula Z-Score masivo usando tu NVIDIA en microsegundos """
    d_close = cp.asarray(close_array)
    # Rellenamos de NaNs los primeros "window" campos
    z_out = cp.full_like(d_close, cp.nan)
    
    if len(d_close) >= window:
        rw = rolling_window_gpu(d_close, window)
        means = rw.mean(axis=-1)
        stds = rw.std(axis=-1, ddof=1)
        
        # Ojo: cp.where o mask array preclásico, valid std > 0
        valid = stds > 0
        z_valid = (d_close[window-1:][valid] - means[valid]) / stds[valid]
        
        # Re-assign mapping
        temp = cp.zeros_like(means)
        temp[valid] = z_valid
        z_out[window-1:] = temp
        
    return cp.asnumpy(z_out)


def construct_signals(data: pd.DataFrame, hyp: dict):
    """
    Construcción vectorizada. 
    Usa numpy crudo o GPU para no reventar la memoria con Pandas `rolling`.
    """
    close = data['close'].values
    p = hyp['params']
    
    # Preasignar memoria cruda = MÁS ESTABLE, la PC no se traba.
    entries = np.zeros(len(close), dtype=bool)
    exits = np.zeros(len(close), dtype=bool)
    
    # ----- CÁLCULO DE INDICADORES (CPU LIGERO o GPU) -----
    if "ZScore" in hyp['trigger']:
        if GPU_AVAILABLE:
            z_score = gpu_zscore(close, p['z_window'])
        else:
            # Optimizacion numpy crudo usando convolucion para la media. MUCHO más rapido q Pandas.
            window = p['z_window']
            z_score = np.full_like(close, np.nan)
            if len(close) >= window:
                # Pandas rolling here is necessary for raw std, but we do it cleanly
                s = pd.Series(close)
                z_score = ((s - s.rolling(window).mean()) / s.rolling(window).std()).values
                
        if hyp['trigger'] == "ZScore_Reversion":
            entries = z_score < -p['z_thresh']
        elif hyp['trigger'] == "ZScore_Momentum":
            entries = z_score > p['z_thresh']
            
    elif "RSI" in hyp['trigger']:
        # vectorbt usa Numba (CPU Paralelo nativo en C), corre ultra rápido
        rsi = vbt.RSI.run(pd.Series(close), window=p['rsi_window']).rsi.values
        if hyp['trigger'] == "RSI_Oversold":
            entries = rsi < 30
        elif hyp['trigger'] == "RSI_Overbought":
            entries = rsi > 70
            
    elif "MA_Cross" in hyp['trigger'] or "Trend" in hyp['regime']:
        ma_f = vbt.MA.run(pd.Series(close), window=p['ma_fast']).ma.values
        ma_s = vbt.MA.run(pd.Series(close), window=p['ma_slow']).ma.values
        if hyp['trigger'] == "MA_Cross":
            entries[1:] = (ma_f[1:] > ma_s[1:]) & (ma_f[:-1] <= ma_s[:-1])
            
    # ------ RETORNO ESTRUCTURADO (EVITA FUGAS MAIN MEMORY) ------
    if hyp['exit'] == "Time_Based":
        # Shift rápido en numpy (Time-based exit = N bars later)
        ex_idx = np.where(entries)[0] + p['hold_bars']
        ex_idx = ex_idx[ex_idx < len(close)] # Límite de array
        exits[ex_idx] = True
        
    # Pandas Series Return required for VBT stats
    idx = data.index
    return pd.Series(entries, index=idx), pd.Series(exits, index=idx)


def run_fast_filter(data_chunk, entries_, exits_):
    """Calculates basic metrics natively with VBT to see if it's worth torturing."""
    pf = vbt.Portfolio.from_signals(
        data_chunk['close'],
        entries=entries_,
        exits=exits_,
        init_cash=100_000,
        fees=0.00005,  # Low friction baseline (approx 0.5 pips)
        freq='1min'
    )
    stats = pf.stats()
    trades = stats.get('Total Trades', 0)
    if pd.isna(trades): trades = 0
    
    wr = stats.get('Win Rate [%]', 0)
    expect = stats.get('Expectancy', 0)
    pf_factor = stats.get('Profit Factor', 0)
    
    metrics = {
        'trades': trades,
        'wr': wr if not pd.isna(wr) else 0,
        'expectancy': expect if not pd.isna(expect) else 0,
        'profit_factor': pf_factor if not pd.isna(pf_factor) else 0,
        'max_dd': stats.get('Max Drawdown [%]', 0)
    }
    
    if trades < 30: return False, "Low Trades", metrics
    if metrics['wr'] > 95: return False, "Illusion WR", metrics
    if metrics['expectancy'] <= 0.2: return False, "Low Exp", metrics
    if metrics['profit_factor'] < 1.05: return False, "Low PF", metrics
        
    return True, "Passed Filter", metrics

def run_torture_chamber(data_full, entries, exits):
    """Phase 6: Retraso algoritmico + Comisiones brutales."""
    delayed_entries = entries.shift(1).fillna(False)
    delayed_exits = exits.shift(1).fillna(False)
    
    torture_pf = vbt.Portfolio.from_signals(
        data_full['close'],
        entries=delayed_entries,
        exits=delayed_exits,
        init_cash=100_000,
        fees=0.00015, # 1.5 pips equivalent friction on EURUSD
        freq='1min'
    )
    
    t_stats = torture_pf.stats()
    t_expect = t_stats.get('Expectancy', 0)
    t_pf = t_stats.get('Profit Factor', 0)
    
    if not pd.isna(t_expect) and t_expect > 0.0 and t_pf > 1.01:
        return True, t_stats
    return False, t_stats

def save_edge(hyp, metrics, pass_torture=False):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO validated_edges (
            hypothesis_desc, regime, trigger, exit_logic,
            win_rate, expectancy, total_trades, profit_factor, max_dd, pass_torture
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        hyp['desc'], hyp['regime'], hyp['trigger'], hyp['exit'],
        float(metrics['wr']), float(metrics['expectancy']), int(metrics['trades']),
        float(metrics['profit_factor']), float(metrics['max_dd']), pass_torture
    ))
    conn.commit()
    conn.close()

def main():
    logger.info("=========================================")
    logger.info(" INFINITE QUANT MINER V2 (MEM-SAFE GPU)  ")
    logger.info("=========================================")
    
    if GPU_AVAILABLE:
         logger.info(">>> GPU NVIDIA DETECTADA Y ACTIVADA VIA CUPY <<<")
    else:
         logger.info(">>> MODO CPU (CUDA/NVIDIA no instalado). Usando Numba optimizado. <<<")
    
    init_db()
    
    try:
        data = load_processed()
        logger.info(f"Full Dataset Loaded: {len(data):,} rows.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    iteration = 0
    start_time = time.time()
    
    CHUNK_SIZE = 500_000 # ~1.5 años en M1. (Previene que la PC se trabe).
    
    while True:
        iteration += 1
        
        # 1. SAMPLING: Extraer un bloque aleatorio pequeño para el Fast Filter (evita colapso de RAM)
        if len(data) > CHUNK_SIZE:
            start_idx = random.randint(0, len(data) - CHUNK_SIZE)
            data_fast = data.iloc[start_idx:start_idx+CHUNK_SIZE]
        else:
            data_fast = data

        hyp = generate_hypothesis()
        
        # 2. SEÑALES EN EL BLOQUE PEQUEÑO
        entries_f, exits_f = construct_signals(data_fast, hyp)
        
        if not entries_f.any():
            continue
            
        # 3. FAST FILTER
        is_valid, reason, metrics = run_fast_filter(data_fast, entries_f, exits_f)
        
        if is_valid:
            logger.info(f"[POTENTIAL EDGE] {hyp['desc']} | P: {hyp['params']} | PF: {metrics['profit_factor']:.2f}")
            logger.info(" -> Encontrado en bloque. Sometiendo a DATASET COMPLETO (10 Años) + TORTURA...")
            
            # 4. TORTURA EN TODA LA HISTORIA RECIÉN SI PASÓ EL FILTRO
            # Generar señales en data completa solo para los fuertes
            full_entries, full_exits = construct_signals(data, hyp)
            survived, t_stats = run_torture_chamber(data, full_entries, full_exits)
            
            if survived:
                logger.warning(f"!!! EDGE SURVIVED 10 YR TORTURE !!! Net Exp: {t_stats.get('Expectancy', 0):.3f}")
                save_edge(hyp, metrics, pass_torture=True)
            else:
                logger.info(f" -> Destruido en 10A Tortura Real.")
                
        # 5. GARBAGE COLLECTION: Protege a tu PC de congelarse en la RAM
        if iteration % 100 == 0:
            gc.collect() 
            elapsed = time.time() - start_time
            logger.info(f"--- Iter {iteration} | Speed: {iteration/elapsed:.2f} it/s (RAM Cleared) ---")

if __name__ == "__main__":
    main()
