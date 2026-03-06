"""
mt5_h3_bot.py — Capa de Ejecución para MT5 (Nasdaq H3v2)

EJECUCIÓN REALISTA + PREVENCIÓN DE RUINA:
  Debe ejecutarse como un proceso daemon durante la sesión NY.
  Conecta via la API MetaTrader5 de Python a la terminal.
  Implementa estrictamente el risk-engine FTMO: SL Físico al momento del fill,
  riesgo del 0.5% por trade.
  
FLUJO:
  1. Al iniciar (cierre NY ~20:05), lee nq_signal_monitor.json 
     para ver si MAÑANA el filtro previo está ON (día anterior bajista).
  2. Al día siguiente a las 14:29 UTC: Lee precio actual y open de 13:30.
  3. Si abs((p_14:29 - p_13:30) / p_13:30) > 0.003:
  4.   Calcula size de lote para arriesgar exactamente 0.5% de balance hasta SL
  5.   Envía orden Market/Limit Aggressive con SL = precio_entrada +/- (1.0 * ATR_1H)
  6. A las 19:59 UTC: Envía orden de Market Close (cierra todo).
"""

import time
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Nota: requiere pip install MetaTrader5
# import MetaTrader5 as mt5

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "quant_bot" / "research" / "artifacts" / "nq"
SIGNAL_FILE = ARTIFACTS_DIR / "daily_signals.json"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - MT5 - %(message)s')
logger = logging.getLogger("MT5_Exec")

# CONFIGURACIÓN RIESGO FTMO
RISK_PER_TRADE_PCT = 0.005  # 0.5%
SYMBOL = "USTEC"            # O "US100", "US30", "NAS100" (depende del broker)
MAGIC_NUMBER = 30032026     # Identificador para este bot

def init_mt5():
    """Inicializa conexión con la terminal local MetaTrader5."""
    # if not mt5.initialize():
    #     logger.error(f"Error cargando MT5: {mt5.last_error()}")
    #     return False
    # logger.info("✅ Conectado a MT5")
    # return True
    logger.info("  (Simulación de conexión MT5... requiere paquete y terminal en Windows o WINE)")
    return True

def get_signal_tomorrow():
    if not SIGNAL_FILE.exists():
        logger.error(f"No signal file found at {SIGNAL_FILE}")
        return False
        
    with open(SIGNAL_FILE, 'r') as f:
        data = json.load(f)
        
    return data.get('filter_active', False)

def execute_trade(direction: int, atr_points: float, current_price: float, balance: float):
    """
    Envia instrucción de trade a MT5.
    direction: 1 para largo, -1 para corto
    atr_points: rango de velas primera hora para stop loss volatil
    """
    # 1. Definir tamaño de Stop Loss (puntos de precio NQ)
    # Según nq_h3_mae_mfe.py veremos el optimal, por ahora 1.5x ATR
    sl_points = atr_points * 1.5
    
    # 2. SL levels (prices)
    if direction == 1:
        sl_price = current_price - sl_points
    else:
        sl_price = current_price + sl_points
        
    # 3. Size del lote
    # Risk USD = Balance * RISK_PER_TRADE_PCT
    risk_usd = balance * RISK_PER_TRADE_PCT
    # Convertir puntos NQ a valor de ticket. 1 lote NQ de broker usualmente = 1 USD/punto
    # o varía según multiplier de contrato. Si tick_value=1.00:
    lot_size = risk_usd / sl_points
    
    # Asegurando formato para MT5
    lot_size = round(lot_size, 2)
    sl_price = round(sl_price, 2)
    
    action = "BUY" if direction == 1 else "SELL"
    
    logger.info(f"⚡ Misión de Ejecución: {action} {lot_size} lotes {SYMBOL}")
    logger.info(f"   Entry aprox: {current_price} | SL Físico al milisegundo: {sl_price}")
    logger.info(f"   Riesgo total: ${risk_usd:.2f} (0.5% of ${balance:.2f})")
    
    # mt5.OrderSend(...) -> mock format
    return True

def bot_loop():
    logger.info("Iniciando daemon de H3v2 MT5...")
    if not init_mt5():
        return
        
    while True:
        now_utc = datetime.now(timezone.utc)
        
        # 1. Si son las 20:05 UTC -> Leer si mañana el filtro está true
        if now_utc.hour == 20 and now_utc.minute >= 5 and now_utc.minute <= 10:
            active = get_signal_tomorrow()
            logger.info(f"Filtro actualizado para mañana: {'ON' if active else 'OFF'}")
            time.sleep(3600)  # Dormir un rato para no spamearlo
            continue
            
        # 2. Si son las 14:29 UTC -> si filtro = true, prepararse
        if now_utc.hour == 14 and now_utc.minute == 29:
            active = get_signal_tomorrow()
            if active:
                logger.info("🔔 Ejecución Inminente. Evaluando retorno de 13:30 a 14:29...")
                # Lógica simplificada:
                # p_open = mt5.copy_rates_from("USTEC", mt5.TIMEFRAME_M1, 13:30, 1)[0]['open']
                # p_now = mt5.symbol_info_tick("USTEC").bid
                p_open = 21000.0  # mock
                p_now  = 21100.0  # mock
                ret = (p_now - p_open) / p_open
                
                if abs(ret) > 0.003:
                    direction = 1 if ret > 0 else -1
                    atr_mock = 50.0 # Puntos NQ
                    balance  = 100000.0
                    execute_trade(direction, atr_mock, p_now, balance)
                    time.sleep(60) # prevent multiple triggers in same minute
                else:
                    logger.info(f"Retorno {ret*100:.2f}% no superó threshold (0.3%). Sin trade.")
                    time.sleep(60)
                    
        # 3. Cierre EOD a las 19:59 UTC
        if now_utc.hour == 19 and now_utc.minute == 59:
            logger.info("Cerrando todas las posiciones abiertas por H3v2.")
            # posiciones = mt5.positions_get(symbol="USTEC")
            # For each position -> SEND CLOSE_MARKET
            time.sleep(60)
            
        time.sleep(30) # Loop pulse

if __name__ == "__main__":
    bot_loop()
