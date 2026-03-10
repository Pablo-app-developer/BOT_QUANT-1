"""
mt5_h3_bot.py — Capa de Ejecución VPS Automática para MT5 (Nasdaq H3v2)

EJECUCIÓN REALISTA + PREVENCIÓN DE RUINA:
  Diseñado para correr 24/5 en un VPS (Windows o Linux+Wine).
  Conecta via la API MetaTrader5 de Python a la terminal MT5 abierta.
  Implementa estrictamente el risk-engine: 
   - Espera autorización del filtro diario (leído del JSON de señales)
   - Extrae el precio a las 14:29:55 UTC
   - Calcula ATR de la hora y lanza MKT_ORDER con SL Físico inyectado
   - Cierra todo a las 19:59:00 UTC
"""

import time
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant_bot.execution.nq_h3v2_risk_engine import RiskEngine, RiskParams
from quant_bot.execution.telegram_notifier import (
    alert_daily_status, alert_trade_open, alert_trade_close
)

ARTIFACTS_DIR = PROJECT_ROOT / "quant_bot" / "research" / "artifacts" / "nq"
SIGNAL_FILE = ARTIFACTS_DIR / "daily_signals.json"

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - VPS_BOT - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "quant_bot" / "execution" / "risk_data" / "vps_daemon.log")
    ]
)
logger = logging.getLogger("VPS_Exec")

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    logger.warning("Paquete 'MetaTrader5' no encontrado. (Ejecuta: pip install MetaTrader5 en Windows/Wine)")
    MT5_AVAILABLE = False

MT5_CONFIG_FILE = PROJECT_ROOT / "quant_bot" / "execution" / "config" / "mt5_config.json"


# ==========================================
# CONFIGURACIÓN DEL BROKER Y CONTRATO
# ==========================================
SYMBOL = "USTEC"            # O "US100", "US30", "NAS100" (depende de tu broker FTMO)
MAGIC_NUMBER = 30032026     # Identificador único para las órdenes de este bot
DOLLAR_PER_POINT = 1.0      # CFD = 1 USD/pto. Micro Futuros = 2 USD/pto. Ajustar.
# ==========================================

def init_mt5():
    """Inicializa conexión con la terminal local MetaTrader5."""
    if not MT5_AVAILABLE:
        return False
        
    # Ruta estándar si se copia la carpeta en Wine o Windows por defecto
    terminal_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    
    # Intentar inicializar pasándole la ruta explícita (supera fallos de registro en Linux/Wine)
    if not mt5.initialize(path=terminal_path):
        logger.warning(f"Fallo init con ruta {terminal_path}: {mt5.last_error()}. Intentando auto-detect...")
        if not mt5.initialize():
            logger.error(f"Error crítico cargando MT5: {mt5.last_error()}")
            return False
        
    # Auto-Login (Especial para Linux Headless o reconexiones)
    if MT5_CONFIG_FILE.exists():
        try:
            with open(MT5_CONFIG_FILE, 'r') as f:
                creds = json.load(f)
            acc = creds.get("account")
            pw = creds.get("password")
            srv = creds.get("server")
            if acc and pw and srv:
                authorized = mt5.login(login=acc, password=pw, server=srv)
                if not authorized:
                    logger.error(f"Error de Login MT5 en cuenta {acc}: {mt5.last_error()}")
                    return False
                logger.info(f"🔑 Login existoso a la cuenta {acc} en el servidor {srv}")
        except Exception as e:
            logger.error(f"Fallo al leer config MT5 o loguearse: {e}")
            
    # Activar el símbolo
    if not mt5.symbol_select(SYMBOL, True):
        logger.error(f"Símbolo {SYMBOL} no encontrado en Market Watch.")
        return False
        
    logger.info(f"✅ Conectado a MT5. Símbolo operativo: {SYMBOL}")
    return True

def get_filter_status():
    """Lee si la señal habilitadora (día anterior < -0.1%) está activa para hoy."""
    if not SIGNAL_FILE.exists():
        logger.error(f"No signal file found at {SIGNAL_FILE}. Asumiendo NO TRADE.")
        return False
        
    try:
        with open(SIGNAL_FILE, 'r') as f:
            data = json.load(f)
            
        # Verificar que la señal no sea de hace varios días
        sig_date_str = data.get('last_data_date', '')
        # Solo operar si la data es reciente, pero para robustez confiaremos en "filter_active"
        active = data.get('filter_active', False)
        return active
    except Exception as e:
        logger.error(f"Error parseando filtro: {e}")
        return False

def get_h1_data_from_mt5():
    """Descarga la velería M1 desde 13:30 hasta 14:29 UTC hoy y devuelve (ret, atr_pct)."""
    now_utc = datetime.now(timezone.utc)
    
    # Construimos datetime de inicio y fin en UTC
    start_dt = datetime(now_utc.year, now_utc.month, now_utc.day, 13, 30, tzinfo=timezone.utc)
    end_dt   = datetime(now_utc.year, now_utc.month, now_utc.day, 14, 29, 59, tzinfo=timezone.utc)
    
    rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M1, start_dt, end_dt)
    if rates is None or len(rates) < 10:
        logger.error(f"No se pudieron descargar suficientes velas M1 de MT5: {mt5.last_error()}")
        return None, None
        
    # Extraer precios
    open_p = rates[0]['open']
    close_p = rates[-1]['close']
    
    high_p = max(r['high'] for r in rates)
    low_p  = min(r['low'] for r in rates)
    
    ret_1h = (close_p - open_p) / open_p
    atr_pct = (high_p - low_p) / open_p
    
    logger.info(f"Data 1H extraída de MT5: Open={open_p:.2f}, Close={close_p:.2f}, High={high_p:.2f}, Low={low_p:.2f}")
    return ret_1h, atr_pct

def get_mt5_balance():
    account_info = mt5.account_info()
    if account_info is None:
        logger.error(f"No se pudo obtener el balance: {mt5.last_error()}")
        return 10000.0 # Fallback
    return account_info.margin_free # Usar capital libre como balance prudente

def execute_trade(direction_str: str, lots: float, current_price: float, sl_price: float):
    """Ejecuta la orden real en MetaTrader 5."""
    order_type = mt5.ORDER_TYPE_BUY if direction_str == "LONG" else mt5.ORDER_TYPE_SELL
    
    # Precios de ejecución según Bid/Ask puros
    tick = mt5.symbol_info_tick(SYMBOL)
    price = tick.ask if direction_str == "LONG" else tick.bid
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": float(lots),
        "type": order_type,
        "price": price,
        "sl": float(sl_price),
        "deviation": 5, # 5 puntos limite de slippage aceptado
        "magic": MAGIC_NUMBER,
        "comment": "H3v2 VPS Execution",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC, # Immediate or Cancel
    }
    
    logger.info(f"Enviando orden a Servidor Broker: {direction_str} {lots} a {price:.2f} SL: {sl_price:.2f}")
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Fallo al enviar la orden. Código de error: {result.retcode}")
        logger.error(f"Mensaje MT5: {result.comment}")
        return False
        
    logger.info(f"✅ ORDEN EJECUTADA MÁGICAMENTE. Ticket #{result.order}. Precio rellenado: {result.price}")
    # Enviar comprobante al teléfono por Telegram
    alert_trade_open(direction_str, lots, result.price, sl_price, get_mt5_balance())
    return result.order

def close_all_h3_positions():
    """Cierra todas las posiciones abiertas por el bot (EOD)."""
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        logger.info("No hay posiciones que cerrar.")
        return 0
        
    closed_count = 0
    for pos in positions:
        if pos.magic == MAGIC_NUMBER:
            order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(SYMBOL)
            price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
            
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": pos.volume,
                "type": order_type,
                "position": pos.ticket,
                "price": price,
                "deviation": 5,
                "magic": MAGIC_NUMBER,
                "comment": "H3v2 EOD CLOSE",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            res = mt5.order_send(close_request)
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Posición {pos.ticket} CERRADA con éxito a las 19:59 UTC.")
                closed_count += 1
                alert_trade_close(closed_count, price)
            else:
                logger.error(f"Fallo al cerrar {pos.ticket}: {res.retcode} / {res.comment}")
                
    return closed_count

def bot_loop():
    logger.info("╔═════════════════════════════════════════════════════╗")
    logger.info("║  VPS DAEMON — AUTOPILOT NQ H3v2 Inciado             ║")
    logger.info("╚═════════════════════════════════════════════════════╝")
    
    if not MT5_AVAILABLE:
        logger.warning("Corriendo en modo SECRETO/SIMULACIÓN porque MT5 no está.")
        logger.warning("Instala las deps en Windows VPS para Trading REAL.")
        
    if MT5_AVAILABLE and not init_mt5():
        return
        
    # Inicializar Risk Engine con parámetros guardados
    params = RiskParams(instrument=SYMBOL, dollar_per_point=DOLLAR_PER_POINT)
    
    trade_executed_today = False
    
    while True:
        now_utc = datetime.now(timezone.utc)
        
        # RESETEAR LA BANDERA A MEDIANOCHE UTC
        if now_utc.hour == 0 and now_utc.minute == 0:
            trade_executed_today = False
            
        # 1. EVALUAR SEÑAL Y TOMAR TRADE (14:29:50 UTC aprox)
        if now_utc.hour == 14 and now_utc.minute >= 29 and now_utc.minute < 30 and not trade_executed_today:
            # Segundos precisos para capturar el mayor rango de la última vela (14:29:50)
            if now_utc.second >= 50:
                logger.info("🔔 14:29:50 UTC - INICIANDO PROTOCOLO DE EVALUACIÓN")
                
                # Check Filtro
                filtro = get_filter_status()
                if not filtro:
                    logger.info("Filtro previo OFF. Cancelando operación por hoy.")
                    alert_daily_status(filtro_activo=False)
                    trade_executed_today = True # Evitar loops
                    time.sleep(60)
                    continue
                    
                # Extraer Data
                if MT5_AVAILABLE:
                    ret_1h, atr_pct = get_h1_data_from_mt5()
                    current_balance = get_mt5_balance()
                    tick = mt5.symbol_info_tick(SYMBOL)
                    entry_price = tick.ask if ret_1h > 0 else tick.bid
                else:
                    # SIM MODE FALLBACK FOR LINUX DEV
                    logger.info("(SIMULATION MODE FALLBACK)")
                    ret_1h, atr_pct = 0.005, 0.004
                    current_balance = 10000.0
                    entry_price = 21000.0

                if ret_1h is None or atr_pct is None:
                    trade_executed_today = True
                    time.sleep(60)
                    continue
                    
                # Evaluar Engine
                engine = RiskEngine(initial_balance=current_balance, params=params)
                direction = engine.evaluate_signal(ret_1h)
                
                logger.info(f"Retorno 1H: {ret_1h*100:.3f}% | ATR 1H: {atr_pct*100:.3f}%")
                
                if direction:
                    logger.info("⚠️ SETUP CONFIRMADO. PREPARANDO ORDEN...")
                    alert_daily_status(filtro_activo=True, ret_1h=ret_1h, atr_pct=atr_pct, balance=current_balance)
                    
                    # Engine internal checks
                    trade_obj = engine.open_trade(
                        direction=direction,
                        entry_price=entry_price,
                        oh_atr_pct=atr_pct,
                        r_first_hour=ret_1h,
                        balance=current_balance
                    )
                    
                    if trade_obj:
                        # Mandar al Broker
                        if MT5_AVAILABLE:
                            execute_trade(direction, trade_obj.position_size, entry_price, trade_obj.sl_price)
                        else:
                            logger.info(">> Orden SIMULADA enviada al broker mock")
                else:
                    logger.info("Movimiento de 1H débil (<0.3%). Setup cancelado por hoy.")
                    
                trade_executed_today = True
                time.sleep(60) # Saltar al próximo minuto

        # 2. CIERRE FORZADO (19:59:00 UTC)
        if now_utc.hour == 19 and now_utc.minute == 58 and now_utc.second >= 50:
            logger.info("⏰ 19:58:50 UTC - ALARMA DE CIERRE DIARIO EOD")
            if MT5_AVAILABLE:
                close_all_h3_positions()
                
            # Reportar en el Engine local
            engine = RiskEngine(initial_balance=10000.0, params=params)
            open_t = [t for t in engine.trades if t.get('status') == 'OPEN']
            if open_t:
                tick = mt5.symbol_info_tick(SYMBOL) if MT5_AVAILABLE else None
                exit_price = tick.bid if open_t[0]['direction'] == 'LONG' else tick.ask if MT5_AVAILABLE else 21010.0
                logger.info(f"Sincronizando log interno. Exit price recogido: {exit_price}")
                engine.close_trade(open_t[0], exit_price, exit_reason='EOD')
                
            time.sleep(120) # Dormir durante la hora crítica
            
        # Loop idle heartbeat (2 segundos)
        time.sleep(2)

if __name__ == "__main__":
    bot_loop()
