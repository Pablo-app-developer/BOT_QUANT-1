import json
import urllib.request
import urllib.parse
import logging
from pathlib import Path

logger = logging.getLogger("VPS_Exec.Telegram")

CONFIG_FILE = Path(__file__).resolve().parent.parent / "execution" / "config" / "telegram_config.json"

def send_telegram_message(message: str) -> bool:
    """Envía un mensaje de texto plano o HTML por Telegram Bot."""
    if not CONFIG_FILE.exists():
        return False
        
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            
        bot_token = config.get("bot_token")
        chat_id = config.get("chat_id")
        
        if not bot_token or not chat_id:
            return False
            
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        req = urllib.request.Request(
            url, 
            data=urllib.parse.urlencode(data).encode('utf-8'),
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )
        
        with urllib.request.urlopen(req) as response:
            return response.status == 200
            
    except Exception as e:
        logger.error(f"Error enviando mensaje por Telegram: {e}")
        return False

def alert_trade_open(direction: str, lots: float, price: float, sl: float, balance: float):
    msg = (
        f"🚨 <b>H3v2 TRADE ABIERTO</b> 🚨\n\n"
        f"<b>Dirección:</b> {direction}\n"
        f"<b>Tamaño:</b> {lots} Lotes (0.5% Riesgo)\n"
        f"<b>Entrada Exacta:</b> {price:,.2f}\n"
        f"<b>Stop Loss Físico:</b> {sl:,.2f} pts\n"
        f"<b>Capital Activo:</b> ${balance:,.2f}\n\n"
        f"<i>El bot ha inyectado la orden en MT5 VPS.</i>"
    )
    send_telegram_message(msg)

def alert_trade_close(closed_count: int, exit_price: float):
    msg = (
        f"✅ <b>H3v2 CIERRE DIARIO (19:59)</b>\n\n"
        f"<b>Posiciones liquidadas:</b> {closed_count}\n"
        f"<b>Último precio cotizado:</b> {exit_price:,.2f}\n\n"
        f"<i>La orden Market de salida se ha ejecutado. Revisa tus beneficios en tu app FTMO/OANDA.</i>"
    )
    send_telegram_message(msg)
    
def alert_daily_status(filtro_activo: bool, ret_1h: float | None = None, atr_pct: float | None = None, balance: float | None = None):
    if not filtro_activo:
        msg = "⏸️ <b>H3v2 REPORTE DIARIO</b>\n\nFiltro día previo apagado. No hay operabilidad hoy. El servidor duerme hasta mañana."
    else:
        if ret_1h is not None:
            movimiento = "alcista" if ret_1h > 0 else "bajista"
            estado = "⚠️ SETUP CANCELADO" if abs(ret_1h) < 0.003 else "🔥 SETUP CONFIRMADO"
            msg = (
                f"🔎 <b>H3v2 REPORTE DE MERCADO (14:29 UTC)</b>\n\n"
                f"<b>Sesión 1H:</b> {ret_1h*100:+.3f}% ({movimiento})\n"
                f"<b>Volatilidad (ATR):</b> {atr_pct*100:.3f}%\n"
                f"<b>Balance VPS:</b> ${balance:,.2f}\n\n"
                f"<b>Veredicto:</b> {estado}"
            )
        else:
            msg = "⚡ <b>H3v2 FILTRO ACTIVO</b>\n\nEl día previo cerró negativo. Evaluando mercado inminente..."
            
    send_telegram_message(msg)
