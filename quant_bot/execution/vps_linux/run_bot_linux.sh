#!/bin/bash
# run_bot_linux.sh — Ejecución Headless del Bot NQ H3v2 en un VPS Linux
# =====================================================================

set -e

# Configurar variables de entorno Wine / Monitor Virtual
export WINEARCH=win64
export WINEPREFIX=~/.wine_mt5

# Identificar la carpeta principal del proyecto
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$DIR")")")"

BOT_SCRIPT="$PROJECT_ROOT/quant_bot/execution/mt5_h3_bot.py"
LOG_OUT="$PROJECT_ROOT/quant_bot/execution/risk_data/vps_headless_out.log"
LOG_ERR="$PROJECT_ROOT/quant_bot/execution/risk_data/vps_headless_err.log"

echo "╔═════════════════════════════════════════════════════════════╗"
echo "║   🚀 EJECUCIÓN DEL VPS LINUX DAEMON CON XVFB & WINE         ║"
echo "╚═════════════════════════════════════════════════════════════╝"
echo "[*] Proyecto NQ100 detectado en: $PROJECT_ROOT"
echo "[*] Script a ejecutar en modo Windows: $BOT_SCRIPT"
echo ""

# xvfb-run abre un "monitor fantasma" para que Wine o MT5 puedan
# arrancar sus ventanas gráficas subyacentes sin crashear intentando 
# dibujar la pantalla (ya que en un VPS remoto Linux no hay monitor).
# Luego usamos "wine python" usando el intérprete de Python-Windows
# que instalamos en setup_vps.sh para que el paquete 'MetaTrader5.pyd' corra.

echo ">>> Inicializando Xvfb y lanzando el BOT Quant (Segundo Plano)..."
echo ">>> Logs de Salida: $LOG_OUT"
echo ">>> Logs Err/WINE: $LOG_ERR"

mkdir -p "$(dirname "$LOG_OUT")"

# Evitar crasheos de handles de streams en Python 3.11 + Wine
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export WINEDEBUG=-all

# Lanza el bot como proceso background (&) usando nohup para que no muera al cerrar ssh
# Usamos 'pythonw' porque no intenta inicializar streams de consola estándar, evitando el error [WinError 6]
nohup xvfb-run -a -s "-screen 0 1024x768x24" wine pythonw "$BOT_SCRIPT" < /dev/null > "$LOG_OUT" 2> "$LOG_ERR" &

# Guardar el ID del proceso
PID=$!
echo "✅ BOT H3v2 lanzado con éxito con PID $PID"
echo "--------------------------------------------------------"
echo "Para verificar cómo va tu bot o detenerlo:"
echo "   tail -f $LOG_OUT"
echo "   kill $PID (o killall wine python)"
echo "--------------------------------------------------------"
