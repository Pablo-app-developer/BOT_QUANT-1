#!/bin/bash
export WINEPREFIX=/root/.wine_mt5
export WINEARCH=win64
export WINEDEBUG=-all
export PYTHONUNBUFFERED=1
export DISPLAY=:99

# Limpiar procesos previos agresivamente
ps -ef | grep -E 'wine|python|terminal64|Xvfb' | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
wineserver -k 2>/dev/null || true
rm -f /tmp/.X99-lock
rm -f /tmp/.X10*-lock
rm -rf /tmp/.xvfb-run.*

# Iniciar X virtual
Xvfb :99 -screen 0 1024x768x24 &
sleep 5

# Cerrador de diálogos en background (Cierra el Wizard de cuenta nueva, etc)
(
    while true; do
        xdotool key Escape 2>/dev/null
        sleep 15
    done
) &

# Iniciar MT5 primero
echo ">>> Iniciando Terminal MT5..."
wine "C:/mt5/terminal64.exe" &
sleep 90

# Ejecutar el bot con reinicio automático
while true; do
    echo ">>> Iniciando Bot Quant..."
    cd /root/.wine_mt5/drive_c/bot
    wine "C:/python311/python.exe" "C:/bot/quant_bot/execution/mt5_h3_bot.py"
    echo ">>> Bot terminó. Reiniciando en 10s..."
    sleep 10
done
