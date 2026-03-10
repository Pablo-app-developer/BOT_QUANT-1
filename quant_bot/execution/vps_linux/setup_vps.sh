#!/bin/bash
# setup_vps.sh — Instalación Automatizada para correr MT5 y Bot Quant en Linux VPS (Ubuntu/Debian)
# ==============================================================================================

set -e

echo "╔═════════════════════════════════════════════════════════════╗"
echo "║   🚀 INSTALADOR VPS LINUX: MetaTrader 5 + Python Windows    ║"
echo "╚═════════════════════════════════════════════════════════════╝"

# 1. Instalar dependencias base del VPS Linux
echo ">> [1/4] Instalando paquetes de Linux (Xvfb, Wine, dependencias)..."
sudo dpkg --add-architecture i386
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    wine64 \
    wine32 \
    xvfb \
    winetricks \
    wget \
    cabextract \
    software-properties-common

# 2. Configurar el entorno de Wine
export WINEARCH=win64
export WINEPREFIX=~/.wine_mt5
echo ">> [2/4] Configurando el entorno virtual de Windows (Wine)..."
wineboot -u
# Esperar que Wine cree el prefix inicial
sleep 5

# 3. Descargar e Instalar Python para WINDOWS dentro de Wine (Requisito estricto del paquete MT5)
echo ">> [3/4] Instalando Python para Windows dentro de Linux..."
PYTHON_INSTALLER="python-3.11.8-amd64.exe"
if [ ! -f "$PYTHON_INSTALLER" ]; then
    wget "https://www.python.org/ftp/python/3.11.8/$PYTHON_INSTALLER" -O "$PYTHON_INSTALLER"
fi

# Instalar silenciosamente Python en Wine
# Añadimos Python al PATH de Windows/Wine
wine "$PYTHON_INSTALLER" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
sleep 10
rm "$PYTHON_INSTALLER"

# 4. Instalar las librerías Python necesarias (incluyendo el ansiado MetaTrader5.pyd de Windows)
echo ">> [4/4] Instalando MetaTrader5 y dependencias Quant en el Python de Windows..."
# Ejecutamos PIP del Python que acabamos de instalar en Wine
wine python -m pip install --upgrade pip
wine python -m pip install MetaTrader5 pandas numpy scipy pyarrow

echo ""
echo "✅ ¡ENTORNO LINUX CONFIGURADO CON ÉXITO!"
echo "--------------------------------------------------------"
echo "Para correr tu bot o ver MT5 gráficamente, necesitarás subir tu"
echo "ejecutable de MT5 (mt5setup.exe de FTMO o tu broker) e instalarlo:"
echo "   wine mt5setup.exe"
echo ""
echo "Luego, usa el script 'run_bot_linux.sh' para lanzar tu bot de"
echo "forma invisible (headless) usando Xvfb y Wine."
echo "--------------------------------------------------------"
