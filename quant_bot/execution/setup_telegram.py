import json
from pathlib import Path

# setup_telegram.py — Pequeño script interactivo para configurar Telegram localmente

CONFIG_DIR = Path(__file__).resolve().parent / "config"
CONFIG_FILE = CONFIG_DIR / "telegram_config.json"

def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║   📡 CONFIGURACIÓN DE NOTIFICACIONES TELEGRAM        ║")
    print("╚══════════════════════════════════════════════════════╝")
    print("Para recibir alertas de cada trade del VPS en tu móvil:")
    print("  1. Habla con @BotFather en Telegram y crea un Nuevo Bot.")
    print("  2. Copia el API Token proporcionado.")
    print("  3. Habla con @userinfobot (o similar) para sacar tu ID Numérico de Chat.\n")
    
    token = input("Ingresa el Bot Token: ").strip()
    if not token:
        print("Cancelado.")
        return
        
    chat_id = input("Ingresa tu Chat ID Numérico: ").strip()
    if not chat_id:
        print("Cancelado.")
        return
        
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    data = {
        "bot_token": token,
        "chat_id": chat_id
    }
    
    with open(CONFIG_FILE, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"\n✅ Guardado en: {CONFIG_FILE}")
    print("Para probarlo, ejecuta el bot_mt5 en modo demo o espera a las 14:29 UTC.")
    
if __name__ == "__main__":
    main()
