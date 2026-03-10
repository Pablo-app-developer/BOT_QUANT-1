import json
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent / "config"
CONFIG_FILE = CONFIG_DIR / "mt5_config.json"

def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║   🏦 CONFIGURACIÓN DE CREDENCIALES MT5 (BROKER)      ║")
    print("╚══════════════════════════════════════════════════════╝")
    print("Esta información se guardará localmente y NO se subirá a GitHub.")
    
    try:
        account = input("Número de Cuenta (Login): ").strip()
        account = int(account)
    except ValueError:
        print("El número de cuenta debe ser un número entero. Cancelado.")
        return
        
    password = input("Contraseña: ").strip()
    if not password:
        print("Cancelado.")
        return
        
    server = input("Servidor del Broker (Exacto como sale en MT5, Ej: FTMO-Demo): ").strip()
    if not server:
        print("Cancelado.")
        return
        
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    data = {
        "account": account,
        "password": password,
        "server": server
    }
    
    with open(CONFIG_FILE, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"\n✅ Configuración MT5 guardada de forma segura en: {CONFIG_FILE}")

if __name__ == "__main__":
    main()
