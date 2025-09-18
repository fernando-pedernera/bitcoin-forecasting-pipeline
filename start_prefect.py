#!/usr/bin/env python3
"""
Script para iniciar Prefect - Versión Windows
"""

import subprocess
import time
import os
import sys

def run_command(command, check=True):
    """Ejecuta un comando en Windows"""
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        return process.returncode == 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def start_prefect_server():
    """Inicia servidor Prefect en puerto 4200"""
    print("🚀 Iniciando servidor Prefect...")
    
    # Usar start para abrir en nueva ventana (opcional)
    server_process = subprocess.Popen([
        "prefect", "server", "start", "--port", "4200"
    ])
    
    time.sleep(10)  # Dar más tiempo en Windows
    return server_process

def main():
    print("=" * 60)
    print("🔧 CONFIGURACIÓN AUTOMÁTICA DE PREFECT (Windows)")
    print("=" * 60)
    
    # 1. Iniciar servidor
    server_process = start_prefect_server()
    
    # 2. Configurar API URL
    os.environ["PREFECT_API_URL"] = "http://localhost:4200/api"
    print(f"✅ PREFECT_API_URL = {os.environ['PREFECT_API_URL']}")
    
    # 3. Crear work pool (si no existe)
    print("🏊 Creando work pool 'local-pool'...")
    run_command(["prefect", "work-pool", "create", "--type", "process", "local-pool"])
    
    # 4. Iniciar worker
    print("👷 Iniciando worker...")
    worker_process = subprocess.Popen([
        "prefect", "worker", "start", "--pool", "local-pool"
    ])
    
    # Guardar PIDs para poder detener después
    with open("prefect_pids.txt", "w") as f:
        f.write(f"server:{server_process.pid}\n")
        f.write(f"worker:{worker_process.pid}\n")
    
    print("=" * 60)
    print("🎉 ¡CONFIGURACIÓN COMPLETADA!")
    print("📊 Prefect UI: http://localhost:4200")
    print("⏹️  Para detener: python stop_prefect.py")
    print("=" * 60)

if __name__ == "__main__":
    main()