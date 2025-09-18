#!/usr/bin/env python3
"""
Script para detener Prefect - Versión Windows
"""

import os
import signal
import subprocess
import time
import psutil

def kill_process(pid, name):
    """Mata un proceso por PID en Windows"""
    try:
        process = psutil.Process(pid)
        process.terminate()
        print(f"✅ {name} (PID: {pid}) detenido")
        time.sleep(2)
    except psutil.NoSuchProcess:
        print(f"⚠️  {name} (PID: {pid}) no encontrado")
    except psutil.AccessDenied:
        print(f"❌ No se tiene permisos para detener {name} (PID: {pid})")
    except Exception as e:
        print(f"❌ Error al detener {name} (PID: {pid}): {e}")

def kill_process_by_name(name):
    """Mata procesos por nombre en Windows"""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if name.lower() in cmdline.lower():
                    proc.terminate()
                    print(f"✅ Proceso '{name}' (PID: {proc.info['pid']}) detenido")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        print(f"❌ Error al buscar procesos: {e}")

def main():
    print("=" * 50)
    print("🛑 DETENIENDO SERVICIOS PREFECT (Windows)")
    print("=" * 50)
    
    # Matar procesos por PID si existe el archivo
    if os.path.exists("prefect_pids.txt"):
        print("📋 Deteniendo procesos por PID...")
        with open("prefect_pids.txt", "r") as f:
            for line in f:
                if ":" in line:
                    name, pid_str = line.strip().split(":")
                    try:
                        kill_process(int(pid_str), name)
                    except ValueError:
                        continue
        os.remove("prefect_pids.txt")
    
    # Forzar cierre de procesos Prefect por nombre
    print("🔍 Buscando procesos residuales...")
    kill_process_by_name("prefect server")
    kill_process_by_name("prefect worker")
    
    # Limpiar variable de entorno
    if "PREFECT_API_URL" in os.environ:
        del os.environ["PREFECT_API_URL"]
        print("✅ Variable de entorno limpiada")
    
    # Forzar cierre de procesos en puerto 4200
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        for line in result.stdout.split('\n'):
            if ':4200' in line and 'LISTENING' in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    subprocess.run(["taskkill", "/PID", pid, "/F"], check=False)
                    print(f"✅ Proceso en puerto 4200 (PID: {pid}) eliminado")
    except:
        pass
    
    print("=" * 50)
    print("✅ Todos los servicios detenidos")
    print("=" * 50)

if __name__ == "__main__":
    main()

