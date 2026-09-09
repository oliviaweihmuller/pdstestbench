#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 19:19:12 2026

@author: oliviaweihmuller
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

#%%
fs = 1000       # Hz
N = 1000        # Cantidad de muestras
Ps = 1          # W, potencia de la señal
VFS = 2         # V, rango del ADC +- 2V
f0 = fs / N     # Frecuencia fundamental (1 Hz)

#%%

def paso_cuantizacion(B, VFS):
    q = (2 * VFS) / (2**B)
    return q

def cuantizacion(x, q, VFS):
    x_recortada = np.clip(x, -VFS, VFS) 
    x_cuantizada = q * np.round(x_recortada / q)
    return x_cuantizada

def espectro_db(x, N, eps=1e-15):
    fft_x = np.fft.fft(x)
    pot_por_bin = (np.abs(fft_x)**2) / (N**2)
    return 10 * np.log10(pot_por_bin[:N//2] + eps)

def generar_senoidal_con_snr(Ps, f0, N, fs, B, VFS, kn):
    nn = np.arange(N)
    tt = nn / fs
    
    amp = np.sqrt(2 * Ps)   # despejado de P = A^2 / 2
    senal = amp * np.sin(2 * np.pi * f0 * tt)
    
    q = paso_cuantizacion(B, VFS)
    Pq = (q**2) / 12        # Potencia del ruido de cuantización
    Pn = kn * Pq            # Potencia del ruido analógico
    
    ruido = rng.normal(0, np.sqrt(Pn), N)
    x = senal + ruido
    
    return tt, senal, x, Pn, Pq, q


def simular(B, kn):
    # 1. Generar señal con ruido analógico y cuantizar
    tt, senal, senal_ruido_analogico, Pn, Pq, q = generar_senoidal_con_snr(
        Ps=Ps, f0=f0, N=N, fs=fs, B=B, VFS=VFS, kn=kn
    )
    
    senal_ruido_cuantizada = cuantizacion(x=senal_ruido_analogico, q=q, VFS=VFS)
    error_cuant = senal_ruido_cuantizada - senal_ruido_analogico

    # 2. Imprimir potencias medidas y teóricas
    print(f"=== RESULTADOS PARA B={B} bits, kn={kn} ===")
    print("Potencia señal (medida):", np.var(senal), "(teórica:", Ps, ")")
    print("Potencia ruido analógico Pn (medida):", np.var(senal_ruido_analogico - senal), "(teórica Pn:", Pn, ")")
    print("Potencia error cuantización Pq (medida):", np.var(error_cuant), "(teórica Pq:", Pq, ")")
    print("-" * 50)

    # 3. Gráfico en el tiempo
    plt.figure(figsize=(10, 4))
    plt.plot(tt, senal_ruido_analogico, color='red', linestyle=':', label='s + n (entrada al ADC)', zorder=2)
    plt.plot(tt, senal_ruido_cuantizada, color='lightseagreen', linestyle='-.', label='Salida ADC (cuantizada)', zorder=1)
    plt.plot(tt, senal, color='blue', linestyle='-', label='Señal analógica (s)', zorder=3)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [V]')
    plt.title(f'Dominio del tiempo (B={B} bits, kn={kn})')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # 4. Gráfico en frecuencia (Espectro + Pisos teóricos)
    freqs = np.fft.fftfreq(N, d=1/fs)[:N//2]
    db_cuant = espectro_db(senal_ruido_cuantizada, N)
    
    piso_analogico_db = 10 * np.log10(Pn / N)
    piso_digital_db   = 10 * np.log10((Pn + Pq) / N)
    
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, db_cuant, label='Salida ADC (Sq)')
    plt.axhline(piso_analogico_db, color='green', linestyle='--', label=f'Piso analógico teórico ({piso_analogico_db:.1f} dB)')
    plt.axhline(piso_digital_db, color='red', linestyle='--', label=f'Piso digital teórico ({piso_digital_db:.1f} dB)')
    plt.ylim(-100, 0)
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Densidad de Potencia [dB]')
    plt.title(f'Espectro - B={B} bits, kn={kn}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 5. Histograma del error de cuantización
    plt.figure(figsize=(7, 4))
    plt.hist(error_cuant, bins=10, alpha=0.7)
    plt.xlabel("Error de cuantización [V]")
    plt.ylabel("Frecuencia")
    plt.title(f"Histograma del error de cuantización (B={B} bits, kn={kn})")
    plt.tight_layout()
    plt.show()
#%%

# --- PARTE A ---
simular(B=4, kn=1)

# b) y bonus
simular(B=4, kn=0.1)
simular(B=4, kn=10)
simular(B=8, kn=1)
simular(B=8, kn=0.1)
simular(B=8, kn=10)
simular(B=16, kn=1)
simular(B=16, kn=0.1)
simular(B=16, kn=10)




