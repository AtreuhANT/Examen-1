import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt

# Datos experimentales
f_data = np.array([
    100, 120, 145, 170, 200, 235, 270, 310, 355, 405, 460, 520, 585, 655, 730,
    810, 895, 985, 1080, 1180, 1290, 1410, 1540, 1680, 1830, 1990, 2160, 2340,
    2530, 2730
], dtype=float)

Z_data = np.array([
    152.3, 149.1, 146.8, 144.9, 142.0, 139.5, 137.9, 136.1, 134.8, 133.6, 132.7,
    131.9, 131.4, 131.1, 130.9, 131.0, 131.3, 131.9, 132.7, 133.8, 135.2, 136.9,
    138.9, 141.1, 143.5, 146.1, 149.0, 152.2, 155.6, 159.2
], dtype=float)

# ── Resultados en terminal ─────────────────────────────────────────────────────
idx_min = np.argmin(Z_data)
idx_max = np.argmax(Z_data)

print("=" * 60)
print("  PARTE A — ANÁLISIS EXPLORATORIO DE BIOIMPEDANCIA")
print("=" * 60)
print(f"  Puntos experimentales  : {len(f_data)}")
print(f"  Rango de frecuencia    : {f_data[0]:.0f} Hz — {f_data[-1]:.0f} Hz")
print()
print(f"  |Z| máxima  : {Z_data[idx_max]:.1f} Ω   @ f = {f_data[idx_max]:.0f} Hz")
print(f"  |Z| mínima  : {Z_data[idx_min]:.1f} Ω   @ f = {f_data[idx_min]:.0f} Hz")
print(f"  |Z| media   : {Z_data.mean():.4f} Ω")
print(f"  Desv. típica: {Z_data.std():.4f} Ω")
print()
print("  Tabla de datos (f, |Z|):")
print(f"  {'f (Hz)':<10} | {'|Z| (Ω)':<10}")
print("  " + "-" * 23)
for f, z in zip(f_data, Z_data):
    print(f"  {f:<10.0f} | {z:<10.1f}")
print("=" * 60)
print("  -> Gráfico guardado como: grafico_A_datos.png")
print("=" * 60)

# ── Gráfico ───────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.scatter(f_data, Z_data, color='#1e3d59', edgecolor='k', s=45,
            label='Mediciones (f, |Z|)')
plt.plot(f_data, Z_data, '--', color='#17b978', alpha=0.5,
         label='Tendencia visual')
plt.plot(f_data[idx_min], Z_data[idx_min], 'rv', markersize=10,
         label=f'Mínimo: {Z_data[idx_min]:.1f} Ω @ {f_data[idx_min]:.0f} Hz')
plt.title('Parte A: Análisis Exploratorio de Bioimpedancia |Z|(f) a 37 °C',
          fontsize=12, fontweight='bold')
plt.xlabel('Frecuencia f (Hz)', fontsize=10)
plt.ylabel('Magnitud de Impedancia |Z| (Ω)', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('grafico_A_datos.png', dpi=300)
plt.show()
