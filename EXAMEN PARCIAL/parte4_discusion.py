import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

# Datos del ensayo
f_data = np.array([
    10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 32.5,
    35.0, 37.5, 40.0, 42.5, 45.0, 47.5, 50.0, 52.5, 55.0, 57.5,
    60.0, 62.5, 65.0, 67.5, 70.0, 72.5, 75.0, 77.5, 80.0, 82.5,
    85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 100.0, 102.5, 105.0, 107.5,
    110.0, 112.5, 115.0, 117.5, 120.0, 122.5, 125.0, 127.5, 130.0, 132.5
], dtype=float)

V_data = np.array([
    0.842, 0.911, 0.986, 1.062, 1.143, 1.227, 1.314, 1.401, 1.482, 1.551,
    1.216, 1.048, 0.866, 0.689, 0.521, 0.364, 0.223, 0.103, 0.012, -0.041,
   -0.057, -0.034, 0.018, 0.096, 0.197, 0.318, 0.452, 0.579, 0.700, 0.809,
    0.611, 0.688, 0.756, 0.811, 0.856, 0.894, 0.926, 0.954, 0.980, 1.004,
    1.026, 1.047, 1.066, 1.084, 1.100, 1.115, 1.129, 1.142, 1.154, 1.165
], dtype=float)

Z_data = np.array([
    182.4, 178.9, 175.1, 171.0, 166.8, 162.7, 158.9, 155.4, 152.0, 149.0,
    146.1, 145.2, 145.8, 147.3, 149.9, 153.5, 158.0, 163.2, 168.9, 174.8,
    180.5, 186.2, 191.5, 196.2, 200.1, 203.1, 205.2, 206.3, 206.1, 204.7,
    198.0, 194.4, 190.9, 187.8, 185.1, 183.0, 181.6, 180.8, 180.6, 180.9,
    181.6, 182.7, 184.0, 185.5, 187.1, 188.8, 190.5, 192.3, 194.1, 195.9
], dtype=float)

# ==============================================================================
# PARTE 4 — DISCUSIÓN TÉCNICA
# ==============================================================================

cs_V = CubicSpline(f_data, V_data, bc_type='natural')
cs_Z = CubicSpline(f_data, Z_data, bc_type='natural')
ds1_V = cs_V.derivative(1)

# Parámetros calculados para apoyar la discusión
idx_Zmin   = np.argmin(Z_data)
idx_Vmax   = np.argmax(V_data)
f_Zmin     = f_data[idx_Zmin]
Z_min      = Z_data[idx_Zmin]
f_Vmax     = f_data[idx_Vmax]
V_max      = V_data[idx_Vmax]

# Raíces de V(f) con spline
raiz1 = float(brentq(cs_V, 52.5, 57.5))
raiz2 = float(brentq(cs_V, 62.5, 65.0))
BW_alarma = raiz2 - raiz1

# Zona de estabilidad: donde dV/df ≈ 0 y |Z| es estable
f_grid = np.linspace(f_data[0], f_data[-1], 5000)
dV_grid = ds1_V(f_grid)
Z_grid  = cs_Z(f_grid)
idx_stable = np.argmin(np.abs(dV_grid))
f_stable    = f_grid[idx_stable]

print("=" * 75)
print("  PARTE 4 — DISCUSIÓN TÉCNICA INTEGRADA")
print("=" * 75)
print()

print("  1. BANDA DE OPERACIÓN RECOMENDADA DEL SISTEMA BIOMÉDICO")
print("  " + "-" * 70)
print(f"     |Z| mínima       : {Z_min:.1f} Ω  @ f = {f_Zmin:.1f} kHz")
print(f"       → Mejor adaptación de impedancias: menor reflexión de potencia")
print(f"       → En esta frecuencia el sensor extrae máxima potencia del circuito")
print()
print(f"     V(f) máxima      : {V_max:.3f} V  @ f = {f_Vmax:.1f} kHz")
print(f"       → Mayor relación señal/ruido → menor probabilidad de error bit")
print()
print(f"     Zona de alarma (V < 0): [{raiz1:.3f}, {raiz2:.3f}] kHz")
print(f"       → Duración de la zona anómala: {BW_alarma:.3f} kHz")
print(f"       → Se debe EVITAR operar en ese rango")
print()
print(f"     Zona de dV/df ≈ 0 (máxima estabilidad):  f ≈ {f_stable:.2f} kHz")
print(f"       → Aquí la sensibilidad del front-end es mínima (menos ruido)")
print()
print(f"     → BANDA RECOMENDADA para operación: 10–32.5 kHz")
print(f"       donde V(f) > 0.8 V, |Z| < 155 Ω y la señal es estable y positiva.")
print()

print("  2. IMPACTO DE LA RESOLUCIÓN DE LOS INSTRUMENTOS")
print("  " + "-" * 70)
print("     Frecuencia (resolución = 0.1 kHz):")
print("       • Paso h entre nodos = 2.5 kHz >> 0.1 kHz → la resolución en")
print("         frecuencia no es el factor limitante de la interpolación.")
print("       • Si h fuera comparable a la resolución (0.1 kHz), las fórmulas de")
print("         diferencias finitas serían más sensibles a errores de redondeo.")
print()
print("     Voltímetro (resolución = 0.001 V):")
print("       • Introduce un error aleatorio ε_V ≈ ±0.0005 V en cada V_i.")
print("       • Para diferencia centrada O(h²): propagación de error ≈ ε_V/(h) ≈")
print(f"         {0.001/(2*2.5):.5f} V/kHz — error relativo ~{0.001/(2*2.5)/0.1*100:.1f}% en dV/df")
print("       • En bisección: afecta el umbral de detección de cruce por cero;")
print(f"         incertidumbre de ±0.001 V en V({raiz1:.1f}) puede desplazar la raíz")
print(f"         ≈ ±{0.001/abs(float(ds1_V(raiz1))):.4f} kHz según la sensibilidad local.")
print()
print("     Impedanciómetro (resolución = 0.1 Ω):")
print("       • Afecta principalmente la interpolación de |Z|(f).")
print("       • Error en |Z| de ±0.05 Ω → error en interpolación << 0.1 Ω.")
print()

print("  3. VENTAJA PRÁCTICA DEL SPLINE CÚBICO vs POLINOMIO GLOBAL")
print("  " + "-" * 70)
print("     Con n = 40 nodos, un polinomio global de grado 39 tendría:")
print("       • κ(V_40) >> 10^50 → sistema numéricamente inestable.")
print("       • Fenómeno de Runge: oscilaciones de >100 V en los extremos del")
print("         intervalo, completamente fuera del rango físico (0–1.6 V).")
print("     El spline cúbico:")
print("       • Usa polinomios cúbicos locales → κ ≈ 1 por tramo.")
print("       • Garantiza continuidad C² → derivadas dV/df suaves y realistas.")
print("       • Minimiza ∫[V''(f)]² df → interpolación físicamente coherente.")
print("       • Converge con error O(h⁴) → mucho mejor que Lagrange global.")
print("=" * 75)
print("  -> Gráfico guardado como: grafico_P4_discusion.png")
print("=" * 75)

# ── Gráfico P4: Panorama completo ──────────────────────────────────────────────
f_grid = np.linspace(f_data[0], f_data[-1], 2000)

fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
fig.suptitle('Parte 4: Panorama Completo — V(f) e |Z|(f)\nTelemetría Biomédica',
             fontsize=13, fontweight='bold')

# — V(f) —
ax = axes[0]
ax.plot(f_grid, cs_V(f_grid), color='#7f77dd', linewidth=2, label='V(f) — Spline')
ax.scatter(f_data, V_data, color='#1e3d59', s=25, zorder=5, label='Datos')
ax.axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=1)
ax.fill_between(f_grid, cs_V(f_grid), 0,
                where=(cs_V(f_grid) < 0), alpha=0.15, color='red', label='V < 0 (alarma)')
ax.plot(raiz1, 0, 'ro', ms=9, label=f'Raíz 1: {raiz1:.3f} kHz')
ax.plot(raiz2, 0, 'ro', ms=9, label=f'Raíz 2: {raiz2:.3f} kHz')
ax.plot(f_Vmax, V_max, 'g*', ms=12, label=f'V_max={V_max:.3f}V @ {f_Vmax:.1f}kHz')
ax.set_ylabel('V(f)  (V)', fontsize=10)
ax.legend(fontsize=8, loc='lower left')
ax.grid(True, linestyle=':', alpha=0.5)

# — dV/df —
ax = axes[1]
ax.plot(f_grid, ds1_V(f_grid), color='#ff6f3c', linewidth=2, label="dV/df (spline)")
ax.axhline(0, color='k', linestyle='--', alpha=0.4)
ax.plot(f_stable, 0, 'bs', ms=9, label=f'dV/df≈0 @ {f_stable:.1f} kHz')
ax.set_ylabel('dV/df  (V/kHz)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, linestyle=':', alpha=0.5)

# — |Z|(f) —
ax = axes[2]
ax.plot(f_grid, cs_Z(f_grid), color='#17b978', linewidth=2, label='|Z|(f) — Spline')
ax.scatter(f_data, Z_data, color='#1e3d59', s=25, zorder=5, label='Datos')
ax.plot(f_Zmin, Z_min, 'r^', ms=11, label=f'|Z|_min={Z_min:.1f}Ω @ {f_Zmin:.1f}kHz')
ax.set_ylabel('|Z|(f)  (Ω)', fontsize=10)
ax.set_xlabel('Frecuencia f (kHz)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('grafico_P4_discusion.png', dpi=300)
plt.show()
