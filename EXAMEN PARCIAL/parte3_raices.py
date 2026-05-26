import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

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

# ==============================================================================
# PARTE 3 — RAÍCES: CRUCES POR CERO DE V(f)
# ==============================================================================

cs_V = CubicSpline(f_data, V_data, bc_type='natural')

# ── 1. Identificar intervalos con cambio de signo ─────────────────────────────
cruces = []
for i in range(len(V_data) - 1):
    if V_data[i] * V_data[i+1] < 0:
        cruces.append((i, f_data[i], V_data[i], f_data[i+1], V_data[i+1]))

print("=" * 70)
print("  PARTE 3 — BÚSQUEDA DE RAÍCES: CRUCES POR CERO DE V(f)")
print("=" * 70)
print(f"\n  Intervalos con cambio de signo detectados: {len(cruces)}")
for k, (i, fa, Va, fb, Vb) in enumerate(cruces, 1):
    print(f"    Cruce {k}: V({fa:.1f}) = {Va:.3f} V  →  V({fb:.1f}) = {Vb:.3f} V")

# ── 2. Método de Bisección ────────────────────────────────────────────────────
def biseccion(func, a, b, tol=1e-8, max_iter=200):
    fa = func(a)
    if fa * func(b) > 0:
        raise ValueError("Sin cambio de signo en [a, b].")
    history = [a, b]
    for _ in range(max_iter):
        c  = (a + b) / 2.0
        fc = func(c)
        history.append(c)
        if abs(fc) < tol or (b - a) / 2.0 < tol:
            return c, len(history) - 2, history
        if fa * fc < 0:
            b = c
        else:
            a, fa = c, fc
    return (a + b) / 2.0, len(history) - 2, history

# ── 3. Bisección sobre los datos crudos (interpolación lineal por tramo) ───────
def V_linear(f_eval, fa, Va, fb, Vb):
    """Interpolación lineal local para bisección sobre datos."""
    return Va + (Vb - Va) * (f_eval - fa) / (fb - fa)

raices_bis = []
raices_spl = []

for k, (i, fa, Va, fb, Vb) in enumerate(cruces, 1):
    # Bisección usando spline (más precisa)
    r_spl, n_spl, _ = biseccion(lambda f: float(cs_V(f)), fa, fb)
    # Bisección usando interpolación lineal entre los dos nodos
    r_lin, n_lin, _ = biseccion(lambda f: V_linear(f, fa, Va, fb, Vb), fa, fb)
    raices_bis.append(r_lin)
    raices_spl.append(r_spl)

    print(f"\n  ─── Cruce {k}: intervalo [{fa:.1f}, {fb:.1f}] kHz ───")
    print(f"    Bisección (interp. lineal)  : f_raíz = {r_lin:.8f} kHz  ({n_lin} iter.)")
    print(f"    Bisección (spline cúbico)   : f_raíz = {r_spl:.8f} kHz  ({n_spl} iter.)")
    print(f"    Diferencia entre métodos    : {abs(r_spl - r_lin):.6f} kHz")
    print(f"    V_spline en la raíz         : {float(cs_V(r_spl)):.2e} V  (≈ 0)")

print()
print("  Interpretación:")
if len(raices_spl) >= 1:
    print(f"    1ª raíz ≈ {raices_spl[0]:.4f} kHz → primera vez que V(f) = 0 (alarma activada)")
if len(raices_spl) >= 2:
    print(f"    2ª raíz ≈ {raices_spl[1]:.4f} kHz → V(f) cruza cero nuevamente (alarma desactivada?)")
print("=" * 70)
print("  -> Gráfico guardado como: grafico_P3_raices.png")
print("=" * 70)

# ── Gráfico Parte 3 ────────────────────────────────────────────────────────────
f_grid = np.linspace(f_data[0], f_data[-1], 2000)

plt.figure(figsize=(10, 5))
plt.plot(f_grid, cs_V(f_grid), color='#7f77dd', linewidth=2, label='Spline cúbico V(f)')
plt.scatter(f_data, V_data, color='#1e3d59', s=25, zorder=5, label='Datos medidos')
plt.axhline(0, color='k', linewidth=1.2, linestyle='--', label='V = 0 (umbral de alarma)')

# Marcar raíces
for k, (r_bis, r_spl) in enumerate(zip(raices_bis, raices_spl), 1):
    plt.axvline(r_bis, color='green', linestyle=':', alpha=0.8, linewidth=1.2,
                label=f'Bisección raíz {k}: {r_bis:.3f} kHz')
    plt.plot(r_spl, 0, 'ro', markersize=10, zorder=6,
             label=f'Spline raíz {k}: {r_spl:.4f} kHz')

# Sombrear zona de V < 0
f_fine = np.linspace(f_data[0], f_data[-1], 2000)
V_fine = cs_V(f_fine)
plt.fill_between(f_fine, V_fine, 0, where=(V_fine < 0),
                 alpha=0.15, color='red', label='Zona V < 0 (alarma activa)')

plt.title('Parte 3: Cruces por Cero de V(f) — Detección de Alarma',
          fontsize=12, fontweight='bold')
plt.xlabel('Frecuencia f (kHz)', fontsize=10)
plt.ylabel('V(f)  (V)', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig('grafico_P3_raices.png', dpi=300)
plt.show()
