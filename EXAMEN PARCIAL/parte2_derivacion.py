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
# PARTE 2 — DERIVACIÓN NUMÉRICA DE V(f)
# ==============================================================================

cs_V = CubicSpline(f_data, V_data, bc_type='natural')
ds1_V = cs_V.derivative(1)

h = 2.5  # espaciado uniforme entre nodos (kHz)

def idx_of(f_val):
    """Retorna el índice del nodo exactamente igual a f_val."""
    return int(np.where(np.isclose(f_data, f_val))[0][0])

# ── Diferencia centrada orden 2: f'(x) ≈ [f(x+h) - f(x-h)] / (2h) ────────────
def dc_ord2(ys, i, h):
    return (ys[i+1] - ys[i-1]) / (2*h)

# ── Diferencia centrada orden 4: f'(x) ≈ [-f(x+2h)+8f(x+h)-8f(x-h)+f(x-2h)] / (12h)
def dc_ord4(ys, i, h):
    return (-ys[i+2] + 8*ys[i+1] - 8*ys[i-1] + ys[i-2]) / (12*h)

# ── Diferencia progresiva orden 2: f'(x) ≈ [-3f(x)+4f(x+h)-f(x+2h)] / (2h) ──
def dp_ord2(ys, i, h):
    return (-3*ys[i] + 4*ys[i+1] - ys[i+2]) / (2*h)

# ── Puntos pedidos ─────────────────────────────────────────────────────────────
f_interior = [40.0, 70.0, 100.0]
f_extremo  = 10.0

resultados = {}
for fv in f_interior:
    i = idx_of(fv)
    r2 = dc_ord2(V_data, i, h)
    # Orden 4 solo si hay 2 nodos a cada lado
    if i >= 2 and i <= len(f_data) - 3:
        r4 = dc_ord4(V_data, i, h)
    else:
        r4 = None
    spl = float(ds1_V(fv))
    resultados[fv] = {'dc2': r2, 'dc4': r4, 'spline': spl}

# Extremo inferior
i0 = idx_of(f_extremo)
dp2_extremo = dp_ord2(V_data, i0, h)
spl_extremo = float(ds1_V(f_extremo))

# ── Resultados en terminal ─────────────────────────────────────────────────────
print("=" * 75)
print("  PARTE 2 — DERIVACIÓN NUMÉRICA: dV/df")
print("=" * 75)
print(f"\n  h = {h} kHz  (espaciado uniforme entre nodos)")
print()
print(f"  {'f (kHz)':<12} | {'Cent. O(h²)':<14} | {'Cent. O(h⁴)':<14} | {'Spline':<14} | {'|Δ| O2-Spl':<12}")
print("  " + "-" * 72)
for fv in f_interior:
    r = resultados[fv]
    dc4_str = f"{r['dc4']:.6f}" if r['dc4'] is not None else "N/A"
    delta = abs(r['dc2'] - r['spline'])
    print(f"  {fv:<12.1f} | {r['dc2']:<14.6f} | {dc4_str:<14} | {r['spline']:<14.6f} | {delta:<12.6f}")
print()
print(f"  Extremo inferior f = {f_extremo} kHz (fórmula PROGRESIVA orden 2):")
print(f"    dV/df (prog. O2) : {dp2_extremo:.6f} V/kHz")
print(f"    dV/df (spline)   : {spl_extremo:.6f} V/kHz")
print(f"    Diferencia       : {abs(dp2_extremo - spl_extremo):.6f} V/kHz")
print()
print("  Interpretación física del signo:")
print(f"    f=40.0 kHz: dV/df = {resultados[40.0]['dc2']:.4f} V/kHz  → V(f) DECRECE (zona de caída)")
print(f"    f=70.0 kHz: dV/df = {resultados[70.0]['dc2']:.4f} V/kHz  → V(f) CRECE (zona de recuperación)")
print(f"    f=100  kHz: dV/df = {resultados[100.0]['dc2']:.4f} V/kHz  → V(f) CRECE (zona estable)")
print(f"    f=10.0 kHz: dV/df = {dp2_extremo:.4f} V/kHz  → V(f) CRECE (zona ascendente inicial)")
print("=" * 75)
print("  -> Gráfico guardado como: grafico_P2_derivada.png")
print("=" * 75)

# ── Gráfico Parte 2 ────────────────────────────────────────────────────────────
f_grid = np.linspace(f_data[0], f_data[-1], 1000)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle('Parte 2: Derivación numérica de V(f)', fontsize=13, fontweight='bold')

ax1.plot(f_grid, cs_V(f_grid), color='#7f77dd', linewidth=2, label='Spline V(f)')
ax1.scatter(f_data, V_data, color='#1e3d59', s=25, zorder=5, label='Datos')
ax1.axhline(0, color='k', linestyle='--', alpha=0.4)
for fv in f_interior:
    ax1.axvline(fv, color='orange', linestyle=':', alpha=0.7)
ax1.axvline(f_extremo, color='green', linestyle=':', alpha=0.7)
ax1.set_ylabel('V(f)  (V)', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(True, linestyle=':', alpha=0.5)

ax2.plot(f_grid, ds1_V(f_grid), color='#ff6f3c', linewidth=2, label="dV/df (spline)")
ax2.axhline(0, color='k', linestyle='--', alpha=0.4)
# Marcar los puntos de diferencias finitas
for fv in f_interior:
    ax2.plot(fv, resultados[fv]['dc2'], 'gs', markersize=8, label=f'Cent. O2 @ {fv}')
    if resultados[fv]['dc4'] is not None:
        ax2.plot(fv, resultados[fv]['dc4'], 'b^', markersize=7, label=f'Cent. O4 @ {fv}')
ax2.plot(f_extremo, dp2_extremo, 'r*', markersize=11, label=f'Prog. O2 @ {f_extremo}')
ax2.set_ylabel('dV/df  (V/kHz)', fontsize=10)
ax2.set_xlabel('Frecuencia f (kHz)', fontsize=10)
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('grafico_P2_derivada.png', dpi=300)
plt.show()
