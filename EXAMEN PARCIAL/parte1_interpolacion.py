import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# ==============================================================================
# DATOS DEL ENSAYO — Telemetría biomédica (50 mediciones)
# ==============================================================================
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

N = len(f_data)

# ==============================================================================
# PARTE 1 — INTERPOLACIÓN: LAGRANGE GRADO 2 + SPLINE CÚBICO NATURAL
# ==============================================================================

# ── Función: Lagrange de grado 2 con los 3 nodos más cercanos ─────────────────
def lagrange_g2(x_eval, xs, ys):
    """Interpolación de Lagrange de grado 2 (3 puntos más cercanos a x_eval)."""
    dists = np.abs(xs - x_eval)
    idx   = np.argsort(dists)[:3]          # índices de los 3 más cercanos
    idx   = np.sort(idx)                    # mantener orden creciente en f
    x0, x1, x2 = xs[idx]
    y0, y1, y2 = ys[idx]
    L0 = ((x_eval-x1)*(x_eval-x2)) / ((x0-x1)*(x0-x2))
    L1 = ((x_eval-x0)*(x_eval-x2)) / ((x1-x0)*(x1-x2))
    L2 = ((x_eval-x0)*(x_eval-x1)) / ((x2-x0)*(x2-x1))
    return y0*L0 + y1*L1 + y2*L2, idx

# ── Splines cúbicos naturales ──────────────────────────────────────────────────
cs_V = CubicSpline(f_data, V_data, bc_type='natural')
cs_Z = CubicSpline(f_data, Z_data, bc_type='natural')

# ── Puntos de interés ──────────────────────────────────────────────────────────
puntos = [41.0, 73.0]
resultados = {}

for fp in puntos:
    V_lag, idx_V = lagrange_g2(fp, f_data, V_data)
    Z_lag, idx_Z = lagrange_g2(fp, f_data, Z_data)
    V_spl = float(cs_V(fp))
    Z_spl = float(cs_Z(fp))
    resultados[fp] = {
        'V_lag': V_lag, 'Z_lag': Z_lag,
        'V_spl': V_spl, 'Z_spl': Z_spl,
        'nodos_V': f_data[idx_V], 'nodos_Z': f_data[idx_Z]
    }

# ── Resultados en terminal ─────────────────────────────────────────────────────
print("=" * 70)
print("  PARTE 1 — INTERPOLACIÓN (LAGRANGE G2 vs SPLINE CÚBICO NATURAL)")
print("=" * 70)

for fp in puntos:
    r = resultados[fp]
    print(f"\n  f = {fp} kHz")
    print(f"  {'Cantidad':<25} | {'Lagrange G2':<15} | {'Spline Cúbico':<15} | {'|Δ|':<10}")
    print("  " + "-" * 70)
    print(f"  {'V(' + str(fp) + ' kHz)':<25} | {r['V_lag']:<15.6f} | {r['V_spl']:<15.6f} | {abs(r['V_lag']-r['V_spl']):<10.6f}")
    print(f"  {'|Z|(' + str(fp) + ' kHz)':<25} | {r['Z_lag']:<15.6f} | {r['Z_spl']:<15.6f} | {abs(r['Z_lag']-r['Z_spl']):<10.6f}")
    print(f"  Nodos usados (V) : f = {r['nodos_V']} kHz")
    print(f"  Nodos usados (Z) : f = {r['nodos_Z']} kHz")

print("\n" + "=" * 70)
print("  -> Gráfico guardado como: grafico_P1_interpolacion.png")
print("=" * 70)

# ── Gráfico Parte 1 ────────────────────────────────────────────────────────────
f_grid = np.linspace(f_data[0], f_data[-1], 1000)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Parte 1: Interpolación de V(f) e |Z|(f)', fontsize=13, fontweight='bold')

# — V(f) —
ax1.scatter(f_data, V_data, color='#1e3d59', s=30, zorder=5, label='Datos medidos')
ax1.plot(f_grid, cs_V(f_grid), color='#7f77dd', linewidth=2, label='Spline cúbico natural')
ax1.axhline(0, color='k', linestyle='--', alpha=0.4)
for fp in puntos:
    r = resultados[fp]
    ax1.plot(fp, r['V_lag'], 'g^', markersize=9, zorder=6, label=f'Lagrange f={fp}')
    ax1.plot(fp, r['V_spl'], 'rv', markersize=9, zorder=6, label=f'Spline f={fp}')
ax1.set_ylabel('V(f)  (V)', fontsize=10)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(fontsize=8, loc='upper right')

# — |Z|(f) —
ax2.scatter(f_data, Z_data, color='#1e3d59', s=30, zorder=5, label='Datos medidos')
ax2.plot(f_grid, cs_Z(f_grid), color='#ff6f3c', linewidth=2, label='Spline cúbico natural')
for fp in puntos:
    r = resultados[fp]
    ax2.plot(fp, r['Z_lag'], 'g^', markersize=9, zorder=6, label=f'Lagrange f={fp}')
    ax2.plot(fp, r['Z_spl'], 'rv', markersize=9, zorder=6, label=f'Spline f={fp}')
ax2.set_ylabel('|Z|(f)  (Ω)', fontsize=10)
ax2.set_xlabel('Frecuencia f (kHz)', fontsize=10)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(fontsize=8, loc='lower right')

plt.tight_layout()
plt.savefig('grafico_P1_interpolacion.png', dpi=300)
plt.show()
