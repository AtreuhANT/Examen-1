import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator
import random

random.seed(42)
np.random.seed(42)

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

# ── Funciones ──────────────────────────────────────────────────────────────────
def piecewise_lagrange(x_eval, xs, ys, degree):
    results = []
    for x in np.atleast_1d(x_eval):
        idx   = np.argmin(np.abs(xs - x))
        half  = degree // 2
        start = max(0, idx - half)
        end   = min(len(xs) - 1, start + degree)
        start = max(0, end - degree)
        poly  = BarycentricInterpolator(xs[start:end+1], ys[start:end+1])
        results.append(poly(x))
    return np.array(results) if np.ndim(x_eval) > 0 else results[0]

def global_vandermonde(x_eval, xs, ys):
    V     = np.vander(xs, increasing=True)
    coefs = np.linalg.solve(V, ys)
    return np.vander(x_eval, N=len(xs), increasing=True) @ coefs, np.linalg.cond(V)

def leave_one_out(xs, ys, degree, num_points=5):
    indices = sorted(random.sample(range(len(xs)), num_points))
    results = []
    for idx in indices:
        pred    = piecewise_lagrange(xs[idx], np.delete(xs, idx), np.delete(ys, idx), degree)
        abs_err = abs(pred - ys[idx])
        results.append({'f': xs[idx], 'real': ys[idx], 'pred': pred,
                        'abs_err': abs_err, 'rel_err': abs_err / ys[idx] * 100})
    return results

# ── Cálculo ────────────────────────────────────────────────────────────────────
f_grid = np.linspace(100, 2730, 1000)
Z_vander, cond_v = global_vandermonde(f_grid, f_data, Z_data)
Z_p5  = piecewise_lagrange(f_grid, f_data, Z_data, degree=5)
Z_p10 = piecewise_lagrange(f_grid, f_data, Z_data, degree=10)
Z_p15 = piecewise_lagrange(f_grid, f_data, Z_data, degree=15)

val_p5_1000 = piecewise_lagrange(1000.0, f_data, Z_data, degree=5)
loo         = leave_one_out(f_data, Z_data, degree=5)

# ── Resultados en terminal ─────────────────────────────────────────────────────
print("=" * 65)
print("  PARTE B1 — INTERPOLACIÓN POLINÓMICA (LAGRANGE + VANDERMONDE)")
print("=" * 65)
print(f"  Condición de Vandermonde (grado 29) : {cond_v:.4e}")
print(f"  -> Matriz severamente mal condicionada (κ >> 10^12)")
print()
print(f"  |Z| interpolada en f = 1000 Hz:")
print(f"    Polinomio Grado 5  (a trozos) : {val_p5_1000:.6f} Ω")
print()
print("  Validación Leave-One-Out (LOO) — Grado 5:")
print(f"  {'Punto retirado':<20} | {'Real (Ω)':<12} | {'Pred. (Ω)':<12} | {'Err. abs.':<10} | {'Err. rel.':<10}")
print("  " + "-" * 72)
for r in loo:
    print(f"  ({r['f']:.0f} Hz){'':<12} | {r['real']:<12.4f} | {r['pred']:<12.4f} | {r['abs_err']:<10.4f} | {r['rel_err']:<10.4f}%")
print("  " + "-" * 72)
avg = np.mean([r['rel_err'] for r in loo])
mx  = max(r['rel_err'] for r in loo)
print(f"  Error relativo promedio : {avg:.6f}%")
print(f"  Error relativo máximo   : {mx:.6f}%")
print("=" * 65)
print("  -> Gráfico guardado como: grafico_B1_runge.png")
print("=" * 65)

# ── Gráfico B1 ─────────────────────────────────────────────────────────────────
Z_vander_clip = np.clip(Z_vander, 90, 200)

plt.figure(figsize=(10, 6))
plt.scatter(f_data, Z_data, color='#1e3d59', s=35, label='Datos medidos', zorder=5)
plt.plot(f_grid, Z_p5,  color='#17b978', linewidth=2,    label='Grado 5 a trozos')
plt.plot(f_grid, Z_p10, color='#ff6f3c', linestyle='--', linewidth=1.5, label='Grado 10 a trozos')
plt.plot(f_grid, Z_p15, color='#ffc93c', linestyle='-.', linewidth=1.5, label='Grado 15 a trozos')
plt.plot(f_grid, Z_vander_clip, color='#ff4b5c', linestyle=':', linewidth=1.8,
         label='Grado 29 (Vandermonde — Runge)')
plt.title('Parte B1: Fenómeno de Runge vs Polinomiales a Trozos',
          fontsize=12, fontweight='bold')
plt.xlabel('Frecuencia f (Hz)')
plt.ylabel('Impedancia |Z| (Ω)')
plt.ylim(118, 175)
plt.xlim(50, 2800)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3)
plt.tight_layout()
plt.savefig('grafico_B1_runge.png', dpi=300)
plt.show()
