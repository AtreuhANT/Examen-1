import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# ==============================================================================
# DATOS DEL EXPERIMENTO
# ==============================================================================
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

# ==============================================================================
# PARTE D — BÚSQUEDA DE RAÍCES: |Z|(f) = 150 Ω
# ==============================================================================
Z_TH = 150.0    # Umbral de impedancia (Ω)

# ---- CÓDIGO ------------------------------------------------------------------

cs  = CubicSpline(f_data, Z_data, bc_type='natural')
ds1 = cs.derivative(1)

def g(f):
    """Función objetivo: g(f) = S(f) - Z_th = 0"""
    return cs(f) - Z_TH

def dg(f):
    """Derivada de la función objetivo"""
    return ds1(f)

# — Método de Bisección —
def biseccion(func, a, b, tol=1e-6, max_iter=100):
    fa, fb = func(a), func(b)
    if fa * fb > 0:
        raise ValueError("No hay cambio de signo en [a, b].")
    history = []
    for _ in range(max_iter):
        c  = (a + b) / 2.0
        fc = func(c)
        history.append(c)
        if abs(fc) < tol or (b - a) / 2.0 < tol:
            return c, len(history), history
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return (a + b) / 2.0, len(history), history

# — Método de Newton-Raphson —
def newton_raphson(func, dfunc, x0, tol=1e-6, max_iter=100):
    x = x0
    history = []
    for _ in range(max_iter):
        fx  = func(x)
        dfx = dfunc(x)
        if abs(dfx) < 1e-12:
            break
        x_new = x - fx / dfx
        history.append(x_new)
        if abs(x_new - x) < tol or abs(fx) < tol:
            return x_new, len(history), history
        x = x_new
    return x, len(history), history

# — Raíz 1: entre 100 y 200 Hz (|Z| desciende de 152.3 a 142.0 Ω) —
r1_bis, n1_bis, h1_bis = biseccion(g, 100.0, 200.0)
r1_nr,  n1_nr,  h1_nr  = newton_raphson(g, dg, 110.0)

# — Raíz 2: entre 2160 y 2340 Hz (|Z| sube de 149.0 a 152.2 Ω) —
r2_bis, n2_bis, h2_bis = biseccion(g, 2160.0, 2340.0)
r2_nr,  n2_nr,  h2_nr  = newton_raphson(g, dg, 2200.0)

# — Sensibilidad en la raíz de alta frecuencia —
deriv_r2 = float(ds1(r2_nr))
sens_r2  = 1.0 / deriv_r2      # df/d|Z| en Hz/Ω

# — Gráfico D —
f_grid = np.linspace(100, 2730, 1000)
plt.figure(figsize=(8.5, 5))
plt.plot(f_grid, cs(f_grid), color='#7f77dd', linewidth=2,
         label='Spline Cúbico Natural')
plt.axhline(Z_TH, color='r', linestyle='--', linewidth=1.5,
            label=f'Umbral Z_th = {Z_TH:.0f} Ω')
plt.scatter([r1_nr, r2_nr], [Z_TH, Z_TH],
            color='black', edgecolor='red', s=60, zorder=5,
            label=f'Raíces (f₁ = {r1_nr:.2f} Hz, f₂ = {r2_nr:.2f} Hz)')
plt.title('Parte D: Frecuencias Límite donde |Z| = 150 Ω',
          fontsize=12, fontweight='bold')
plt.xlabel('Frecuencia f (Hz)')
plt.ylabel('Impedancia |Z| (Ω)')
plt.xlim(80, 2750)
plt.ylim(128, 162)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('grafico_D_raices.png', dpi=300)
plt.close()

# ---- EXPLICACIÓN -------------------------------------------------------------
print("=" * 80)
print(f" PARTE D: BÚSQUEDA DE RAÍCES  (|Z|(f) = {Z_TH:.0f} Ω)")
print("=" * 80)
print(" -> Gráfico guardado como: 'grafico_D_raices.png'")
print()
print(" Función objetivo:")
print(f"   g(f) = S(f) - {Z_TH:.0f} = 0")
print("   Donde S(f) es el spline cúbico natural construido sobre los datos.")
print()
print(" Resultados — Comparación de métodos:")
print(f"   {'Raíz':<8} | {'Método':<16} | {'f aproximada (Hz)':<22} | {'Iteraciones':<12} | {'|g(f)| final':<14}")
print("   " + "-" * 80)
print(f"   {'Raíz 1':<8} | {'Bisección':<16} | {r1_bis:<22.6f} | {n1_bis:<12} | {abs(g(r1_bis)):<14.2e}")
print(f"   {'Raíz 1':<8} | {'Newton-Raphson':<16} | {r1_nr:<22.6f} | {n1_nr:<12} | {abs(g(r1_nr)):<14.2e}")
print(f"   {'Raíz 2':<8} | {'Bisección':<16} | {r2_bis:<22.6f} | {n2_bis:<12} | {abs(g(r2_bis)):<14.2e}")
print(f"   {'Raíz 2':<8} | {'Newton-Raphson':<16} | {r2_nr:<22.6f} | {n2_nr:<12} | {abs(g(r2_nr)):<14.2e}")
print("   " + "-" * 80)
print()
print(" Interpretación física:")
print(f"   La impedancia |Z| ≤ 150 Ω solo en el rango de frecuencias:")
print(f"   [{r1_nr:.4f} Hz , {r2_nr:.4f} Hz]")
print(f"   Ancho de banda seguro estimado: {r2_nr - r1_nr:.4f} Hz")
print()
print(" Discusión de convergencia y robustez:")
print("   • Bisección:")
print("     Garantiza convergencia si g cambia de signo en [a,b].")
print("     Convergencia lineal: requiere ~20 iteraciones para tol = 1e-6.")
print("     No requiere derivada.")
print("   • Newton-Raphson:")
print("     Convergencia cuadrática: tipicamente 4–6 iteraciones para tol = 1e-6.")
print("     Requiere la derivada S'(f) y una aproximación inicial cercana a la raíz.")
print("     En este caso, la derivada analítica del spline permite funcionamiento")
print("     impecable del método.")
print()
print(f" Análisis de sensibilidad en la raíz f₂ ≈ {r2_nr:.4f} Hz:")
print(f"   d|Z|/df en la raíz   : {deriv_r2:.6f} Ω/Hz")
print(f"   Sensibilidad df/d|Z| : {sens_r2:.6f} Hz/Ω")
print()
print(f"   Interpretación: una variación de ±1.0 Ω en la medición de impedancia")
print(f"   desplaza la raíz estimada en ≈ {abs(sens_r2):.4f} Hz.")
print("   La pendiente pronunciada indica una raíz estable y poco sensible al")
print("   ruido de medición (pequeñas variaciones de |Z| no mueven drásticamente")
print("   la frecuencia límite del sistema).")
print("=" * 80)
