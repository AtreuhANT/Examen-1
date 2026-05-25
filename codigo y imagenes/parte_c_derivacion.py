import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

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
# PARTE C — DERIVACIÓN NUMÉRICA SOBRE EL SPLINE
# ==============================================================================

# ---- CÓDIGO ------------------------------------------------------------------

# — Construcción del spline y sus derivadas analíticas —
cs  = CubicSpline(f_data, Z_data, bc_type='natural')
ds1 = cs.derivative(1)    # Primera derivada  d|Z|/df
ds2 = cs.derivative(2)    # Segunda derivada  d²|Z|/df²

# — Evaluación de derivadas sobre malla fina —
f_grid = np.linspace(100, 2730, 1000)
dZ_df  = ds1(f_grid)
d2Z_df2 = ds2(f_grid)

# — Localización del mínimo: raíz de ds1 en (700, 900) Hz —
f_min   = brentq(ds1, 700.0, 900.0)
Z_min   = float(cs(f_min))
d2Z_min = float(ds2(f_min))

# — Derivadas en todos los nodos experimentales —
d1_pts = ds1(f_data)
d2_pts = ds2(f_data)

# — Diferencias finitas centradas (verificación independiente) —
# Usamos los 30 puntos originales para estimar d|Z|/df en los nodos internos
dZ_fd = np.gradient(Z_data, f_data)    # numpy usa esquema centrado en interiores

# — Gráfico C —
plt.figure(figsize=(8, 5))
plt.plot(f_grid, dZ_df, color='#ff6f3c', linewidth=2.5,
         label='d|Z|/df  (derivada analítica del spline)')
plt.plot(f_data, dZ_fd, 'o', color='#1e3d59', markersize=4,
         label='d|Z|/df  (diferencias finitas centradas)')
plt.axhline(0, color='k', linestyle='--', alpha=0.5)
plt.plot(f_min, 0, 'ro', markersize=7,
         label=f'Mínimo en f = {f_min:.4f} Hz')
plt.title('Parte C: Primera Derivada d|Z|/df vs Frecuencia',
          fontsize=12, fontweight='bold')
plt.xlabel('Frecuencia f (Hz)')
plt.ylabel('Derivada d|Z|/df (Ω/Hz)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('grafico_C_derivada.png', dpi=300)
plt.close()

# ---- EXPLICACIÓN -------------------------------------------------------------
print("=" * 80)
print(" PARTE C: DERIVACIÓN NUMÉRICA")
print("=" * 80)
print(" -> Gráfico guardado como: 'grafico_C_derivada.png'")
print()
print(" Método empleado:")
print("   La derivada se obtiene de forma analítica desde el spline cúbico S(f).")
print("   Dado que cada tramo es un polinomio cúbico, su derivada es un polinomio")
print("   cuadrático exacto, sin error de truncamiento adicional.")
print("   Como verificación, también se calculan diferencias finitas centradas")
print("   usando np.gradient sobre los 30 nodos experimentales.")
print()
print(" Mínimo local de |Z|(f):")
print(f"   Frecuencia del mínimo (f_min) : {f_min:.6f} Hz")
print(f"   Impedancia en el mínimo       : {Z_min:.6f} Ω")
print(f"   Segunda derivada en f_min     : {d2Z_min:.6e} Ω/Hz²")
print()
print(f" Criterio de la segunda derivada (confirmación del mínimo):")
if d2Z_min > 0:
    print(f"   S''(f_min) = {d2Z_min:.4e} > 0  →  Mínimo local confirmado (convexo).")
else:
    print(f"   S''(f_min) = {d2Z_min:.4e} < 0  →  Máximo local (cóncavo).")
print()
print(" Derivadas en los nodos experimentales (analítica vs. diferencias finitas):")
print(f"   {'f (Hz)':<10} | {'d|Z|/df spline':<18} | {'d|Z|/df dif. finitas':<22} | {'Δ (Ω/Hz)':<12}")
print("   " + "-" * 70)
for i, f in enumerate(f_data):
    diff = abs(d1_pts[i] - dZ_fd[i])
    print(f"   {f:<10.1f} | {d1_pts[i]:<18.6f} | {dZ_fd[i]:<22.6f} | {diff:<12.6f}")
print()
print(" Dependencia del error con el espaciado h:")
print("   Para un spline cúbico con espaciado máximo h entre nodos:")
print("     Error en la función       : O(h⁴)")
print("     Error en la 1ª derivada   : O(h³)")
print("     Error en la 2ª derivada   : O(h²)")
print("   La 2ª derivada es intrínsecamente más sensible al espaciado entre nodos.")
print()
print(" Propuesta de mejora experimental:")
print("   Muestreo adaptativo no uniforme: concentrar mediciones cerca del mínimo")
print("   (700–1000 Hz) reduciendo localmente h y disminuyendo el error de O(h²)")
print("   en la segunda derivada.")
print("=" * 80)
