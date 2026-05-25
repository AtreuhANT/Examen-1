import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
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
# PARTE E — APLICACIONES TÉCNICAS Y DISCUSIÓN
# ==============================================================================
# Esta sección no genera código computacional nuevo; consolida los resultados
# de las partes anteriores y analiza su impacto en los tres subsistemas.

# ---- CÁLCULOS DE APOYO -------------------------------------------------------
cs  = CubicSpline(f_data, Z_data, bc_type='natural')
ds1 = cs.derivative(1)
ds2 = cs.derivative(2)

# Mínimo de impedancia (desde Parte C)
f_min = brentq(ds1, 700.0, 900.0)
Z_min = float(cs(f_min))

# Frecuencias límite para |Z| = 150 Ω (desde Parte D)
f_low  = brentq(lambda f: cs(f) - 150.0, 100.0, 200.0)
f_high = brentq(lambda f: cs(f) - 150.0, 2160.0, 2340.0)
BW     = f_high - f_low

# Sensibilidades en ambas raíces
sens_low  = 1.0 / float(ds1(f_low))
sens_high = 1.0 / float(ds1(f_high))

# ---- EXPLICACIÓN -------------------------------------------------------------
print("=" * 80)
print(" PARTE E: APLICACIONES TÉCNICAS Y DISCUSIÓN INTEGRADA")
print("=" * 80)
print()
print(" Resumen de parámetros clave del sistema de bioimpedancia:")
print(f"   Frecuencia del mínimo de |Z|     : {f_min:.4f} Hz")
print(f"   Impedancia mínima                : {Z_min:.4f} Ω")
print(f"   Rango seguro (|Z| ≤ 150 Ω)       : [{f_low:.4f}, {f_high:.4f}] Hz")
print(f"   Ancho de banda seguro            : {BW:.4f} Hz")
print(f"   Sensibilidad df/d|Z| en f_low    : {sens_low:.4f} Hz/Ω")
print(f"   Sensibilidad df/d|Z| en f_high   : {sens_high:.4f} Hz/Ω")
print()

print(" a) Subsistema Biomédico:")
print(f"   El mínimo de impedancia en f ≈ {f_min:.1f} Hz representa la frecuencia")
print("   característica donde la corriente penetra de forma óptima las células,")
print("   reflejando el punto de máxima conductividad tisular.")
print("   Variaciones en f_min o en Z_min a lo largo del tiempo son biomarcadores")
print("   clínicos de isquemia tisular, perfusión, o viabilidad celular:")
print("   - Isquemia → altera gradientes iónicos → cambia propiedades dieléctricas")
print("     de las membranas → desplaza f_min.")
print("   - Deshidratación → aumenta resistividad extracelular → eleva Z_min.")
print()

print(" b) Subsistema Eléctrico/Electrónico:")
print(f"   La impedancia varía de {Z_data.min():.1f} Ω (mínimo) a {Z_data.max():.1f} Ω (extremos)")
print("   dentro del rango de medición. Si la etapa de acondicionamiento no cuenta")
print("   con una impedancia de entrada >> 160 Ω (búfer de alta impedancia), la")
print("   carga variable desplaza la frecuencia de corte del filtro analógico")
print("   dinámicamente con la señal de excitación, distorsionando la calibración.")
print("   Recomendación: usar amplificador de instrumentación con Rin > 10 MΩ.")
print()

print(" c) Subsistema de Telecomunicaciones:")
print(f"   Fuera del rango [{f_low:.1f}, {f_high:.1f}] Hz, la impedancia del sensor")
print("   supera 150 Ω. Esto genera desajuste con la línea de transmisión (típic.")
print("   50 Ω), elevando el coeficiente de reflexión Γ y la VSWR, lo que:")
print("   - Reduce la potencia efectiva entregada al tejido.")
print("   - Incrementa la potencia reflejada hacia el transceptor (riesgo de daño).")
print("   - Acorta la duración de la batería en dispositivos inalámbricos portátiles.")
print()

print(" Mejoras prácticas en la recolección de datos:")
print()
print("   1. Muestreo logarítmico o adaptativo:")
print("      Concentrar mediciones (mayor densidad de puntos) en zonas de curvatura")
print("      alta, especialmente cerca del mínimo (700–1000 Hz). Esto reduce")
print("      localmente el paso h y minimiza el error O(h²) de la 2ª derivada.")
print()
print("   2. Control térmico de precisión:")
print("      La impedancia tisular varía ~2–3% por grado Celsius. Estabilizar la")
print("      temperatura a 37 °C con control PID reduce ruido sistemático en los")
print("      datos y previene derivas espurias en el cálculo de las raíces.")
print()
print("   3. Promediado de mediciones (ensemble averaging):")
print("      Adquirir y promediar múltiples ciclos de la señal reduce el ruido")
print("      aleatorio de medición en √N, donde N es el número de promedios,")
print("      mejorando la precisión sin necesidad de puntos adicionales.")
print("=" * 80)
