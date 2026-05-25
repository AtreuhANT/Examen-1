# METODOS-NUMERICOS 


## 🎥 Video Explicativo 

Puedes ver la explicación detallada del desarrollo y los resultados en el siguiente enlace:

👉 [https://drive.google.com/drive/folders/161OuathCkbwlIq6Tg1ci85pTZkMtHEBb?usp=drive_link]

LA CARPETA DE " codigo y imagenes " corresponde la tarea 1 y la carpeta " EXAMEN PARCIAL " es de la tarea que se dejo hoy lunes 24/05/2026


Resumen de lo que es la tarea:

Este repositorio contiene la solución completa al Examen/Trabajo Práctico 1 de la asignatura Métodos Numéricos. El objetivo del proyecto es procesar, interpolar, derivar e identificar raíces a partir de datos experimentales de bioimpedancia eléctrica ($|Z|$) en función de la frecuencia de excitación ($f$) para caracterizar el comportamiento fisiológico de un tejido biológico a 37 °C.

El análisis combina fundamentos de fisiología celular (dispersión beta de membranas) con análisis numérico avanzado implementado en Python.

📋 Descripción del Proyecto

El proyecto está estructurado en 5 etapas secuenciales, cada una enfocada en resolver un desafío de ingeniería específico mediante el uso de algoritmos numéricos:

Parte A: Análisis Exploratorio de Datos

Visualización de la curva de bioimpedancia $|Z|(f)$ a partir de 30 puntos experimentales.
Explicación física del comportamiento de dispersión beta celular y comportamiento de las membranas como capacitores.

Parte B: Interpolación Polinómica (Lagrange a Trozos, Vandermonde y Splines)

Ajuste global por Vandermonde y demostración práctica del Mal Condicionamiento (número de condición $\kappa > 10^{102}$) y del Fenómeno de Runge.
Implementación de interpolación de Lagrange a trozos (Barycentric) de distintos grados.
Validación cruzada Leave-One-Out (LOO) para medir el error de generalización en puntos no muestreados.
Construcción de Splines Cúbicos Naturales.

Parte C: Derivación Numérica y Estabilidad
Cálculo de la primera y segunda derivada analítica a partir del Spline Cúbico Natural vs. Diferencias Finitas.
Localización exacta del punto mínimo de impedancia ($f_{mín} \approx 742.16\text{ Hz}$).
Uso de la segunda derivada ($S'' > 0$) para verificar la convexidad y estabilidad del mínimo.
Análisis de sensibilidad del error en relación al espaciamiento del muestreo ($h$).

Parte D: Búsqueda de Raíces y Límites de Operación

Localización de las frecuencias de corte $f_1$ y $f_2$ donde la impedancia alcanza los $150\ \Omega$.
Implementación del método de bisección y refinamiento mediante Brentq.
Cálculo del Ancho de Banda seguro ($BW \approx 2103\text{ Hz}$) y análisis de sensibilidad de las raíces frente al ruido en la medición.

Parte E: Discusión Técnica e Integración

Correlación del sensor con tres subsistemas de ingeniería: Biomédico (isquemia, deshidratación), Eléctrico/Electrónico (impedancia de entrada INA, filtros activos) y Telecomunicaciones (desajuste de impedancia de RF a $50\ \Omega$, VSWR).
Propuesta de mejoras metodológicas: muestreo adaptativo no uniforme, control PID de temperatura y promediado de señal.

Tecnologías y Librerías Utilizadas:

Lenguaje: Matlab y Python 

Procesamiento Numérico: numpy (esquemas de interpolación y diferencias finitas), scipy (módulo CubicSpline, brentq para raíces).

Visualización: matplotlib

