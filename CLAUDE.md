# Proyecto: Fluctuaciones de Λ en Causal Set Theory

## Qué es este proyecto

Simulador numérico de conjuntos causales (causal sets) mediante sprinkling de
Poisson, construido en fases incrementales. El objetivo final es implementar
y evaluar numéricamente las fluctuaciones de la acción de Benincasa-Dowker,
conectándolas con el modelo fenomenológico "Everpresent Λ" (constante
cosmológica fluctuante derivada de discreción del espacio-tiempo).

Este NO es un ejercicio de reproducir resultados conocidos por reproducirlos:
las fases 1-2 son infraestructura de validación; el valor original del
proyecto está en las fases 3-4, que implementan un cálculo que los propios
autores de la literatura de referencia dejaron como "trabajo futuro".

## Entorno técnico

- Python 3.12.3, en un entorno virtual (`venv/`) dentro de esta carpeta.
  Actívalo siempre antes de trabajar: `source venv/bin/activate`.
- Librerías ya instaladas: numpy, scipy, matplotlib, pytest.
- Para cualquier gráfica con matplotlib, usa el backend 'Agg' (sin GUI) y
  guarda con `plt.savefig()` — no uses `plt.show()`.
- Estructura de código: módulos separados por responsabilidad
  (sprinkling, relación causal, invariantes, acción, fluctuaciones), no
  scripts monolíticos. Cada fase del proyecto se apoya en el código de la
  fase anterior — diseña pensando en que se va a extender.

## Roadmap del proyecto (4 goals secuenciales)

1. **Sprinkling básico + invariantes dimensionales** — sprinkling de Poisson
   en Minkowski 2D/4D, verificación de orden parcial válido, dimensión de
   Myrheim-Meyer convergiendo a d=2/d=4.
2. **Acción de Benincasa-Dowker** — cálculo de la acción BD sobre el
   simulador, verificada en Minkowski (R=0) y extendida a de Sitter
   (R=12H² en 4D).
3. **Kernels de fluctuación** — varianza (ΔS)² de la acción BD, siguiendo
   Moradi, Yazdi & Zilhão (2024/2025), verificada primero en Minkowski.
4. **Extensión a FLRW + Everpresent Λ** — fluctuaciones en espaciotiempos
   FLRW, generación estocástica de Λ(t) siguiendo el modelo Model 1 de
   Das, Nasiri & Yazdi (2023/2024).

Cada goal se lanza como comando `/goal` independiente, con su propio
criterio de éxito y un límite de seguridad de 25 turnos (detente y reporta
el estado si no hay progreso verificable en ese margen, en vez de iterar
indefinidamente).

## Convenciones del proyecto

- Todo resultado numérico que afirme "converge a X" debe verificarse con
  un test automatizado (pytest) que falle explícitamente si no se cumple
  dentro de la tolerancia estadística indicada — nunca fallar en silencio.
- Cita siempre la referencia exacta del método/fórmula usado, en README o
  docstring del módulo correspondiente.
- Antes de dar por completado un criterio de éxito, ejecuta los tests y
  muestra el resultado real (no asumas que pasan).

## Referencias principales

- Myrheim, J. (1978) — dimensión de Myrheim-Meyer.
- Reid, D. (2003), Class. Quantum Grav. 20 — formulación práctica del
  estimador de dimensión.
- Benincasa, D. & Dowker, F. (2010), Phys. Rev. Lett. 104 — acción
  discreta de Benincasa-Dowker.
- Moradi, Yazdi & Zilhão (2024/2025), Class. Quantum Grav. 42,
  arXiv:2407.03395 — kernels analíticos de fluctuación de la acción BD.
- Das, Nasiri & Yazdi, JCAP 10 (2023) 047 y JCAP 10 (2024) 076 —
  modelo fenomenológico Everpresent Λ (Model 1).

## Estado actual

Goal 1 completado (2026-09-29): paquete `causets/`, 43 tests pasando,
figura en `figures/goal1_mm_convergence.png`; ver README. Goals 2-4 no
lanzados todavía — el Goal 2 está completamente especificado y listo;
los Goals 3 y 4 son esbozos que se desarrollarán en detalle antes de
lanzarlos, revisando primero si hay literatura más reciente relevante
(en particular, verificar si el grupo de Yazdi ha publicado avances sobre
la evaluación numérica de fluctuaciones antes de lanzar el Goal 3).
