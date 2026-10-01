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
indefinidamente). "Turno" = cada ronda de llamadas a herramientas del
asistente (no cada mensaje del usuario). Al llegar al límite, avisa y pide
confirmación aunque haya progreso, en vez de seguir sin decir nada.

## Convenciones del proyecto

- Todo resultado numérico que afirme "converge a X" debe verificarse con
  un test automatizado (pytest) que falle explícitamente si no se cumple
  dentro de la tolerancia estadística indicada — nunca fallar en silencio.
- Cita siempre la referencia exacta del método/fórmula usado, en README o
  docstring del módulo correspondiente.
- Antes de dar por completado un criterio de éxito, ejecuta los tests y
  muestra el resultado real (no asumas que pasan).
- Los márgenes estadísticos de los tests se fijan antes de ver los
  resultados (hasta ahora: 4σ combinada). Comprueba que los tests pueden
  fallar introduciendo un error deliberado (y restaurando el código).
- Los cálculos largos van en scripts que guardan resultados intermedios
  (p. ej. JSON en `figures/`), lanzados en segundo plano.

## Referencias principales

- Myrheim, J. (1978) — dimensión de Myrheim-Meyer.
- Reid, D. D. (2003), Phys. Rev. D 67, 024034 (arXiv:gr-qc/0207103) —
  formulación práctica del estimador de dimensión.
- Benincasa, D. & Dowker, F. (2010), Phys. Rev. Lett. 104, 181301 — acción
  discreta de Benincasa-Dowker.
- Moradi, Yazdi & Zilhão, Class. Quantum Grav. 42 (2025) 045017,
  arXiv:2407.03395 — formalismo de fluctuaciones de la acción BD (sólo da
  el formalismo, sin valores numéricos). PDF en `references/`.
- Das, Nasiri & Yazdi, JCAP 10 (2023) 047 y JCAP 10 (2024) 076 —
  modelo fenomenológico Everpresent Λ (Model 1).

## Estado actual

Goal 1 completado (2026-09-29): paquete `causets/`, figura en
`figures/goal1_mm_convergence.png`; ver README.
Goal 2 completado (2026-09-29): acción BD 2D/4D (local y suavizada),
valores esperados exactos a densidad finita, verificación en Minkowski y
de Sitter; ver README.
Goal 3 completado (2026-09-29): formalismo de Moradi-Yazdi-Zilhão en
`causets/fluctuations.py` para la acción BD LOCAL (la suavizada, en la
ampliación de abajo). Minkowski 2D verificado con predicción exacta (5 densidades,
|z| ≤ 1.56; tests). 4D preliminar (plugin, compatible, error 6-10 %) y de
Sitter 2D exploratorio. Conclusión: con la acción local en Minkowski,
ΔS ∝ (ρV)^0.72 (2D), no de tipo Poisson: no respalda Everpresent Λ tal
cual; ver README. La búsqueda de literatura del grupo de Yazdi se hizo
(sept. 2026) y no encontró nada publicado. Repetida el 2026-10-01: MYZ
sólo tiene 2 citas en INSPIRE (Yeats 2412.14036, combinatoria; Dowker-Liu-
Lloyd-Jones 2501.00139, media de la acción con frontera temporal); ninguna
evalúa la varianza ni su escalado. MYZ sólo PLANTEAN si las fluctuaciones de
la acción podrían ser el origen de Everpresent Λ, no lo responden.
Revisado también Ashfaque, "Numerical Explorations of the Benincasa-Dowker
Conjecture" (ResearchGate, abr. 2026, sin arXiv; PDF local en references/,
no versionado): sólo trata la MEDIA de la acción, sin resultados numéricos,
y su suavizado e^(-εn) no es el de Sorkin/Dowker-Glaser; no se solapa.
Ampliación del Goal 3 (2026-10-01): acción BD SUAVIZADA (abundancias
adelgazadas) y |A∩B| exacto en 4D (`causets/overlap.py`, cuadratura).
2D, ε = 0.5 y 0.25, ρV = 25..400: 10/10 puntos con |z| < 4. El suavizado
reduce la amplitud pero no el escalado (p = 0.72 / 0.77). 4D local con
ρV = 25, 50, 100: compatible, ΔS ∝ (ρV)^1.21. Ver README.
Goal 3c completado (2026-10-01): (1) Everpresent Λ (Das-Nasiri-Yazdi
2023, ecs. 2.2, 3.6, 3.10-3.13) exige estrictamente δΛ ∝ V^(-1/2), pero lo
obtiene de la fluctuación de Poisson de N con contribuciones independientes
por elemento, NO de la acción BD; α sólo fija la amplitud. (2) Minkowski 2D
con ρV = 800-3200 (ερV hasta 3200): 9/9 puntos con |z| < 4; el exponente
NO es estable (baja de 0.72-0.77 a 0.59-0.62, más de 7σ) y sigue siendo
incompatible con 0.5 (más de 5σ). Hipótesis a posteriori: (ΔS)² ≈ ρV(a + b ln ρV)
(Poisson con corrección logarítmica). (3) Veredicto: el Goal 4 tal como
estaba planteado NO tiene base; ver la propuesta alternativa en el README.
Total: 107 tests pasando.

Goal 4 no lanzado y por REPLANTEAR según el Goal 3c: no generar Λ(t)
suponiendo que las fluctuaciones de S_BD tienen el escalado de Poisson;
medirlas. Pendientes que le afectan:
- ¿En 4D el exponente efectivo baja con ρV, como en 2D? (Ahora sólo hay
  ρV ≤ 100, con p ≈ 1.2.)
- Comprobar o refutar (ΔS)² ≈ ρV(a + b ln ρV) en 2D y localizar el origen
  del logaritmo (¿frontera?).
- Acción suavizada con ερV ≫ 1 (límite ε → 0): sólo se ha llegado a
  ερV = 100; el escalado asintótico sigue abierto.
- 4D: la predicción con ρV = 100 tiene un error MC del 23 %; la cuadratura
  es lenta (~10–30 min por densidad).
- FLRW puede seguir el patrón de `DeSitterGeometry` (coordenadas conformes,
  peso a(η)^d, volúmenes de intervalos), pero sin simetría máxima los
  volúmenes de intervalos no se reducen a un invariante.
