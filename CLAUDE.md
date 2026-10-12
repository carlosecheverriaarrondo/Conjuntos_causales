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
- `pytest.approx` aplica siempre una tolerancia absoluta de 1e-12 (aunque se
  pase sólo `rel=`): para cantidades pequeñas (p. ej. Λ en unidades de
  Planck), compara cocientes o pasa `abs=0`.
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
Goal 3d (paso 1, 2026-10-01): el logaritmo en 2D viene del término "mismo
par" J¹ (ruido de Poisson del número de intervalos casi nulos, por el rango
de rapidez cortado por el diamante); NO es un efecto de frontera (meseta en
rapidez). Derivado y verificado: (ΔS)² ⊇ b(ε) ρV ln ρV con b = Σ_n f_ε(n)²
(24 / 0.395 / 0.0228). J⁶ conexo crece también (forma sin determinar; con
suavizado aporta ~28 %). Asintótica 2D: ΔS ∝ √(ρV ln ρV). Predicción 4D si
J¹ domina: p = 0.75. Ver README.
Paso 2 (4D, 2026-10-01): J¹ ∝ (ρV)^(2−2/d) exacto → p → 1 − 1/d = 0.75 en
4D (exponente local 1.26 a ρV = 25, 0.77 a 10⁶; el p ≈ 1.2 del Goal 3 era
pre-asintótico). Simulación hasta ρV = 6400: las correlaciones son una
fracción constante de J¹ (resto/J¹ = 0.317 ± 0.014, pendiente 0.6σ). En 4D,
ΔΛ ∝ V^(−1/4): sin escalado de Poisson por una razón estructural (boosts).
Punto 1a (2026-10-02): cono de luz pasado truncado J⁻(q) ∩ {t ≥ 0} en
Minkowski 2D (región de Everpresent Λ, análogo plano). Fórmula exacta de
⟨Σ_pares H⟩ en el cono; MYZ frente a simulación con 8/8 |z| < 4 (ε = 1, 0.5,
ρV = 50–3200); b total = 24.09 ± 0.19 frente a 24.43 ± 0.24 del diamante. En 2D
el escalado ρV ln ρV (mismo b(ε)) no depende de la región; sólo cambia la
constante. Siguiente: 1b, FLRW (el big bang podría cambiar el corte de rapidez).
Punto 1b (2026-10-02): cono pasado hasta el big bang en FLRW 2D, a = η^α
(α = 1, 2), volúmenes exactos (`FLRWConeGeometry`). MYZ frente a simulación
con 8/8 |z| < 4; b = 24.26 ± 0.19 (α = 1) y 24.78 ± 0.24 (α = 2) frente a
24.43 ± 0.24 del diamante. El big bang no cambia el escalado: en 2D,
ΔS ∝ √(ρV ln ρV) con b = Σf² en toda región compacta conformemente plana.
Abierto: 4D (predicción p → 0.75; prefactor dependiente de la geometría).
Punto 1c (2026-10-02): cono pasado truncado en Minkowski 4D. Medias exactas
(integral doble) y prefactor asintótico ⟨N_k⟩ ≈ (ρV)^1.5 G/(8√c₄) Γ(k+½)/k!,
G = ∫dV∫dΩ ℓ² (rango de boosts). Cono/diamante → 0.566, verificado con J¹
exacto hasta 10⁸ (0.562). Simulación ρV = 25–6400: resto/J¹ constante
(0.342 ± 0.017, pendiente 0.05σ). En 4D, el escalado tampoco depende de la
región: ΔS ∝ (ρV)^0.75; la geometría sólo fija la amplitud.
Punto 1d (2026-10-06): cono pasado hasta el big bang en FLRW 4D (a = η^α,
α = 1 radiación, 2 materia), la región física de Everpresent Λ. Volúmenes
exactos (lentes + Gauss-Legendre), prefactor G con el factor de escala
(G/V^1.5 = 1.31 / 1.42) verificado con medias MC hasta ρV = 10⁸ (desviación
≤ 1.9 %), resto/J¹ constante (0.35 / 0.32). ΔS ∝ (ρV)^0.75, ΔΛ ∝ V^(−1/4)
también en FLRW. Las medias MC de FLRW 4D usan muestreo exacto (Beta en η):
el rechazo genérico agotaba la memoria.
Goal 4a (2026-10-08): compatibilidad con el Λ observado (cálculo casi
analítico, `causets/everpresent.py`). Suposición: δΛ·ℓ² = ΔS/(ρV) (reproduce
Model 1 con su acción de Poisson). Acción local EXCLUIDA (δΛ/Λ_obs ~ 10⁶²).
Acción suavizada: amplitud ∝ (ℓ/ℓ_k)⁵; encaja con ℓ_k ≈ 10⁻²² m (por debajo
de la cota LHC 10⁻¹⁹ m, en el rango de los experimentos optomecánicos), pero
es un ajuste y la envolvente tiene w ≈ −½ (Model 1: w ≈ 0). Para comparar con
datos falta la correlación temporal de ΔS entre conos pasados.
Goal 4b (2026-10-10): (1) conos pasados anidados, acción local 4D, 4000
sprinklings (Minkowski y FLRW materia): S(t) es casi un paseo aleatorio, con
memoria positiva pequeña (κ = 0.069 ± 0.015, β = 0.88–0.97); B1 y B3 sí, B2
NO (criterio mal planteado: valor central sin error; documentado, no
cambiado). (2) Λ(t) estocástico con retroacción (`causets/stochastic_lambda.py`),
Model 1 validado frente a las tasas de DNY (añadido a posteriori). Ningún
modelo da historias tipo ΛCDM (≤ 0.1 %); BD con ℓ_k ≈ 2.5·10⁻²³ m da
Ω_Λ0 ≈ h ≈ 0.72 típicos, pero colapsa en el 63–73 % y μ(z), H(z) fallan en
forma. Ver README.
Goal 4c (2026-10-10): Pantheon+SH0ES (`causets/supernovae.py`), ajuste por
semilla como DNY 2024 sec. 4 (10 000 semillas por modelo). V1 NO (χ²_ΛCDM =
1452.0 frente a 1485.3 de DNY; el nuestro coincide con Brout et al. 2022;
umbrales relativos al nuestro). V3 sí (Model 1: 0.25 % buenas, DNY ≈ 0.3 %).
BD (β = 0.9) frente a Model 1: 0.48 % frente a 0.25 % de semillas buenas,
z = 2.7 → indistinguibles; ℓ_k ≈ 3·10⁻²³ m fijado por los datos. Matiza el
4b-2. Datos en data/ (no versionado; scripts/get_pantheon.sh).
Goal 4d (2026-10-10): SN + BAO DESI DR2 + priors de distancia del CMB
(`causets/cmb_bao.py`). W1 (validación con Planck) falló primero (l_A a
−1.45σ, sesgo de r_s por el arranque de radiación pura) y pasó tras corregirlo
(exact_start); W2 sí. 0 de 10 000 semillas buenas en los dos modelos
(f < 3·10⁻⁴); Δχ² mínimo +364 (Model 1) y +124 (BD). Ninguna buena del 4c
sobrevive: corrige la conclusión 1 del 4c. El χ²_SN del ΛCDM combinado
(1486.3) explica probablemente el 1485.3 de DNY. Variante calibrada (a
posteriori): sin cambios.
Goal 4e (2026-10-12): robustez de la exclusión de BD (mismo χ² que el 4d).
E1: β = 0.5, 0.88, 0.97, 1.0 con 10 000 semillas: 0 buenas en todas
(Δχ² mín. +45, +116, +135, +154) → robusta en β. E2: mapa (q, c) de la
familia OU, 1000 semillas por celda: ninguna celda con semillas buenas; la
mejor es q = 1/8, c = 0.25 (+17); los datos sólo toleran Λ casi constante.
Abierto: dinámicas fuera de la familia OU.
Total: 205 tests pasando.

Goal 4 no lanzado y por REPLANTEAR según el Goal 3c: no generar Λ(t)
suponiendo que las fluctuaciones de S_BD tienen el escalado de Poisson;
medirlas. Pendientes que le afectan:
- ¿En 4D el exponente efectivo baja con ρV, como en 2D? (Ahora sólo hay
  ρV ≤ 100, con p ≈ 1.2.)
- 4D con suavizado: no simulado (la predicción es el mismo exponente 0.75).
- Conos pasados: hechos en 2D (Minkowski, 1a; FLRW, 1b) y 4D (Minkowski,
  1c; FLRW, 1d); mismo escalado en todos. Abierto: FLRW con Λ de fondo.
- J⁶ conexo: determinar su crecimiento (dominante con suavizado fuerte).
- Acción suavizada con ερV ≫ 1 (límite ε → 0): sólo se ha llegado a
  ερV = 100; el escalado asintótico sigue abierto.
- 4D: la predicción con ρV = 100 tiene un error MC del 23 %; la cuadratura
  es lenta (~10–30 min por densidad).
- FLRW puede seguir el patrón de `DeSitterGeometry` (coordenadas conformes,
  peso a(η)^d, volúmenes de intervalos), pero sin simetría máxima los
  volúmenes de intervalos no se reducen a un invariante.
