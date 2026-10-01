# Conjuntos causales: sprinkling, dimensión y acción de Benincasa-Dowker

Simulador numérico de conjuntos causales por sprinkling de Poisson. Ver
`CLAUDE.md` para el roadmap completo (Goals 1–4).

## Uso

```bash
source venv/bin/activate
python -m pytest -q                      # tests (≈5 min)
python scripts/goal1_convergence.py      # figura figures/goal1_mm_convergence.png
python scripts/goal2_bd_action.py        # figura figures/goal2_bd_action.png
python scripts/goal3_fluctuations.py run  # cálculo largo (~20 min) -> figures/goal3_results.json
python scripts/goal3_fluctuations.py plot # figura figures/goal3_fluctuations.png
python scripts/smeared_fluctuations.py run2d  # acción suavizada 2D (largo) -> figures/smeared_results_2d.json
python scripts/smeared_fluctuations.py run4d  # 4D con |A∩B| exacto (largo) -> figures/smeared_results_4d.json
python scripts/smeared_fluctuations.py plot   # figura figures/smeared_fluctuations.png
```

## Estructura

| Módulo | Responsabilidad |
|---|---|
| `causets/regions.py` | Regiones: `CausalDiamond` (Minkowski), `ConformalDiamond` (métricas a(η)² η, p. ej. de Sitter), volúmenes de intervalos en de Sitter |
| `causets/sprinkling.py` | Sprinkling de Poisson (`sprinkle`) y muestreo uniforme de N fijo (`sample_uniform`), puntos ordenados por t |
| `causets/causal.py` | Matriz causal de Minkowski y verificación de orden parcial (reflexividad, antisimetría, transitividad) |
| `causets/dimension.py` | Estimador de dimensión de Myrheim-Meyer |
| `causets/intervals.py` | Cardinalidades de intervalos n(y,x) y abundancias N_k |
| `causets/action.py` | Operador B_ε y acción S_ε de Benincasa-Dowker (2D y 4D, local y suavizados) |
| `causets/continuum.py` | Valores esperados exactos a densidad finita (Minkowski y de Sitter) |
| `causets/geometry.py` | Geometrías para el formalismo de fluctuaciones (Minkowski, de Sitter): pesos, volúmenes de intervalos y de su intersección |
| `causets/overlap.py` | Volumen exacto de la intersección de dos intervalos en Minkowski 4D (cuadratura determinista) |
| `causets/fluctuations.py` | Fluctuaciones (ΔS)² de la acción BD (formalismo de Moradi-Yazdi-Zilhão), local y suavizada |
| `tests/` | Tests pytest (tolerancias estadísticas explícitas en cada fichero) |
| `scripts/` | Scripts que generan figuras |

Convenciones: coordenadas `(t, x_1, …, x_{d-1})`, c = 1; `C[i, j] = True` ⇔ i ≺ j
estrictamente. Como los puntos van ordenados por t, `C` es triangular superior.

## Goal 1: sprinkling y dimensión de Myrheim-Meyer (completado)

Para N puntos uniformes en un intervalo causal de Minkowski d-dim, la fracción
esperada de pares relacionados es

  f(d) = Γ(d+1) Γ(d/2) / (2 Γ(3d/2)),  con f(2) = 1/2, f(4) = 1/10,

y el estimador d_MM invierte f(d) = R / (N choose 2).

Criterio de convergencia (en `tests/test_dimension.py`): para M = 20 sprinklings
de N ∈ {1000, 2000}, |⟨d_MM⟩ − d| < 4·SEM **y** |⟨d_MM⟩ − d| < 0.02 (2D) / 0.05 (4D);
además std(d_MM) cae con N de forma compatible con N^(−1/2).

![Convergencia de Myrheim-Meyer](figures/goal1_mm_convergence.png)

## Goal 2: acción de Benincasa-Dowker (completado)

Operador y acción (detalles y convenciones en el docstring de `causets/action.py`):

  (B_ε φ)(x) = α_d ℓ_ε⁻² [β_d φ(x) + ε Σ_{y≺x} f_d(n(y,x), ε) φ(y)],  S_ε = −s_d Σ_x ℓ² (B_ε 1)(x)

con (α, β, C_k) = (4, −1/2, {1,−2,1}) en 2D y (4/√6, −1, {1,−9,16,−8}) en 4D.

**Valores esperados exactos.** Por la fórmula de Mecke del proceso de Poisson,
E[Σ_{y≺x} h(n(y,x))] = ∫_{J⁻(x)} ρ dV_y E[h(Poisson(ρV(y,x)))]. Para x en el
vértice superior de un intervalo plano de volumen Λ (unidades de 1/ρ),
parametrizando y = x + τ(−cosh χ, sinh χ n̂) e integrando χ, la integral se
reduce a una variable:

  W[H](Λ) = ∫₀^Λ H(z) ln(Λ/z) dz                              (2D)
  W[H](Λ) = 3 ∫₀^Λ H(z) [√(Λ/z) − √(z/Λ) − ln(Λ/z)] dz        (4D)

Con los coeficientes BD, los términos en ln Λ y √Λ se cancelan y E[ℓ²B·1] → 0
(R = 0). La misma distribución de volúmenes da E[N_k] y E[S] para el diamante
entero (`continuum.py`). En de Sitter, por simetría máxima, V(y,x) sólo depende
del invariante Z = cosh(Hτ), y E[B·1](x) sólo de τ(p,x): se calcula como una
integral doble (cuadratura de Gauss-Legendre).

**Resultados** (todos con test en `tests/test_action.py` y `tests/test_de_sitter.py`):

1. Abundancias N_0..N_3 y acción media de sprinklings de diamantes 2D y 4D
   coinciden con los valores exactos (< 4 SEM).
2. E[ℓ²B·1] → 0 en Minkowski 2D y 4D, local y con ε = 0.1; la simulación en el
   vértice superior coincide con el valor exacto.
3. Acción 2D del diamante: ⟨S⟩ → 1 (= vértices − aristas + caras del teselado
   nulo, Benincasa-Dowker-Schmitzer 2011; ya es 1 con error < 10⁻⁶ para ρV ≥ 100).
4. de Sitter: E[B·1]/(−R/2) = 0.78 → 0.989 (2D) y 0.52 → 0.961 (4D) al bajar
   ℓH de 0.4 a 0.05, monótonamente; la simulación reproduce el valor exacto a
   densidad finita (< 4 SEM). Es decir, R = 12H² (4D) se recupera en el límite
   continuo, con correcciones grandes a densidad finita (~(ℓH)^1.4 en 4D).

![Goal 2](figures/goal2_bd_action.png)

**Observaciones y limitaciones.**
- La dispersión de la acción y de B·1 crece con ρV (panel izquierdo; en 4D
  local, std(ℓ²B·1) ~ 100 a Λ = 100). Por eso la curvatura no se puede medir
  punto a punto: en de Sitter se promedia B_ε·1 sobre todos los puntos con
  ερV(p,x) ≥ 30 y se usa suavizado ε = 0.05. Estas fluctuaciones son el objeto
  del Goal 3.
- La media exacta sólo depende de ερ (el suavizado equivale a adelgazar el
  conjunto con probabilidad ε).
- La cuadratura de de Sitter está verificada hasta ℓH = 0.05 en 4D (resolución
  1 y 2 coinciden a 10⁻⁴ relativo); por debajo, la cancelación entre términos
  O(1) exige más precisión de la implementada.

## Goal 3: fluctuaciones de la acción BD (formalismo de Moradi-Yazdi-Zilhão)

Referencia: H. Moradi, Y. K. Yazdi & M. Zilhão, "Fluctuations and Correlations
in Causal Set Theory", Class. Quantum Grav. 42 (2025) 045017, arXiv:2407.03395
(MYZ; ecuaciones citadas según la v2 del arXiv). Código: `causets/fluctuations.py`
(formalismo), `causets/geometry.py` (geometrías), `tests/test_fluctuations.py`,
`scripts/goal3_fluctuations.py`.

**Qué acción.** Todo el Goal 3 trata la acción BD **local** (ε = 1), que es la
que MYZ descomponen: S = a_N N + Σ_{k<n_d} a_k N_k con a = (1, −2, 4, −2) en 2D
y (4/√6)(1, −1, 9, −16, 8) en 4D (normalización del Goal 2; la de MYZ, ec. 5.4,
difiere en una constante global, p. ej. factor −√6/2 en 4D). La acción
suavizada (ε < 1) depende de todas las N_n; se trata en la sección siguiente
("Ampliación del Goal 3").

**Qué se implementó del artículo.**
- Indicadores ζ_i (cardinalidad, MYZ 3.2/3.4) y χ (ocupación, 3.16):
  `cardinality_indicator`, `occupation_indicator`. Las reglas de correlación
  ζ-χ (3.31) y ζ-ζ con solapamiento (3.15, 3.34) se verifican directamente
  contra sprinklings (tests).
- (ΔS)² = aᵀ M a con M_ij = K_ij − L_iL_j = Cov(N_i, N_j), i, j = −1…n_d−1
  (MYZ 5.9-5.10), L_i por (5.15), M_{−1,−1} = ρV (5.20), M_{−1,j} de (5.26-5.27)
  — que se reduce a la identidad exacta M_{−1,j} = (j+2)L_j − (j+1)L_{j+1} — y
  K_ij = J¹+…+J⁶ (5.29-5.36) con las subregiones de D_2, D_3, D_6 (5.40-5.44)
  detectadas configuración a configuración.
- Lo que MYZ dejan como trabajo futuro: **evaluar las integrales**. Se hace por
  Monte Carlo con muestreo de importancia adaptado a la medida de volumen de
  intervalos casi nulos; J⁶ se evalúa ya restada L_iL_j (parte conexa) y
  partida en patrones (un extremo dentro del otro intervalo / sólo
  solapamiento) para evitar colas pesadas.
- **Minkowski 2D:** |A∩B| tiene forma cerrada (rectángulos en coordenadas de
  cono de luz) → predicción exacta salvo el error MC, que se reporta.
- **4D y de Sitter:** |A∩B| no tiene forma cerrada (MYZ sólo parametrizan
  J⁻(x₁)∩J⁻(x₂) en 2D, apéndice C.1). Se estima con K puntos MC y se usa la
  fórmula exacta de ⟨ζζ⟩ ("plugin"): sesgo O(1/K), medido comparando K. Hay
  también un modo insesgado (realización de Poisson local + ensayos
  geométricos para 1/|A∩B|), pero su varianza lo hace inútil en 4D.

**Decisión sobre N ≳ 2¹⁶ y termalización.** Cunningham & Krioukov (2018)
necesitan N grandes y termalización porque muestrean *dinámica* de causets con
cadenas de Markov. Aquí no hay cadena: cada sprinkling de Poisson es una
muestra independiente y exacta del ensemble, así que no hay termalización
posible ni necesaria. Y la predicción de MYZ es exacta a densidad finita, así
que la comparación es significativa a cualquier ρV: se usan ρV = 25…400 (2D)
con 4000 sprinklings por punto; el error del estimador de la varianza se
estima como std(d²)/√M con d = S − ⟨S⟩.

**Resultados en Minkowski 2D** (predicción exacta; margen de los tests fijado a
priori: |sim − pred| < 4σ_combinada):

| ρV | (ΔS)² predicción MYZ | (ΔS)² simulación (4000 sprinklings) | z | (ΔS)²/ρV |
|---|---|---|---|---|
| 25 | 638 ± 8 | 651 ± 17 | 0.64 | 26 |
| 50 | 1968 ± 24 | 1971 ± 48 | 0.06 | 39 |
| 100 | 5401 ± 68 | 5626 ± 126 | 1.56 | 54 |
| 200 | 14 099 ± 177 | 14 029 ± 317 | −0.19 | 70 |
| 400 | 34 734 ± 468 | 36 121 ± 830 | 1.46 | 87 |

Todas las densidades cumplen |z| < 4 (máximo 1.56). Los tests
(`tests/test_fluctuations.py`, ρV = 50 y 100) verifican además las reglas de
correlación ζ-χ (3.31) y ζ-ζ (3.34) contra sprinklings, y la fila M_{−1,j}
de la matriz de covarianza. Se comprobó que los tests pueden fallar: al
eliminar el patrón c0 de J⁶ los dos tests de varianza fallan.
Ajuste: (ΔS)² ∝ (ρV)^1.44 en este rango, es decir, ΔS ∝ (ρV)^0.72.

![Goal 3](figures/goal3_fluctuations.png)

**Minkowski 4D (preliminar).** ρV = 50, 6000 sprinklings, predicción plugin:

| K (puntos para \|A∩B\|) | (ΔS)² predicción | (ΔS)² simulación | z |
|---|---|---|---|
| 256 | 10 142 ± 1000 | 10 542 ± 251 | 0.39 |
| 1024 | 9 616 ± 582 | 10 542 ± 251 | 1.46 |

Son compatibles, pero la predicción tiene un error del 6-10 % y un sesgo
O(1/K) que no se ha acotado con precisión en 4D. En Minkowski 2D, donde se
puede comparar con el cálculo exacto, el sesgo medido fue de ~+465 con K = 64
y ~+200 con K = 256 (ρV = 100, sobre ~5400). Es una verificación preliminar,
no estricta, y no tiene test automatizado.

**de Sitter 2D (exploratorio, no es una verificación).** MYZ no tratan de
Sitter. La extensión supone que su formalismo vale para cualquier variedad
(así lo afirman en el resumen) y usa: coordenadas conformes, donde la
relación causal es la de Minkowski; la medida física a(η)² d²x; y volúmenes
de intervalos exactos por simetría máxima. La región es η ∈ [−4, −0.3], H = 1,
con plugin K = 256:

| ρV | (ΔS)² predicción | (ΔS)² simulación | z |
|---|---|---|---|
| 50 | 2292 ± 161 | 1977 ± 47 | −1.88 |
| 100 | 4872 ± 408 | 5845 ± 136 | 2.26 |

Ambas quedan dentro de 4σ, pero con z de signos opuestos, errores del 7-8 %
y el sesgo de K = 256. No permiten afirmar un acuerdo fino: el resultado se
toma como una prueba de que el código funciona en geometría curva, no como una
predicción validada.

**Trabajo futuro.** (La cuadratura de |A∩B| en 4D, la acción suavizada y
más densidades en 4D se hicieron después: ver "Ampliación del Goal 3".)
Queda FLRW (Goal 4).

**¿Validan estos resultados usar las fluctuaciones para modelar Everpresent Λ?**

**No, con la acción local y en Minkowski los números no lo respaldan.**
Everpresent Λ supone fluctuaciones de tipo Poisson: δN ~ √N, de donde sale
ΔΛ ~ V^(−1/2). Si Λ se identificara con las fluctuaciones de la acción a
través del término −2ΛV (MYZ ec. 4.6), haría falta ΔS ~ √(ρV). Lo que se mide
es otra cosa:
- En 2D, (ΔS)² ∝ (ρV)^1.44, es decir, ΔS ∝ (ρV)^0.72. Eso daría
  ΔΛ ∝ V^(−0.28), no V^(−1/2).
- La varianza es de 26 a 87 veces mayor que ρV en 2D (y ~210 veces en 4D a
  ρV = 50). Está dominada por las correlaciones entre intervalos (términos
  J², J³, J⁴, J⁶), no por el término de Poisson de N.

Esto **no** descarta la idea en general. Queda abierto para: la acción
suavizada (pensada precisamente para reducir fluctuaciones); espaciotiempos
FLRW; y otras formas de extraer Λ que no sean la varianza total de S en un
diamante. Una hipótesis no comprobada aquí es que la mayor parte de la
varianza venga de intervalos casi nulos pegados a la frontera del diamante;
si fuera así, restar términos de frontera podría cambiar el escalado.

## Ampliación del Goal 3: acción suavizada y |A∩B| exacto en 4D

Código: `causets/fluctuations.py`, `causets/overlap.py`,
`tests/test_smeared_fluctuations.py`, `tests/test_overlap.py`,
`scripts/smeared_fluctuations.py` (resultados en
`figures/smeared_results_2d.json` y `figures/smeared_results_4d.json`).

**Acción suavizada como abundancias adelgazadas.** La función de suavizado de
Dowker & Glaser (2013) es f_d(n, ε) = Σ_k C_k B_k(n, ε), con
B_k(n, ε) = binom(n, k) ε^k (1−ε)^(n−k). Por tanto
S_ε = a_N N + Σ_{k<n_d} a_k Ñ_k, con Ñ_k = Σ_{y≺x} B_k(n(y,x), ε): el número
esperado de pares con k elementos entre ellos cuando cada elemento del
intervalo se conserva con probabilidad ε (`thinned_abundance_vector`,
`smeared_action_coefficients`). Así el formalismo MYZ se aplica sin cambios a
(N, Ñ_0, …), con estas modificaciones (extensión propia, no está en MYZ):
- ⟨Ñ_k⟩ = ρ²∫∫P_k(ερV): la densidad efectiva es ερ.
- J¹ deja de ser diagonal: los dos factores de un mismo par son
  adelgazamientos independientes del mismo intervalo.
- ⟨ζζ⟩ con solapamiento: adelgazar un proceso de Poisson da procesos de
  Poisson independientes. A∩B se reparte en "retenido por ambos" (ε²),
  "sólo por A" y "sólo por B" (ε(1−ε) cada uno), y la ec. (3.15) de MYZ se
  mantiene con ρ|A| → ερ|A| − ε²ρ|A∩B|, ρ|A∩B| → ε²ρ|A∩B|. Los puntos fijos
  dentro del intervalo se conservan con probabilidad ε (desplazamiento
  binomial) (`thinned_joint_probability`).
- La fila M_{−1,k} = (k+2)L_k − (k+1)L_{k+1} sigue valiendo (fórmula de Mecke).

**|A∩B| exacto en 4D** (`overlap.py`). A t fijo cada cono es una bola y a
(t, z) fijos un disco. El área de la intersección de los 4 discos se calcula
exactamente con el teorema de Green (arcos de circunferencia) y se integra con
Gauss-Legendre en t (partido en los instantes de tangencia, que son lineales
en t) y en z. Verificado contra casos exactos (tolerancia relativa 1e-3) y
contra Monte Carlo con 10⁶ puntos (4σ). Sustituye al estimador plugin, que
tenía un sesgo O(1/K).

Los tests fallan con errores deliberados: con el área de los discos ×1.01
fallan los 4 tests de área/volumen; con ε en vez de ε² en la probabilidad
conjunta fallan el test contra sprinklings y el de la varianza.

**Resultados en Minkowski 2D, acción suavizada** (4000 sprinklings por punto;
predicción con 4·10⁶ (ε = 0.5) y 10⁷ (ε = 0.25) muestras MC por término):

| ρV | ε = 0.5 predicción | ε = 0.5 simulación | z | ε = 0.25 predicción | ε = 0.25 simulación | z |
|---|---|---|---|---|---|---|
| 25 | 12.59 ± 0.27 | 13.36 ± 0.36 | 1.72 | 0.576 ± 0.015 | 0.595 ± 0.016 | 0.86 |
| 50 | 39.8 ± 0.9 | 40.5 ± 1.0 | 0.55 | 2.09 ± 0.05 | 2.10 ± 0.05 | 0.21 |
| 100 | 117.3 ± 2.7 | 113.4 ± 2.6 | −1.03 | 6.46 ± 0.17 | 6.24 ± 0.15 | −0.96 |
| 200 | 295 ± 8 | 299 ± 7 | 0.41 | 17.7 ± 0.5 | 17.2 ± 0.4 | −0.74 |
| 400 | 733 ± 22 | 748 ± 17 | 0.56 | 43.4 ± 1.6 | 45.3 ± 1.0 | 0.98 |

Los 10 puntos cumplen |z| < 4 (máximo 1.72). Ajuste ponderado de la
simulación, ΔS ∝ (ρV)^p:

| ε | p |
|---|---|
| 1 (local) | 0.718 ± 0.006 |
| 0.5 | 0.722 ± 0.006 |
| 0.25 | 0.772 ± 0.006 |

**Resultados en Minkowski 4D, acción local, |A∩B| por cuadratura** (6000
sprinklings por punto):

| ρV | (ΔS)² predicción | (ΔS)² simulación | z | (ΔS)²/ρV |
|---|---|---|---|---|
| 25 | 1899 ± 82 | 1840 ± 97 | −0.46 | 74 |
| 50 | 9775 ± 332 | 10 234 ± 249 | 1.10 | 205 |
| 100 | 51 224 ± 12 003 | 53 932 ± 1144 | 0.22 | 539 |

Con ρV = 50 el error de la predicción baja al 3.4 % (plugin: 6–10 % más un
sesgo sin acotar). Con ρV = 100 el error MC es del 23 % (colas pesadas con
5·10⁵ muestras): el acuerdo es real pero poco exigente. Ajuste de la
simulación: **ΔS ∝ (ρV)^1.21 ± 0.02 en 4D** (la predicción da 1.18 ± 0.04).
Este punto no tiene test automatizado: es caro (~10–30 min por densidad).

![Acción suavizada](figures/smeared_fluctuations.png)

**Conclusión para Everpresent Λ.** El suavizado reduce mucho la amplitud de
las fluctuaciones: con ρV = 400 en 2D, (ΔS)² pasa de ~35 000 (ε = 1) a ~750
(ε = 0.5) y ~45 (ε = 0.25). Pero **no mejora el escalado**: p no baja hacia
el 0.5 de Poisson. Se mantiene en 0.72 con ε = 0.5 y sube a 0.77 con
ε = 0.25. Con ε = 0.25, ερV sólo va de 6 a 100, así que es probable que sea un
régimen pre-asintótico; el escalado para ε → 0 con ερV → ∞ sigue abierto.
En 4D, con la acción local, el escalado es aún peor (p ≈ 1.2, y (ΔS)²/ρV
crece de 74 a 539). Así que, también con la acción suavizada, la varianza
total de S en un diamante de Minkowski no tiene el escalado de Poisson que
supone Everpresent Λ.

## Referencias

- Bombelli, Lee, Meyer & Sorkin (1987), Phys. Rev. Lett. 59, 521 — sprinkling de Poisson.
- Myrheim, J. (1978), "Statistical geometry", CERN preprint TH-2538 — ordering fraction y dimensión.
- Meyer, D. A. (1988), tesis doctoral, MIT — expresión cerrada de f(d).
- Reid, D. D. (2003), Phys. Rev. D 67, 024034, arXiv:gr-qc/0207103 — estimador práctico de dimensión.
- Surya, S. (2019), Living Rev. Relativ. 22, 5 — revisión general.
- Benincasa & Dowker (2010), Phys. Rev. Lett. 104, 181301, arXiv:1001.2725 — acción BD en 4D.
- Sorkin, R. D. (2007), arXiv:gr-qc/0703099 — d'Alembertiano 2D y suavizado ε.
- Benincasa, Dowker & Schmitzer (2011), Class. Quantum Grav. 28, 105018, arXiv:1011.5191 — acción 2D y su media.
- Dowker & Glaser (2013), Class. Quantum Grav. 30, 195016, arXiv:1305.2588 — f_d(n, ε) en dimensión d.
- Moradi, Yazdi & Zilhão, arXiv:2407.03395 — kernels de fluctuación (Goal 3).
- Das, Nasiri & Yazdi, JCAP 10 (2023) 047; JCAP 10 (2024) 076 — Everpresent Λ (Goal 4).
