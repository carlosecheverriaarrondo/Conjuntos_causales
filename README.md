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

## Goal 3c: ¿es robusto el exponente? ¿Qué exige realmente Everpresent Λ?

### Fase 1: qué exige el modelo Everpresent Λ (Model 1)

Fuente: S. Das, A. Nasiri & Y. K. Yazdi, "Aspects of Everpresent Λ (I): A
Fluctuating Cosmological Constant from Spacetime Discreteness", JCAP 10 (2023)
047, arXiv:2304.03819 (PDF en `references/`). Se revisó también la Parte II
(JCAP 10 (2024) 076, arXiv:2307.13743), que contrasta el modelo con supernovas
Ia y CMB y no trata el exponente de escalado.

**1. De dónde sale el exponente 1/2: de la relación número-volumen, no de la
acción BD.** El artículo deriva δΛ ∝ V^(−1/2) en tres pasos:
- Fluctuación de Poisson de N (sec. 2, ec. 2.2): "the standard deviation
  δN = √(ρ_cs V), which is also the smallest possible deviation [29] from an
  exact number-volume correspondence", es decir, δV = √N/ρ_cs.
- Relación de incertidumbre entre Λ y V en gravedad unimodular (ecs. 3.4-3.5):
  δΛ/(8πG) = (ℏ/2)(1/δV).
- Combinando ambas, δΛ = 4π(ℓ_p/ℓ_cs)² (1/√V) (ec. 3.6), y por tanto
  δΛ ∼ H² (ec. 3.7).

La acción de Benincasa-Dowker-Glaser sólo aparece para definir la integral de
camino Z(N) = Σ e^{iS_BDG} (ec. 2.3). **En ningún momento se identifica Λ con
las fluctuaciones de S_BDG.** Esa identificación, a través del término −2ΛV
de MYZ (ec. 4.6), es una hipótesis de este proyecto, no del modelo.

**2. Model 1 impone el exponente 1/2 por construcción** (sec. 3.1).
- "We next make the ansatz that (3.8) is true even element-wise, in the sense
  that each causal set element contributes a random variable with standard
  deviation 8πα/ℓ²_cs to Λ" (ec. 3.10).
- S_Λ es la suma de esas variables independientes (ec. 3.11). Por el teorema
  central del límite, S_Λ ∼ N(0, √N/2) y Λ(x) ∼ N(0, 8πα/√V) (ec. 3.13).
- La recursión (3.14-3.16) añade √ΔV ξ_n con ξ_n normal estándar e
  independiente: es un paseo aleatorio.
- Lo único flexible es la forma de la distribución: "any other distribution
  with mean 0 and standard deviation 8πα/ℓ²_cs would work just as well for
  our purposes".
- El único parámetro libre es α (ec. 3.9), que fija la **amplitud** (se
  exploran α = 0.005-0.012, sec. 5). Ningún parámetro del modelo puede
  absorber otro exponente.
- Los autores admiten que "the true causal set mechanism that gives rise to
  Everpresent Λ is not fully understood".
- Model 2 (sec. 5.4.1) sustituye la dinámica por un proceso gaussiano en
  Ω_Λ, con parámetros α̃ y µ que fijan amplitud y tiempo de correlación. Está
  construido para reproducir la misma propiedad de Model 1: Ω_Λ = O(1) en
  todas las épocas.

**3. Por qué otro exponente no es tolerable** (derivación propia, no está en
el artículo, que no discute otros exponentes). Si las fluctuaciones de la
acción escalan como ΔS ∝ N^p y Λ ∼ S/V, entonces ΔΛ ∝ V^(p−1). Con V ∼ H^(−4):

  ΔΛ / H² ∝ H^(2 − 4p).

Sólo con p = 1/2 la fracción Ω_Λ fluctúa con amplitud constante en todas las
épocas. Eso es lo que da nombre al modelo ("Everpresent") y lo que resuelve el
problema de la constante cosmológica "without fine tuning" (resumen). Con
p = 0.72, ΔΛ/H² ∝ H^(−0.88): la energía oscura crece sin límite respecto a la
densidad crítica a medida que H disminuye. Con p ≈ 1.2 (4D), ΔΛ crece en
términos absolutos con el volumen. Ajustar α sólo puede hacer Ω_Λ ∼ 1 en una
época concreta (hoy), y eso es exactamente el ajuste fino que el modelo
pretende evitar. Además, sin p = 1/2 desaparece el paseo aleatorio de
incrementos independientes (3.15) en el que se basa la simulación de Model 1.

**Respuesta de la Fase 1.** El modelo exige estrictamente el exponente 1/2,
pero lo obtiene de la fluctuación de Poisson del número de elementos (δN =
√N), con contribuciones independientes por elemento. No lo obtiene de la
acción BD. El exponente no es un parámetro ajustable: α sólo controla la
amplitud.

### Fase 2: régimen ερV > 100 (Minkowski 2D)

Código: `scripts/goal3c_asymptotic.py` (resultados en `figures/goal3c_sim.json`,
`figures/goal3c_pred.json` y `figures/goal3c_results.json`), tests en
`tests/test_asymptotic_scaling.py`.

**Coste y alcance.** Sólo 2D. La cuadratura de |A∩B| en 4D cuesta ~10–30 min
por densidad ya con ρV ≤ 100 (ver la ampliación del Goal 3), y la simulación
4D con ρV ≫ 100 no es viable en este equipo. En 2D se llegó a ρV = 3200 con
2000 sprinklings por densidad. Cada sprinkling de ρV = 3200 cuesta varios
segundos (~1.5 h en total). Los mismos sprinklings sirven para ε = 1, 0.5 y
0.25, así que ερV va de 200 a 3200: más de un orden de magnitud por encima de
100.

**Reglas fijadas antes de ver los resultados:**
- acuerdo simulación-predicción: |z| < 4 (σ combinada);
- p compatible con 0.5: |p − 0.5| < 4σ_p;
- p estable entre regímenes: |p_alto − p_bajo| < 4σ combinada.

**Resultados:**

| ρV | ε | (ΔS)² predicción | (ΔS)² simulación | z |
|---|---|---|---|---|
| 800 | 1 | 84 994 ± 852 | 89 698 ± 2856 | 1.58 |
| 800 | 0.5 | 1811 ± 42 | 1810 ± 58 | −0.02 |
| 800 | 0.25 | 103.7 ± 3.3 | 108.8 ± 3.6 | 1.03 |
| 1600 | 1 | 202 790 ± 2209 | 199 363 ± 6130 | −0.53 |
| 1600 | 0.5 | 4113 ± 109 | 4176 ± 128 | 0.37 |
| 1600 | 0.25 | 250.4 ± 9.1 | 268.2 ± 8.4 | 1.43 |
| 3200 | 1 | 457 192 ± 5463 | 456 621 ± 14 357 | −0.04 |
| 3200 | 0.5 | 9601 ± 283 | 9325 ± 292 | −0.68 |
| 3200 | 0.25 | 609 ± 24 | 602 ± 19 | −0.24 |

Los 9 puntos cumplen |z| < 4 (máximo 1.58): el formalismo MYZ extendido
sigue siendo exacto también en este régimen.

Exponentes (ajuste ponderado de la simulación, ΔS ∝ (ρV)^p):

| ε | p (ρV = 25–400) | p (ρV = 800–3200) | salto entre regímenes | p alto vs 0.5 |
|---|---|---|---|---|
| 1 | 0.718 ± 0.006 | 0.587 ± 0.016 | −7.7σ | +5.4σ |
| 0.5 | 0.722 ± 0.006 | 0.591 ± 0.016 | −7.6σ | +5.6σ |
| 0.25 | 0.772 ± 0.006 | 0.617 ± 0.017 | −8.9σ | +7.0σ |

Según las reglas fijadas: **el exponente NO es estable** (baja más de 7σ) y
**sigue sin ser compatible con 0.5** (más de 5σ por encima). Las
predicciones dan lo mismo (p alto = 0.608, 0.601, 0.638).

**El exponente es efectivo, no asintótico** (análisis hecho *a posteriori*,
tras ver los datos). El exponente local entre densidades consecutivas
(predicciones, ρV = 25 → 3200) baja de forma sostenida, sin estabilizarse:

- ε = 1: 0.81, 0.73, 0.69, 0.65, 0.65, 0.63, 0.59
- ε = 0.5: 0.83, 0.78, 0.66, 0.66, 0.65, 0.59, 0.61
- ε = 0.25: 0.93, 0.82, 0.73, 0.65, 0.63, 0.64, 0.64

Una ley de potencias única se rechaza con fuerza al ajustar (ΔS)²/ρV en todo
el rango (χ²/dof = 94, 36 y 43). La forma (ΔS)² = ρV(a + b ln ρV) ajusta
mucho mejor (χ²/dof = 11.7, 1.8 y 2.0; con ε = 1 hace falta algo más en las
densidades bajas). Si esa forma es la correcta, ΔS ∝ √(ρV ln ρV)
asintóticamente: escalado de Poisson **con una corrección logarítmica**, y el
"0.72" del Goal 3 sería un exponente efectivo de densidades pequeñas.
**Esto es una hipótesis, no un resultado verificado**: se formuló tras ver
los datos, sólo cubre 2D y dos décadas de ρV, y no se ha derivado
analíticamente. Una explicación natural, sin comprobar, son contribuciones
de intervalos casi nulos a lo largo de la frontera del diamante.

![Goal 3c](figures/goal3c_asymptotic.png)

**Tests** (`tests/test_asymptotic_scaling.py`, márgenes fijados de antemano):
- un punto calculado en vivo con ε = 0.2 y ρV = 600 (ερV = 120), a 4σ;
- comprobación de los 9 puntos del cálculo largo (|z| < 4) y de que los
  exponentes p_sim y p_pred del régimen alto coinciden a 4σ.

Con el error deliberado ε → ε² en la probabilidad conjunta, el test en vivo
falla.

### Fase 3: veredicto

**¿Hay alguna versión del mecanismo compatible con lo que exige
Everpresent Λ?** Con los datos de este proyecto, **no**:

1. **Exponente.** Ninguna versión probada da p = 0.5: ni la local, ni la
   suavizada con ε = 0.5 o 0.25, ni el régimen ερV de 200 a 3200 (todas más
   de 5σ por encima con la regla fijada). En 4D, la única evidencia
   (acción local, ρV ≤ 100) es p ≈ 1.2, mucho peor. Lo más cerca que se llega
   es la tendencia 2D hacia un escalado de Poisson con corrección
   logarítmica. Aun en ese caso, ΔΛ ∝ V^(−1/2)·√ln V no es exactamente lo que
   pide el modelo, y no hay ningún dato que diga que pase lo mismo en 4D.
2. **Mecanismo.** Aunque el exponente fuera 1/2, el modelo (Fase 1) **no**
   deriva Λ de las fluctuaciones de la acción BD. Las deriva de la
   fluctuación de Poisson de N y de contribuciones independientes por
   elemento (ecs. 2.2, 3.6, 3.10-3.11). Las fluctuaciones de S_BD están
   dominadas por correlaciones entre intervalos, que son justo lo contrario
   de contribuciones independientes. Además, Model 1 usa el cono de luz
   pasado, no un diamante.
3. **Amplitud.** (ΔS)²/ρV va de ~0.2 a ~140 según ε y ρV; no es una
   constante O(1) por elemento. α sólo absorbería una amplitud fija.

**¿Tiene base viable el Goal 4 tal como está planteado?** **No.** Generar
realizaciones estocásticas de Λ(t) con Model 1 "a partir de las
fluctuaciones de la acción BD" supondría un mecanismo que ni el artículo
propone ni los datos de este proyecto respaldan. Model 1 se puede simular
tal cual, pero entonces es independiente de todo el trabajo de los Goals 2-3
y no aportaría nada nuevo.

**Propuesta para el Goal 4** (no implementada):
- Documentar como hallazgo del proyecto que las fluctuaciones de la acción BD
  en un diamante de Minkowski no tienen el escalado de Poisson que requiere
  Everpresent Λ: p efectivo de 0.6–0.8 en 2D y ~1.2 en 4D a las densidades
  accesibles. Incluir las cotas cuantitativas de este README.
- Antes de FLRW, cerrar la pregunta abierta más importante: ¿en 4D el
  exponente efectivo también baja con ρV, como en 2D? La predicción MYZ es
  más barata que la simulación en densidades altas, pero en 4D la cuadratura
  es el cuello de botella: hace falta acelerarla o reducir su precisión de
  forma controlada.
- Comprobar en 2D la hipótesis ρV(a + b ln ρV) con una predicción
  independiente (análisis asintótico del formalismo MYZ, o densidades
  mayores), y localizar de dónde sale el logaritmo, por ejemplo separando
  las contribuciones de la frontera.
- Si se va a FLRW, hacerlo como **medida** y no como supuesto: calcular ΔS
  en conos de luz pasados de FLRW y comparar con δΛ ∝ V^(−1/2), en vez de
  generar Λ(t) suponiendo que la relación se cumple.

## Referencias

- Bombelli, Lee, Meyer & Sorkin (1987), Phys. Rev. Lett. 59, 521 — sprinkling de Poisson.
- Myrheim, J. (1978), "Statistical geometry", CERN preprint TH-2538 — ordering fraction y dimensión.
- Meyer, D. A. (1988), tesis doctoral, MIT — expresión cerrada de f(d).
- Reid, D. D. (2003), Phys. Rev. D 67, 024034, arXiv:gr-qc/0207103 — estimador práctico de dimensión.
- Surya, S. (2019), Living Rev. Relativ. 22, 5 — revisión general.
- Benincasa & Dowker (2010), Phys. Rev. Lett. 104, 181301, arXiv:1001.2725 — acción BD en 4D.
- Sorkin, R. D. (2007), arXiv:gr-qc/0703099 — d'Alembertiano 2D y suavizado ε.
- Benincasa, Dowker & Schmitzer (2011), Class. Quantum Grav. 28, 105018, arXiv:1011.5191 — acción 2D y su media.
- Das, Nasiri & Yazdi (2023), JCAP 10, 047, arXiv:2304.03819 — Everpresent Λ (I), Model 1.
- Das, Nasiri & Yazdi (2024), JCAP 10, 076, arXiv:2307.13743 — Everpresent Λ (II), tests cosmológicos.
- Dowker & Glaser (2013), Class. Quantum Grav. 30, 195016, arXiv:1305.2588 — f_d(n, ε) en dimensión d.
- Moradi, Yazdi & Zilhão, arXiv:2407.03395 — kernels de fluctuación (Goal 3).
