# Conjuntos causales: sprinkling, dimensión y (próximamente) acción BD

Simulador numérico de conjuntos causales por sprinkling de Poisson. Ver
`CLAUDE.md` para el roadmap completo (Goals 1–4).

## Uso

```bash
source venv/bin/activate
python -m pytest -q                      # tests (≈20 s)
python scripts/goal1_convergence.py      # figura figures/goal1_mm_convergence.png
```

## Estructura

| Módulo | Responsabilidad |
|---|---|
| `causets/regions.py` | Regiones del espacio-tiempo (`CausalDiamond` en Minkowski d-dim): volumen, pertenencia, caja envolvente |
| `causets/sprinkling.py` | Sprinkling de Poisson (`sprinkle`) y muestreo uniforme de N fijo (`sample_uniform`), puntos ordenados por t |
| `causets/causal.py` | Matriz causal de Minkowski y verificación de orden parcial (reflexividad, antisimetría, transitividad) |
| `causets/dimension.py` | Estimador de dimensión de Myrheim-Meyer |
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

## Referencias

- Bombelli, Lee, Meyer & Sorkin (1987), Phys. Rev. Lett. 59, 521 — sprinkling de Poisson.
- Myrheim, J. (1978), "Statistical geometry", CERN preprint TH-2538 — ordering fraction y dimensión.
- Meyer, D. A. (1988), tesis doctoral, MIT — expresión cerrada de f(d).
- Reid, D. D. (2003), Phys. Rev. D 67, 024034, arXiv:gr-qc/0207103 — estimador práctico de dimensión.
- Surya, S. (2019), Living Rev. Relativ. 22, 5 — revisión general.
- Benincasa & Dowker (2010), Phys. Rev. Lett. 104, 181301 — acción BD (Goal 2).
- Moradi, Yazdi & Zilhão, arXiv:2407.03395 — kernels de fluctuación (Goal 3).
- Das, Nasiri & Yazdi, JCAP 10 (2023) 047; JCAP 10 (2024) 076 — Everpresent Λ (Goal 4).
