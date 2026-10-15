# Fluctuations of the Benincasa–Dowker action and Everpresent Λ

Code and results for the paper

> C. Echeverría-Arrondo, *Non-Poissonian fluctuations of the Benincasa–Dowker
> action and the fate of Everpresent Λ* (in preparation).

The package simulates causal sets by Poisson sprinkling, evaluates the
Benincasa–Dowker (BD) action and its fluctuations (formalism of Moradi, Yazdi &
Zilhão, arXiv:2407.03395), and confronts stochastic Λ(t) histories with
supernova, BAO and CMB data.

**Main results**

- In *d* dimensions the variance of the BD action is dominated by the Poisson
  noise in the number of nearly null intervals, which grows with the boost
  range of the region: (ΔS)² ∝ (ρV)^(2−2/d). In 4D, ΔS ∝ (ρV)^(3/4) and
  δΛ ∝ V^(−1/4), not the V^(−1/2) of Everpresent Λ.
- The exponent is the same in causal diamonds, truncated past light cones and
  FLRW past cones reaching the big bang, for the local and smeared actions.
- Stochastic Λ(t) histories with this scaling fail Pantheon+SH0ES + DESI DR2
  BAO + Planck distance priors (0 of 10⁴ histories acceptable,
  Δχ²_min = +124), robustly in the memory parameter and across a family of
  Ornstein–Uhlenbeck dynamics.

A detailed lab notebook (in Spanish), with every pre-registered criterion,
intermediate result and deliberate-error check, is in
[`README.es.md`](README.es.md).

## Installation

Tested with Python 3.12.3 on Linux.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m pytest -q          # 209 tests, ~12 min
```

The observational data are not stored in the repository. Download them (with
SHA-256 verification) with

```bash
./scripts/get_pantheon.sh    # Pantheon+SH0ES and DESI DR2 BAO, into data/
```

Tests that need these data are skipped if they are missing.

## Reproducing the results

```bash
./reproduce.sh fast   # all figures and tables from the saved JSON results, then the tests
./reproduce.sh full   # recompute everything from scratch (many hours; fixed seeds)
python paper/make_figures.py   # the figures of the paper, from the saved results
```

Every long computation stores its results as JSON in `figures/`, so the
analysis can be redone without re-running the simulations. All random
numbers come from fixed seeds.

## Where each result of the paper comes from

| Paper | Script | Results | Tests |
|---|---|---|---|
| §2, validation of the MYZ formalism (2D, 4D) | `scripts/goal3_fluctuations.py`, `scripts/smeared_fluctuations.py` | `figures/goal3_results.json`, `figures/smeared_results_{2d,4d}.json` | `test_fluctuations.py`, `test_smeared_fluctuations.py`, `test_overlap.py` |
| §3, logarithm in 2D and rapidity plateau | `scripts/goal3c_asymptotic.py`, `scripts/goal3d_log_origin.py` | `figures/goal3c_*.json`, `figures/goal3d_*.json` | `test_asymptotic_scaling.py`, `test_log_origin.py` |
| §4.2, 4D diamond (figure 1) | `scripts/step2_4d_scaling.py` | `figures/step2_4d_*.json` | `test_4d_scaling.py` |
| §4.3, past cones in 2D | `scripts/cone_fluctuations.py`, `scripts/flrw_fluctuations.py` | `figures/cone_*.json`, `figures/flrw_*.json` | `test_cone.py`, `test_flrw.py` |
| §4.3, past cones in 4D (table 1) | `scripts/cone4d_scaling.py`, `scripts/flrw4d_scaling.py` | `figures/cone4d_*.json`, `figures/flrw4d_*.json` | `test_cone4d.py`, `test_flrw4d.py` |
| §4.4, smeared action (figure 2) | `scripts/goal5_smeared4d.py`, `scripts/goal5b_decomposition.py` | `figures/goal5_*.json`, `figures/goal5b_results.json` | `test_goal5.py`, `test_goal5b.py` |
| §5.1, amplitude and ℓ_k | `scripts/goal4_compatibility.py` | `figures/goal4a_results.json` | `test_goal4a.py` |
| §5.2, correlation along nested cones | `scripts/goal4b_correlations.py` | `figures/goal4b_*.json` | — |
| §6, stochastic Λ(t) and Model 1 validation | `scripts/goal4b_lambda.py` | `figures/goal4b_lambda.json` | `test_goal4b.py` |
| §6, supernovae only | `scripts/goal4c_sn_fit.py` | `figures/goal4c_*.json` | `test_goal4c.py` |
| §6, SN + BAO + CMB (table 2) | `scripts/goal4d_combined.py` | `figures/goal4d_*.json` | `test_goal4d.py` |
| §6, robustness (figure 3) | `scripts/goal4e_robustness.py` | `figures/goal4e_*.json` | `test_goal4e.py` |

Large intermediate files (interval-cardinality histograms of the 4D
sprinklings) go to `data/`, which is not versioned; the scripts regenerate
them.

## Package structure

| Module | Purpose |
|---|---|
| `causets/sprinkling.py` | Poisson sprinkling; points sorted by time |
| `causets/causal.py` | Minkowski causal matrix and partial-order checks |
| `causets/regions.py` | Regions: causal diamond, truncated past cone, FLRW past cone, de Sitter |
| `causets/dimension.py` | Myrheim–Meyer dimension estimator |
| `causets/intervals.py` | Interval cardinalities n(y, x) and abundances N_k |
| `causets/action.py` | BD operator and action, local and smeared, in 2D and 4D |
| `causets/continuum.py` | Exact finite-density means (Mecke formula) in diamonds, cones and de Sitter |
| `causets/geometry.py` | Geometries for the fluctuation formalism: weights, interval and overlap volumes |
| `causets/overlap.py` | Exact volume of the intersection of two causal intervals in 4D Minkowski |
| `causets/fluctuations.py` | MYZ variance formalism (local and smeared), exact same-pair term and Mecke covariance |
| `causets/everpresent.py` | Past-cone volume in ΛCDM, conversion ΔS → δΛ, amplitudes |
| `causets/stochastic_lambda.py` | Stochastic Λ(t) with feedback on the expansion (Model 1 and BD) |
| `causets/supernovae.py` | Pantheon+SH0ES likelihood with exactly compressed χ² |
| `causets/cmb_bao.py` | Sound horizon, z* and z_drag, CMB distance priors, DESI DR2 BAO |
| `scripts/` | One script per computation; long runs save JSON in `figures/` |
| `tests/` | pytest suite with explicit statistical tolerances (4σ, fixed in advance) |
| `paper/` | LaTeX source of the paper and the script for its figures |

Conventions: coordinates `(t, x_1, …, x_{d−1})`, c = 1; `C[i, j] = True`
means i ≺ j strictly. Since points are sorted by t, `C` is upper triangular.

## Citation

If you use this code, please cite the paper and the archived version of the
software (see [`CITATION.cff`](CITATION.cff)).

## License

MIT; see [`LICENSE`](LICENSE).

The development of this code was assisted by Claude (Anthropic).
