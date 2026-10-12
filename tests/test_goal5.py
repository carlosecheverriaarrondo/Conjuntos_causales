"""Goal 5: momentos exactos de la acción suavizada en 4D (``smeared_pair_moment``,
``smeared_pair_mecke_slope``) y el histograma de ``scripts/goal5_smeared4d.py``.

Tolerancias fijadas antes de ejecutar estos tests (4σ en los de Monte Carlo).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from step2_4d_scaling import exact_parts  # noqa: E402

from causets import sprinkle  # noqa: E402
from causets.causal import minkowski_causal_matrix  # noqa: E402
from causets.fluctuations import (  # noqa: E402
    action_coefficients,
    pair_weight,
    smeared_pair_mecke_slope,
    smeared_pair_moment,
    thinned_abundance_vector,
)
from causets.geometry import MinkowskiDiamondGeometry  # noqa: E402

_spec = importlib.util.spec_from_file_location("goal5", ROOT / "scripts" / "goal5_smeared4d.py")
g5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g5)

LAM = 50.0


def test_local_limit_reproduces_step2():
    """Con eps = 1, J¹ y el cruzado N-k coinciden con las fórmulas del Paso 2."""
    ref = exact_parts(LAM)
    a_N = action_coefficients(4, 1.0)[0]
    assert smeared_pair_moment(4, LAM, 1.0, 2) == pytest.approx(ref["J1"], rel=1e-6)
    assert 2 * a_N * LAM * smeared_pair_mecke_slope(4, LAM, 1.0, h=1e-3) == pytest.approx(ref["Nk"], rel=1e-4)


def test_histogram_gives_thinned_action():
    """a_N N + hist · f_eps reproduce a · Ñ (abundancias adelgazadas, Goal 3)."""
    g = MinkowskiDiamondGeometry(4)
    C = minkowski_causal_matrix(sprinkle(g.region, 200 / g.volume, np.random.default_rng(5)))
    eps = 0.3
    a = action_coefficients(4, eps)
    h = g5.histogram(C)
    S_hist = a[0] * C.shape[0] + h @ pair_weight(4, np.arange(len(h)), eps)
    S_thin = a @ thinned_abundance_vector(C, len(a) - 1, eps)
    assert S_hist == pytest.approx(S_thin, rel=1e-10)


def test_mecke_covariance_monte_carlo():
    """Cov(N, F) simulada = rho V g y <F> = E[F] exactos (eps = 0.5), |z| < 4."""
    eps, M = 0.5, 3000
    g = MinkowskiDiamondGeometry(4)
    rng = np.random.default_rng(2026)
    N, F = np.empty(M), np.empty(M)
    for i in range(M):
        C = minkowski_causal_matrix(sprinkle(g.region, LAM / g.volume, rng))
        h = g5.histogram(C)
        N[i], F[i] = C.shape[0], h @ pair_weight(4, np.arange(len(h)), eps)
    dN, dF = N - N.mean(), F - F.mean()
    cov, cov_se = np.sum(dN * dF) / (M - 1), np.std(dN * dF, ddof=1) / np.sqrt(M)
    assert abs(cov - LAM * smeared_pair_mecke_slope(4, LAM, eps)) < 4 * cov_se
    assert abs(F.mean() - smeared_pair_moment(4, LAM, eps, 1)) < 4 * F.std(ddof=1) / np.sqrt(M)
