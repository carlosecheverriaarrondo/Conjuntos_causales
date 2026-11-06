"""Volume of the intersection of two causal intervals in 4D Minkowski,
|I(y1, x1) ∩ I(y2, x2)|, by deterministic quadrature.

There is no closed form (MYZ, arXiv:2407.03395, only parametrise J⁻(x1)∩J⁻(x2)
in 2D, appendix C.1). Here it is reduced to a double integral:

    I(y1,x1) ∩ I(y2,x2) = J⁺(y1) ∩ J⁺(y2) ∩ J⁻(x1) ∩ J⁻(x2).

At fixed time t each cone is a ball of R³ (centre the spatial part of the
vertex, radius |t - t_vertex|), and at fixed (t, z) each ball is a disc of the
(x, y) plane. The area of the intersection of the four discs is computed
EXACTLY with Green's theorem: the boundary of an intersection of discs is
made up of arcs of their circles; for circle i (centre c, radius r) the arcs
are cut at its intersections with the others, the sub-arcs whose midpoint
lies inside all the other discs are kept, and each sub-arc [θ1, θ2]
contributes

    ½ ∮ (x dy - y dx) = ½ [r² (θ2 - θ1) + c_x r (sin θ2 - sin θ1) - c_y r (cos θ2 - cos θ1)].

The volume is  ∫ dt ∫ dz Area(t, z),  with Gauss-Legendre in t and in z over
the intervals where all the balls/discs exist. The integrand has square-root
singularities at the endpoints; the change of variable s -> (3s - s³)/2
(vanishing derivative at s = ±1) smooths them out.

The volume of the slice V3(t) has "kinks" where two spheres become tangent
(one ball enters another or stops intersecting it). For each pair of cones
the condition |Δc| = R_i ± R_j is LINEAR in t (the radii are |t - t_vertex|),
so those instants are computed exactly and the t range is split into panels
at them. Kinks in z remain (and those of triple tangencies), so the
convergence in z is algebraic: the accuracy is checked in the tests against
exact cases and against high-statistics Monte Carlo.
"""

from __future__ import annotations

import numpy as np

_TWO_PI = 2.0 * np.pi


def _nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [-1, 1] after s -> (3s - s³)/2."""
    s, w = np.polynomial.legendre.leggauss(n)
    return (3 * s - s**3) / 2, w * 1.5 * (1 - s**2)


def disk_intersection_area(c: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Exact area of the intersection of k discs. ``c``: (..., k, 2), ``r``:
    (..., k). A radius <= 0 (empty disc) gives area 0."""
    c = np.asarray(c, float)
    r = np.asarray(r, float)
    k = r.shape[-1]
    empty = np.any(r <= 0, axis=-1)
    r = np.where(r > 0, r, 1.0)
    ci, cj = c[..., :, None, :], c[..., None, :, :]
    ri, rj = r[..., :, None], r[..., None, :]
    dvec = cj - ci  # (..., k, k, 2)
    d = np.hypot(dvec[..., 0], dvec[..., 1])
    phi = np.arctan2(dvec[..., 1], dvec[..., 0])
    with np.errstate(invalid="ignore", divide="ignore"):
        cosa = (ri**2 + d**2 - rj**2) / (2 * ri * d)
    cross = (d > np.abs(ri - rj)) & (d < ri + rj)
    alpha = np.arccos(np.clip(np.where(cross, cosa, 1.0), -1, 1))
    ang = np.concatenate([phi - alpha, phi + alpha], axis=-1) % _TWO_PI  # (..., k, 2k)
    ok = np.concatenate([cross, cross], axis=-1)
    # angles without an intersection -> the largest valid one is repeated (arcs of
    # length 0); with no intersection at all, all 0 and the cyclic arc is the whole circle
    vmax = np.max(np.where(ok, ang, -np.inf), axis=-1, keepdims=True)
    vmax = np.where(np.isfinite(vmax), vmax, 0.0)
    ang = np.sort(np.where(ok, ang, vmax), axis=-1)
    start = ang
    stop = np.concatenate([ang[..., 1:], ang[..., :1] + _TWO_PI], axis=-1)
    mid = (start + stop) / 2
    # midpoint of each sub-arc of circle i: (..., k, 2k, 2)
    ri_ = r[..., :, None]
    px = c[..., :, None, 0] + ri_ * np.cos(mid)
    py = c[..., :, None, 1] + ri_ * np.sin(mid)
    inside = np.ones(px.shape, dtype=bool)
    for j in range(k):
        dx = px - c[..., j, None, None, 0]
        dy = py - c[..., j, None, None, 1]
        in_j = dx * dx + dy * dy <= r[..., j, None, None] ** 2 * (1 + 1e-12)
        own = (np.arange(k) == j)[:, None]  # the circle itself is not checked
        inside &= in_j | own
    # coincident circles (shared vertex, e.g. J2/J3): only the one with the
    # lowest index counts
    same = (d <= 1e-12 * ri) & (np.abs(ri - rj) <= 1e-12 * ri) & np.tri(k, k, -1, dtype=bool)
    inside &= ~np.any(same, axis=-1)[..., None]
    cx, cy = c[..., :, None, 0], c[..., :, None, 1]
    green = ri_**2 * (stop - start) + cx * ri_ * (np.sin(stop) - np.sin(start)) - cy * ri_ * (np.cos(stop) - np.cos(start))
    area = 0.5 * np.sum(np.where(inside, green, 0.0), axis=(-1, -2))
    return np.where(empty, 0.0, np.maximum(area, 0.0))


def interval_intersection_volume_4d(y1, x1, y2, x2, n_t: int = 6, n_z: int = 32, chunk: int = 128) -> np.ndarray:
    """|I(y1,x1) ∩ I(y2,x2)| in 4D Minkowski (row by row; 0 if either interval
    is empty or they do not intersect). ``n_t``: nodes per panel in t (there are
    11 panels, see the module docstring); ``n_z``: nodes in z."""
    pts = [np.atleast_2d(np.asarray(p, float)) for p in (y1, x1, y2, x2)]
    m = len(pts[0])
    out = np.zeros(m)
    st, wt = _nodes(n_t)
    sz, wz = _nodes(n_z)
    for s in range(0, m, chunk):
        sl = slice(s, min(s + chunk, m))
        out[sl] = _volume_chunk(*(p[sl] for p in pts), st, wt, sz, wz)
    return out


def _volume_chunk(y1, x1, y2, x2, st, wt, sz, wz):
    tops = np.stack([x1, x2], axis=1)  # (m, 2, 4)
    bots = np.stack([y1, y2], axis=1)
    t_lo = bots[:, :, 0].max(axis=1)
    t_hi = tops[:, :, 0].min(axis=1)
    valid = t_hi > t_lo
    t_hi = np.where(valid, t_hi, t_lo + 1.0)
    # tangency instants of each pair of cones (linear in t)
    ev = []
    for i in range(2):
        for j in range(2):
            dc = np.linalg.norm(tops[:, i, 1:] - bots[:, j, 1:], axis=1)
            s_ij = tops[:, i, 0] + bots[:, j, 0]
            ev += [(s_ij - dc) / 2, (s_ij + dc) / 2]  # |Δc| = |R_top - R_bot|
    ev.append((tops[:, 0, 0] + tops[:, 1, 0] - np.linalg.norm(tops[:, 0, 1:] - tops[:, 1, 1:], axis=1)) / 2)
    ev.append((bots[:, 0, 0] + bots[:, 1, 0] + np.linalg.norm(bots[:, 0, 1:] - bots[:, 1, 1:], axis=1)) / 2)
    edges = np.sort(np.clip(np.column_stack([t_lo, *ev, t_hi]), t_lo[:, None], t_hi[:, None]), axis=1)
    a, b = edges[:, :-1, None], edges[:, 1:, None]  # (m, panels, 1)
    t = ((a + b) / 2 + (b - a) / 2 * st).reshape(len(a), -1)  # (m, nt)
    wt_full = ((b - a) / 2 * wt).reshape(len(a), -1)
    centers = np.concatenate([tops, bots], axis=1)[:, None, :, 1:]  # (m, 1, 4, 3)
    R = np.concatenate([tops[:, None, :, 0] - t[:, :, None], t[:, :, None] - bots[:, None, :, 0]], axis=2)  # (m, nt, 4)
    z_lo = np.max(centers[..., 2] - R, axis=2)
    z_hi = np.min(centers[..., 2] + R, axis=2)
    ok_t = z_hi > z_lo
    z_hi = np.where(ok_t, z_hi, z_lo + 1.0)
    hz = (z_hi - z_lo) / 2
    z = (z_lo + z_hi)[..., None] / 2 + hz[..., None] * sz  # (m, nt, nz)
    dz = z[..., None] - centers[:, :, None, :, 2]  # (m, nt, nz, 4)
    r2 = R[:, :, None, :] ** 2 - dz**2
    r = np.sqrt(np.maximum(r2, 0.0)) * (r2 > 0)
    c2 = np.broadcast_to(centers[:, :, None, :, :2], r.shape + (2,))
    area = disk_intersection_area(c2, r)  # (m, nt, nz)
    slab = np.sum(area * wz, axis=-1) * hz * ok_t  # 3D volume of each slice
    return np.where(valid, np.sum(slab * wt_full, axis=-1), 0.0)
