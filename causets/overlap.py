"""Volumen de la intersección de dos intervalos causales en Minkowski 4D,
|I(y1, x1) ∩ I(y2, x2)|, por cuadratura determinista.

No hay forma cerrada (MYZ, arXiv:2407.03395, sólo parametrizan J⁻(x1)∩J⁻(x2)
en 2D, apéndice C.1). Aquí se reduce a una integral doble:

    I(y1,x1) ∩ I(y2,x2) = J⁺(y1) ∩ J⁺(y2) ∩ J⁻(x1) ∩ J⁻(x2).

A tiempo fijo t cada cono es una bola de R³ (centro la parte espacial del
vértice, radio |t - t_vértice|), y a (t, z) fijos cada bola es un disco del
plano (x, y). El área de la intersección de los cuatro discos se calcula
EXACTAMENTE con el teorema de Green: la frontera de una intersección de
discos está formada por arcos de sus circunferencias; para la circunferencia
i (centro c, radio r) se cortan los arcos por sus intersecciones con las
demás, se conservan los sub-arcos cuyo punto medio está dentro de todos los
otros discos, y cada sub-arco [θ1, θ2] aporta

    ½ ∮ (x dy - y dx) = ½ [r² (θ2 - θ1) + c_x r (sin θ2 - sin θ1) - c_y r (cos θ2 - cos θ1)].

El volumen es  ∫ dt ∫ dz Área(t, z),  con Gauss-Legendre en t y en z sobre
los intervalos donde todas las bolas/discos existen. El integrando tiene
singularidades de tipo raíz en los extremos; el cambio de variable
s -> (3s - s³)/2 (derivada nula en s = ±1) las suaviza.

El volumen del corte V3(t) tiene "codos" donde dos esferas pasan a ser
tangentes (una bola entra en otra o deja de cortarla). Para cada par de conos
la condición |Δc| = R_i ± R_j es LINEAL en t (los radios son |t - t_vértice|),
así que esos instantes se calculan exactamente y el rango de t se parte en
paneles con ellos. Quedan codos en z (y los de tangencias triples), por lo
que la convergencia en z es algebraica: la precisión se comprueba en los tests
contra casos exactos y contra Monte Carlo de alta estadística.
"""

from __future__ import annotations

import numpy as np

_TWO_PI = 2.0 * np.pi


def _nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodos y pesos en [-1, 1] de Gauss-Legendre tras s -> (3s - s³)/2."""
    s, w = np.polynomial.legendre.leggauss(n)
    return (3 * s - s**3) / 2, w * 1.5 * (1 - s**2)


def disk_intersection_area(c: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Área exacta de la intersección de k discos. ``c``: (..., k, 2), ``r``:
    (..., k). Un radio <= 0 (disco vacío) da área 0."""
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
    # ángulos sin corte -> se repite el máximo válido (arcos de longitud 0);
    # sin ningún corte, todos 0 y el arco cíclico es la circunferencia entera
    vmax = np.max(np.where(ok, ang, -np.inf), axis=-1, keepdims=True)
    vmax = np.where(np.isfinite(vmax), vmax, 0.0)
    ang = np.sort(np.where(ok, ang, vmax), axis=-1)
    start = ang
    stop = np.concatenate([ang[..., 1:], ang[..., :1] + _TWO_PI], axis=-1)
    mid = (start + stop) / 2
    # punto medio de cada sub-arco de la circunferencia i: (..., k, 2k, 2)
    ri_ = r[..., :, None]
    px = c[..., :, None, 0] + ri_ * np.cos(mid)
    py = c[..., :, None, 1] + ri_ * np.sin(mid)
    inside = np.ones(px.shape, dtype=bool)
    for j in range(k):
        dx = px - c[..., j, None, None, 0]
        dy = py - c[..., j, None, None, 1]
        in_j = dx * dx + dy * dy <= r[..., j, None, None] ** 2 * (1 + 1e-12)
        own = (np.arange(k) == j)[:, None]  # la propia circunferencia no se comprueba
        inside &= in_j | own
    # circunferencias coincidentes (vértice compartido, p. ej. J2/J3): sólo
    # cuenta la de menor índice
    same = (d <= 1e-12 * ri) & (np.abs(ri - rj) <= 1e-12 * ri) & np.tri(k, k, -1, dtype=bool)
    inside &= ~np.any(same, axis=-1)[..., None]
    cx, cy = c[..., :, None, 0], c[..., :, None, 1]
    green = ri_**2 * (stop - start) + cx * ri_ * (np.sin(stop) - np.sin(start)) - cy * ri_ * (np.cos(stop) - np.cos(start))
    area = 0.5 * np.sum(np.where(inside, green, 0.0), axis=(-1, -2))
    return np.where(empty, 0.0, np.maximum(area, 0.0))


def interval_intersection_volume_4d(y1, x1, y2, x2, n_t: int = 6, n_z: int = 32, chunk: int = 128) -> np.ndarray:
    """|I(y1,x1) ∩ I(y2,x2)| en Minkowski 4D (fila a fila; 0 si alguno de los
    intervalos es vacío o no se cortan). ``n_t``: nodos por panel en t (hay 11
    paneles, ver docstring del módulo); ``n_z``: nodos en z."""
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
    # instantes de tangencia de cada par de conos (lineales en t)
    ev = []
    for i in range(2):
        for j in range(2):
            dc = np.linalg.norm(tops[:, i, 1:] - bots[:, j, 1:], axis=1)
            s_ij = tops[:, i, 0] + bots[:, j, 0]
            ev += [(s_ij - dc) / 2, (s_ij + dc) / 2]  # |Δc| = |R_top - R_bot|
    ev.append((tops[:, 0, 0] + tops[:, 1, 0] - np.linalg.norm(tops[:, 0, 1:] - tops[:, 1, 1:], axis=1)) / 2)
    ev.append((bots[:, 0, 0] + bots[:, 1, 0] + np.linalg.norm(bots[:, 0, 1:] - bots[:, 1, 1:], axis=1)) / 2)
    edges = np.sort(np.clip(np.column_stack([t_lo, *ev, t_hi]), t_lo[:, None], t_hi[:, None]), axis=1)
    a, b = edges[:, :-1, None], edges[:, 1:, None]  # (m, paneles, 1)
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
    slab = np.sum(area * wz, axis=-1) * hz * ok_t  # volumen 3D de cada corte
    return np.where(valid, np.sum(slab * wt_full, axis=-1), 0.0)
