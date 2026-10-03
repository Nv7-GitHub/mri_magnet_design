"""
Magnetic forces on the ring-holder stack and the resulting bolt loads.

Force on each magnet: a uniformly magnetized cube is equivalent to magnetic surface charge sigma = +-M on its N and S
faces, so F = M * (integral of B_ext over the N face - integral over the S face), with B_ext the field of all other
magnets (Gauss-Legendre quadrature on the faces). This is exact for rigid magnets (mu_r = 1), unlike a dipole
approximation, which is poor for cubes 1-2 mm apart.

Bolt loads: for each joint between slabs (and the top end cap), the load on everything above the joint is the force on
the magnets in the slabs above plus the upward push of the joint's own magnets on their lid (the slab above). The
separating force and tilting moment are shared by the bolts as a rigid-plate joint, each bolt in proportion to its
tensile stiffness (stress area). Bolt positions follow make_ring_cad.py with the same arguments.

Usage:
  python check_bolt_loads.py results/BEST_bore100_cyl40x40_649mag_427ppm.xlsx --inner-holes 6
"""

import argparse

import numpy as np
import pandas as pd

import magsimulator as ms
from make_ring_cad import load_ledger, outer_hole_angles, place_inner_holes, pocket_outline, notch_outline

STRESS_AREA = {2.0: 2.07, 2.5: 3.39, 3.0: 5.03, 4.0: 8.78}     # ISO metric coarse, mm^2


def magnet_forces(L, Br=1.32, n_quad=6):
    """force (N) on each magnet and its moment about the origin (N m), plus the quadrature points (mm)"""
    mu0 = 4e-7*np.pi
    M = Br/mu0
    a = L['Magnet_length'].values[0]
    col = ms.build_magnet_collection(L, [Br*1000, 0, 0])
    cubes = col.children
    g, w = np.polynomial.legendre.leggauss(n_quad)
    g, w = g*a/2, w*a/2
    GY, GZ = np.meshgrid(g, g)
    W = np.outer(w, w).ravel()
    loc = np.vstack([np.c_[np.full(GY.size, s*a/2), GY.ravel(), GZ.ravel()] for s in (1, -1)])
    sgn = np.repeat([1.0, -1.0], GY.size)
    Wf = np.tile(W, 2)
    pts = np.array([c.orientation.apply(loc) + c.position for c in cubes])
    B = col.getB(pts.reshape(-1, 3)).reshape(len(cubes), -1, 3)
    Bself = np.array([c.getB(p) for c, p in zip(cubes, pts)])
    dF = M*sgn[None, :, None]*(B - Bself)*1e-3*(Wf*1e-6)[None, :, None]
    return dF.sum(1), np.cross(pts*1e-3, dF).sum(1), pts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ledger')
    ap.add_argument('--Br', type=float, default=1.32, help='remanence, T')
    ap.add_argument('--tol', type=float, nargs=3, default=[0.15, 0.15, 0.3])
    ap.add_argument('--bore', type=float, default=100.0)
    ap.add_argument('--min-wall', type=float, default=1.5)
    ap.add_argument('--relief', type=float, default=0.4)
    ap.add_argument('--outer-holes', type=int, default=8)
    ap.add_argument('--outer-bolt', type=float, default=3.0)
    ap.add_argument('--outer-tol', type=float, default=0.4)
    ap.add_argument('--hole-key', type=float, default=10.0)
    ap.add_argument('--inner-holes', type=int, default=6)
    ap.add_argument('--inner-bolt', type=float, default=2.0)
    ap.add_argument('--inner-tol', type=float, default=0.4)
    ap.add_argument('--hole-wall', type=float, default=0.8)
    ap.add_argument('--yield-mpa', type=float, default=150.0,
                    help='bolt yield stress for the margin, MPa (150 = conservative for brass CW614N / C360)')
    args = ap.parse_args()

    L = load_ledger(args.ledger)
    a = L['Magnet_length'].values[0]
    tx, ty, tz = args.tol
    wx, wy, depth = a + tx, a + ty, a + tz

    # bolt positions exactly as make_ring_cad.py places them
    pockets = [pocket_outline(r['X-pos'], r['Y-pos'], r['Z-rot'], wx + 2*args.relief, wy + 2*args.relief) for _, r in L.iterrows()]
    tris = [notch_outline(r['X-pos'], r['Y-pos'], r['Z-rot'], wx, 2.0, 1.0) for _, r in L.iterrows()]
    d_out, d_in = args.outer_bolt + args.outer_tol, args.inner_bolt + args.inner_tol
    r_out = max(np.hypot(*P.T).max() for P in pockets + tris) + args.min_wall + d_out/2
    bolts = [(r_out*np.cos(np.deg2rad(t)), r_out*np.sin(np.deg2rad(t)), args.outer_bolt)
             for t in outer_hole_angles(args.outer_holes, args.hole_key)]
    if args.inner_holes:
        r_in = args.bore/2 + args.hole_wall + d_in/2
        ang, _ = place_inner_holes(args.inner_holes, d_in, r_in, pockets + tris, args.hole_wall)
        bolts += [(r_in*np.cos(np.deg2rad(t)), r_in*np.sin(np.deg2rad(t)), args.inner_bolt) for t in ang]
    bx = np.array([b[0] for b in bolts])*1e-3
    by = np.array([b[1] for b in bolts])*1e-3
    size = np.array([b[2] for b in bolts])
    k = np.array([STRESS_AREA[s] for s in size])

    def bolt_loads(Fz, Mx, My):
        A = np.array([[k.sum(), (k*by).sum(), -(k*bx).sum()],
                      [(k*by).sum(), (k*by*by).sum(), -(k*bx*by).sum()],
                      [-(k*bx).sum(), -(k*bx*by).sum(), (k*bx*bx).sum()]])
        u, ax, ay = np.linalg.solve(A, [Fz, Mx, -My])
        return k*(u + ax*by - ay*bx)

    F, Mo, pts = magnet_forces(L, args.Br)
    Fm = np.linalg.norm(F, axis=1)
    print(f'force per magnet: median {np.median(Fm):.2f} N, max {Fm.max():.2f} N (net on whole array {np.linalg.norm(F.sum(0)):.1e} N)')

    zs = L['Z-pos'].round(3).values
    rings = np.sort(np.unique(zs))
    tops = rings + depth/2
    print('\njoint (plane)          separating   shear   tilt moment   torque    max bolt load ' +
          ' / '.join(f'M{s:g}' for s in np.unique(size)[::-1]))
    worst = {s: 0.0 for s in np.unique(size)}
    for i, z in enumerate(rings):
        zi = tops[i]*1e-3
        above, up = zs > z, (zs == z) & (F[:, 2] > 0)
        Fu = F[above].sum(0) + [0, 0, F[up, 2].sum()]
        lever = np.c_[pts[up][:, :, :2].mean(1)*1e-3, np.full(up.sum(), zi)]
        Mu = Mo[above].sum(0) + np.cross(lever, np.c_[np.zeros((up.sum(), 2)), F[up, 2]]).sum(0) - np.cross([0, 0, zi], Fu)
        fb = bolt_loads(Fu[2], Mu[0], Mu[1])
        per = {s: fb[size == s].max() for s in worst}
        for s in worst:
            worst[s] = max(worst[s], per[s])
        name = 'top end cap' if i == len(rings) - 1 else f'slab {i}/{i+1}'
        print(f'{name:12s} z {tops[i]:+7.2f}  {Fu[2]:+7.1f} N  {np.hypot(*Fu[:2]):5.1f} N  {np.hypot(*Mu[:2]):7.3f} N m  '
              f'{Mu[2]:+6.3f} N m   ' + ' / '.join(f'{per[s]:+5.1f} N' for s in sorted(worst, reverse=True)))
    print('(the bottom end cap carries the mirror image of the top one)')
    for s in sorted(worst, reverse=True):
        cap = STRESS_AREA[s]*args.yield_mpa
        print(f'M{s:g}: worst tension from magnet forces {worst[s]:.1f} N; yield at {args.yield_mpa:g} MPa '
              f'{cap:.0f} N -> margin {cap/max(worst[s], 1e-9):.0f}x')


if __name__ == '__main__':
    main()
