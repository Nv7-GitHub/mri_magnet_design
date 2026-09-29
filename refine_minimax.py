# -*- coding: utf-8 -*-
"""
Minimax refinement of an aligned-ring Halbach design: minimizes the largest field deviation over the target
cylinder directly (SLSQP: minimize t subject to |B_i - mean(B)| <= t at every check point).

Variables: ring slice positions along z (mirror symmetric), each ring's rotation offset and, with --radial,
a small radial shift of each ring. Magnet counts per ring are then changed by +-1, or one magnet moved between rings, where that helps (total count <= --max-magnets).
Constraints: B0 >= min, slice gaps >= cube + 3 mm, total length <= max, inner rings outside the bore wall,
layers at least sqrt(3)*cube + 3 mm apart, holder OD <= max, and every ring's magnets fit around it.

usage:
    python refine_minimax.py <ledger.xlsx> <out.xlsx> [--radial] [--min-b0 45] [--max-length 150] [--bore 100] [--max-od 200] [--rounds 3]
"""

import argparse
import time
import numpy as np
import pandas as pd
import magpylib as magpy
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import magsimulator
from refine_design import Design, rings_from_ledger, COLS


class RadialDesign(Design):
    def __init__(self, rings, a, br):
        super().__init__(rings, a, br)
        self.roff = {k: 0.0 for k in self.keys}

    def radius(self, key):
        return self.radii[key[0]] + self.roff[key]

    def magnets(self):
        pos, rotz = [], []
        for key in self.keys:
            k = key[1]
            r, n, ph = self.radius(key), self.n[key], self.phase[key]
            th = ph + 2*np.pi*np.arange(n)/n
            for z in ([0.0] if k == 0 else [self.zpos[k-1], -self.zpos[k-1]]):
                pos.append(np.stack([r*np.cos(th), r*np.sin(th), np.full(n, z)], axis=1))
                rotz.append(2*th)
        return np.vstack(pos), np.concatenate(rotz)


def field_x(design, sensors):
    pos, rotz = design.magnets()
    n_m, n_s = len(pos), len(sensors)
    B = magpy.getB('Cuboid', observers=np.tile(sensors, (n_m, 1)), position=np.repeat(pos, n_s, axis=0),
                   orientation=R.from_euler('z', np.repeat(rotz, n_s)[:, None]), magnetization=(design.br, 0, 0),
                   dimension=(design.a,)*3)
    return B[:, 0].reshape(n_m, n_s).sum(axis=0)


def ppm(B):
    return 1e6*(B.max()-B.min())/B.mean()


def field_ring(design, key, sensors):
    k = key[1]
    r, n, ph = design.radius(key), design.n[key], design.phase[key]
    th = ph + 2*np.pi*np.arange(n)/n
    zs = [0.0] if k == 0 else [design.zpos[k-1], -design.zpos[k-1]]
    pos = np.vstack([np.stack([r*np.cos(th), r*np.sin(th), np.full(n, z)], axis=1) for z in zs])
    rotz = np.concatenate([2*th for _ in zs])
    n_m, n_s = len(pos), len(sensors)
    B = magpy.getB('Cuboid', observers=np.tile(sensors, (n_m, 1)), position=np.repeat(pos, n_s, axis=0),
                   orientation=R.from_euler('z', np.repeat(rotz, n_s)[:, None]), magnetization=(design.br, 0, 0),
                   dimension=(design.a,)*3)
    return B[:, 0].reshape(n_m, n_s).sum(axis=0)


class Problem:
    PH = 10.0  # phase scaling so all variables are of order 1-10

    def __init__(self, d, sensors, args):
        self.d, self.sensors, self.args = d, sensors, args
        self.radial = args.radial
        self.nz, self.nk = len(d.zpos), len(d.keys)
        self.cache_x, self.cache = None, None
        # which rings each variable moves (a z variable moves every ring in that slice)
        self.affects = [[k for k in d.keys if k[1] == i+1] for i in range(self.nz)] + [[k] for k in d.keys]
        if self.radial:
            self.affects += [[k] for k in d.keys]

    def pack(self, t):
        v = list(self.d.zpos) + [self.d.phase[k]*self.PH for k in self.d.keys]
        if self.radial:
            v += [self.d.roff[k] for k in self.d.keys]
        return np.array(v + [t/100])

    def unpack(self, x):
        d = self.d
        d.zpos = list(x[:self.nz])
        for i, k in enumerate(d.keys):
            d.phase[k] = x[self.nz+i]/self.PH
            if self.radial:
                d.roff[k] = x[self.nz+self.nk+i]
        return x[-1]*100

    # per-ring fields at x, cached; the total field is their sum
    def rings(self, x):
        if self.cache_x is None or not np.array_equal(x, self.cache_x):
            self.unpack(x)
            self.cache = {k: field_ring(self.d, k, self.sensors) for k in self.d.keys}
            self.cache_x = x.copy()
        return self.cache

    def geometry(self, x):
        self.unpack(x)
        d, a, args = self.d, self.d.a, self.args
        zs = np.sort(np.array(d.zpos))
        g = [np.diff(zs)-(a+3)]
        if d.has_center:
            g.append([zs[0]-(a+3)])
        g.append([args.max_length-(2*zs.max()+a)])
        rmin_inner = args.bore/2+3+a/np.sqrt(2)
        rmax_outer = args.max_od/2-3-a/np.sqrt(2)
        dr = np.sqrt(2)*a+args.layer_wall
        for k in d.keys:
            r = d.radius(k)
            g.append([2*np.pi/d.n[k]-2*np.arcsin(np.sqrt(3)*a/2/r)])   # magnets fit around the ring
            if self.radial:
                g.append([r-rmin_inner, rmax_outer-r])
                if (k[0]+1, k[1]) in d.n:
                    g.append([d.radius((k[0]+1, k[1]))-r-dr])
        return np.concatenate([np.atleast_1d(np.asarray(v, float)) for v in g])

    def cons(self, x):
        t = x[-1]*100
        B = sum(self.rings(x).values())
        m = B.mean()
        dev = 1e6*(B-m)/m
        return np.concatenate([t-dev, t+dev, [m-self.args.min_b0], self.geometry(x)])

    def jac(self, x):
        base = self.rings(x)
        B = sum(base.values()); m = B.mean(); ns = len(B)
        nv = len(x)
        J = np.zeros((2*ns+1, nv))
        h = 1e-3
        for j, keys in enumerate(self.affects):
            xp = x.copy(); xp[j] += h
            self.unpack(xp)
            dB = sum(field_ring(self.d, k, self.sensors)-base[k] for k in keys)/h
            dm = dB.mean()
            ddev = 1e6*(dB*m-B*dm)/m**2
            J[:ns, j], J[ns:2*ns, j], J[2*ns, j] = -ddev, ddev, dm
        self.unpack(x)
        J[:2*ns, -1] = 100
        g0 = self.geometry(x)
        Jg = np.zeros((len(g0), nv))
        for j in range(nv-1):
            xp = x.copy(); xp[j] += h
            Jg[:, j] = (self.geometry(xp)-g0)/h
        self.unpack(x)
        return np.vstack([J, Jg])

    def solve(self, maxiter):
        B = field_x(self.d, self.sensors)
        x0 = self.pack(0.5*ppm(B)*1.05)
        res = minimize(lambda x: x[-1], x0, jac=lambda x: np.eye(len(x))[-1], method='SLSQP',
                       constraints=[{'type': 'ineq', 'fun': self.cons, 'jac': self.jac}],
                       options={'maxiter': maxiter, 'ftol': 1e-4})
        ok = np.all(self.cons(res.x) > -1e-3)
        self.unpack(res.x)
        return ok


def score(d, sensors, min_b0):
    B = field_x(d, sensors)
    return ppm(B) if B.mean() >= min_b0-1e-3 else 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ledger'); ap.add_argument('out')
    ap.add_argument('--radial', action='store_true')
    ap.add_argument('--min-b0', type=float, default=45); ap.add_argument('--max-length', type=float, default=150)
    ap.add_argument('--bore', type=float, default=100); ap.add_argument('--max-od', type=float, default=200)
    ap.add_argument('--cyl-d', type=float, default=40); ap.add_argument('--cyl-h', type=float, default=40)
    ap.add_argument('--br', type=float, default=1320); ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--sensors', type=int, default=400); ap.add_argument('--max-magnets', type=int, default=650)
    ap.add_argument('--layer-wall', type=float, default=2, help='mm of resin between layers (layer spacing = sqrt(2)*cube + this)')
    ap.add_argument('--top', type=int, default=12, help='magnet-count moves fully re-tuned per step, after screening')
    args = ap.parse_args()

    L = pd.read_excel(args.ledger)
    a = float(L['Magnet_length'].iloc[0])
    d = RadialDesign(rings_from_ledger(L), a, args.br)
    sensors = magsimulator.define_sensor_points_on_cylinder(args.sensors, args.cyl_d/2, args.cyl_h, [0,0,0]).position
    sensors = sensors[sensors[:, 2] >= 0]   # mirror symmetric in z

    t0 = time.time()
    best = score(d, sensors, args.min_b0)
    print(f'start: {len(L)} magnets, {best:.0f} ppm, B0 {field_x(d, sensors).mean():.2f} mT, {len(sensors)} check points, radial={args.radial}', flush=True)

    def count():
        return sum(n*(1 if k[1] == 0 else 2) for k, n in d.n.items())

    def state():
        return (dict(d.n), list(d.zpos), dict(d.phase), dict(d.roff))

    def restore(s):
        d.n, d.zpos, d.phase, d.roff = dict(s[0]), list(s[1]), dict(s[2]), dict(s[3])

    for rnd in range(args.rounds):
        saved = state()
        ok = Problem(d, sensors, args).solve(300)
        c = score(d, sensors, args.min_b0)
        if ok and c < best:
            best = c
        else:
            restore(saved)
        print(f'round {rnd+1} minimax: {best:.0f} ppm (feasible={ok})  ({time.time()-t0:.0f} s)', flush=True)
        improved = False
        moves = [((key, dn),) for key in d.keys for dn in (-1, 1)]
        moves += [((k1, -1), (k2, 1)) for k1 in d.keys for k2 in d.keys if k1 != k2 and (k1[1] == 0) == (k2[1] == 0)]
        for step in range(20):
            # screen every move without re-tuning, then fully re-tune the most promising ones
            base = state(); screened = []
            for move in moves:
                for key, dn in move:
                    d.n[key] += dn
                if min(d.n.values()) >= 4 and count() <= args.max_magnets:
                    screened.append((score(d, sensors, args.min_b0), move))
                restore(base)
            screened.sort(key=lambda c: c[0])
            best_move, best_state = None, None
            for _, move in screened[:args.top]:
                for key, dn in move:
                    d.n[key] += dn
                ok = Problem(d, sensors, args).solve(60)
                c = score(d, sensors, args.min_b0)
                if ok and c < best - 1:
                    best, best_move, best_state = c, move, state()
                restore(base)
            if best_move is None:
                break
            restore(best_state); improved = True
            print('   ' + ', '.join(f'ring {k}: {base[0][k]} -> {d.n[k]}' for k, _ in best_move) + f' magnets (total {count()}), {best:.0f} ppm  ({time.time()-t0:.0f} s)', flush=True)
        if not improved:
            break

    out = d.ledger()
    out.to_excel(args.out, index=False)
    B = field_x(d, sensors)
    radii = sorted({round(d.radius(k), 2) for k in d.keys})
    print(f'done: {len(out)} magnets, {ppm(B):.0f} ppm, B0 {B.mean():.2f} mT, length {np.ptp(d.slices())+a:.1f} mm, '
          f'ring radii {radii[0]:.1f}-{radii[-1]:.1f} mm  ({time.time()-t0:.0f} s) -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
