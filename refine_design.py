# -*- coding: utf-8 -*-
"""
Local refinement of an aligned-ring Halbach design (a ledger .xlsx written by optimize_robust_aligned.py).

The genetic algorithm finds a good layout but is poor at fine-tuning continuous values. This script keeps the
number of magnets per ring fixed and tunes the ring positions along z (mirror symmetric) and each ring's
rotation offset to minimize the peak-to-peak field variation over a target cylinder, then tries +-1 magnet
changes per ring. Constraints (penalties): B0 >= min, ring slice spacing >= cube + 3 mm, total length <= max.

usage:
    python refine_design.py <ledger.xlsx> <out.xlsx> [--min-b0 45] [--max-length 150] [--cyl-d 40] [--cyl-h 40] [--rounds 3]
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
import magpylib as magpy
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import magsimulator

COLS = ['X-pos','Y-pos','Z-pos','X-rot','Y-rot','Z-rot','Searched','CostValue','Used','Placement_index','Bmag','Magnet_length','Tag']


# rings from a ledger: list of dicts {r, z, n, phase}; mirror pairs share a slice index and phase
def rings_from_ledger(L):
    L = L.copy()
    L['r'] = np.hypot(L['X-pos'], L['Y-pos']).round(2)
    L['z'] = L['Z-pos'].round(3)
    rings = []
    for (r, z), g in L.groupby(['r', 'z']):
        n = len(g)
        ang = np.mod(np.arctan2(g['Y-pos'], g['X-pos']), 2*np.pi/n)
        rings.append({'r': r, 'z': z, 'n': n, 'phase': float(np.median(ang))})
    return rings


class Design:
    def __init__(self, rings, a, br):
        self.a, self.br = a, br
        self.radii = sorted({rg['r'] for rg in rings})
        self.zpos = sorted({abs(rg['z']) for rg in rings if abs(rg['z']) > 1e-6})
        self.has_center = any(abs(rg['z']) <= 1e-6 for rg in rings)
        # one entry per (layer, slice); slice 0 = center, slice k>0 = +-zpos[k-1]
        self.n = {}
        self.phase = {}
        for rg in rings:
            k = 0 if abs(rg['z']) <= 1e-6 else self.zpos.index(abs(rg['z'])) + 1
            key = (self.radii.index(rg['r']), k)
            self.n[key] = rg['n']
            self.phase[key] = rg['phase']
        self.keys = sorted(self.n)

    def vector(self):
        return np.array(self.zpos + [self.phase[k] for k in self.keys])

    def set_vector(self, v):
        nz = len(self.zpos)
        self.zpos = list(v[:nz])
        for k, p in zip(self.keys, v[nz:]):
            self.phase[k] = p

    def magnets(self):
        pos, rotz = [], []
        for (li, k) in self.keys:
            r, n, ph = self.radii[li], self.n[(li, k)], self.phase[(li, k)]
            th = ph + 2*np.pi*np.arange(n)/n
            zs = [0.0] if k == 0 else [self.zpos[k-1], -self.zpos[k-1]]
            for z in zs:
                pos.append(np.stack([r*np.cos(th), r*np.sin(th), np.full(n, z)], axis=1))
                rotz.append(2*th)
        return np.vstack(pos), np.concatenate(rotz)

    def ledger(self):
        pos, rotz = self.magnets()
        rows = [[p[0], p[1], p[2], 0, 0, np.rad2deg(t), 0, 0, 0, 0, 0, self.a, ''] for p, t in zip(pos, rotz)]
        return pd.DataFrame(rows, columns=COLS)

    def slices(self):
        z = ([0.0] if self.has_center else []) + self.zpos + [-z for z in self.zpos]
        return np.sort(np.array(z))


# field of all cubes on the sensors, vectorized with the functional magpylib interface (only the Bx component is used)
def field_x(design, sensors):
    pos, rotz = design.magnets()
    n_m, n_s = len(pos), len(sensors)
    B = magpy.getB('Cuboid', observers=np.tile(sensors, (n_m, 1)), position=np.repeat(pos, n_s, axis=0),
                   orientation=R.from_euler('z', np.repeat(rotz, n_s)[:, None]), magnetization=(design.br, 0, 0),
                   dimension=(design.a,)*3)
    return B[:, 0].reshape(n_m, n_s).sum(axis=0)


def ppm(B):
    return 1e6*(B.max()-B.min())/B.mean()


def cost(design, sensors, min_b0, max_len):
    B = field_x(design, sensors)
    c = ppm(B)
    s = design.slices()
    gap = np.diff(s).min() if len(s) > 1 else 1e9
    c += 1e5*max(0, min_b0-B.mean()) + 1e5*max(0, design.a+3-gap) + 1e5*max(0, s.max()-s.min()+design.a-max_len)
    c += 1e5*max(0, design.a+3-min(design.zpos)) if design.zpos and design.has_center else 0
    return c


def polish(design, sensors, min_b0, max_len, maxfev):
    x0 = design.vector()
    def f(v):
        design.set_vector(v)
        return cost(design, sensors, min_b0, max_len)
    res = minimize(f, x0, method='Powell', options={'maxfev': maxfev, 'xtol': 1e-3, 'ftol': 1e-4})
    design.set_vector(res.x)
    return res.fun


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ledger'); ap.add_argument('out')
    ap.add_argument('--min-b0', type=float, default=45); ap.add_argument('--max-length', type=float, default=150)
    ap.add_argument('--cyl-d', type=float, default=40); ap.add_argument('--cyl-h', type=float, default=40)
    ap.add_argument('--br', type=float, default=1320); ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--sensors', type=int, default=400)
    args = ap.parse_args()

    L = pd.read_excel(args.ledger)
    a = float(L['Magnet_length'].iloc[0])
    d = Design(rings_from_ledger(L), a, args.br)
    sensors = magsimulator.define_sensor_points_on_cylinder(args.sensors, args.cyl_d/2, args.cyl_h, [0,0,0]).position
    sensors = sensors[sensors[:, 2] >= 0] # the design is mirror symmetric in z, so the z >= 0 half sees the full field range
    nmax = [int(magsimulator.get_max_magnets_per_radius(r, a))-1 for r in d.radii]

    t0 = time.time()
    best = cost(d, sensors, args.min_b0, args.max_length)
    print(f'start: {len(L)} magnets, {ppm(field_x(d, sensors)):.0f} ppm, B0 {field_x(d, sensors).mean():.2f} mT', flush=True)
    for rnd in range(args.rounds):
        best = polish(d, sensors, args.min_b0, args.max_length, 4000)
        print(f'round {rnd+1} continuous polish: cost {best:.0f}  ({time.time()-t0:.0f} s)', flush=True)
        # integer moves: +-1 magnet per ring, each followed by a short polish, kept only if the cost improves
        improved = False
        for key in list(d.keys):
            for dn in (-1, 1):
                n_new = d.n[key] + dn
                if n_new < 4 or n_new > nmax[key[0]]:
                    continue
                saved_n, saved_v = d.n[key], d.vector()
                d.n[key] = n_new
                c = polish(d, sensors, args.min_b0, args.max_length, 600)
                if c < best - 1:
                    best, improved = c, True
                    print(f'   ring {key}: {saved_n} -> {n_new} magnets, cost {best:.0f}', flush=True)
                else:
                    d.n[key] = saved_n
                    d.set_vector(saved_v)
        if not improved:
            break

    B = field_x(d, sensors)
    out = d.ledger()
    out.to_excel(args.out, index=False)
    print(f'done: {len(out)} magnets, {ppm(B):.0f} ppm, B0 {B.mean():.2f} mT, length {np.ptp(d.slices())+a:.1f} mm, '
          f'min slice gap {np.diff(d.slices()).min():.2f} mm  ({time.time()-t0:.0f} s) -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
