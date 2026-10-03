"""
Generate printable ring holders (a stack of slabs with a bore) from a magnet ledger.

Each distinct magnet z in the ledger becomes one slab. A slab is an annulus whose top face is level with the top of its
magnet pockets, so the next slab up is the lid that holds the magnets in; slab i spans from the top of slab i-1 to the top
of its own pockets. Slab 0 has a solid floor (--base) under its pockets. The top slab's pockets are open: the end cap
(not generated here) closes them. Stacking the slabs therefore sets every ring's z with no separate spacers.

Every pocket is the magnet cube plus a clearance on each local axis (x = magnetization, y, z = depth) and has:
  - an engraved triangle on the slab's top face, just outside the N face (local +x, the magnetization direction),
    pointing away from the pocket,
  - an air hole from the pocket floor through the bottom of the slab,
  - optional corner reliefs (SLA rounds inside corners, which would otherwise stop the cube seating).
All slabs share M3 bolt holes outside the magnets (--outer-holes, evenly spaced with one shifted off the pattern so the
stack only goes together one way: no wrong clocking, no flipped slab) and optionally smaller holes between the bore and
the inner magnets (--inner-holes), placed automatically where they clear the pockets of every slab. A V notch on the outside of every slab marks +x (the B0 direction).

The script checks wall thickness between pockets, to the bore, to the outside and to the bolt holes, and the floor under
each pocket, and prints warnings for anything thinner than --min-wall.

Usage:
  python make_ring_cad.py results/BEST_bore100_cyl40x40_649mag_427ppm.xlsx --out cad/best
  python make_ring_cad.py design.xlsx --tol 0.15 0.15 0.3 --outer-holes 8 --inner-holes 6 --inner-bolt 2 --inner-tol 0.4 --magnets

Outputs in --out:
  slab_<i>_z<z>.step/.stl   one per ring, numbered bottom (-z) to top (+z)
  assembly.step             all parts in place (plus the magnets with --magnets)
  slab_<i>_z<z>.png         top-view loading guide: pockets, N triangles and magnet counts per slab
  slabs.csv                 z range, thickness, magnet count and minimum walls per slab

Ledger format as in the MRI4ALL scripts (X-pos, Y-pos, Z-pos, X-rot, Y-rot, Z-rot, Magnet_length; mm and degrees,
extrinsic rotations, magnetization along the cube's local x as in magsimulator.build_magnet_collection).
Only rotations about z are supported (X-rot = Y-rot = 0), which is what all Halbach ledgers here use.

Printing: the Form 4 build volume is 200 x 125 x 210 mm, so rings wider than 125 mm print standing on edge. Ring-to-ring
differences and out-of-roundness matter, uniform shrink does not (see DESIGN_NOTES.md): print every slab with the same
resin, wash and cure, and rotate every other slab 90 degrees about its own axis on the build plate so any print
ovality alternates direction through the stack.
"""

import argparse
import os

import cadquery as cq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def load_ledger(fn):
    L = pd.read_excel(fn)
    if (np.abs(L['X-rot']) > 1e-9).any() or (np.abs(L['Y-rot']) > 1e-9).any():
        raise SystemExit('only rotations about z are supported (X-rot and Y-rot must be 0)')
    return L


def rot2(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def pocket_outline(x, y, zrot, wx, wy):
    """corners of a pocket footprint (local x along the magnetization), counterclockwise"""
    loc = np.array([[-wx/2, -wy/2], [wx/2, -wy/2], [wx/2, wy/2], [-wx/2, wy/2]])
    return loc @ rot2(zrot).T + [x, y]


def triangle_outline(x, y, zrot, wx, side, gap):
    """engraved N marker: base parallel to the N face, `gap` outside it, apex pointing along local +x"""
    x0 = wx/2 + gap
    loc = np.array([[x0, -side/2], [x0 + side*np.sqrt(3)/2, 0], [x0, side/2]])
    return loc @ rot2(zrot).T + [x, y]


def pocket_boundary(x, y, zrot, wx, wy, relief, n_edge=60, n_arc=24):
    """points on the true pocket outline: the square plus the corner relief circles"""
    P = pocket_outline(x, y, zrot, wx, wy)
    pts = [P[i] + (P[(i+1) % 4] - P[i])*t for i in range(4) for t in np.linspace(0, 1, n_edge, endpoint=False)]
    if relief > 0:
        th = np.linspace(0, 2*np.pi, n_arc, endpoint=False)
        pts += [c + relief*np.array([np.cos(t), np.sin(t)]) for c in P for t in th]
    return np.array(pts)


def seg_point_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab)/np.dot(ab, ab), 0, 1)
    return np.linalg.norm(p - (a + t*ab))


def polygon_gap(P, Q):
    """distance between two convex polygons; negative if they overlap"""
    def separated(P, Q):
        for poly in (P, Q):
            for i in range(len(poly)):
                e = poly[(i+1) % len(poly)] - poly[i]
                n = np.array([-e[1], e[0]])
                if (P @ n).max() < (Q @ n).min() or (Q @ n).max() < (P @ n).min():
                    return True
        return False
    if not separated(P, Q):
        return -1.0
    d = min(seg_point_dist(p, Q[i], Q[(i+1) % len(Q)]) for p in P for i in range(len(Q)))
    return min(d, min(seg_point_dist(q, P[i], P[(i+1) % len(P)]) for q in Q for i in range(len(P))))


def point_polygon_dist(p, Q):
    """distance from point p to convex polygon Q (counterclockwise); negative inside"""
    d = min(seg_point_dist(p, Q[i], Q[(i+1) % len(Q)]) for i in range(len(Q)))
    def cross(u, v):
        return u[0]*v[1] - u[1]*v[0]
    inside = all(cross(Q[(i+1) % len(Q)] - Q[i], p - Q[i]) >= 0 for i in range(len(Q)))
    return -d if inside else d


def hole_clearance(xy, d, obstacles):
    """resin left between a hole of diameter d at xy and the nearest obstacle polygon"""
    near = [Q for Q in obstacles if np.linalg.norm(Q.mean(0) - xy) < d + 12]
    return min([point_polygon_dist(np.asarray(xy), Q) for Q in near] + [np.inf]) - d/2


def outer_hole_angles(n, key_offset):
    """evenly spaced, offset by half a pitch so no hole sits on the +x notch, one hole shifted by key_offset"""
    a = (np.arange(n) + 0.5)*360.0/n
    if n > 0:
        a[0] += key_offset
    return a


def place_inner_holes(n, d, r, obstacles, min_clear, step=0.25):
    """choose n hole angles at radius r, as evenly spread as possible, each with >= min_clear resin to every obstacle.
    returns (angles, clearances); fewer than n angles if they do not fit."""
    th = np.arange(0, 360, step)
    c = np.array([hole_clearance((r*np.cos(np.deg2rad(t)), r*np.sin(np.deg2rad(t))), d, obstacles) for t in th])
    ok = c >= min_clear
    if not ok.any():
        return np.array([]), np.array([])
    # windows of consecutive good angles (wrapping at 360); one hole per window, at its best clearance
    edges = np.flatnonzero(np.diff(np.r_[ok[-1], ok].astype(int)))
    starts = [e for e in edges if ok[e]]
    wins = []
    for s0 in starts:
        k, idx = s0, []
        while ok[k % len(th)] and len(idx) < len(th):
            idx.append(k % len(th)); k += 1
        best = idx[int(np.argmax(c[idx]))]
        wins.append((th[best], c[best]))
    if len(wins) == 0:      # every angle is good
        wins = [(t, cc) for t, cc in zip(th, c)]
    wins = sorted(wins)
    # greedily pick the window farthest (in angle) from those already chosen, starting from the best one
    chosen = [max(wins, key=lambda w: w[1])]
    while len(chosen) < min(n, len(wins)):
        def gap(w):
            return min(abs((w[0] - cw[0] + 180) % 360 - 180) for cw in chosen)
        chosen.append(max((w for w in wins if w not in chosen), key=gap))
    chosen.sort()
    return np.array([w[0] for w in chosen]), np.array([w[1] for w in chosen])


def pocket_cutters(x, y, zrot, z_floor, wx, wy, depth, floor, relief=0.4, air_d=1.5, tri_side=2.5, tri_gap=0.4,
                   tri_depth=0.4):
    """solids to subtract for one magnet pocket open at the top face (z_floor + depth): the pocket (poking 1 mm out of
    the top), corner reliefs, an air hole through the `floor` below it and the engraved N triangle on the top face"""
    parts = [cq.Solid.makeBox(wx, wy, depth + 1.0, pnt=cq.Vector(-wx/2, -wy/2, 0))]
    if relief > 0:
        for sx in (-1, 1):
            for sy in (-1, 1):
                parts.append(cq.Solid.makeCylinder(relief, depth + 1.0, cq.Vector(sx*wx/2, sy*wy/2, 0)))
    if air_d > 0 and floor > 0:
        parts.append(cq.Solid.makeCylinder(air_d/2, floor + 1.0, cq.Vector(0, 0, -floor - 0.5)))
    if tri_side > 0:
        T = triangle_outline(0, 0, 0, wx, tri_side, tri_gap)
        parts.append(cq.Workplane('XY').workplane(offset=depth - tri_depth)
                     .polyline([tuple(p) for p in T]).close().extrude(tri_depth + 1.0).val())
    c = cq.Vector(x, y, z_floor)
    return [s.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), zrot).translate(c) for s in parts]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ledger')
    ap.add_argument('--out', default='cad/out')
    ap.add_argument('--tol', type=float, nargs=3, default=[0.15, 0.15, 0.3], metavar=('X', 'Y', 'Z'),
                    help='total clearance added to the cube on each local axis, mm (x = magnetization, z = depth); '
                         'tune with a test coupon')
    ap.add_argument('--bore', type=float, default=100.0, help='inner diameter, mm')
    ap.add_argument('--od', type=float, default=None, help='outer diameter, mm (default: just fits pockets and outer holes)')
    ap.add_argument('--base', type=float, default=3.0, help='solid floor under the lowest ring of pockets, mm')
    ap.add_argument('--min-wall', type=float, default=1.5, help='warn below this wall/floor thickness, mm')
    ap.add_argument('--air-d', type=float, default=1.5, help='air hole diameter, mm (0 = none)')
    ap.add_argument('--relief', type=float, default=0.4, help='corner relief radius, mm (0 = none)')
    ap.add_argument('--tri-side', type=float, default=2.5, help='N triangle side length, mm')
    ap.add_argument('--tri-gap', type=float, default=0.4, help='gap between pocket N face and triangle, mm')
    ap.add_argument('--tri-depth', type=float, default=0.4, help='N triangle engraving depth, mm')
    ap.add_argument('--outer-holes', type=int, default=8, help='bolt holes through the stack outside the magnets (0 = none)')
    ap.add_argument('--outer-bolt', type=float, default=3.0, help='outer bolt diameter, mm (3 = M3)')
    ap.add_argument('--outer-tol', type=float, default=0.4, help='clearance added to the outer bolt diameter, mm')
    ap.add_argument('--hole-key', type=float, default=10.0,
                    help='angular offset of one outer hole so the slabs only stack one way, deg')
    ap.add_argument('--inner-holes', type=int, default=6,
                    help='bolt holes between the bore and the inner magnets, placed automatically in the gaps between '
                         'pockets of all slabs (fewer are made if they do not fit)')
    ap.add_argument('--inner-bolt', type=float, default=2.0,
                    help='inner bolt diameter, mm (2 = M2; M3 only fits at one angle in the 100 mm bore design)')
    ap.add_argument('--inner-tol', type=float, default=0.4, help='clearance added to the inner bolt diameter, mm')
    ap.add_argument('--hole-wall', type=float, default=0.8,
                    help='minimum resin between an inner hole and the bore or any pocket, mm')
    ap.add_argument('--notch', type=float, default=1.0, help='depth of the +x (B0) notch on the outside, mm (0 = none)')
    ap.add_argument('--magnets', action='store_true', help='include the magnet cubes in assembly.step')
    ap.add_argument('--no-step', action='store_true', help='skip STEP export (faster; STL and PNG only)')
    args = ap.parse_args()
    args.outer_d = args.outer_bolt + args.outer_tol
    args.inner_d = args.inner_bolt + args.inner_tol

    L = load_ledger(args.ledger)
    tx, ty, tz = args.tol
    zkey = L['Z-pos'].round(3)
    zs = np.sort(zkey.unique())
    os.makedirs(args.out, exist_ok=True)

    a_all = L['Magnet_length'].values
    if np.ptp(a_all) > 1e-9:
        raise SystemExit('all magnets must be the same size (slab boundaries assume one pocket depth)')
    a = a_all[0]
    wx, wy, depth = a + tx, a + ty, a + tz
    rel = args.relief

    # outline of everything cut into the top face, per magnet: pocket (grown by the corner reliefs) and triangle
    pockets = [pocket_outline(r['X-pos'], r['Y-pos'], r['Z-rot'], wx + 2*rel, wy + 2*rel) for _, r in L.iterrows()]
    tris = [triangle_outline(r['X-pos'], r['Y-pos'], r['Z-rot'], wx, args.tri_side, args.tri_gap) for _, r in L.iterrows()]

    r_corner_max = max(np.hypot(*P.T).max() for P in pockets + tris)
    r_out = r_corner_max + args.min_wall + args.outer_d/2
    od = args.od if args.od else 2*(r_out + args.outer_d/2 + args.min_wall)
    if args.od:
        r_out = od/2 - args.min_wall - args.outer_d/2
    holes = []      # (x, y, diameter), through every slab
    for t in outer_hole_angles(args.outer_holes, args.hole_key):
        holes.append((r_out*np.cos(np.deg2rad(t)), r_out*np.sin(np.deg2rad(t)), args.outer_d))
    r_in = args.bore/2 + args.hole_wall + args.inner_d/2
    if args.inner_holes > 0:
        ang_in, clr_in = place_inner_holes(args.inner_holes, args.inner_d, r_in, pockets + tris, args.hole_wall)
        print(f'inner holes: {len(ang_in)} of {args.inner_holes} fit (d {args.inner_d} mm at r {r_in:.2f}, '
              f'{args.hole_wall} mm to the bore) at ' + ', '.join(f'{t:.1f} deg ({c:.2f} mm to pockets)'
                                                         for t, c in zip(ang_in, clr_in)))
        for t in ang_in:
            holes.append((r_in*np.cos(np.deg2rad(t)), r_in*np.sin(np.deg2rad(t)), args.inner_d))

    # slab z boundaries: slab i spans [bottom_i, top_i], top_i = top of its pockets
    tops = zs + depth/2
    bottoms = np.concatenate([[zs[0] - depth/2 - args.base], tops[:-1]])

    print(f'{len(L)} magnets, {len(zs)} slabs; pocket {wx:.2f} x {wy:.2f} x {depth:.2f} mm; '
          f'bore {args.bore:.1f}, OD {od:.1f} mm; {args.outer_holes} outer holes x {args.outer_d} mm at r {r_out:.1f}')
    if od > 200:
        print(f'  WARNING: OD {od:.1f} mm exceeds the Form 4 build width (200 mm)')

    rows, warnings = [], []
    parts = []
    for i, z in enumerate(zs):
        idx = np.where(zkey.values == round(z, 3))[0]
        sub = L.iloc[idx]
        z_bot, z_top = bottoms[i], tops[i]
        floor = (z - depth/2) - z_bot

        # wall checks in the plane of this slab
        # thinnest resin between neighbouring pockets, from their true outlines (square + corner reliefs)
        outl = [pocket_boundary(L['X-pos'].iloc[j], L['Y-pos'].iloc[j], L['Z-rot'].iloc[j], wx, wy, rel) for j in idx]
        min_pp = np.inf
        for j in range(len(idx)):
            for k in range(j+1, len(idx)):
                if np.linalg.norm(outl[j].mean(0) - outl[k].mean(0)) < 2*np.sqrt(2)*max(wx, wy) + 1:
                    min_pp = min(min_pp, cKDTree(outl[k]).query(outl[j])[0].min())
        min_tri = np.inf
        for j in idx:
            for k in idx:
                if j != k and np.linalg.norm(tris[j].mean(0) - pockets[k].mean(0)) < 2*max(wx, wy) + args.tri_side:
                    min_tri = min(min_tri, polygon_gap(tris[j], pockets[k]))
        min_bore = min(np.hypot(*pockets[j].T).min() for j in idx) - args.bore/2
        min_od = od/2 - max(np.hypot(*np.vstack([pockets[j], tris[j]]).T).max() for j in idx)
        obst = [pockets[j] for j in idx] + [tris[j] for j in idx]
        clr = [hole_clearance((hx, hy), hd, obst) for hx, hy, hd in holes]
        n_out = args.outer_holes
        min_rod = min(clr[:n_out] + [np.inf])
        min_inner = min(clr[n_out:] + [np.inf])
        if min_inner < args.hole_wall:
            warnings.append(f'slab {i} (z {z:+.2f}): inner bolt hole wall {min_inner:.2f} mm < {args.hole_wall} mm')
        for name, v in [('pocket-pocket wall', min_pp), ('floor under pockets', floor), ('wall to bore', min_bore),
                        ('wall to outside', min_od), ('wall to bolt hole', min_rod)]:
            if v < args.min_wall:
                warnings.append(f'slab {i} (z {z:+.2f}): {name} {v:.2f} mm < {args.min_wall} mm')
        if min_tri < 0:
            warnings.append(f'slab {i} (z {z:+.2f}): an N triangle runs into a neighbouring pocket '
                            f'(reduce --tri-side/--tri-gap); markers stay readable but check the loading guide')

        # solid
        slab = cq.Workplane('XY').workplane(offset=z_bot).circle(od/2).circle(args.bore/2).extrude(z_top - z_bot).val()
        cutters = []
        for _, r in sub.iterrows():
            cutters += pocket_cutters(r['X-pos'], r['Y-pos'], r['Z-rot'], z - depth/2, wx, wy, depth, floor, rel,
                                      args.air_d, args.tri_side, args.tri_gap, args.tri_depth)
        for (x, y, hd) in holes:
            cutters.append(cq.Solid.makeCylinder(hd/2, z_top - z_bot + 2, cq.Vector(x, y, z_bot - 1)))
        if args.notch > 0:
            n = args.notch
            v = (cq.Workplane('XY').workplane(offset=z_bot - 1)
                 .polyline([(od/2 - n, 0), (od/2 + 1, -(n + 1)), (od/2 + 1, n + 1)]).close().extrude(z_top - z_bot + 2).val())
            cutters.append(v)
        slab = slab.cut(*cutters)

        # sanity check: pocket centres are empty, walls next to the N and side faces are solid,
        # and the triangle on the N side is engraved
        for _, r in sub.iterrows():
            R2 = rot2(r['Z-rot'])
            c2 = np.array([r['X-pos'], r['Y-pos']])
            def solid_at(local_xy, zz):
                p = c2 + R2 @ np.array(local_xy)
                return slab.isInside(cq.Vector(p[0], p[1], zz), 1e-4)
            assert not solid_at((0, 0), z), 'pocket centre is not empty'
            tri_c = triangle_outline(0, 0, 0, wx, args.tri_side, args.tri_gap).mean(0)
            assert not solid_at(tri_c, z_top - args.tri_depth/2), 'N triangle missing'
            assert solid_at((-(wx/2 + rel + 0.3), 0), z), 'no wall behind the S face'
        name = f'slab_{i}_z{z:+.2f}'
        cq.exporters.export(cq.Workplane().add(slab), os.path.join(args.out, name + '.stl'), tolerance=0.01, angularTolerance=0.1)
        if not args.no_step:
            cq.exporters.export(cq.Workplane().add(slab), os.path.join(args.out, name + '.step'))
        parts.append((name, slab))

        # loading guide
        fig, ax = plt.subplots(figsize=(9, 9))
        th = np.linspace(0, 2*np.pi, 400)
        for rr in (args.bore/2, od/2):
            ax.plot(rr*np.cos(th), rr*np.sin(th), 'k', lw=1)
        for (x, y, hd) in holes:
            ax.add_patch(plt.Circle((x, y), hd/2, fill=False, color='0.4'))
        for j in idx:
            r = L.iloc[j]
            P = pocket_outline(r['X-pos'], r['Y-pos'], r['Z-rot'], wx, wy)
            ax.fill(*np.vstack([P, P[:1]]).T, fc='0.85', ec='k', lw=0.6)
            ax.fill(*tris[j].T, fc='tab:red', ec='none')
        ax.annotate('+x (B0), notch', (od/2, 0), (od/2 + 4, 6), fontsize=8)
        ax.set_aspect('equal')
        ax.set_title(f'{name}: {len(idx)} magnets, slab {z_bot:+.2f} to {z_top:+.2f} mm (thickness {z_top - z_bot:.2f}), '
                     f'viewed from +z\nred triangle = N face of the magnet (magnetization direction)', fontsize=9)
        ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
        fig.savefig(os.path.join(args.out, name + '.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

        rows.append(dict(slab=i, ring_z=z, bottom=z_bot, top=z_top, thickness=z_top - z_bot, magnets=len(idx),
                         floor=floor, min_wall_pockets=min_pp, min_wall_bore=min_bore, min_wall_outside=min_od,
                         min_wall_outer_bolts=min_rod, min_wall_inner_bolts=min_inner))
        print(f'  slab {i}: z {z:+7.2f}, {len(idx):3d} magnets, {z_bot:+7.2f} to {z_top:+7.2f} mm '
              f'({z_top - z_bot:5.2f} thick), floor {floor:.2f}, min walls: pockets {min_pp:.2f}, bore {min_bore:.2f}, '
              f'outside {min_od:.2f}, outer bolts {min_rod:.2f}, inner bolts {min_inner:.2f}')


    if not args.no_step:
        assy = cq.Assembly()
        for name, s in parts:
            assy.add(s, name=name, color=cq.Color(0.8, 0.8, 0.85, 1))
        if args.magnets:
            for j, r in L.iterrows():
                cube = (cq.Solid.makeBox(a, a, a, pnt=cq.Vector(-a/2, -a/2, -a/2))
                        .rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), r['Z-rot'])
                        .translate(cq.Vector(r['X-pos'], r['Y-pos'], r['Z-pos'])))
                assy.add(cube, name=f'magnet_{j}', color=cq.Color(0.8, 0.1, 0.1, 1))
        assy.save(os.path.join(args.out, 'assembly.step'))

    pd.DataFrame(rows).to_csv(os.path.join(args.out, 'slabs.csv'), index=False)
    print(f'stack {bottoms[0]:+.2f} to {tops[-1]:+.2f} mm ({tops[-1] - bottoms[0]:.1f} mm long, without end caps)')
    for w in warnings:
        print('  WARNING:', w)
    print(f'wrote {len(parts)} parts to {args.out}')


if __name__ == '__main__':
    main()
