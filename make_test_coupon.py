"""
Tolerance test coupon for the ring holders made by make_ring_cad.py: print it with the same resin, orientation, wash and
cure as the rings, try the magnets and bolts in it, then pass the values that fit to make_ring_cad.py.

The pockets are cut with the same function as the rings (pocket_cutters: corner reliefs, air hole, N triangle), so
the fit you find here is the fit you get there.

Layout (viewed from the top face; values are engraved next to each feature):
  - magnet pockets: one column per in-plane clearance (--pocket-tols, added to the cube on both in-plane axes), one row
    per pocket rotation (--angles). The rings have pockets at every angle and SLA accuracy depends on how an edge lines
    up with the printer's pixel grid and layers, so check the fit at every angle.
  - thin-wall pairs: two pockets at the default clearance separated by each wall in --walls (the ring design has walls
    down to ~0.7 mm between neighbouring pockets), to check that they print and survive pushing magnets in.
  - bolt holes: one row per bolt size (--bolts), one hole per clearance in --hole-tols (hole = bolt + clearance).

Usage:
  python make_test_coupon.py --out cad/coupon
  python make_test_coupon.py --pocket-tols 0.05 0.1 0.15 0.2 0.25 0.3 --angles 0 22.5 45 --bolts 3 2 \\
      --hole-tols 0.1 0.2 0.3 0.4 0.5 0.6 --walls 0.6 0.8 1.0

Then e.g.: python make_ring_cad.py design.xlsx --tol 0.15 0.15 0.3 --outer-tol 0.3 --inner-tol 0.3

Print the coupon standing on its long edge, like the rings (which print upright on the Form 4), so the pocket axes are
horizontal on the build plate the same way. Test the z play with --tol-z: the pocket depth is cube + tol-z, and in the
rings the next slab closes the pocket, so z clearance is only rattle room.
"""

import argparse
import os

import cadquery as cq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from make_ring_cad import pocket_cutters, pocket_outline, triangle_outline


def label(txt, x, y, z_top, size, depth):
    return (cq.Workplane('XY', origin=(x, y, z_top - depth))
            .text(txt, size, depth + 1.0, combine=False, halign='center', valign='center').val())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='cad/coupon')
    ap.add_argument('--cube', type=float, default=6.35, help='magnet cube side, mm')
    ap.add_argument('--pocket-tols', type=float, nargs='+', default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                    help='in-plane pocket clearances to test, mm (added to the cube on x and y)')
    ap.add_argument('--angles', type=float, nargs='+', default=[0, 22.5, 45], help='pocket rotations to test, deg')
    ap.add_argument('--tol-z', type=float, default=0.3, help='depth clearance, mm')
    ap.add_argument('--walls', type=float, nargs='+', default=[0.6, 0.8, 1.0],
                    help='thin walls between pocket pairs to test, mm')
    ap.add_argument('--wall-tol', type=float, default=0.15, help='pocket clearance used for the thin-wall pairs, mm')
    ap.add_argument('--bolts', type=float, nargs='+', default=[3.0, 2.0], help='bolt diameters, mm (3 = M3, 2 = M2)')
    ap.add_argument('--hole-tols', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    help='bolt hole clearances to test, mm (hole = bolt + clearance)')
    ap.add_argument('--floor', type=float, default=3.0, help='floor under the pockets, mm')
    ap.add_argument('--relief', type=float, default=0.4, help='corner relief radius, mm (as in make_ring_cad.py)')
    ap.add_argument('--air-d', type=float, default=1.5, help='air hole diameter, mm')
    ap.add_argument('--text', type=float, default=2.5, help='label height, mm')
    ap.add_argument('--text-depth', type=float, default=0.4, help='label engraving depth, mm')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    a, depth, margin, pitch = args.cube, args.cube + args.tol_z, 5.0, args.cube + 6.5
    ncol = max(len(args.pocket_tols), len(args.hole_tols), 2*len(args.walls))
    width = 2*margin + 8 + ncol*pitch                     # 8 mm on the left for row labels
    hole_pitch = pitch
    rows_h = [('labels', 5.0)] + [('pocket', pitch)]*len(args.angles) + [('labels', 5.0), ('walls', pitch)] \
        + [('holes', max(args.bolts) + 6.5)]*len(args.bolts)
    height = 2*margin + sum(h for _, h in rows_h)
    thick = args.floor + depth
    z_floor = args.floor
    print(f'coupon {width:.1f} x {height:.1f} x {thick:.2f} mm')

    block = cq.Workplane('XY').box(width, height, thick, centered=False).val()
    cutters, texts, drawn = [], [], []
    x0 = margin + 8
    y = height - margin

    def col_x(k):
        return x0 + (k + 0.5)*pitch

    # pocket grid
    y -= 5.0
    for k, t in enumerate(args.pocket_tols):
        texts.append((f'{t:.2f}', col_x(k), y + 2.5))
    for ang in args.angles:
        yc = y - pitch/2
        texts.append((f'{ang:g}', margin + 4, yc))
        for k, t in enumerate(args.pocket_tols):
            w = a + t
            cutters += pocket_cutters(col_x(k), yc, ang, z_floor, w, w, depth, args.floor, args.relief, args.air_d)
            drawn.append(('pocket', col_x(k), yc, ang, w))
        y -= pitch

    # thin-wall pairs: two pockets side by side along x, separated by the wall (between the corner reliefs)
    y -= 5.0
    w = a + args.wall_tol
    for k, wall in enumerate(args.walls):
        xc = x0 + (2*k + 1)*pitch
        texts.append((f'w{wall:.1f}', xc, y + 2.5))
        dx = w/2 + args.relief + wall/2
        for sx in (-1, 1):      # N triangles point away from the shared wall
            ang = 0 if sx > 0 else 180
            cutters += pocket_cutters(xc + sx*dx, y - pitch/2, ang, z_floor, w, w, depth, args.floor, args.relief, args.air_d)
            drawn.append(('pocket', xc + sx*dx, y - pitch/2, ang, w))
    y -= pitch

    # bolt holes
    for b in args.bolts:
        h = max(args.bolts) + 6.5
        yc = y - h/2
        texts.append((f'M{b:g}', margin + 4, yc))
        for k, t in enumerate(args.hole_tols):
            d = b + t
            cutters.append(cq.Solid.makeCylinder(d/2, thick + 2, cq.Vector(col_x(k), yc + 1.2, -1)))
            texts.append((f'{t:.1f}', col_x(k), yc - b/2 - 1.6))
            drawn.append(('hole', col_x(k), yc + 1.2, 0, d))
        y -= h

    block = block.cut(*cutters)
    for txt, x, yy in texts:
        block = block.cut(label(txt, x, yy, thick, args.text, args.text_depth))

    for ext in ('step', 'stl'):
        cq.exporters.export(cq.Workplane().add(block), os.path.join(args.out, f'tolerance_coupon.{ext}'),
                            **({'tolerance': 0.01, 'angularTolerance': 0.1} if ext == 'stl' else {}))

    # layout drawing
    fig, ax = plt.subplots(figsize=(8, 8*height/width))
    ax.add_patch(plt.Rectangle((0, 0), width, height, fill=False, color='k'))
    for kind, x, yy, ang, s in drawn:
        if kind == 'pocket':
            P = pocket_outline(x, yy, ang, s, s)
            ax.fill(*np.vstack([P, P[:1]]).T, fc='0.85', ec='k', lw=0.6)
            ax.fill(*triangle_outline(x, yy, ang, s, 2.5, 0.4).T, fc='tab:red')
        else:
            ax.add_patch(plt.Circle((x, yy), s/2, fill=False, color='k'))
    for txt, x, yy in texts:
        ax.text(x, yy, txt, ha='center', va='center', fontsize=7)
    ax.set_aspect('equal'); ax.set_xlim(-2, width + 2); ax.set_ylim(-2, height + 2)
    ax.set_title('tolerance coupon (top face): pocket clearance (mm) per column, rotation (deg) per row;\n'
                 'w = wall between pocket pairs (mm); bolt rows: clearance (mm) under each hole', fontsize=8)
    fig.savefig(os.path.join(args.out, 'tolerance_coupon.png'), dpi=150, bbox_inches='tight')
    print(f'wrote {args.out}/tolerance_coupon.step/.stl/.png')


if __name__ == '__main__':
    main()
