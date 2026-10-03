"""
Tolerance test coupon for the ring holders made by make_ring_cad.py: print it with the same resin, orientation, wash and
cure as the rings, try the magnets and bolts in it, then pass the values that fit to make_ring_cad.py.

The pockets are cut with the same function as the rings (pocket_cutters: corner reliefs, air hole, N notch), so
the fit you find here is the fit you get there.

Layout (viewed from the top face; values are engraved next to each feature):
  - magnet pockets: one column per in-plane clearance (--pocket-tols, added to the cube on both in-plane axes), one row
    per pocket rotation (--angles). The rings have pockets at every angle and SLA accuracy depends on how an edge lines
    up with the printer's pixel grid and layers, so check the fit at every angle.
  - thin-wall pairs: two pockets at the default clearance separated by each wall in --walls (the ring design has walls
    down to ~0.7 mm between neighbouring pockets), to check that they print and survive pushing magnets in.
  - bolt holes: one row per bolt size (--bolts), one column per clearance in --hole-tols (hole = bolt + clearance).

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

from make_ring_cad import notch_outline, pocket_cutters, pocket_outline


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
    ap.add_argument('--floor', type=float, default=2.0, help='floor under the pockets, mm')
    ap.add_argument('--margin', type=float, default=2.0, help='border around the features, mm')
    ap.add_argument('--gap', type=float, default=1.8, help='minimum resin between neighbouring test features, mm')
    ap.add_argument('--relief', type=float, default=0.4, help='corner relief radius, mm (as in make_ring_cad.py)')
    ap.add_argument('--air-d', type=float, default=1.5, help='air hole diameter, mm')
    ap.add_argument('--text', type=float, default=2.2, help='label height, mm')
    ap.add_argument('--text-depth', type=float, default=0.4, help='label engraving depth, mm')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    a, depth, m, gap = args.cube, args.cube + args.tol_z, args.margin, args.gap
    rel, th = args.relief, args.text
    wmax = a + max(args.pocket_tols + [args.wall_tol])

    def extent(w, ang):
        """half-size of a pocket's footprint (with reliefs and N notch) along x and y"""
        P = np.vstack([pocket_outline(0, 0, ang, w + 2*rel, w + 2*rel), notch_outline(0, 0, ang, w, 2.0, 1.0)])
        return np.abs(P).max(0)

    pitch = max(2*extent(wmax, ang)[0] for ang in args.angles) + gap                  # column pitch of the pocket grid
    row_h = [2*extent(wmax, ang)[1] + gap for ang in args.angles]
    w_pair = a + args.wall_tol
    pair_w = [2*(w_pair + 2*rel) + wall + 2*1.0 for wall in args.walls]                 # two pockets, wall, notches
    hole_pitch = max(args.bolts) + max(args.hole_tols) + max(gap, 1.2*th)
    x0 = m + 1.5*th                                                                     # room for row labels
    width = max(x0 + len(args.pocket_tols)*pitch, x0 + sum(pair_w) + gap*(len(args.walls) - 1),
                x0 + len(args.hole_tols)*hole_pitch) + m
    pair_h = 2*extent(w_pair, 0)[1] + gap
    bolt_rows = [b + max(args.hole_tols) + gap for b in args.bolts]
    height = m + (th + 1) + sum(row_h) + (th + 1) + pair_h + (th + 1) + sum(bolt_rows) + m
    thick = args.floor + depth
    z_floor = args.floor
    print(f'coupon {width:.1f} x {height:.1f} x {thick:.2f} mm')

    block = cq.Workplane('XY').box(width, height, thick, centered=False).val()
    cutters, texts, drawn = [], [], []
    y = height - m

    def col_x(k, p=pitch):
        return x0 + (k + 0.5)*p

    # pocket grid: clearance per column, rotation per row
    y -= th + 1
    for k, t in enumerate(args.pocket_tols):
        texts.append((f'{t:.2f}'.lstrip('0'), col_x(k), y + (th + 1)/2))
    for ang, h in zip(args.angles, row_h):
        yc = y - h/2
        texts.append((f'{ang:g}', m + 0.75*th, yc))
        for k, t in enumerate(args.pocket_tols):
            w = a + t
            cutters += pocket_cutters(col_x(k), yc, ang, z_floor, w, w, depth, args.floor, rel, args.air_d)
            drawn.append(('pocket', col_x(k), yc, ang, w))
        y -= h

    y_split = y         # below here only the thin-wall pairs and bolt holes: trim the block to their width
    lower_w = max(x0 + sum(pair_w) + gap*(len(args.walls) - 1), x0 + len(args.hole_tols)*hole_pitch) + m
    if width - lower_w > 1:
        cutters.append(cq.Solid.makeBox(width - lower_w + 1, y_split + 1, thick + 2, pnt=cq.Vector(lower_w, -1, -1)))

    # thin-wall pairs: two pockets side by side, `wall` of resin between their corner reliefs, notches facing out
    y -= th + 1
    x = x0
    for wall, pw in zip(args.walls, pair_w):
        xc = x + pw/2
        texts.append((f'w{wall:g}', xc, y + (th + 1)/2))
        dx = w_pair/2 + rel + wall/2
        for sx in (-1, 1):
            ang = 0 if sx > 0 else 180
            cutters += pocket_cutters(xc + sx*dx, y - pair_h/2, ang, z_floor, w_pair, w_pair, depth, args.floor, rel, args.air_d)
            drawn.append(('pocket', xc + sx*dx, y - pair_h/2, ang, w_pair))
        x += pw + gap
    y -= pair_h

    # bolt holes: clearance per column (one label row shared by all bolt sizes), bolt size per row
    y -= th + 1
    for k, t in enumerate(args.hole_tols):
        texts.append((f'{t:.1f}'.lstrip('0'), col_x(k, hole_pitch), y + (th + 1)/2))
    for b, h in zip(args.bolts, bolt_rows):
        yc = y - h/2
        texts.append((f'M{b:g}', m + 0.75*th, yc))
        for k, t in enumerate(args.hole_tols):
            d = b + t
            cutters.append(cq.Solid.makeCylinder(d/2, thick + 2, cq.Vector(col_x(k, hole_pitch), yc, -1)))
            drawn.append(('hole', col_x(k, hole_pitch), yc, 0, d))
        y -= h

    block = block.cut(*cutters)
    for txt, x, yy in texts:
        block = block.cut(label(txt, x, yy, thick, args.text, args.text_depth))

    for ext in ('step', 'stl'):
        cq.exporters.export(cq.Workplane().add(block), os.path.join(args.out, f'tolerance_coupon.{ext}'),
                            **({'tolerance': 0.01, 'angularTolerance': 0.1} if ext == 'stl' else {}))

    # layout drawing
    fig, ax = plt.subplots(figsize=(8, 8*height/width))
    ax.plot([0, width, width, lower_w, lower_w, 0, 0], [height, height, y_split, y_split, 0, 0, height], 'k')
    for kind, x, yy, ang, s in drawn:
        if kind == 'pocket':
            P = pocket_outline(x, yy, ang, s, s)
            ax.fill(*np.vstack([P, P[:1]]).T, fc='0.85', ec='k', lw=0.6)
            ax.fill(*notch_outline(x, yy, ang, s, 2.0, 1.0).T, fc='tab:red')
        else:
            ax.add_patch(plt.Circle((x, yy), s/2, fill=False, color='k'))
    for txt, x, yy in texts:
        ax.text(x, yy, txt, ha='center', va='center', fontsize=7)
    ax.set_aspect('equal'); ax.set_xlim(-2, width + 2); ax.set_ylim(-2, height + 2)
    ax.set_title('tolerance coupon (top face): pocket clearance (mm) per column, rotation (deg) per row;\n'
                 'w = wall between pocket pairs (mm); bolt holes: clearance (mm) per column, bolt per row', fontsize=8)
    fig.savefig(os.path.join(args.out, 'tolerance_coupon.png'), dpi=150, bbox_inches='tight')
    print(f'wrote {args.out}/tolerance_coupon.step/.stl/.png')


if __name__ == '__main__':
    main()
