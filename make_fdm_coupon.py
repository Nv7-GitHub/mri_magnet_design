"""
FDM test coupon for the spacers made by make_ring_cad.py: print it with the same printer, material, layer height and
slicer settings as the spacers, then

  1. measure each height sample with calipers at its corners and centre (the spacer heights set the ring spacing, which needs
     ~0.05 mm: see DESIGN_NOTES.md). The average (printed - nominal) is the printer's height error: pass minus that to
     make_ring_cad.py as --spacer-comp. If the error grows with height (a scale error rather than an offset), print
     at a layer height that divides the spacer heights or fix the printer's z steps first.
  2. test-fit the bolts in the hole bars and pass the clearance that slides freely as --fdm-outer-tol / --fdm-inner-tol.

Layout, all joined by a thin base plate:
  - one height sample per spacer height (--heights): a solid block (sliced with the spacers' walls and infill, like a
    piece of a spacer) with its nominal height engraved on the plate in front of it,
  - one bar per bolt size (--bolts) with one hole per clearance (--hole-tols), the clearance engraved next to each hole.

Usage:
  python make_fdm_coupon.py --out cad/fdm_coupon
  python make_fdm_coupon.py --heights 6.06 10.21 17.22 --bolts 3 2 --hole-tols 0.2 0.3 0.4 0.5 0.6 0.7

The default heights are the three spacer heights of the recommended design (cad/best/spacers.csv).
"""

import argparse
import os

import cadquery as cq


def engrave(txt, x, y, z_top, size, depth):
    return (cq.Workplane('XY', origin=(x, y, z_top - depth))
            .text(txt, size, depth + 1.0, combine=False, halign='center', valign='center').val())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='cad/fdm_coupon')
    ap.add_argument('--heights', type=float, nargs='+', default=[6.06, 10.21, 17.22],
                    help='spacer heights to test, mm (from spacers.csv)')
    ap.add_argument('--bolts', type=float, nargs='+', default=[3.0, 2.0], help='bolt diameters, mm')
    ap.add_argument('--hole-tols', type=float, nargs='+', default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                    help='bolt hole clearances to test, mm (FDM holes print small)')
    ap.add_argument('--plate', type=float, default=1.6, help='base plate joining the samples, mm')
    ap.add_argument('--sample', type=float, default=20.0, help='height sample footprint, mm (big enough to get infill)')
    ap.add_argument('--bar-h', type=float, default=6.0, help='height of the bolt hole bars, mm')
    ap.add_argument('--text', type=float, default=4.0, help='label height, mm')
    ap.add_argument('--text-depth', type=float, default=0.6, help='label engraving depth, mm')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    m, gap, s, th = 3.0, 4.0, args.sample, args.text
    hole_pitch = max(args.bolts) + max(args.hole_tols) + 4.0
    bar_w = max(args.bolts) + max(args.hole_tols) + 3.0 + th + 1.0       # holes along one side, labels along the other
    width = 2*m + max(len(args.heights)*(s + gap) - gap, len(args.hole_tols)*hole_pitch)
    depth_y = m + (th + 2) + s + gap + len(args.bolts)*(bar_w + gap) - gap + m
    print(f'FDM coupon {width:.1f} x {depth_y:.1f} mm, plate {args.plate} mm, tallest {max(args.heights):.2f} mm')

    body = cq.Workplane('XY').box(width, depth_y, args.plate, centered=False).val()
    adds, cuts = [], []

    # height samples: solid blocks, nominal height engraved on the plate in front of each
    y_lab = m + (th + 2)/2
    y0 = m + th + 2
    for k, h in enumerate(args.heights):
        x0 = m + k*(s + gap)
        adds.append(cq.Solid.makeBox(s, s, h, pnt=cq.Vector(x0, y0, 0)))
        cuts.append(engrave(f'{h:.2f}', x0 + s/2, y_lab, args.plate, th*0.8, min(args.text_depth, args.plate/2)))

    # bolt hole bars
    y = y0 + s + gap
    for b in args.bolts:
        bar = cq.Solid.makeBox(width - 2*m, bar_w, args.bar_h, pnt=cq.Vector(m, y, 0))
        adds.append(bar)
        yh = y + 1.5 + (max(args.bolts) + max(args.hole_tols))/2
        for k, t in enumerate(args.hole_tols):
            xh = m + (k + 0.5)*hole_pitch
            cuts.append(cq.Solid.makeCylinder((b + t)/2, args.bar_h + 2, cq.Vector(xh, yh, -1)))
            cuts.append(engrave(f'{t:.1f}'.lstrip('0'), xh, y + bar_w - 1 - th/2, args.bar_h, th*0.8, args.text_depth))
        cuts.append(engrave(f'M{b:g}', width - m - th, y + bar_w/2, args.bar_h, th*0.8, args.text_depth)
                    if len(args.hole_tols)*hole_pitch + 2*th < width - 2*m else
                    engrave(f'M{b:g}', m + 0.6*th, y + bar_w/2, args.plate, th*0.6, args.plate/2))
        y += bar_w + gap

    body = body.fuse(*adds).clean()
    body = body.cut(*cuts)
    for ext in ('step', 'stl'):
        cq.exporters.export(cq.Workplane().add(body), os.path.join(args.out, f'fdm_coupon.{ext}'),
                            **({'tolerance': 0.02, 'angularTolerance': 0.1} if ext == 'stl' else {}))
    print(f'wrote {args.out}/fdm_coupon.step/.stl')


if __name__ == '__main__':
    main()
