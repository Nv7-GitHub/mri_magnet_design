"""
Put all test coupons into one STEP file as separate, named bodies of a single part (no assembly structure), so CAD
programs import it as one part studio with every coupon in it (Onshape: one Part Studio, one part per coupon).

Reads the coupons made by make_test_coupon.py (SLA) and make_fdm_coupon.py (FDM); any that are missing are generated
first with default settings. To use non-default coupon settings, run those scripts with your options first, then this.

Usage:
  python make_coupons_step.py                       # -> cad/test_coupons.step
  python make_coupons_step.py --out cad/test_coupons.step --gap 15
"""

import argparse
import os
import re
import subprocess
import sys

import cadquery as cq
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

# (body name, coupon STEP file, script that makes it, its default output folder)
COUPONS = [
    ('sla_tolerance_coupon', 'cad/coupon/tolerance_coupon.step', 'make_test_coupon.py', 'cad/coupon'),
    ('fdm_tolerance_coupon', 'cad/fdm_coupon/fdm_coupon.step', 'make_fdm_coupon.py', 'cad/fdm_coupon'),
]


def write_flat_step(solids, names, path, product='test_coupons'):
    """one STEP product holding the solids as separate bodies, each body (MANIFOLD_SOLID_BREP) named"""
    w = STEPControl_Writer()
    Interface_Static.SetIVal_s('write.step.assembly', 0)      # after creating the writer, which resets it
    Interface_Static.SetCVal_s('write.step.product.name', product)
    w.Transfer(cq.Compound.makeCompound(solids).wrapped, STEPControl_AsIs)
    w.Write(path)
    txt = open(path).read()
    assert 'NEXT_ASSEMBLY_USAGE_OCCURRENCE' not in txt, 'STEP was written as an assembly'
    # the bodies are written in compound order: name them in that order
    it = iter(names)
    txt, n = re.subn(r"MANIFOLD_SOLID_BREP\(''", lambda m: f"MANIFOLD_SOLID_BREP('{next(it)}'", txt)
    assert n == len(names), f'expected {len(names)} bodies, found {n}'
    open(path, 'w').write(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='cad/test_coupons.step')
    ap.add_argument('--gap', type=float, default=10.0, help='space between coupons, mm')
    args = ap.parse_args()

    solids, names, x = [], [], 0.0
    for name, fn, script, outdir in COUPONS:
        if not os.path.exists(fn):
            print(f'{fn} missing: running {script} with default settings')
            subprocess.run([sys.executable, script, '--out', outdir], check=True)
        s = cq.importers.importStep(fn).solids().vals()
        assert len(s) == 1, f'{fn}: expected one solid, found {len(s)}'
        bb = s[0].BoundingBox()
        solids.append(s[0].translate(cq.Vector(x - bb.xmin, -bb.ymin, -bb.zmin)))   # side by side on z = 0
        names.append(name)
        print(f'{name}: {bb.xlen:.1f} x {bb.ylen:.1f} x {bb.zlen:.1f} mm from {fn}')
        x += bb.xlen + args.gap

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    write_flat_step(solids, names, args.out)
    print(f'wrote {args.out}: {len(solids)} bodies in one part ({", ".join(names)})')


if __name__ == '__main__':
    main()
