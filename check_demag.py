# -*- coding: utf-8 -*-
"""
Checks a magnet ledger (the .xlsx written by the optimization scripts) for demagnetization risk:
computes the reverse H field on every magnet and the minimum iHc the magnets need.

usage:
    python check_demag.py <ledger.xlsx> [--Br 1270] [--max-temp 40] [--safety 1.2] [--plot]
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import magsimulator


def main():
    parser = argparse.ArgumentParser(description='Worst-case reverse H field and required iHc for a magnet ledger')
    parser.add_argument('ledger', help='ledger .xlsx file (columns X-pos, Y-pos, Z-pos, X-rot, Y-rot, Z-rot, ..., Magnet_length)')
    parser.add_argument('--Br', type=float, default=1270, help='remanence in mT, magnetization along local x (default 1270, as in the optimization scripts)')
    parser.add_argument('--max-temp', type=float, default=40, help='highest magnet temperature in C, in operation or during assembly (default 40)')
    parser.add_argument('--safety', type=float, default=1.2, help='safety factor on the reverse field (default 1.2)')
    parser.add_argument('--knee', type=float, default=0.9, help='knee field as a fraction of iHc (default 0.9)')
    parser.add_argument('--intrinsic-rot', action='store_true', help='ledger uses intrinsic rotations (the scripts use extrinsic, the default)')
    parser.add_argument('--points', type=int, default=5, help='sample points per axis inside each magnet (default 5)')
    parser.add_argument('--out', help='write the per-magnet results to this .xlsx file')
    parser.add_argument('--plot', action='store_true', help='3D plot of the magnets colored by reverse field')
    args = parser.parse_args()

    ledger = pd.read_excel(args.ledger)
    print(f'{len(ledger)} magnets in {args.ledger}, computing fields...')

    res = magsimulator.compute_demag_fields(ledger, mag_constant=[args.Br,0,0], extrinsicrot=not args.intrinsic_rot, points_per_axis=args.points)

    worst = res.sort_values('Hrev_max_kAm', ascending=False)
    print('\nworst magnets (reverse H in kA/m, positive = opposing magnetization):')
    print(worst[['X-pos','Y-pos','Z-pos','Z-rot','Hrev_center_kAm','Hrev_max_kAm']].head(10).round(1).to_string())

    H_center = res['Hrev_center_kAm'].max()
    H_max = res['Hrev_max_kAm'].max()
    ihc_needed = magsimulator.required_ihc(H_max, args.max_temp, args.safety, args.knee)
    grade = magsimulator.suggest_grade(ihc_needed, args.max_temp)

    print(f'\nworst reverse H at a magnet center:     {H_center:7.0f} kA/m  ({H_center*4*np.pi/1000:5.2f} kOe)')
    print(f'worst reverse H inside a magnet:         {H_max:7.0f} kA/m  ({H_max*4*np.pi/1000:5.2f} kOe)')
    print(f'required iHc at 20 C (datasheet value):  {ihc_needed:7.0f} kA/m  ({ihc_needed*4*np.pi/1000:5.2f} kOe)'
          f'  [max temp {args.max_temp:.0f} C, safety {args.safety}, knee {args.knee}]')
    print('lowest sufficient grade series: ' + (f'{grade} (e.g. N42{grade if grade != "N" else ""})' if grade else 'none in the CY-Mag table'))

    if args.out:
        res.to_excel(args.out, index=False)
        print('per-magnet results written to ' + args.out)

    if args.plot:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        sc = ax.scatter(res['X-pos'], res['Y-pos'], res['Z-pos'], c=res['Hrev_max_kAm'], cmap='jet')
        fig.colorbar(sc, label='max reverse H (kA/m)')
        ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.set_zlabel('z (mm)')
        plt.show()


if __name__ == '__main__':
    main()
