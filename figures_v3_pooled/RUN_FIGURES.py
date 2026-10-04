#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pooled_stats import BASE_DEFAULT, OUTPUT_DEFAULT

SCRIPTS = [
    '4.2_pooled.py',
    '4.3_pooled.py',
    '4.4_pooled.py',
    '4.5_pooled.py',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', type=Path, default=BASE_DEFAULT)
    ap.add_argument('--output-dir', type=Path, default=OUTPUT_DEFAULT)
    ap.add_argument('--force-recompute', action='store_true',
                    help='Recompute the expensive pooled blocked correlations.')
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    out = args.output_dir.absolute(); out.mkdir(parents=True, exist_ok=True)

    print('\nPOOLED THREE-LOCATION PAPER FIGURES')
    print('=' * 88)
    print(f'Python      : {sys.executable}')
    print(f'Base        : {args.base_dir.absolute()}')
    print(f'Output      : {out}')
    print('Pooling rule: average city repetitions within outcome BEFORE inference')
    print('=' * 88)

    for script in SCRIPTS:
        cmd = [
            sys.executable,
            str(here / script),
            '--base-dir', str(args.base_dir.absolute()),
            '--output-dir', str(out),
        ]
        if args.force_recompute and script in {'4.4_pooled.py', '4.5_pooled.py'}:
            cmd.append('--force-recompute')
        print('\nRUNNING:', ' '.join(cmd))
        subprocess.run(cmd, check=True)

    print('\nDONE')
    print(f'Figures and exact statistics: {out}')


if __name__ == '__main__':
    main()
