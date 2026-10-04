#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DEFAULT = Path('/scratch/victoria.estanislau/ai-summary')

SCRIPTS = [
    'plot_4_2_three_locations.py',
    'plot_4_3_three_locations.py',
    'plot_4_4_three_locations.py',
    'plot_4_5_three_locations.py',
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Generate final three-location figures for Sections 4.2-4.5.'
    )
    ap.add_argument('--base-dir', type=Path, default=BASE_DEFAULT)
    ap.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Default: <base-dir>/figures_three_locations',
    )
    args = ap.parse_args()

    base = args.base_dir.absolute()
    out = (args.output_dir or (base / 'figures_three_locations')).absolute()
    out.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).absolute().parent
    python = sys.executable

    print('=' * 88)
    print('THREE-LOCATION PAPER FIGURES')
    print('=' * 88)
    print(f'Python     : {python}')
    print(f'Base       : {base}')
    print(f'Output     : {out}')
    print('Locations  : Dallas · New York · Los Angeles')
    print('=' * 88)

    for script_name in SCRIPTS:
        script = here / script_name
        cmd = [
            python,
            str(script),
            '--base-dir', str(base),
            '--output-dir', str(out),
        ]
        print('\n' + '-' * 88)
        print('RUNNING:', ' '.join(cmd))
        print('-' * 88)
        subprocess.run(cmd, check=True)

    print('\n' + '=' * 88)
    print('DONE')
    print('=' * 88)
    print(f'All figures and exact plotted statistics are in:\n{out}')


if __name__ == '__main__':
    main()
