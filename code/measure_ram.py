"""Measure CPU evaluation RAM on Windows, including Python launcher children.

Install measurement-only dependency: python -m pip install psutil
Run: python measure_ram.py --checkpoint runs/student-deep-16000/checkpoint.pt
"""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    output = args.checkpoint.parent / 'test_ram_measurement.json'
    command = [sys.executable, 'evaluate.py', '--checkpoint', str(args.checkpoint),
               '--split', 'test', '--device', 'cpu', '--precision', 'fp32',
               '--output', str(output)]
    peak_bytes = 0
    peak_processes = []
    with subprocess.Popen(command) as child:
        root = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                family = [root, *root.children(recursive=True)]
                observed = []
                current = 0
                for process in family:
                    try:
                        rss = process.memory_info().rss
                        current += rss
                        observed.append((process.pid, process.name(), rss))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                if current > peak_bytes:
                    peak_bytes = current
                    peak_processes = observed
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.05)
        if child.returncode:
            raise SystemExit(f'Evaluator failed with exit code {child.returncode}')
    print(f'Peak process-tree RSS: {peak_bytes / 2**30:.3f} GiB ({peak_bytes} bytes)')
    print('Processes at peak:', peak_processes)
    if peak_bytes < 64 * 2**20:
        print('WARNING: This result is implausibly low for PyTorch; do not report it.')
    print(f'Evaluator JSON: {output}')


if __name__ == '__main__':
    main()
