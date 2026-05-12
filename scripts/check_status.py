"""
Status checker for all running background processes.
Run periodically to monitor progress.

Usage:
    .venv/bin/python scripts/check_status.py

    # Auto-refresh every 5 minutes:
    while true; do .venv/bin/python scripts/check_status.py; sleep 300; done
"""

import json
import subprocess
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_DIR / 'data' / 'processed'
CORRECTED_DIR = PROJECT_DIR / 'data' / 'corrected'
TEACHER_CHECKPOINT_DIR = PROCESSED_DIR / 'teacher_checkpoints'
SONNET_CHECKPOINT_DIR = CORRECTED_DIR / 'sonnet_checkpoints'
TOTAL_DOMAINS = 96986

def check_process(name):
    result = subprocess.run(
        ['pgrep', '-f', name], capture_output=True, text=True
    )
    pids = result.stdout.strip().split('\n') if result.stdout.strip() else []
    return pids

def count_sonnet():
    results = {}
    for f in sorted(SONNET_CHECKPOINT_DIR.glob('checkpoint_*.json')):
        with open(f) as fh:
            results.update(json.load(fh))
    valid = sum(1 for v in results.values() if not v.get('category', '').startswith('_'))
    errors = len(results) - valid
    return valid, errors

def count_teacher():
    results = {}
    for f in sorted(TEACHER_CHECKPOINT_DIR.glob('checkpoint_*.json')):
        with open(f) as fh:
            results.update(json.load(fh))
    valid = sum(1 for v in results.values()
               if '_ERROR' not in v and '_PARSE_ERROR' not in v and '_INVALID_CATEGORIES' not in v)
    errors = len(results) - valid
    return valid, errors

def check_modernbert():
    result = subprocess.run(
        ['pgrep', '-f', 'nbconvert.*05_modernbert'], capture_output=True, text=True
    )
    if result.stdout.strip():
        return 'RUNNING'
    model_dir = PROJECT_DIR / 'models' / 'modernbert_v1_best'
    if model_dir.exists():
        meta_path = model_dir / 'meta.json'
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            return f"DONE - Val Top-1: {meta['val_top1_accuracy']*100:.1f}%"
        return 'DONE (no meta)'
    return 'NOT RUNNING'

def main():
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'\n{"="*70}')
    print(f'  STATUS CHECK: {now}')
    print(f'{"="*70}')

    # Sonnet
    print(f'\n  [1] SONNET DATA CORRECTION (run_sonnet_correction.py)')
    sonnet_pids = check_process('run_sonnet_correction')
    sonnet_valid, sonnet_errors = count_sonnet()
    sonnet_pct = sonnet_valid / TOTAL_DOMAINS * 100
    sonnet_remaining = TOTAL_DOMAINS - sonnet_valid
    sonnet_eta = sonnet_remaining / 13.0 / 60 if sonnet_remaining > 0 else 0
    status = 'RUNNING' if sonnet_pids else 'STOPPED'
    print(f'      Status: {status} (PID: {", ".join(sonnet_pids) if sonnet_pids else "none"})')
    print(f'      Progress: {sonnet_valid:,} / {TOTAL_DOMAINS:,} ({sonnet_pct:.1f}%)')
    print(f'      Errors: {sonnet_errors}')
    print(f'      ETA: ~{sonnet_eta:.0f} min ({sonnet_eta/60:.1f} hrs)' if status == 'RUNNING' else '      ETA: N/A')

    # Teacher
    print(f'\n  [2] OPUS TEACHER LABELING (run_teacher_labeling.py)')
    teacher_pids = check_process('run_teacher_labeling')
    teacher_valid, teacher_errors = count_teacher()
    teacher_pct = teacher_valid / TOTAL_DOMAINS * 100
    status = 'RUNNING' if teacher_pids else 'STOPPED'
    print(f'      Status: {status} (PID: {", ".join(teacher_pids) if teacher_pids else "none"})')
    print(f'      Progress: {teacher_valid:,} / {TOTAL_DOMAINS:,} ({teacher_pct:.1f}%)')
    print(f'      Errors: {teacher_errors}')
    if status == 'RUNNING':
        teacher_remaining = TOTAL_DOMAINS - teacher_valid
        teacher_eta = teacher_remaining / 2.5 / 60
        print(f'      ETA (full coverage): ~{teacher_eta:.0f} min ({teacher_eta/60:.1f} hrs)')
        print(f'      Note: Full coverage not required; 16K+ is sufficient for distillation')

    # ModernBERT
    print(f'\n  [3] MODERNBERT FINE-TUNING (notebook 05)')
    bert_status = check_modernbert()
    print(f'      Status: {bert_status}')

    print(f'\n{"="*70}\n')

if __name__ == '__main__':
    main()
