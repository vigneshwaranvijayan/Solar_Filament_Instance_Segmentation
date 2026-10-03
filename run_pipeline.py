"""Run the notebook's stages as standalone subprocesses using explicit paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
STAGES = ('prepare', 'weights', 'smoke', 'detector', 'patches', 'refiner', 'evaluate', 'predict')


def load_config(path):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    defaults = yaml.safe_load((ROOT / 'configs/default.yaml').read_text())
    if not isinstance(cfg, dict) or set(cfg) - set(defaults):
        raise ValueError('Configuration must be a mapping with the keys in configs/default.yaml')
    cfg = {**defaults, **cfg}
    folds = cfg['folds']
    if not isinstance(folds, list) or not folds or len(set(folds)) != len(folds) or any(type(f) is not int or f not in range(5) for f in folds):
        raise ValueError('folds must contain distinct integers from 0 through 4')
    for key in ('detector_epochs', 'detector_size', 'detector_batch', 'refiner_epochs', 'refiner_steps', 'refiner_batch', 'refiner_size'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    if type(cfg['seed']) is not int or cfg['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not isinstance(cfg['detector_model'], str) or Path(cfg['detector_model']).name != cfg['detector_model'] or not cfg['detector_model'].endswith('-seg.pt'):
        raise ValueError('detector_model must be a segmentation checkpoint filename ending in -seg.pt')
    if not isinstance(cfg['run_tag'], str) or not cfg['run_tag'].strip():
        raise ValueError('run_tag must be a nonempty string')
    return cfg


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch_weights(detector, encoder):
    from ultralytics.utils.downloads import attempt_download_asset
    import torch
    detector.parent.mkdir(parents=True, exist_ok=True)
    if not detector.is_file():
        attempt_download_asset(str(detector))
    if not detector.is_file():
        raise FileNotFoundError('Detector checkpoint was not downloaded: ' + str(detector))
    if not encoder.is_file():
        torch.hub.download_url_to_file('https://download.pytorch.org/models/resnet18-f37072fd.pth', str(encoder), hash_prefix='f37072fd')
    if not digest(encoder).startswith('f37072fd'):
        raise ValueError('Encoder checkpoint checksum mismatch')
    (detector.parent / 'pretrained_manifest.json').write_text(json.dumps({p.name: digest(p) for p in (detector, encoder)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/default.yaml')
    parser.add_argument('--input', type=Path, help='Official competition input tree; required for prepare')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--work-root', type=Path, default=ROOT / 'experiments')
    parser.add_argument('--pretrained-root', type=Path, default=ROOT / 'pretrained')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/submission_v5.csv')
    parser.add_argument('--stages', nargs='+', choices=('all',) + STAGES, default=['all'])
    parser.add_argument('--dry-run', action='store_true', help='Print the stage commands without executing or creating outputs')
    args = parser.parse_args()
    if 'all' in args.stages and len(args.stages) > 1:
        parser.error('Use all by itself or list specific stages')
    selected = STAGES if args.stages == ['all'] else tuple(s for s in STAGES if s in args.stages)
    if 'prepare' in selected and args.input is None:
        parser.error('--input is required for the prepare stage')
    cfg = load_config(args.config)
    training = {k: v for k, v in cfg.items() if k != 'folds'}
    training['pipeline_version'] = 'detref-v5-1'
    config_id = hashlib.sha256(json.dumps(training, sort_keys=True).encode()).hexdigest()[:10]
    data, work, pretrained, submission = (p.resolve() for p in (args.data_root, args.work_root, args.pretrained_root, args.out))
    annotations = data / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json'
    images, test = data / 'train_images', data / 'test_images'
    experiments = {fold: work / f'fold{fold}_{config_id}' for fold in cfg['folds']}
    detector = pretrained / cfg['detector_model']
    encoder = pretrained / 'resnet18-f37072fd.pth'
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    if not args.dry_run:
        work.mkdir(parents=True, exist_ok=True)
        for experiment in experiments.values():
            experiment.mkdir(parents=True, exist_ok=True)
            path = experiment / 'training_config.json'
            if path.exists() and json.loads(path.read_text()) != training:
                raise ValueError('Existing experiment has a different configuration: ' + str(experiment))
            path.write_text(json.dumps(training, indent=2))
        with (work / 'requirements-observed.txt').open('w') as stream:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
    journal = work / ('stage_history_' + config_id + '.jsonl')

    def record(label, began, status):
        if not args.dry_run:
            with journal.open('a') as stream:
                stream.write(json.dumps({'stage': label, 'started_unix': began, 'seconds': time.time() - began, 'status': status}) + '\n')

    def run(label, script, *values):
        command = [sys.executable, str(ROOT / script), *map(str, values)]
        print(label + ': ' + shlex.join(command), flush=True)
        if args.dry_run:
            return
        began = time.time()
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except BaseException:
            record(label, began, 'interrupted_or_failed')
            raise
        record(label, began, 'completed')

    if 'prepare' in selected:
        run('prepare_data', 'prepare_data.py', '--input', args.input.resolve(), '--output', data, '--folds', work / 'all_folds.csv')
        if not args.dry_run:
            for directory, expected in ((images, 707), (test, 180)):
                count = len([p for p in directory.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
                if count != expected:
                    raise ValueError(f'Expected {expected} images in {directory}, found {count}')
        for fold, experiment in experiments.items():
            run(f'prepare_fold_{fold}', 'prepare_experiment.py', '--annotations', annotations, '--images', images, '--out', experiment, '--fold', fold, '--seed', cfg['seed'])
    if 'weights' in selected:
        print('weights: download or reuse ' + str(detector) + ' and ' + str(encoder), flush=True)
        if not args.dry_run:
            began = time.time()
            try:
                fetch_weights(detector, encoder)
            except BaseException:
                record('weights', began, 'interrupted_or_failed')
                raise
            record('weights', began, 'completed')
    if 'smoke' in selected:
        first = experiments[cfg['folds'][0]]
        image = '<first-training-image-from-partitions.json>' if args.dry_run else images / json.loads((first / 'partitions.json').read_text())['partitions']['train'][0]
        run('smoke', 'smoke_test.py', '--detector-weights', detector, '--encoder-weights', encoder, '--image', image, '--size', cfg['refiner_size'])
    for fold, experiment in experiments.items():
        seed = cfg['seed'] + fold
        if 'detector' in selected:
            run(f'detector_fold_{fold}', 'train_detector.py', '--experiment', experiment, '--weights', detector, '--epochs', cfg['detector_epochs'], '--imgsz', cfg['detector_size'], '--batch', cfg['detector_batch'], '--seed', seed)
            run(f'select_fold_{fold}', 'select_detector.py', '--experiment', experiment, '--images', images, '--annotations', annotations, '--imgsz', cfg['detector_size'])
        if 'patches' in selected:
            run(f'patches_fold_{fold}', 'patches.py', '--experiment', experiment, '--annotations', annotations, '--images', images, '--detector', experiment / 'detector_selected.pt', '--size', cfg['refiner_size'], '--imgsz', cfg['detector_size'])
        if 'refiner' in selected:
            run(f'refiner_fold_{fold}', 'train_refiner.py', '--experiment', experiment, '--encoder-weights', encoder, '--epochs', cfg['refiner_epochs'], '--steps', cfg['refiner_steps'], '--batch', cfg['refiner_batch'], '--seed', seed)
        if 'evaluate' in selected:
            run(f'evaluate_fold_{fold}', 'evaluate_experiment.py', '--experiment', experiment, '--images', images, '--annotations', annotations, '--imgsz', cfg['detector_size'])
    if 'predict' in selected:
        if not args.dry_run:
            submission.parent.mkdir(parents=True, exist_ok=True)
        experiment = experiments[cfg['folds'][0]]
        run('test_predict', 'predict_test.py', '--experiment', experiment, '--images', test, '--out', submission)
        run('submission_check', 'validate_submission.py', '--submission', submission, '--images', test)
    print('Planned stages displayed.' if args.dry_run else 'Requested stages completed.', flush=True)


if __name__ == '__main__':
    main()
