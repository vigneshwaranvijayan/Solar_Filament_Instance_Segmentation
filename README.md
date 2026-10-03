# Solar Filament Instance Segmentation

An instance segmentation pipeline for solar filaments in full-disk H-alpha observations. A pretrained YOLO11 segmentation model identifies individual filament candidates. A ResNet18 based refinement model reconstructs each candidate mask from its local image context. Calibration chooses the output method and filtering settings used for validation and test prediction.

The repository contains the complete Python pipeline and a self-contained Kaggle notebook. It includes the OpenCV array-layout correction for COCO mask polygon conversion.

## Contents

| File or folder | Purpose |
|---|---|
| [`V5.ipynb`](V5.ipynb) | Complete Kaggle workflow, with the pipeline Python source embedded |
| [`run_pipeline.py`](run_pipeline.py) | Command-line runner for the same preparation, training and prediction stages |
| [`configs/default.yaml`](configs/default.yaml) | Default command-line training configuration |
| [`requirements.txt`](requirements.txt) | Dependencies for a standalone environment |
| [`requirements-kaggle.txt`](requirements-kaggle.txt) | Minimal additions to Kaggle's existing runtime |
| [`filament/`](filament/) | Annotation handling, image channels, temporal folds and reference metrics |
| [`docs/METHOD.md`](docs/METHOD.md) | Architecture, supervision, losses and evaluation details |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | Data, OpenCV, GPU and recovery troubleshooting |
| [`scripts/`](scripts/) | Environment checks and notebook/source synchronization |
| [`tests/`](tests/) | CPU geometry/metric tests and native polygon regression tests |
| [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) | Dependency attribution and model references |

## Method at a glance

1. Decode the competition masks and keep observations from nearby dates in the same temporal group.
2. Separate model training, calibration and outer validation images before fitting.
3. Fine tune COCO-pretrained `yolo11m-seg.pt` on mask-derived polygons at 1536 pixel image size.
4. Select a detector checkpoint by calibration Panoptic Quality (PQ).
5. Build 384 pixel instance crops from original RLE masks, together with background examples and false detector proposals from the training partition.
6. Train a ResNet18 encoder with a GroupNorm decoder to predict a mask and foreground presence.
7. Calibrate detector-only and refined outputs, freeze the selected settings, and evaluate on the outer fold.
8. Predict masks for all test images, encode them as compressed COCO RLE and validate the submission CSV.

The refinement stage is retained only when calibration selects it. The pipeline does not assume that a second model always improves the result. With multiple folds configured, each fold is trained and evaluated separately; test inference uses the **first configured fold**, not a fold ensemble.

## Required data

Obtain the official data through the [competition page](https://www.kaggle.com/competitions/filament-segmentation-2026).

| Input | Expected size | Use |
|---|---|---|
| Training H-alpha images | 707 physical images, 2048 × 2048 | Training, calibration and validation |
| `MAGFiLO_1.0_Annotations_kaggle2026_train.json` | Competition annotation file | Mask supervision and evaluation |
| Test H-alpha images | 180 images, 2048 × 2048 | Final inference |

Training images and the annotation JSON are used together. Test images are never included in supervised fitting, negative mining or threshold selection. The test prediction command accepts images and frozen model settings; it does not accept annotation JSON.

The code accepts extracted images and supported image ZIPs. Keep the training JSON separately accessible in the input tree. Extracted test images should be under a folder named `test_images`; a ZIP named `test.zip` is also recognized. Data preparation checks for conflicting duplicate files and overlap between training and test filenames. Prepared image folders can contain symlinks, so the original input must remain available during the run.

Images, training annotations, downloaded weights and generated experiments are not bundled in this source package.

## Run on Kaggle

1. Create a Kaggle notebook and import [`V5.ipynb`](V5.ipynb).
2. Attach the official training images, training JSON and test images as inputs.
3. Enable a GPU. Enable Internet access for dependency installation and pretrained-weight downloads, or attach the required weights as inputs.
4. Review the settings in the notebook setup cell.
5. Run cells in order: preparation, pretrained weights and smoke checks, detector, instance crops, refiner, calibration/evaluation, and test prediction.
6. Review the validation comparison and segmentation previews, then download the submission CSV and results archive.

The notebook creates the Python files in `/kaggle/working/filament_project_v5`. It writes the test CSV to `/kaggle/working/filament_project_v5/submission_v5.csv` and the recovery archive to `/kaggle/working/filament_v5_results.zip`.

Kaggle's matching Torch and Torchvision installations are retained. The notebook adds pinned Ultralytics and pycocotools versions and records the actual installed environment. You do not need to upload the extracted repository as a second Kaggle input: the notebook embeds the pipeline files it uses.

### Default settings

| Setting | Value |
|---|---|
| Outer folds | `[0]` |
| Detector | `yolo11m-seg.pt` |
| Detector input size | 1536 |
| Detector epochs / batch | Up to 30 / 2 |
| Refiner encoder | ImageNet-pretrained ResNet18 |
| Refiner crop size | 384 |
| Refiner epochs / steps / batch | Up to 15 / 150 per epoch / 8 |
| Seed | 2026 |
| Refiner inference | Four flip views averaged |

Runtime depends on GPU, image preparation, checkpoint selection and proposal counts. Estimate it from measured stage and epoch timings rather than assuming a fixed duration. The notebook prints training-time estimates and saves recovery state at epoch boundaries.

## Run on a standalone GPU machine

Use Python 3.12 or a compatible environment with an NVIDIA GPU. The model stages use CUDA device 0.

### 1 Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows, activate with `.venv\Scripts\activate` instead.

Install a **matching CUDA-enabled Torch and Torchvision pair** using the command generated by the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Choose the platform and CUDA build supported by your machine. Then install this project's dependencies:

```bash
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

The environment check verifies GPU availability, the compiled CUDA NMS operator and package imports. `requirements.txt` pins Ultralytics, pycocotools and OpenCV; it is not a complete platform-specific lockfile. Each real command-line run records `requirements-observed.txt` under the experiment root.

### 2 Configure the experiment

Edit [`configs/default.yaml`](configs/default.yaml), or copy it to a separate YAML file. The notebook settings and command-line YAML are independent entry points; changing one does not automatically change the other.

Configuration changes produce a different experiment directory. To start a new experiment with otherwise identical settings, change `run_tag`. Rerunning the same configuration reuses its existing directory and recovery checkpoints.

### 3 Preview the stage commands

```bash
python run_pipeline.py \
  --config configs/default.yaml \
  --input /path/to/official_competition_input \
  --dry-run
```

The dry run prints paths and planned subprocess commands without creating outputs, downloading weights or training models. Replace the example input path with the directory containing your official data. The prepared data directory must be outside that input tree.

### 4 Execute the pipeline

```bash
python run_pipeline.py \
  --config configs/default.yaml \
  --input /path/to/official_competition_input
```

Defaults use `data/`, `experiments/`, `pretrained/` and `artifacts/submission_v5.csv` beneath the repository. Explicit paths are also supported:

```bash
python run_pipeline.py \
  --config configs/default.yaml \
  --input /path/to/official_competition_input \
  --data-root /path/to/prepared_data \
  --work-root /path/to/experiments \
  --pretrained-root /path/to/pretrained_weights \
  --out /path/to/output/submission_v5.csv
```

The runner uses `sys.executable`, so every stage runs in the active Python environment. It records elapsed stage times and completion/failure status in a JSONL journal. Model-specific recovery and completion files remain inside each experiment.

### Execute or resume selected stages

```bash
# Prepare data, fetch weights and run the GPU smoke checks.
python run_pipeline.py --input /path/to/official_competition_input \
  --stages prepare weights smoke

# Train and select the detector, then build crops and train the refiner.
python run_pipeline.py --stages detector patches refiner

# Freeze calibration settings, evaluate and create the test CSV.
python run_pipeline.py --stages evaluate predict
```

Use the same configuration and path arguments across these commands. Selected stages execute in pipeline order. `--input` is required only when `prepare` is selected. A later stage needs the earlier stage's saved files. Training modules skip completed stages and resume from compatible recovery checkpoints; they do not automatically retrain a completed model.

## Validation and model selection

The default split uses five date-grouped folds and runs fold 0. Nearby observing dates are linked when the gap is no more than three days. All annotator records of a physical image stay together. Within the non-validation groups, an inner calibration partition reserves 15% of groups.

Detector checkpoint selection and final filtering are performed on calibration images. The selected method, confidence threshold, mask threshold and minimum area are frozen before outer-fold evaluation. Final calibration compares 40 operating points across detector-only output and three refiner mask thresholds.

Reports include PQ, segmentation quality (SQ), recognition quality (RQ), TP, FP, FN, split and merge diagnostics, and matched mask Dice/IoU distributions. Predictions are evaluated separately against available annotators and the resulting counts are pooled. Consequently, pooled TP/FP/FN counts are not unique physical filament counts. Splits and merges are overlap diagnostics, not additional terms secretly inserted into the PQ formula.

The evaluator uses an IoU threshold strictly greater than 0.5 and preserves the provided reference metric's pair-count semantics. Compare it with the [organizer self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook) for the competition evaluation version you use. A local validation score is not an official hidden-test score. If a fold has already influenced development choices, treat it as a development fold and obtain additional fold evidence.

Detailed formulas and model definitions are in [`docs/METHOD.md`](docs/METHOD.md).

## Generated files

An experiment directory is named `fold<index>_<configuration hash>`.

| Output | Meaning |
|---|---|
| `training_config.json`, `partitions.json`, `folds.csv` | Settings and disjoint data partitions |
| `polygon_conversion.csv` | Original-mask versus polygon quality |
| `detector_resume.pt` | Detector recovery checkpoint |
| `detector_selected.pt`, `detector_selection.json` | Detector chosen by calibration PQ |
| `patches.json` and `patches/` | Refiner crop manifest and cached arrays |
| `refiner_best.pt`, `refiner_last.pt` | Selected weights and recoverable training state |
| `selected_settings.json`, `calibration_trials.csv` | Frozen inference rule and calibration trials |
| `validation_summary.json`, `validation_comparison.csv` | Outer-fold result and detector/refiner comparison |
| `validation_per_image.csv`, `validation_per_annotator.csv` | Detailed evaluation records |
| `validation_matched_overlaps.csv`, `preview_*.jpg` | Mask overlaps and morphology examples |
| `submission_v5.csv`, its `.metrics.json` companion | Encoded test predictions and inference summary |

Cached predictions carry fingerprints for models, images, source and runtime settings. They reuse inference when those fingerprints match. An image with no retained predictions has no instance row in the CSV and is listed in the prediction summary. The format checker verifies identifiers, native 2048 pixel mask dimensions, decoding, nonempty masks and disjoint ownership; actual detection coverage must also be reviewed.

For Kaggle recovery, attach `filament_v5_results.zip` in a fresh session and use the same training configuration. The notebook restores saved experiments and pretrained weights. Official input images are still needed to regenerate prepared data and crop arrays. The standalone runner resumes from saved directories; it does not import the Kaggle recovery ZIP automatically.

## Verify the source package

```bash
python scripts/sync_notebook.py
python -m unittest discover -s tests -v
```

The synchronization check requires only standard Python. Core metric and geometry tests run on CPU with NumPy. Native polygon tests also need the project libraries, including Torch and Ultralytics, but do not need a GPU. GPU forward/backward, detector execution and RLE round-trip checks are run by `smoke_test.py` before training.

After editing pipeline Python files, update the notebook's embedded copies:

```bash
python scripts/sync_notebook.py --write
python scripts/sync_notebook.py
```

The notebook embeds 18 pipeline source files. Repository-only helpers, tests, configuration and the command-line runner remain separate. Refreshing embedded sources clears notebook outputs and execution counts.

The source package has passed syntax/source consistency and CPU checks, and the OpenCV polygon fix has been exercised with native COCO masks. Full model training and competition scores are measured when the workflow is executed; this package does not claim a winning score.

## Share the repository on GitHub

Extract the ZIP, open a terminal inside its `filament-segmentation-v5` folder, and create an empty GitHub repository. Then run:

```bash
git init -b main
git add .
git commit -m "Add solar filament segmentation pipeline and Kaggle notebook"
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the URL of your own empty repository. The included `.gitignore` keeps images, annotations, checkpoints, caches and generated submissions out of ordinary commits. The README is at the repository root so GitHub displays it automatically. Keep large trained model artifacts separately when sharing experiment results.

## References and attribution

- [Competition overview](https://www.kaggle.com/competitions/filament-segmentation-2026)
- [Organizer self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook)
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics segmentation label format](https://docs.ultralytics.com/datasets/segment/)
- [Ultralytics model training and recovery](https://docs.ultralytics.com/modes/train/)
- [Torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [Detect and refine workflow study](https://github.com/HeShen-1/filament-segmentation-2026)

Dependency and model licensing information is documented in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
