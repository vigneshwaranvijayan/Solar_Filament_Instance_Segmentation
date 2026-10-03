# Troubleshooting and recovery

## OpenCV fillPoly reports an incompatible output layout

The corrected `prepare_experiment.py` allocates its drawing image with:

```python
rendered = np.zeros(mask.shape, dtype=np.uint8, order='C')
polygon = np.ascontiguousarray(np.rint(points), dtype=np.int32)
cv2.fillPoly(rendered, [polygon], 1)
```

It also makes the decoded mask contiguous. COCO mask arrays can be column-major; `np.zeros_like` can preserve that layout and make the destination unsuitable for OpenCV drawing. Use the corrected notebook or synchronize it after changing the Python file:

```bash
python scripts/sync_notebook.py --write
python scripts/sync_notebook.py
```

Data preparation precedes training. If it failed at this point, the detector has not started. Rerun preparation after applying the correction; existing label files are regenerated.

## CUDA unavailable or Torchvision NMS fails

On Kaggle, enable a GPU and retain the existing matching Torch/Torchvision pair. On a standalone machine, install a matching pair through the official PyTorch selector and run:

```bash
python scripts/check_environment.py
```

Changing only one of Torch and Torchvision can break compiled operators. Resolve the environment check before starting the training stages.

## Missing training JSON or test images

Place the training JSON in the input tree as an accessible file. For extracted test images, use a folder named `test_images`. For ZIP inputs, retain a `test_images` path in the ZIP or name the archive `test.zip`. Prepared data must be written outside the input tree. Expected counts are 707 training and 180 test images.

Conflicting duplicate JSONs or image files are rejected. Use one consistent official input set rather than suppressing those checks.

## GPU out of memory

Start by reducing the affected batch size. If necessary, reduce detector image size or refiner crop size and create a separate experiment configuration. The default four-view refiner inference is also evaluated by `evaluate_experiment.py`; its `--no-tta` option can be used for a controlled speed/accuracy comparison.

Keep calibration and test settings consistent. Do not silently change only inference resolution while assuming the old calibration result remains applicable.

## Interrupted training

Detector recovery uses `detector_resume.pt`. Refiner recovery uses `refiner_last.pt`, including optimizer, scheduler and scaler state. Repeat the same stage with the same configuration and experiment directory to continue from the saved epoch. Work since the last saved epoch can be lost.

On Kaggle, preserve outputs with Save Version and download the results archive before ending the session. Attach the archive and official inputs to a fresh session to restore the saved stages. A changed configuration hash creates a new experiment instead of resuming the old one.

The command-line runner resumes from existing directories. It does not automatically extract a Kaggle archive. The standalone stages print errors and record stage status; they do not guarantee completion within a particular server session limit.

## A completed training stage is skipped

Completion markers intentionally reuse finished stages. Set a different `run_tag` for a fresh experiment. Changing epoch limits also creates a different configuration directory through the runner/notebook settings; it is not an automatic continuation of a completed model.

## Detector polygon quality is low

Inspect `polygon_conversion.csv`. External-contour polygons can omit holes and alter thin or disconnected structures. Refiner training uses original RLE masks, but detector label quality still matters. Review low-overlap cases before extending an expensive training search.

## More pooled predictions than unique instances

Metrics compare one physical prediction set against multiple annotators. TP and FP can therefore count annotation-level comparisons rather than unique masks. Use the submission row count for unique exported instances and per-annotator reports to understand pooled metrics.

## Empty predictions on an image

The prediction summary lists images with no retained instances. A CSV instance format contains no row for such images. The structural checker does not establish whether a real filament was missed; inspect the image and the detector/refiner outputs when reviewing coverage.

## Runtime is longer than expected

Separate preparation, detector training, checkpoint selection, crop construction, refiner training, calibration and test prediction. Checkpoint selection and Python mask processing can add substantial time beyond the training epoch estimates. The runner writes a stage timing journal; the notebook saves model histories and prints epoch estimates. Cached inference reduces repeated threshold-search work but does not make the first pass free.
