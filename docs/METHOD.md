# Method and evaluation

## Data and temporal partitions

All annotations belonging to the same physical filename stay in the same partition. Five outer folds are built with `GroupKFold`. Observing dates are linked into temporal groups when successive gaps are no more than three days. Within the non-validation groups, `GroupShuffleSplit` selects an inner calibration set with `test_size=0.15` and seed `2026 + fold`.

Only the inner training partition provides gradient updates. Calibration is used for checkpoint selection and inference settings. The outer partition supplies frozen-system evaluation. Test inputs are used only after selecting the pipeline.

## Detector supervision

For a multiply annotated image, choose the annotator mask set with the greatest average agreement with the others. Agreement between two sets is the sum of each mask's maximum IoU with the other set in both directions, divided by their combined mask count. This is a mask-derived medoid policy, not an additional ground-truth label source.

OpenCV extracts external contours and simplifies them with a 0.5 pixel tolerance. Disconnected contours are combined with Ultralytics' multi-segment converter. Very thin masks can require a box-shaped fallback. `polygon_conversion.csv` records overlap with each original RLE mask because polygon conversion can bridge components or omit holes and fine structures. The refiner uses original RLE masks rather than these converted polygons.

COCO decoding can produce column-major arrays. The conversion function explicitly makes the mask and integer contour arrays contiguous and allocates a row-major drawing destination. This prevents the OpenCV `fillPoly` output-layout error.

The detector fine tunes `yolo11m-seg.pt` using 1536 pixel input, batch 2 and up to 30 epochs. It uses AdamW, initial learning rate 0.001, cosine scheduling, final learning-rate factor 0.05 and patience 10. Mosaic probability is 0.15 and mosaic closes for the final five epochs. Mixup and copy-paste are disabled; modest geometric and brightness augmentations are used.

The detector uses the pinned Ultralytics instance-segmentation training objective. Its training metrics are retained, but selection among available saved checkpoints is based on pooled calibration PQ. This avoids assuming that the best library mAP checkpoint is also the best PQ checkpoint.

## Instance refinement

Each target box receives square context with padding factor 1.5 and minimum side 48, then is resized to 384 pixels. Channels are grayscale, CLAHE-enhanced intensity and a binary candidate box. Targets are exact original masks. Other annotated foreground is ignored where it does not belong to the chosen target. Negative seed boxes must contain no annotated foreground from any annotator.

Training-only detector false proposals and dark background regions provide negatives. Weighted sampling draws approximately 75% positive and 25% negative examples. Crop augmentation uses quarter turns, horizontal flips and brightness scaling from 0.85 to 1.15.

`Refiner` uses an ImageNet-pretrained ResNet18 encoder. Encoder stage widths are 64, 128, 256 and 512. The skip-connected decoder has widths 256, 128, 64 and 32, with GroupNorm and SiLU. One head predicts the mask; another applies global pooling and a linear projection to predict foreground presence. Encoder BatchNorm running statistics stay fixed while trainable weights are updated.

The implemented objective is:

$$L_{ref} = L_{boundary\ BCE} + L_{positive\ Dice} + 0.3L_{presence\ BCE}.$$

For probability $p_i$, binary mask $y_i$ and validity $v_i$, boundary BCE weights are $v_i(1+3b_i)$, where $b$ is the 3-by-3 target dilation minus erosion. Weighted BCE is divided by the sum of valid weights.

$$L_{Dice} = 1 - \frac{2\sum_i p_i y_i v_i + 1}{\sum_i p_i v_i + \sum_i y_i v_i + 1}.$$

Dice is averaged only over positive examples. Presence BCE uses a target of one for positive crops and zero for negatives. Training uses batch 8, 150 steps per epoch, AdamW with learning rate 0.0003 and weight decay 0.0001, mixed precision and gradient clipping at norm 5. Cosine scheduling approaches learning rate 0.00001. Calibration patch loss selects weights; five non-improving checks stop the stage.

## Inference and calibration

Detector proposal confidence begins at 0.03 with box NMS IoU 0.5 and at most 100 candidates per image. Native-resolution masks are requested. The refiner averages predictions for the original crop and horizontal, vertical and combined flips, reverses the transforms, and maps probabilities to the original image.

The refined ranking score is:

$$s_{refined} = s_{YOLO}\,\sigma(z_{presence}).$$

This combined score is used for empirically calibrated ranking and filtering; it is not treated as a calibrated probability of correctness.

Calibration searches detector-only output and refiner masks at thresholds 0.4, 0.5 and 0.6, with final score thresholds 0.03, 0.08, 0.15, 0.25 and 0.40, and minimum areas 32 or 128 pixels. This creates 40 final rules. Ties favor fewer false positives and detector-only output when exactly tied.

Candidates receive disjoint pixel ownership in score order before final confidence/area filtering, using base area 8. This fixed ownership policy makes cached IoU matrices reusable across final operating points. It also means removing a candidate later does not reassign its pixels to a weaker candidate. Disconnected fragments belonging to a retained candidate stay in one instance mask.

## Evaluation

For each physical image, compare predictions independently against each annotator set and pool the results. A pair qualifies only if IoU is strictly greater than 0.5. Preserve the reference evaluator's qualifying-pair counts; no alternative assignment algorithm is substituted.

$$PQ = \frac{\sum_{(g,r)\in M} IoU(g,r)}{TP + 0.5FP + 0.5FN},\quad SQ = \frac{\sum_{(g,r)\in M} IoU(g,r)}{TP},\quad RQ = \frac{TP}{TP + 0.5FP + 0.5FN}.$$

Thus $PQ=SQ\,RQ$. Split and merge diagnostics use any positive overlap. Dice and IoU distributions report qualifying matched pairs. A 1,000-draw paired temporal-group bootstrap describes the variation of refiner-minus-detector pooled PQ. The interval is unavailable with fewer than two temporal groups.

Preview panels show low-scoring and middle-scoring images with a labeled annotator record. They complement the numerical metrics: a higher pooled score should correspond to plausible, complete filament morphology.

## Scope

The current workflow evaluates configured folds independently and predicts test images using the first configured fold. It does not implement a fold ensemble, automatic full-data refit, test-label learning or a guaranteed score target. Those would require separate experiments and validation.
