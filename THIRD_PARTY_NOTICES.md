# Attribution and dependency notices

This repository uses external libraries and pretrained model weights. Their licenses and attribution remain with their respective maintainers; this source bundle does not include pretrained weights or the competition data.

| Component | Source and role |
|---|---|
| Ultralytics YOLO11 | [Official repository](https://github.com/ultralytics/ultralytics), [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/); instance detector and multi-segment conversion utility |
| PyTorch and Torchvision | [PyTorch](https://pytorch.org/), [Torchvision](https://github.com/pytorch/vision); neural training and ResNet18 encoder |
| COCO API and pycocotools | [COCO API](https://github.com/cocodataset/cocoapi); compressed mask encoding and overlap operations |
| OpenCV | [OpenCV](https://opencv.org/); preprocessing, contour conversion and visualizations |
| NumPy, pandas and SciPy | Array processing, records and numerical utilities |
| scikit-learn | [scikit-learn](https://scikit-learn.org/); grouped data splitting |

Ultralytics publishes its software and models under AGPL-3.0 and offers Enterprise licensing. Review the official [license information](https://docs.ultralytics.com/license/) for the distribution and use you intend. This archive does not assign an additional license to original project code on the owner's behalf.

Model references:

- Glenn Jocher and Jing Qiu. Ultralytics YOLO11. 2024. https://github.com/ultralytics/ultralytics
- Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385
- Torchvision ResNet18 weight definition. https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html

Task and workflow references:

- Solar Filament Segmentation Challenge 2026. https://www.kaggle.com/competitions/filament-segmentation-2026
- Organizer Self Evaluation Notebook. https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook
- Detect and refine workflow study. https://github.com/HeShen-1/filament-segmentation-2026

Competition data access and use are governed by the competition terms. This source package expects users to obtain the official inputs separately.
