"""Check CUDA, required imports and the Torch/Torchvision compiled operator pair."""
import importlib.metadata
import json
import torch
import torchvision
import cv2
from pycocotools import mask
from ultralytics import YOLO


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA-enabled PyTorch build and NVIDIA GPU are required for training and inference.')
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device='cuda')
    scores = torch.tensor([.9, .8], device='cuda')
    assert torchvision.ops.nms(boxes, scores, .5).numel() == 1
    packages = ('torch', 'torchvision', 'ultralytics', 'pycocotools', 'numpy', 'pandas', 'scikit-learn', 'PyYAML')
    print(json.dumps({'gpu': torch.cuda.get_device_name(0), 'opencv': cv2.__version__, 'versions': {p: importlib.metadata.version(p) for p in packages}, 'cuda_nms': 'passed'}, indent=2))


if __name__ == '__main__':
    main()
