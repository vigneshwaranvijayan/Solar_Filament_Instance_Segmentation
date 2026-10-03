from __future__ import annotations
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from pycocotools import mask as coco
from sklearn.model_selection import GroupKFold


def annotation_rle(annotation, height=2048, width=2048):
    seg = annotation["segmentation"]
    if isinstance(seg, list):
        if not seg: raise ValueError(f"Empty polygon: {annotation['id']}")
        return coco.merge(coco.frPyObjects(seg, height, width))
    if isinstance(seg["counts"], list): return coco.frPyObjects(seg, height, width)
    return {"size": list(seg["size"]), "counts": seg["counts"].encode() if isinstance(seg["counts"], str) else seg["counts"]}


def encode_mask(mask):
    rle = coco.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")}


def encode_pixels(ys, xs, shape=(2048, 2048)):
    height, width = shape
    positions = np.unique(np.asarray(xs, dtype=np.int64)*height + np.asarray(ys, dtype=np.int64))
    if not len(positions): counts = [height*width]
    else:
        cuts = np.where(np.diff(positions) != 1)[0]
        starts = positions[np.r_[0, cuts+1]]
        ends = positions[np.r_[cuts, len(positions)-1]]+1
        counts = np.column_stack([starts-np.r_[0,ends[:-1]], ends-starts]).ravel().tolist()
        counts.append(int(height*width-ends[-1]))
    rle = coco.frPyObjects({"size":[height,width], "counts":counts},height,width)
    return {"size":[height,width], "counts":rle["counts"].decode("ascii")}


class AnnotationIndex:
    def __init__(self, annotation_path):
        data=json.loads(Path(annotation_path).read_text())
        self.images={str(i["id"]):i for i in data["images"]}
        self.annotations=defaultdict(list)
        for a in data["annotations"]:
            if str(a["image_id"]) not in self.images: raise ValueError("Missing image record")
            self.annotations[str(a["image_id"])].append(a)
        self.by_filename=defaultdict(list)
        for image_id, image in self.images.items(): self.by_filename[Path(image["file_name"]).name].append(image_id)
        self.filenames=sorted(self.by_filename)

    def rles(self, image_id):
        image=self.images[image_id]
        return [annotation_rle(a,image["height"],image["width"]) for a in self.annotations[image_id]]

    def labels(self,image_id):
        image=self.images[image_id]
        labels=np.zeros((image["height"],image["width"]),np.int32)
        for k,rle in enumerate(self.rles(image_id),1):
            mask=coco.decode(rle).astype(bool); overlap=mask&(labels!=0)
            labels[mask&(labels==0)]=k; labels[overlap]=-1
        return labels


def make_folds(index,n_splits=5,gap_days=3):
    names=np.array(index.filenames); mapping={}; group=-1; previous=None
    for date,name in sorted((datetime.strptime(Path(n).stem[:8],"%Y%m%d"),n) for n in names):
        if previous is None or (date-previous).days>gap_days: group+=1
        mapping[name]=group; previous=date
    groups=np.array([mapping[n] for n in names])
    if len(set(groups))<n_splits: raise ValueError("Too few independent temporal groups")
    result=[]
    for fold,(_,val) in enumerate(GroupKFold(n_splits).split(names,groups=groups)):
        for n in names[val]: result.append({"filename":str(n),"fold":fold,"temporal_group":mapping[n],"station":Path(n).stem[-2:],"date":str(n)[:8]})
    return sorted(result,key=lambda r:r["filename"])


def image_channels(path):
    image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if image is None: raise ValueError(f"Cannot read {path}")
    contrast=cv2.createCLAHE(clipLimit=1.5,tileGridSize=(8,8)).apply(image)
    contrast[image==0]=0
    # Counterfactual channel: estimate how much darker each pixel is than its
    # local surroundings at two solar-feature scales.  Filaments should remain
    # dark in both the raw image and this explicit local-contrast view, while
    # many limb/texture artefacts do not.
    small=cv2.GaussianBlur(image,(0,0),9)
    large=cv2.GaussianBlur(image,(0,0),33)
    darkness=np.maximum(small.astype(np.float32)-image,large.astype(np.float32)-image)
    darkness=np.clip(darkness,0,64)*(255./64.)
    darkness[image==0]=0
    return np.stack([image,contrast,darkness]).astype(np.float32)/255.


def slice_pair(shape,offset):
    h,w=shape;dy,dx=offset
    if abs(dy)>=h or abs(dx)>=w: return (slice(0,0),slice(0,0)),(slice(0,0),slice(0,0))
    return (slice(max(0,-dy),min(h,h-dy)),slice(max(0,-dx),min(w,w-dx))), (slice(max(0,dy),min(h,h+dy)),slice(max(0,dx),min(w,w+dx)))


def affinity_targets(labels,offsets):
    targets=np.zeros((len(offsets),*labels.shape),np.float32);valid=np.zeros_like(targets)
    for k,offset in enumerate(offsets):
        a,b=slice_pair(labels.shape,offset); left,right=labels[a],labels[b]
        usable=(left>0)&(right>0)
        targets[k][a]=(left==right)&usable;valid[k][a]=usable
    return targets,valid
