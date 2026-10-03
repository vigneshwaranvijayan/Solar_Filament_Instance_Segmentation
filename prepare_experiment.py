"""Build date-grouped partitions and masks-derived YOLO segmentation labels."""
import argparse, hashlib, json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import yaml
from pycocotools import mask as coco
from sklearn.model_selection import GroupShuffleSplit
from ultralytics.data.converter import merge_multi_segment
from filament.data import AnnotationIndex, make_folds
from filament.metrics import pairwise_iou
from geometry import assert_partitions


def choose_annotation(index, name):
    ids = sorted(index.by_filename[name]); candidates = [index.rles(k) for k in ids]
    if len(ids) == 1: return ids[0]
    scores = []
    for i, rles in enumerate(candidates):
        values = []
        for j, other in enumerate(candidates):
            if i == j: continue
            iou = pairwise_iou(rles, other)
            a = float(iou.max(1).sum()) if iou.shape[1] else 0.
            b = float(iou.max(0).sum()) if iou.shape[0] else 0.
            values.append((a+b)/max(1,len(rles)+len(other)))
        scores.append(np.mean(values))
    return ids[int(np.argmax(scores))]


def polygon_for(rle):
    mask = np.ascontiguousarray(coco.decode(rle), dtype=np.uint8); h,w = mask.shape
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = [cv2.approxPolyDP(c,.5,True).reshape(-1,2) for c in contours]
    contours = [c for c in contours if len(c)>=3]
    if not contours:
        x,y,bw,bh=map(float,coco.toBbox(rle))
        contours=[np.array([[x,y],[x+bw,y],[x+bw,y+bh],[x,y+bh]])]
    polygons = [c.reshape(-1).tolist() for c in contours]
    points = contours[0] if len(contours)==1 else np.concatenate(merge_multi_segment(polygons),axis=0)
    # COCO decode returns column-major masks; OpenCV drawing needs a row-major destination.
    rendered = np.zeros(mask.shape, dtype=np.uint8, order='C')
    polygon = np.ascontiguousarray(np.rint(points), dtype=np.int32)
    cv2.fillPoly(rendered, [polygon], 1)
    intersection = np.logical_and(mask,rendered).sum(); union=np.logical_or(mask,rendered).sum()
    quality = float(intersection/max(1,union))
    return np.clip(points/np.array([w,h]),0,1).reshape(-1),quality


def main():
    p=argparse.ArgumentParser()
    for name in ('annotations','images','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--fold',type=int,default=0);p.add_argument('--seed',type=int,default=2026)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    index=AnnotationIndex(args.annotations); folds=pd.DataFrame(make_folds(index));folds.to_csv(out/'folds.csv',index=False)
    groups=folds.set_index('filename').temporal_group.to_dict()
    validation=folds.loc[folds.fold==args.fold,'filename'].tolist()
    remaining=folds.loc[folds.fold!=args.fold,'filename'].to_numpy()
    train_ids,cal_ids=next(GroupShuffleSplit(n_splits=1,test_size=.15,random_state=args.seed+args.fold).split(remaining,groups=[groups[n] for n in remaining]))
    partitions={'train':sorted(remaining[train_ids].tolist()),'calibration':sorted(remaining[cal_ids].tolist()),'validation':sorted(validation)}
    assert_partitions(partitions,groups)
    manifest={'fold':args.fold,'seed':args.seed,'partitions':partitions,'annotation_sha256':hashlib.sha256(Path(args.annotations).read_bytes()).hexdigest(),
        'target_policy':'medoid annotator chosen using allowed masks only; refiner uses original RLE masks',
        'test_images_used_for_training':False}
    manifest_path=out/'partitions.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:raise ValueError('Existing experiment has different data/partitions; choose a new output directory.')
    manifest_path.write_text(json.dumps(manifest,indent=2)); quality=[]
    for split in ('train','calibration'):
        images=out/'detector_data'/'images'/split;labels=out/'detector_data'/'labels'/split
        images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
        for i,name in enumerate(partitions[split]):
            image_id=choose_annotation(index,name);lines=[]
            for j,rle in enumerate(index.rles(image_id)):
                points,iou=polygon_for(rle);lines.append('0 '+' '.join(f'{v:.7f}' for v in points))
                quality.append({'filename':name,'image_id':image_id,'instance':j,'polygon_mask_iou':iou})
            (labels/(Path(name).stem+'.txt')).write_text('\n'.join(lines)+'\n')
            link=images/name
            if not link.exists():link.symlink_to((Path(args.images)/name).resolve())
            if (i+1)%50==0:print(f'Prepared {split} {i+1}/{len(partitions[split])}',flush=True)
    data={'path':str((out/'detector_data').resolve()),'train':'images/train','val':'images/calibration','names':{0:'filament'}}
    (out/'detector_data.yaml').write_text(yaml.safe_dump(data))
    pd.DataFrame(quality).to_csv(out/'polygon_conversion.csv',index=False)
    print(json.dumps({'partitions':{k:len(v) for k,v in partitions.items()},'polygon_quality_mean':float(np.mean([r['polygon_mask_iou'] for r in quality])),
                     'polygon_quality_below_0.9':sum(r['polygon_mask_iou']<.9 for r in quality)},indent=2))

if __name__=='__main__':main()
