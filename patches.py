"""Use original masks for refiner targets and dark/false-proposal background patches."""
import argparse,hashlib,json
from pathlib import Path
import cv2
import numpy as np
from pycocotools import mask as coco
from ultralytics import YOLO
from filament.data import AnnotationIndex,image_channels
from geometry import context_crop,box_iou


def patch_input(channels,box,size=256):
    _,h,w=channels.shape;crop=context_crop(box,w,h);x1,y1,x2,y2=crop
    gray=cv2.resize(channels[0,y1:y2,x1:x2],(size,size),interpolation=cv2.INTER_AREA)
    contrast=cv2.resize(channels[1,y1:y2,x1:x2],(size,size),interpolation=cv2.INTER_AREA)
    seed=np.zeros((y2-y1,x2-x1),np.uint8)
    xa=max(0,int(np.floor(box[0]))-x1);ya=max(0,int(np.floor(box[1]))-y1)
    xb=min(seed.shape[1],int(np.ceil(box[2]))-x1);yb=min(seed.shape[0],int(np.ceil(box[3]))-y1)
    if xb>xa and yb>ya:seed[ya:yb,xa:xb]=1
    seed=cv2.resize(seed,(size,size),interpolation=cv2.INTER_NEAREST)
    return np.stack([gray,contrast,seed]).astype(np.float32),crop


def save_patch(path,channels,box,truth,union,size):
    x,crop=patch_input(channels,box,size);x1,y1,x2,y2=crop
    target=cv2.resize(truth[y1:y2,x1:x2],(size,size),interpolation=cv2.INTER_NEAREST)
    occupied=cv2.resize(union[y1:y2,x1:x2],(size,size),interpolation=cv2.INTER_NEAREST)
    valid=((occupied==0)|(target>0)).astype(np.uint8)
    np.savez_compressed(path,x=np.round(x*255).astype(np.uint8),target=target,valid=valid)


def main():
    p=argparse.ArgumentParser()
    for name in ('experiment','annotations','images','detector'):p.add_argument('--'+name,required=True)
    p.add_argument('--size',type=int,default=256);p.add_argument('--imgsz',type=int,default=1536)
    args=p.parse_args();out=Path(args.experiment);root=out/'patches';root.mkdir(exist_ok=True)
    descriptor={'detector_sha256':hashlib.sha256(Path(args.detector).read_bytes()).hexdigest(),'size':args.size,'imgsz':args.imgsz,
        'annotations_sha256':hashlib.sha256(Path(args.annotations).read_bytes()).hexdigest(),'partitions_sha256':hashlib.sha256((out/'partitions.json').read_bytes()).hexdigest()}
    if (out/'patches.json').exists():
        saved=json.loads((out/'patches.json').read_text())
        if saved.get('descriptor')!=descriptor:raise ValueError('Patch cache belongs to a different model or experiment.')
        if all(Path(r['path']).is_file() for r in saved['records']):print('Patch generation already complete; using cached targets.');return
    progress=out/'patch_generation.json'
    if progress.exists() and json.loads(progress.read_text())!=descriptor:raise ValueError('Partial patch cache belongs to a different experiment.')
    progress.write_text(json.dumps(descriptor))
    partitions=json.loads((out/'partitions.json').read_text())['partitions'];index=AnnotationIndex(args.annotations)
    detector=YOLO(args.detector);rows=[];rng=np.random.default_rng(2026)
    for split in ('train','calibration'):
        for i,name in enumerate(partitions[split]):
            channels=image_channels(Path(args.images)/name);_,h,w=channels.shape
            rles=[r for k in index.by_filename[name] for r in index.rles(k)]
            union=np.zeros((h,w),np.uint8);boxes=[]
            for r in rles:
                union|=coco.decode(r).astype(np.uint8);x,y,bw,bh=map(float,coco.toBbox(r));boxes.append([x,y,x+bw,y+bh])
            for j,(rle,box) in enumerate(zip(rles,boxes)):
                key=f'{split}_{Path(name).stem}_positive_{j:03d}.npz';path=root/key
                if not path.exists():save_patch(path,channels,box,coco.decode(rle).astype(np.uint8),union,args.size)
                rows.append({'path':str(path.resolve()),'split':split,'positive':1,'filename':name,'kind':'mask_target'})
            negatives=[]
            if split=='train':
                result=detector.predict(str(Path(args.images)/name),imgsz=args.imgsz,conf=.08,iou=.5,max_det=100,retina_masks=True,device=0,verbose=False)[0]
                if result.boxes is not None:
                    for box in result.boxes.xyxy.cpu().numpy():
                        xa,ya,xb,yb=np.round(box).astype(int);xa,ya=max(0,xa),max(0,ya);xb,yb=min(w,xb),min(h,yb)
                        if xb>xa and yb>ya and union[ya:yb,xa:xb].sum()==0 and (not boxes or box_iou(box,boxes).max()<.1):
                            negatives.append((box.tolist(),'detector_false_proposal'))
            dark=(channels[2]>.15)&(channels[0]>.05)&(union==0);ys,xs=np.where(dark)
            for attempt in range(60):
                if len(negatives)>=6 or not len(xs):break
                j=int(rng.integers(len(xs)));side=int(rng.integers(40,160));cx,cy=int(xs[j]),int(ys[j])
                box=[max(0,cx-side//2),max(0,cy-side//2),min(w,cx+side//2),min(h,cy+side//2)]
                xa,ya,xb,yb=box
                if xb>xa and yb>ya and union[ya:yb,xa:xb].sum()==0:negatives.append((box,'dark_background'))
            for j,(box,kind) in enumerate(negatives[:10]):
                key=f'{split}_{Path(name).stem}_negative_{j:03d}.npz';path=root/key
                if not path.exists():save_patch(path,channels,box,np.zeros((h,w),np.uint8),union,args.size)
                rows.append({'path':str(path.resolve()),'split':split,'positive':0,'filename':name,'kind':kind})
            if (i+1)%25==0:print(f'Refiner patches {split} {i+1}/{len(partitions[split])}',flush=True)
    if not any(r['positive']==0 and r['split']=='train' for r in rows):raise RuntimeError('No valid negative training patches.')
    payload={'size':args.size,'records':rows,'descriptor':descriptor}
    (out/'patches.json').write_text(json.dumps(payload))
    print(json.dumps({'patches':len(rows),'negative_training_patches':sum(r['split']=='train' and not r['positive'] for r in rows)},indent=2))

if __name__=='__main__':main()
