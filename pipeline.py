"""Detector masks and optional instance-conditioned refinement; all outputs disjoint."""
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from filament.data import encode_mask,encode_pixels,image_channels
from geometry import ordered_disjoint
from patches import patch_input
from refiner import Refiner


class Pipeline:
    def __init__(self,detector,refiner=None,imgsz=1536,tta=True):
        self.detector=YOLO(str(detector));self.imgsz=imgsz;self.tta=tta;self.refiner=None
        if refiner:
            saved=torch.load(refiner,map_location='cpu',weights_only=True);self.refiner=Refiner().cuda();self.refiner.load_state_dict(saved['model']);self.refiner.eval();self.size=saved['size']
    @torch.inference_mode()
    def candidates(self,path,mask_thresholds=(.4,.5,.6)):
        result=self.detector.predict(str(path),imgsz=self.imgsz,conf=.03,iou=.5,max_det=100,retina_masks=True,device=0,verbose=False)[0]
        records=[]
        if result.boxes is None or result.masks is None:return records
        boxes=result.boxes.xyxy.cpu().numpy();scores=result.boxes.conf.cpu().numpy();shape=result.orig_shape
        channels=image_channels(path) if self.refiner is not None else None
        for i,(box,score) in enumerate(zip(boxes,scores)):
            raw=result.masks.data[i].cpu().numpy()>0
            if raw.shape!=shape:raise ValueError('retina_masks did not return original image resolution')
            refined={};presence=1.
            if self.refiner is not None:
                patch,crop=patch_input(channels,box,self.size);x=torch.from_numpy(patch[None]).cuda();values=[];presences=[]
                transforms=[(),(-1,),(-2,),(-2,-1)] if self.tta else [()]
                inputs=torch.cat([x.flip(dims) if dims else x for dims in transforms],0)
                with torch.autocast('cuda',dtype=torch.float16):logits,objectness=self.refiner(inputs)
                for j,dims in enumerate(transforms):
                    probability=logits[j:j+1].float().sigmoid();probability=probability.flip(dims) if dims else probability
                    values.append(probability);presences.append(objectness[j:j+1].float().sigmoid())
                probability=torch.stack(values).mean(0)[0,0].cpu().numpy();presence=float(torch.stack(presences).mean())
                x1,y1,x2,y2=crop;probability=cv2.resize(probability,(x2-x1,y2-y1),interpolation=cv2.INTER_LINEAR)
                for threshold in mask_thresholds:
                    ys,xs=np.where(probability>=threshold)
                    refined[str(threshold)]=encode_pixels(ys+y1,xs+x1,shape)
            records.append({'raw':encode_mask(raw),'refined':refined,'score':float(score),'refined_score':float(score)*presence})
        return records


def prepare_candidates(records,mode,mask_threshold=.5,shape=(2048,2048)):
    """Fix overlap ownership before confidence/area calibration for cacheable IoUs."""
    from pycocotools import mask as coco
    patches=[]
    for r in records:
        score=r['score'] if mode=='detector' else r['refined_score']
        rle=r['raw'] if mode=='detector' else r['refined'][str(mask_threshold)]
        mask=coco.decode(rle).astype(bool)
        if mask.shape!=shape:raise ValueError('Unexpected mask dimensions')
        patches.append({'crop':(0,0,shape[1],shape[0]),'mask':mask,'score':score})
    return [{'rle':encode_mask(r['mask']),'score':r['score'],'area':int(r['mask'].sum())}
            for r in ordered_disjoint(patches,shape,8)]


def select_prepared(records,settings):
    return [r['rle'] for r in records if r['score']>=settings['confidence'] and r['area']>=settings['min_area']]
