"""Organizer v6 semantics: all qualifying pairs, per-annotator comparison, pooled counts."""
from dataclasses import dataclass,asdict
import numpy as np
from pycocotools import mask as coco

@dataclass
class Counts:
    tp:int=0
    fp:int=0
    fn:int=0
    sum_iou:float=0.
    split_gt:int=0
    merged_predictions:int=0
    def add(self,other):
        for key in asdict(self): setattr(self,key,getattr(self,key)+getattr(other,key))
        return self
    def report(self):
        d=self.tp+.5*self.fp+.5*self.fn
        return {**asdict(self),"pq":self.sum_iou/d if d else 0.,"sq":self.sum_iou/self.tp if self.tp else 0.,"rq":self.tp/d if d else 0.}

def overlap_counts(iou):
    iou=np.asarray(iou,np.float64);hit=iou>.5;overlap=iou>0
    return Counts(int(hit.sum()),int((hit.sum(0)==0).sum()),int((hit.sum(1)==0).sum()),float(iou[hit].sum()),int((overlap.sum(1)>1).sum()),int((overlap.sum(0)>1).sum()))

def pairwise_iou(gt,pred):
    if not gt or not pred:return np.zeros((len(gt),len(pred)),np.float64)
    return np.asarray(coco.iou(pred,gt,[0]*len(gt)),np.float64).T

def evaluate_image(index,filename,predicted_rles):
    total,rows=Counts(),[]
    for image_id in index.by_filename[filename]:
        count=overlap_counts(pairwise_iou(index.rles(image_id),predicted_rles));total.add(count)
        rows.append({"filename":filename,"annotator_image":image_id,**count.report()})
    return total,rows
