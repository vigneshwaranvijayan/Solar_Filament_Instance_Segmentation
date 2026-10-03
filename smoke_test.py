"""One forward/backward pass and round-trip encoding before expensive training."""
import argparse,json
import numpy as np
import torch
from ultralytics import YOLO
from pycocotools import mask as coco
from filament.data import encode_mask
from refiner import Refiner,objective
from prepare_experiment import polygon_for


def main():
    p=argparse.ArgumentParser();p.add_argument('--detector-weights',required=True);p.add_argument('--encoder-weights');p.add_argument('--image',required=True);p.add_argument('--size',type=int,default=384);args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('Enable a GPU.')
    for disconnected in (False, True):
        mask=np.zeros((64,80),np.uint8,order='F');mask[5:25,8:20]=1
        if disconnected:mask[40:55,50:70]=1
        points,quality=polygon_for(encode_mask(mask))
        assert points.size>=6 and np.isfinite(points).all() and ((points>=0)&(points<=1)).all()
        assert 0<quality<=1, 'Polygon conversion failed before training.'
    model=Refiner(pretrained=True,encoder_weights=args.encoder_weights).cuda().train()
    x=torch.rand(2,3,args.size,args.size,device='cuda');y=torch.zeros(2,1,args.size,args.size,device='cuda');y[0,0,80:180,100:110]=1
    valid=torch.ones_like(y);positive=torch.tensor([1.,0.],device='cuda')
    logits,presence=model(x);loss=objective(logits,presence,y,valid,positive)
    assert logits.shape==y.shape and presence.shape==(2,) and torch.isfinite(loss)
    loss.backward();gradients=[p.grad for p in model.parameters() if p.grad is not None];assert gradients and all(torch.isfinite(g).all() for g in gradients)
    mask=np.zeros((2048,2048),np.uint8);mask[1:20,2040:2046]=1
    assert np.array_equal(coco.decode(encode_mask(mask)),mask)
    del model,x,y,valid,positive,logits,presence,loss;torch.cuda.empty_cache()
    result=YOLO(args.detector_weights).predict(args.image,imgsz=512,device=0,retina_masks=True,verbose=False)[0]
    if result.masks is not None:assert tuple(result.masks.data.shape[-2:])==tuple(result.orig_shape)
    print(json.dumps({'smoke_test':'passed','gpu':torch.cuda.get_device_name(0),'refiner_backward':True,'rle_round_trip':True,'polygon_conversion':True,'detector_api':True}))

if __name__=='__main__':main()
