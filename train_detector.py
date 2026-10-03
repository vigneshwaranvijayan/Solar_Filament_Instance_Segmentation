"""Train YOLO11 segmentation, preserving an unstripped recovery checkpoint."""
import argparse,json,shutil,time
from pathlib import Path
import torch
from ultralytics import YOLO,settings


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--weights',default='yolo11m-seg.pt')
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--imgsz',type=int,default=1536)
    p.add_argument('--batch',type=int,default=2);p.add_argument('--seed',type=int,default=2026)
    args=p.parse_args();out=Path(args.experiment);done=out/'detector_complete.json'
    if done.exists():print('Detector stage already complete; reusing saved checkpoints.');return
    if not torch.cuda.is_available():raise RuntimeError('Enable a GPU.')
    settings.update({'wandb':False,'mlflow':False,'comet':False,'clearml':False,'tensorboard':False})
    recovery=out/'detector_resume.pt';started=time.monotonic();epoch_times=[];timer=[started]
    def preserve(trainer):
        shutil.copy2(trainer.last,recovery.with_suffix('.tmp'));recovery.with_suffix('.tmp').replace(recovery)
        now=time.monotonic();epoch_times.append(now-timer[0]);timer[0]=now
        if len(epoch_times)>=2:
            median=sorted(epoch_times[-3:])[len(epoch_times[-3:])//2]
            print(json.dumps({'detector_epoch':trainer.epoch+1,'recent_seconds_per_epoch':median,
                'estimated_remaining_training_hours':max(0,trainer.epochs-trainer.epoch-1)*median/3600}),flush=True)
    model=YOLO(str(recovery) if recovery.exists() else args.weights);model.add_callback('on_model_save',preserve)
    if recovery.exists():
        model.train(resume=True,data=str(out/'detector_data.yaml'),device=0,workers=2)
    else:
        model.train(data=str(out/'detector_data.yaml'),epochs=args.epochs,imgsz=args.imgsz,batch=args.batch,
            device=0,workers=2,project=str(out),name='detector',exist_ok=True,save_period=5,
            optimizer='AdamW',lr0=.001,lrf=.05,cos_lr=True,patience=10,seed=args.seed,
            amp=True,cache=False,mask_ratio=2,overlap_mask=False,
            mosaic=.15,close_mosaic=5,mixup=0.,copy_paste=0.,degrees=15.,translate=.05,scale=.15,
            flipud=.5,fliplr=.5,hsv_h=0.,hsv_s=0.,hsv_v=.10,plots=False)
    done.write_text(json.dumps({'seconds':time.monotonic()-started,'epochs_requested':args.epochs,'imgsz':args.imgsz,'batch':args.batch},indent=2))

if __name__=='__main__':main()
