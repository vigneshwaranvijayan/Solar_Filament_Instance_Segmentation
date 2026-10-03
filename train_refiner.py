"""Balanced positive/negative patch training with epoch recovery checkpoints."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler
from refiner import Refiner,objective


class Patches(Dataset):
    def __init__(self,rows,augment=False,seed=2026):self.rows=rows;self.augment=augment;self.epoch=0;self.seed=seed
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        with np.load(row['path']) as data:x=data['x'].astype(np.float32)/255.;y=data['target'][None].astype(np.float32);valid=data['valid'][None].astype(np.float32)
        if self.augment:
            rng=np.random.default_rng(self.seed+self.epoch*100000+i);k=int(rng.integers(4))
            x,y,valid=[np.rot90(a,k,axes=(-2,-1)) for a in (x,y,valid)]
            if rng.random()<.5:x,y,valid=[a[...,::-1] for a in (x,y,valid)]
            x[:2]=np.clip(x[:2]*rng.uniform(.85,1.15),0,1)
        return tuple(torch.from_numpy(np.ascontiguousarray(a)) for a in (x,y,valid))+(torch.tensor(float(row['positive'])),)


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--encoder-weights')
    p.add_argument('--epochs',type=int,default=15);p.add_argument('--steps',type=int,default=150);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--seed',type=int,default=2026);args=p.parse_args();out=Path(args.experiment)
    if (out/'refiner_complete.json').exists():print('Refiner stage complete; reusing saved model.');return
    if not torch.cuda.is_available():raise RuntimeError('Enable a GPU.')
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    payload=json.loads((out/'patches.json').read_text());train=[r for r in payload['records'] if r['split']=='train'];val=[r for r in payload['records'] if r['split']=='calibration']
    npos=sum(r['positive'] for r in train);nneg=len(train)-npos
    if not npos or not nneg or not val:raise ValueError('Need positive, negative and calibration patches.')
    last=out/'refiner_last.pt';model=Refiner(pretrained=not last.exists(),encoder_weights=args.encoder_weights if not last.exists() else None).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=1e-5)
    scaler=torch.amp.GradScaler('cuda');start=0;best=float('inf');bad=0;history=[]
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=True);model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler'])
        start=saved['epoch'];best=saved['best'];bad=saved['bad'];history=saved['history']
    weights=[.75/npos if r['positive'] else .25/nneg for r in train]
    ds=Patches(train,True,args.seed);vd=Patches(val,False,args.seed)
    vl=DataLoader(vd,batch_size=args.batch,shuffle=False,num_workers=2,pin_memory=True)
    for epoch in range(start,args.epochs):
        began=time.monotonic();ds.epoch=epoch;generator=torch.Generator().manual_seed(args.seed+epoch)
        sampler=WeightedRandomSampler(weights,args.steps*args.batch,replacement=True,generator=generator)
        loader=DataLoader(ds,batch_size=args.batch,sampler=sampler,num_workers=2,pin_memory=True)
        model.train();train_loss=0.
        for x,y,valid,positive in loader:
            x,y,valid,positive=[a.cuda(non_blocking=True) for a in (x,y,valid,positive)];optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):mask,presence=model(x);loss=objective(mask,presence,y,valid,positive)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss; recovery checkpoint remains at previous epoch.')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            scaler.step(optimizer);scaler.update();train_loss+=float(loss.detach())
        model.eval();val_loss=0.;n=0
        with torch.inference_mode():
            for x,y,valid,positive in vl:
                x,y,valid,positive=[a.cuda(non_blocking=True) for a in (x,y,valid,positive)]
                with torch.autocast('cuda',dtype=torch.float16):mask,presence=model(x);loss=objective(mask,presence,y,valid,positive)
                val_loss+=float(loss)*len(x);n+=len(x)
        val_loss/=max(1,n);scheduler.step();improved=val_loss<best-1e-5
        if improved:best=val_loss;bad=0
        else:bad+=1
        row={'epoch':epoch+1,'train_loss':train_loss/args.steps,'calibration_patch_loss':val_loss,'seconds':time.monotonic()-began};history.append(row)
        print(json.dumps({**row,'estimated_remaining_refiner_hours':row['seconds']*max(0,args.epochs-epoch-1)/3600}),flush=True)
        checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
            'epoch':epoch+1,'best':best,'bad':bad,'history':history,'size':payload['size'],'config':vars(args)}
        torch.save(checkpoint,last.with_suffix('.tmp'));last.with_suffix('.tmp').replace(last)
        if improved:torch.save({'model':model.state_dict(),'size':payload['size'],'epoch':epoch+1},out/'refiner_best.pt')
        (out/'refiner_history.json').write_text(json.dumps(history,indent=2))
        if bad>=5:print('Refiner stopped after five calibration-loss checks without improvement.',flush=True);break
    (out/'refiner_complete.json').write_text(json.dumps({'epochs_finished':len(history),'best_calibration_patch_loss':best},indent=2))

if __name__=='__main__':main()
