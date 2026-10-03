"""Test inference uses test images and frozen settings; no annotation input."""
import argparse,importlib.metadata,json,time
from pathlib import Path
import pandas as pd
from pipeline import Pipeline,prepare_candidates,select_prepared
from cached_evaluation import sha,atomic_json


def main():
    p=argparse.ArgumentParser()
    for name in ('experiment','images','out'):p.add_argument('--'+name,required=True)
    args=p.parse_args();experiment=Path(args.experiment);settings=json.loads((experiment/'selected_settings.json').read_text())
    detector=experiment/'detector_selected.pt';refiner=experiment/'refiner_best.pt' if settings['mode']=='refiner' else None
    descriptor={'detector':sha(detector),'refiner':sha(refiner) if refiner else None,'runtime_versions':{p:importlib.metadata.version(p) for p in ('torch','torchvision','ultralytics','numpy','pycocotools')},'settings':settings,'version':'detref-v5-1','source_sha256':{f:sha(Path(__file__).parent/f) for f in ('pipeline.py','patches.py','geometry.py','refiner.py')}}
    pipeline=None;rows=[];empty=[];names=sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {'.jpeg','.jpg','.png'})
    if len(names)!=180:raise ValueError(f'Expected 180 competition test images, found {len(names)}')
    cache_dir=experiment/'test_cache';cache_dir.mkdir(exist_ok=True);start=time.monotonic()
    for i,path in enumerate(names):
        fingerprint={**descriptor,'image_sha256':sha(path)};cache=cache_dir/(path.stem+'.json');data=json.loads(cache.read_text()) if cache.exists() else None
        if data is None or data.get('fingerprint')!=fingerprint:
            if pipeline is None:pipeline=Pipeline(detector,refiner,settings['imgsz'],settings['tta'])
            candidates=pipeline.candidates(path);prepared=prepare_candidates(candidates,settings['mode'],settings['mask_threshold'])
            rles=select_prepared(prepared,settings);data={'fingerprint':fingerprint,'rles':rles};atomic_json(cache,data)
        rles=data['rles'];rows.extend({'filament_id':f'{path.stem}_{j}','segmentation_rle':r['counts']} for j,r in enumerate(rles,1))
        if not rles:empty.append(path.name)
        if (i+1)%20==0:print(f'Predicted {i+1}/{len(names)} | elapsed {(time.monotonic()-start)/60:.1f} min',flush=True)
    out=Path(args.out);pd.DataFrame(rows,columns=['filament_id','segmentation_rle']).to_csv(out,index=False)
    summary={'images':len(names),'predicted_instances':len(rows),'images_with_no_predictions':empty,'seconds':time.monotonic()-start,'settings':settings,'annotations_used':False}
    out.with_suffix('.metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
