"""Select among saved detector checkpoints by pooled PQ on calibration only."""
import argparse,json,shutil
from pathlib import Path
import pandas as pd
from filament.data import AnnotationIndex
from cached_evaluation import collect,tune,sha


def main():
    p=argparse.ArgumentParser()
    for name in ('experiment','images','annotations'):p.add_argument('--'+name,required=True)
    p.add_argument('--imgsz',type=int,default=1536);args=p.parse_args();out=Path(args.experiment)
    if (out/'detector_selection.json').exists() and (out/'detector_selected.pt').exists():print('Detector PQ selection already complete.');return
    names=json.loads((out/'partitions.json').read_text())['partitions']['calibration'];index=AnnotationIndex(args.annotations)
    candidates=sorted((out/'detector'/'weights').glob('epoch*.pt'))
    candidates += [out/'detector'/'weights'/'best.pt',out/'detector'/'weights'/'last.pt']
    candidates=[p for p in candidates if p.is_file()];seen=set();trials=[];selected=None;best=-1.
    for checkpoint in candidates:
        digest=sha(checkpoint)
        if digest in seen:continue
        seen.add(digest);print('Calibration PQ for',checkpoint.name,flush=True)
        rows=collect(names,args.images,index,args.annotations,checkpoint,None,args.imgsz,out/'detector_selection_cache'/digest[:12],tta=False)
        rule,table=tune(rows,False);score=max(r['pq'] for r in table)
        trials.append({'checkpoint':str(checkpoint),'pq':score,'rule':rule})
        if score>best:best=score;selected=checkpoint
    if selected is None:raise RuntimeError('No detector checkpoint available.')
    shutil.copy2(selected,out/'detector_selected.pt')
    (out/'detector_selection.json').write_text(json.dumps({'selected':str(selected),'calibration_pq':best,'trials':trials},indent=2))
    print('Selected by calibration PQ:',selected,'PQ:',best,flush=True)

if __name__=='__main__':main()
