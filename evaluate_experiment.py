"""Freeze calibration settings, compare held-out systems, export morphology examples."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as coco
from filament.data import AnnotationIndex
from cached_evaluation import collect,tune,totals,method_key
from metric_core import report,counts,grouped_interval
from pipeline import select_prepared


def preview(rows,index,images,rule,out):
    for previous in out.glob('preview_*.jpg'):previous.unlink()
    # Deterministic low-scoring and middle-scoring cases, with labels for the chosen annotator.
    scores=[report(totals([row],rule).sum(0))['pq'] for row in rows];order=np.argsort(scores)
    chosen=list(order[:4])+list(order[len(order)//2:len(order)//2+4]);details=[]
    for n,i in enumerate(chosen):
        row=rows[int(i)];name=row['filename'];image_id=sorted(index.by_filename[name])[0]
        gray=cv2.imread(str(Path(images)/name),cv2.IMREAD_GRAYSCALE);gray=cv2.resize(gray,(640,640));base=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        panels=[]
        pred=select_prepared(row['methods'][method_key(rule)],rule)
        for label,rles in [('Ground truth annotator '+str(image_id),index.rles(image_id)),('Frozen prediction',pred)]:
            panel=base.copy()
            for j,rle in enumerate(rles):
                mask=cv2.resize(coco.decode(rle).astype(np.uint8),(640,640),interpolation=cv2.INTER_NEAREST)
                contours=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
                color=[(0,255,255),(0,255,0),(255,150,0),(180,50,255)][j%4];cv2.drawContours(panel,contours,-1,color,1)
            cv2.putText(panel,label,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            panels.append(panel)
        canvas=np.concatenate(panels,axis=1);cv2.imwrite(str(out/f'preview_{n+1:02d}_{Path(name).stem}.jpg'),canvas)
        details.append({'filename':name,'pooled_image_pq':scores[int(i)],'displayed_annotator_image':image_id,'selection':'lowest PQ' if n<4 else 'middle PQ'})
    pd.DataFrame(details).to_csv(out/'preview_cases.csv',index=False)


def main():
    p=argparse.ArgumentParser()
    for name in ('experiment','annotations','images'):p.add_argument('--'+name,required=True)
    p.add_argument('--imgsz',type=int,default=1536);p.add_argument('--no-tta',action='store_true')
    args=p.parse_args();out=Path(args.experiment);index=AnnotationIndex(args.annotations)
    partitions=json.loads((out/'partitions.json').read_text())['partitions'];detector=out/'detector_selected.pt';refiner=out/'refiner_best.pt'
    calibration=collect(partitions['calibration'],args.images,index,args.annotations,detector,refiner,args.imgsz,out/'calibration_cache',not args.no_tta)
    chosen,trials=tune(calibration,True);detector_rule,detector_trials=tune(calibration,False)
    refinement_trials=[r for r in trials if r['mode']=='refiner'];best_refiner=max(refinement_trials,key=lambda r:(r['pq'],-r['fp']))
    refiner_rule={k:best_refiner[k] for k in ('mode','mask_threshold','confidence','min_area')}
    settings={**chosen,'imgsz':args.imgsz,'tta':not args.no_tta,'overlap_policy':'all candidates above base confidence 0.03 get score-ordered disjoint ownership before final filters',
              'base_min_area':8,'selection_partition':'calibration'}
    (out/'selected_settings.json').write_text(json.dumps(settings,indent=2));pd.DataFrame(trials).to_csv(out/'calibration_trials.csv',index=False)
    print('Frozen before held-out evaluation:',json.dumps(settings),flush=True)
    validation=collect(partitions['validation'],args.images,index,args.annotations,detector,refiner,args.imgsz,out/'validation_cache',not args.no_tta)
    baseline_counts=totals(validation,detector_rule);refiner_counts=totals(validation,refiner_rule);chosen_counts=totals(validation,chosen)
    comparisons=[{'system':'detector_only','selected_on_calibration':chosen['mode']=='detector',**report(baseline_counts.sum(0))},
                 {'system':'detector_plus_refiner','selected_on_calibration':chosen['mode']=='refiner',**report(refiner_counts.sum(0))}]
    pd.DataFrame(comparisons).to_csv(out/'validation_comparison.csv',index=False)
    folds=pd.read_csv(out/'folds.csv');groups=folds.set_index('filename').temporal_group.to_dict()
    interval=grouped_interval(baseline_counts,refiner_counts,[groups[r['filename']] for r in validation])
    summary={'fold':json.loads((out/'partitions.json').read_text())['fold'],'selected_settings':settings,
        'validation':report(chosen_counts.sum(0)),'comparisons':comparisons,'refiner_minus_detector_pq_group_bootstrap_95pct':interval,
        'validation_images':len(validation),'calibration_images':len(calibration),
        'interpretation':'Model and postprocessing settings are selected on the calibration partition, then applied unchanged to the held-out validation partition. The bootstrap interval summarizes temporal-group variability.'}
    (out/'validation_summary.json').write_text(json.dumps(summary,indent=2))
    per_image=[];per_annotator=[];matched=[];key=method_key(chosen)
    for row,total in zip(validation,chosen_counts):
        per_image.append({'filename':row['filename'],**report(total)})
        candidates=row['methods'][key];keep=np.array([r['score']>=chosen['confidence'] and r['area']>=chosen['min_area'] for r in candidates],bool)
        for annotation in row['annotations']:
            iou=np.asarray(annotation['iou'][key],float).reshape(annotation['gt_count'],len(candidates))[:,keep]
            per_annotator.append({'filename':row['filename'],'annotator_image':annotation['image_id'],**report(counts(iou))})
            for value in iou[iou>.5]:matched.append({'filename':row['filename'],'annotator_image':annotation['image_id'],'iou':float(value),'dice':float(2*value/(1+value))})
    pd.DataFrame(per_image).to_csv(out/'validation_per_image.csv',index=False)
    pd.DataFrame(per_annotator).to_csv(out/'validation_per_annotator.csv',index=False)
    pd.DataFrame(matched,columns=['filename','annotator_image','iou','dice']).to_csv(out/'validation_matched_overlaps.csv',index=False)
    preview(validation,index,args.images,chosen,out)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
