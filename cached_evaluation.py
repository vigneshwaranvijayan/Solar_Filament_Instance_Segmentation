"""One IoU calculation per method/image; threshold comparisons reuse matrices."""
import hashlib,importlib.metadata,json,time
from pathlib import Path
import numpy as np
from filament.metrics import pairwise_iou
from metric_core import counts,report
from pipeline import Pipeline,prepare_candidates


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data));temp.replace(path)


def method_key(settings):
    return 'detector' if settings['mode']=='detector' else f"refiner_{settings['mask_threshold']}"


def collect(names,images,index,annotations,detector,refiner,imgsz,out,tta=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    descriptor={'detector_sha256':sha(detector),'refiner_sha256':sha(refiner) if refiner else None,
        'runtime_versions':{p:importlib.metadata.version(p) for p in ('torch','torchvision','ultralytics','numpy','pycocotools')},
        'annotations_sha256':sha(annotations),'imgsz':imgsz,'tta':tta,'version':'detref-v5-1',
        'source_sha256':{f:sha(Path(__file__).parent/f) for f in ('pipeline.py','patches.py','geometry.py','refiner.py','cached_evaluation.py')}}
    fingerprint=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest()
    pipeline=None;rows=[];started=time.monotonic()
    for i,name in enumerate(names):
        path=Path(images)/name;key=hashlib.sha256((fingerprint+sha(path)).encode()).hexdigest();cache=out/(Path(name).stem+'.json')
        row=json.loads(cache.read_text()) if cache.exists() else None
        if row is None or row.get('fingerprint')!=key:
            if pipeline is None:pipeline=Pipeline(detector,refiner,imgsz,tta)
            candidates=pipeline.candidates(path);methods={'detector':prepare_candidates(candidates,'detector')}
            if refiner:
                for threshold in (.4,.5,.6):methods[f'refiner_{threshold}']=prepare_candidates(candidates,'refiner',threshold)
            annotations_rows=[]
            for image_id in index.by_filename[name]:
                gt=index.rles(image_id)
                annotations_rows.append({'image_id':image_id,'gt_count':len(gt),
                    'iou':{k:pairwise_iou(gt,[r['rle'] for r in records]).tolist() for k,records in methods.items()}})
            row={'filename':name,'fingerprint':key,'methods':methods,'annotations':annotations_rows};atomic_json(cache,row)
        rows.append(row)
        if (i+1)%20==0 or i+1==len(names):print(f'Cached predictions {i+1}/{len(names)} | elapsed {(time.monotonic()-started)/60:.1f} min',flush=True)
    return rows


def image_counts(row,rule):
    key=method_key(rule);candidates=row['methods'][key]
    keep=np.array([r['score']>=rule['confidence'] and r['area']>=rule['min_area'] for r in candidates],bool)
    result=np.zeros(6)
    for annotation in row['annotations']:
        iou=np.asarray(annotation['iou'][key],float).reshape(annotation['gt_count'],len(candidates))
        result+=counts(iou[:,keep])
    return result


def totals(rows,rule):return np.array([image_counts(row,rule) for row in rows])


def rule_grid(include_refiner=True):
    methods=[('detector',.5)]+([('refiner',t) for t in (.4,.5,.6)] if include_refiner else [])
    return [{'mode':mode,'mask_threshold':threshold,'confidence':confidence,'min_area':minimum}
        for mode,threshold in methods for confidence in (.03,.08,.15,.25,.4) for minimum in (32,128)]


def tune(rows,include_refiner=True):
    trials=[{**rule,**report(totals(rows,rule).sum(0))} for rule in rule_grid(include_refiner)]
    # Fewer false positives breaks exact PQ ties; detector wins identical scores.
    ordered=sorted(trials,key=lambda r:(-r['pq'],r['fp'],r['mode']!='detector',r['confidence'],r['min_area']))
    best={k:ordered[0][k] for k in ('mode','mask_threshold','confidence','min_area')}
    return best,trials
