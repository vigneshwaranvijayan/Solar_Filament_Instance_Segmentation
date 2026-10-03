"""Decode every test mask, validate IDs and enforce disjoint instances."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pycocotools import mask as coco


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--submission',required=True)
    parser.add_argument('--images',required=True)
    args=parser.parse_args()
    table=pd.read_csv(args.submission,dtype=str,keep_default_na=False)
    if list(table.columns)!=['filament_id','segmentation_rle']:
        raise ValueError('Incorrect submission columns')
    if table.filament_id.duplicated().any():raise ValueError('Duplicate filament IDs')
    if not table.filament_id.str.fullmatch(r'[0-9]{14}[A-Za-z]{2}_[1-9][0-9]*').all():raise ValueError('Invalid filament IDs')
    names={p.stem for p in Path(args.images).iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}}
    if not names:raise ValueError('No test images found')
    table['image_id']=table.filament_id.str.rsplit('_',n=1).str[0]
    if not set(table.image_id)<=names:raise ValueError('Submission contains unknown image IDs')
    areas=[]
    for name,group in table.groupby('image_id',sort=False):
        occupied=np.zeros((2048,2048),bool)
        for row in group.itertuples(index=False):
            rle={'size':[2048,2048],'counts':row.segmentation_rle.encode('ascii')}
            mask=coco.decode(rle).astype(bool)
            area=int(mask.sum())
            if mask.shape!=(2048,2048) or area==0:raise ValueError(f'Empty or invalid mask: {row.filament_id}')
            if np.any(occupied&mask):raise ValueError(f'Overlapping predicted instances: {name}')
            occupied|=mask;areas.append(area)
    report={'status':'passed','test_images':len(names),'predicted_instances':len(table),
            'images_with_predictions':int(table.image_id.nunique()),
            'images_without_predictions':sorted(names-set(table.image_id)),
            'minimum_instance_area':min(areas) if areas else None,
            'all_masks_decoded':True,'prediction_instances_disjoint':True}
    Path(args.submission).with_suffix('.checks.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
