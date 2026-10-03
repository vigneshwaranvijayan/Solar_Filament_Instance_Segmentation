"""Unify split ZIP uploads or extracted competition folders without mixing train/test."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path
import pandas as pd
from filament.data import AnnotationIndex,make_folds

def main():
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output",required=True);p.add_argument("--folds",default="reports/folds.csv");args=p.parse_args()
    source=Path(args.input).resolve();out=Path(args.output).resolve()
    if source==out or source in out.parents:raise ValueError("Output must be outside the input tree")
    candidates=list(source.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if not candidates:raise FileNotFoundError("Attach the training JSON separately or use an extracted competition dataset")
    if len({hashlib.sha256(p.read_bytes()).hexdigest() for p in candidates})!=1:raise ValueError("Conflicting annotation JSON copies")
    out.mkdir(parents=True,exist_ok=True);annotations=out/candidates[0].name;shutil.copyfile(candidates[0],annotations)
    index=AnnotationIndex(annotations);train_names=set(index.filenames)
    for split in ["train_images","test_images"]:(out/split).mkdir(exist_ok=True)
    def place(name,split,raw=None,path=None):
        dest=out/split/name
        if dest.exists():
            incoming=raw if raw is not None else path.read_bytes()
            if hashlib.sha256(dest.read_bytes()).digest()!=hashlib.sha256(incoming).digest():raise ValueError(f"Conflicting image: {name}")
        elif raw is not None:dest.write_bytes(raw)
        else:dest.symlink_to(path.resolve())
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".jpg",".jpeg",".png"}:continue
        if path.name in train_names:place(path.name,"train_images",path=path)
        elif "test_images" in path.parts:place(path.name,"test_images",path=path)
    for archive in sorted(source.rglob("*.zip")):
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                name=Path(item.filename).name
                if item.is_dir() or Path(name).suffix.lower() not in {".jpg",".jpeg",".png"}:continue
                if name in train_names:split="train_images"
                elif "test_images" in Path(item.filename).parts or archive.name.lower()=="test.zip":split="test_images"
                else:continue
                place(name,split,raw=z.read(item))
    missing=train_names-{p.name for p in (out/"train_images").glob("*")}
    if missing:raise ValueError(f"Missing {len(missing)} training images; examples: {sorted(missing)[:5]}")
    test_names={p.name for p in (out/"test_images").glob("*")}
    if train_names&test_names:raise ValueError("Training/test filename overlap")
    folds=Path(args.folds);folds.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(make_folds(index)).to_csv(folds,index=False)
    print(json.dumps({"train_images":len(train_names),"test_images":len(test_names),"annotations":str(annotations),"train_directory":str(out/"train_images"),"test_directory":str(out/"test_images")},indent=2))

if __name__=="__main__":main()
