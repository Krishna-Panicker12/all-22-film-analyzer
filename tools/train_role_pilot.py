"""Experimental crop classifier trained on explicit weak role labels, split by play."""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['S', 'CB', 'LB']


def network():
    return nn.Sequential(nn.Conv2d(3,16,5,padding=2),nn.ReLU(),nn.MaxPool2d(2),
        nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
        nn.Conv2d(32,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((3,3)),
        nn.Flatten(),nn.Dropout(.3),nn.Linear(288,3))


def crop_tensor(image, x, y):
    image = cv2.resize(image,(768,432))
    # Context around a reviewed point, not a fabricated detector bounding box.
    image = cv2.copyMakeBorder(image,80,80,80,80,cv2.BORDER_REFLECT_101)
    crop = image[y:y+160,x:x+160]
    crop = cv2.cvtColor(cv2.resize(crop,(64,64)),cv2.COLOR_BGR2RGB)
    return torch.from_numpy(crop.transpose(2,0,1).copy()).float()/255


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--allow-unverified',action='store_true')
    args=parser.parse_args()
    if not args.allow_unverified:
        parser.error('This pilot requires --allow-unverified; labels are NOT verified ground truth.')
    if args.epochs < 1: parser.error('epochs must be positive')
    if args.output.exists(): parser.error('Choose a new output directory')
    torch.set_num_threads(2)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    source=ROOT/'data/metadata/player_role_annotations.json'
    labels=json.loads(source.read_text())
    # Whole clips, never neighboring frames, define the split. Only one game exists.
    splits={'train':[1,2,3,5,6,8,10,13,14], 'val':[4,7,11], 'test':[9,12,15]}
    samples={}; manifest={}
    for split, numbers in splits.items():
        xs=[]; ys=[]; records=[]
        for n in numbers:
            play=f'pass_{n:03d}'
            for phase,points in labels['plays'][play].items():
                path=ROOT/f'data/defensive_shells/player_roles/{play}_{phase}_original.jpg'
                image=cv2.imread(str(path))
                if image is None: raise ValueError(f'Missing image {path}')
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                for role,x,y in points:
                    xs.append(crop_tensor(image,x,y)); ys.append(CLASSES.index(role))
                    records.append(dict(play=play,phase=phase,role=role,point=[x,y],
                        image=str(path.relative_to(ROOT)),sha256=digest))
        samples[split]=(torch.stack(xs),torch.tensor(ys))
        manifest[split]=records
    args.output.mkdir(parents=True)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    model=network(); optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    counts=torch.bincount(samples['train'][1],minlength=3)
    loss_fn=nn.CrossEntropyLoss(weight=counts.sum()/(3*counts.float()))
    best_loss=float('inf'); best=None; history=[]; best_epoch=0
    for epoch in range(args.epochs):
        model.train(); x,y=samples['train']; order=torch.randperm(len(y)); losses=[]
        for batch in order.split(16):
            optimizer.zero_grad(); loss=loss_fn(model(x[batch]),y[batch])
            loss.backward(); optimizer.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad(): val_loss=loss_fn(model(samples['val'][0]),samples['val'][1]).item()
        history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),val_loss=val_loss))
        if val_loss < best_loss:
            best_loss=val_loss; best=copy.deepcopy(model.state_dict()); best_epoch=epoch+1
        if (epoch+1)%10==0: print(f'Epoch {epoch+1}: validation loss {val_loss:.4f}',flush=True)
    model.load_state_dict(best); model.eval()
    metrics={}
    baseline=int(counts.argmax())
    with torch.no_grad():
        for split,(x,y) in samples.items():
            predicted=model(x).argmax(1); confusion=torch.zeros(3,3,dtype=torch.int64)
            for truth,guess in zip(y,predicted): confusion[truth,guess]+=1
            metrics[split]=dict(samples=len(y),accuracy=float((predicted==y).float().mean()),
                majority_baseline_accuracy=float((y==baseline).float().mean()),
                confusion_matrix=confusion.tolist(),
                recall_by_class={name:float(confusion[i,i]/confusion[i].sum()) if confusion[i].sum() else None for i,name in enumerate(CLASSES)})
    torch.save(dict(state_dict=best,classes=CLASSES,input_size=64,context_size=160,
        reference_size=[768,432],experimental=True),args.output/'best.pt')
    report=dict(status='complete',seed=42,epochs=args.epochs,best_epoch=best_epoch,
        device='cpu',model='small context-crop CNN, random initialization',
        label_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),splits=splits,
        classes=CLASSES,metrics=metrics,history=history,production_promoted=False,
        limitations=['Weak agent labels, not human ground truth.',
          'Same-game held-out plays do not establish generalization to new games.',
          'Classifies supplied player points; does not locate players or classify coverage.',
          'Tiny dataset; annotated diagrams excluded from training.'])
    (args.output/'training.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(best_epoch=best_epoch,metrics=metrics),indent=2))


if __name__=='__main__': main()
