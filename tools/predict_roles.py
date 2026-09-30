"""Run the experimental role classifier at supplied player centers, without detecting players."""
import argparse
import json
from pathlib import Path
import cv2
import torch
from tools.train_role_pilot import crop_tensor, network


class RolePredictor:
    def __init__(self, checkpoint):
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.model=network(); self.model.load_state_dict(saved['state_dict']); self.model.eval()
        self.classes=saved['classes']

    def predict(self, image, points):
        height,width=image.shape[:2]
        if not points: return []
        for x,y in points:
            if not 0 <= x < width or not 0 <= y < height:
                raise ValueError('Player centers must lie inside the image')
        tensors=torch.stack([crop_tensor(image,min(767,int(x*768/width)),
                                               min(431,int(y*432/height))) for x,y in points])
        with torch.no_grad(): scores=self.model(tensors).softmax(1).tolist()
        return [dict(point=list(point),role_candidate=self.classes[max(range(len(s)),key=s.__getitem__)],
                     uncalibrated_scores=dict(zip(self.classes,s)),status='experimental_weak_label_model')
                for point,s in zip(points,scores)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--point',type=float,nargs=2,action='append',required=True,metavar=('X','Y'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Choose a new output file')
    image=cv2.imread(str(args.image))
    if image is None: parser.error('Image could not be read')
    result=RolePredictor(args.checkpoint).predict(image,args.point)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
