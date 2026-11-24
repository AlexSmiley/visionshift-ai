import argparse
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from model import StyleNet
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('target');p.add_argument('--epochs',type=int,default=10);p.add_argument('--output',default='style.pt');a=p.parse_args()
def load(p): return torch.from_numpy(np.array(Image.open(p).convert('RGB').resize((256,256))).copy()).permute(2,0,1).float().unsqueeze(0)/255
pairs=[(x,Path(a.target)/x.name) for x in Path(a.source).iterdir() if x.suffix.lower() in ('.jpg','.png','.jpeg') and (Path(a.target)/x.name).is_file()]
if not pairs: raise ValueError('No matching image pairs')
d='cuda' if torch.cuda.is_available() else 'cpu';m=StyleNet().to(d);o=torch.optim.Adam(m.parameters())
for epoch in range(a.epochs):
    for x,y in pairs:
        o.zero_grad();loss=torch.nn.functional.l1_loss(m(load(x).to(d)),load(y).to(d));loss.backward();o.step()
    print(epoch,float(loss))
torch.save(m.state_dict(),a.output)
torch.onnx.export(m.cpu().eval(),torch.zeros(1,3,256,256),a.output+'.onnx',input_names=['image'],output_names=['styled'])
