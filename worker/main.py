import io,os
from fastapi import FastAPI,Request,HTTPException,Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image,UnidentifiedImageError
import numpy as np
import torch
from model import StyleNet
app=FastAPI();app.add_middleware(CORSMiddleware,allow_origins=os.getenv('ORIGINS','http://localhost:8080').split(','),allow_methods=['POST','GET'],allow_headers=['*'])
device='cuda' if torch.cuda.is_available() else 'cpu';model=None
path=os.getenv('MODEL_PATH','style.pt')
if os.path.isfile(path):
    model=StyleNet().to(device);model.load_state_dict(torch.load(path,map_location=device,weights_only=True));model.eval()
@app.get('/health')
def health(): return {'ready':model is not None,'device':device}
@app.post('/infer')
async def infer(request:Request):
    if model is None: raise HTTPException(503,'Trained model required')
    data=bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data)>5_000_000: raise HTTPException(413,'Frame too large')
    try:
        im=Image.open(io.BytesIO(data));im.thumbnail((1280,720));im=im.convert('RGB');size=im.size
    except (UnidentifiedImageError,OSError): raise HTTPException(400,'Invalid image')
    x=torch.from_numpy(np.array(im.resize((256,256))).copy()).permute(2,0,1).float().unsqueeze(0).to(device)/255
    with torch.inference_mode(): y=model(x)[0].permute(1,2,0).cpu().numpy()
    out=io.BytesIO();Image.fromarray((y.clip(0,1)*255).astype('uint8')).resize(size).save(out,format='JPEG');return Response(out.getvalue(),media_type='image/jpeg')
