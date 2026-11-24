import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'worker'))
import torch
from model import StyleNet
def test_shape():
    y=StyleNet()(torch.zeros(1,3,32,32));assert y.shape==(1,3,32,32);assert y.min()>=0 and y.max()<=1
