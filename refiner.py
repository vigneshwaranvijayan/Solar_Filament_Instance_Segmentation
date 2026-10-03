"""ResNet18 encoder, GroupNorm decoder, mask and foreground-presence heads."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18,ResNet18_Weights


class Block(nn.Sequential):
    def __init__(self,inp,out):
        super().__init__(nn.Conv2d(inp,out,3,padding=1,bias=False),nn.GroupNorm(8,out),nn.SiLU(),
                         nn.Conv2d(out,out,3,padding=1,bias=False),nn.GroupNorm(8,out),nn.SiLU())


class Refiner(nn.Module):
    def __init__(self,pretrained=False,encoder_weights=None):
        super().__init__()
        backbone=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained and not encoder_weights else None)
        if encoder_weights:backbone.load_state_dict(torch.load(encoder_weights,map_location='cpu',weights_only=True))
        self.stem=nn.Sequential(backbone.conv1,backbone.bn1,backbone.relu)
        self.pool=backbone.maxpool;self.e1=backbone.layer1;self.e2=backbone.layer2;self.e3=backbone.layer3;self.e4=backbone.layer4
        self.d3=Block(512+256,256);self.d2=Block(256+128,128);self.d1=Block(128+64,64);self.d0=Block(64+64,32)
        self.mask=nn.Sequential(Block(32,32),nn.Conv2d(32,1,1))
        self.presence=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(512,1))
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
    def train(self,mode=True):
        super().train(mode)
        # Frozen encoder running statistics support small instance batches.
        for module in self.modules():
            if isinstance(module,nn.BatchNorm2d):module.eval()
        return self
    def forward(self,x):
        size=x.shape[-2:];x=(x-self.mean)/self.std;s=self.stem(x)
        a=self.e1(self.pool(s));b=self.e2(a);c=self.e3(b);d=self.e4(c)
        up=lambda x,y:F.interpolate(x,size=y.shape[-2:],mode='bilinear',align_corners=False)
        y=self.d3(torch.cat([up(d,c),c],1));y=self.d2(torch.cat([up(y,b),b],1))
        y=self.d1(torch.cat([up(y,a),a],1));y=self.d0(torch.cat([up(y,s),s],1))
        return F.interpolate(self.mask(y),size=size,mode='bilinear',align_corners=False),self.presence(d).flatten()


def objective(logits,presence,truth,valid,positive):
    valid=valid.float();prob=logits.sigmoid()
    dilated=F.max_pool2d(truth,3,1,1);eroded=-F.max_pool2d(-truth,3,1,1)
    weight=valid*(1+3*(dilated-eroded).clamp(0,1))
    bce=(F.binary_cross_entropy_with_logits(logits,truth,reduction='none')*weight).sum()/weight.sum().clamp_min(1)
    dims=(1,2,3);intersection=(prob*truth*valid).sum(dims)
    dice=1-(2*intersection+1)/((prob*valid).sum(dims)+(truth*valid).sum(dims)+1)
    positive_dice=dice[positive>0].mean() if (positive>0).any() else logits.sum()*0
    detection=F.binary_cross_entropy_with_logits(presence,positive.float())
    return bce+positive_dice+.3*detection
