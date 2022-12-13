import os
import torch
from torch import nn

from models.tfeat_utils import describe_opencv

class TNet(nn.Module):
    """
    TFeat model definition
    """
    def __init__(self):
        super(TNet, self).__init__()
        self.features = nn.Sequential(
            nn.InstanceNorm2d(1, affine=False),
            nn.Conv2d(1, 32, kernel_size=7),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=6),
            nn.Tanh()
        )
        self.descr = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.descr(x)
        return x


def eva_tfeat(device, net_name='liberty', weight_root='weights'):
    TFeat = TNet()
    TFeat.to(device)
    TFeat.load_state_dict(
        torch.load(os.path.join(weight_root, "tfeat-" + net_name + ".params")))
    TFeat.eval()
    return TFeat


class TFeat:
    def __init__(self, weight_file, device=torch.device('cuda' if torch.cuda.is_available else 'cpu'),
                 ks=11, c=4):
        TFeat = TNet().to(device)
        TFeat.load_state_dict(torch.load(weight_file))
        TFeat.eval()
        self.net = TFeat
        
        self.dist_thresh = 8
        self.mag_factor = 3
        self.N = 32
        self.set_params(ks=ks)
        
    def set_params(self, ks):
        self.ks = ks
        
    def compute(self, im, kp):
        des = describe_opencv(self.net, im, kp, self.N, self.mag_factor)
        return kp, des
        