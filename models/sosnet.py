import torch
import torch.nn as nn

from models.tfeat_utils import describe_opencv


class SOSNet32x32(nn.Module):
    """
    128-dimensional SOSNet model definition trained on 32x32 patches
    """
    def __init__(self, dim_desc=128, drop_rate=0.1, eps_fea_norm=1e-5, eps_l2_norm=1e-10):
        super(SOSNet32x32, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate
        self.eps_l2_norm = eps_l2_norm

        norm_layer = nn.BatchNorm2d
        activation = nn.ReLU()

        self.layers = nn.Sequential(
            nn.InstanceNorm2d(1, affine=False, eps=eps_fea_norm),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32, affine=False, eps=eps_fea_norm),
            activation,

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(64, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64, affine=False, eps=eps_fea_norm),
            activation,

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
            activation,

            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
        )

        self.desc_norm = nn.Sequential(
            nn.LocalResponseNorm(2 * self.dim_desc, alpha=2 * self.dim_desc, beta=0.5, k=0)
        )

        return

    def forward(self, patch):
        descr = self.desc_norm(self.layers(patch) + self.eps_l2_norm)
        descr = descr.view(descr.size(0), -1)
        return descr

    
class SOSNet:
    def __init__(self, weight_file,
                 device=torch.device('cuda' if torch.cuda.is_available else 'cpu')):
        sosnet32 = SOSNet32x32()
        sosnet32.to(device)
        sosnet32.load_state_dict(torch.load(weight_file))
        sosnet32.eval()
        self.net = sosnet32
        
        self.dist_thresh = 1.2
        self.mag_factor = 3
        self.N = 32
        
    def compute(self, im, kp):
        des = describe_opencv(self.net, im, kp, self.N, self.mag_factor)
        return kp, des